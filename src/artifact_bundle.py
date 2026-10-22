"""Create and securely provision model bundles for source-only deployments."""

import hashlib
import json
import os
import re
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO

import pandas as pd
import requests

MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
REQUIRED_ARTIFACTS = {
    "churn_model.joblib",
    "model_metadata.json",
    "reference_features.csv",
}
SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ArtifactBundleError(RuntimeError):
    """Raised when model artifact configuration or bundle validation fails."""


def _feature_frame(dataset_path: Path, metadata_path: Path) -> pd.DataFrame:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata.get("features")
    if not isinstance(features, list) or not features or not all(
        isinstance(feature, str) for feature in features
    ):
        raise ArtifactBundleError("Model metadata must define a non-empty feature list.")

    dataset = pd.read_csv(dataset_path)
    missing = set(features) - set(dataset.columns)
    if missing:
        raise ArtifactBundleError(
            f"Reference dataset is missing model features: {sorted(missing)}"
        )
    reference = dataset.loc[:, features].copy()
    if reference.empty:
        raise ArtifactBundleError("Reference dataset must contain at least one row.")
    if "TotalCharges" in reference.columns:
        reference["TotalCharges"] = pd.to_numeric(
            reference["TotalCharges"], errors="coerce"
        ).fillna(0)
    return reference


def create_artifact_bundle(
    dataset_path: Path,
    model_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> str:
    """Package trusted local model files and a feature-only drift baseline."""
    for path in (dataset_path, model_path, metadata_path):
        if not path.is_file():
            raise ArtifactBundleError(f"Required artifact input is missing: {path}")

    reference = _feature_frame(dataset_path, metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as staging:
        staging_dir = Path(staging)
        reference_path = staging_dir / "reference_features.csv"
        reference.to_csv(reference_path, index=False)
        files = {
            "churn_model.joblib": model_path,
            "model_metadata.json": metadata_path,
            "reference_features.csv": reference_path,
        }
        with tarfile.open(output_path, mode="w:gz") as bundle:
            for name, path in files.items():
                bundle.add(path, arcname=name, recursive=False)

    return _sha256_file(output_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_member(member: tarfile.TarInfo, bundle: tarfile.TarFile, destination: Path) -> None:
    if member.name not in REQUIRED_ARTIFACTS:
        raise ArtifactBundleError(f"Unexpected file in model bundle: {member.name}")
    if not member.isfile() or member.size > MAX_ARTIFACT_BYTES:
        raise ArtifactBundleError(f"Invalid model bundle entry: {member.name}")

    source: BinaryIO | None = bundle.extractfile(member)
    if source is None:
        raise ArtifactBundleError(f"Could not read model bundle entry: {member.name}")
    target = destination / member.name
    with source, target.open("xb") as output:
        remaining = member.size
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ArtifactBundleError(f"Truncated model bundle entry: {member.name}")
            output.write(chunk)
            remaining -= len(chunk)
        if source.read(1):
            raise ArtifactBundleError(f"Model bundle entry exceeded declared size: {member.name}")


def _extract_validated_bundle(archive_path: Path, staging_dir: Path) -> None:
    try:
        with tarfile.open(archive_path, mode="r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(REQUIRED_ARTIFACTS) or set(names) != REQUIRED_ARTIFACTS:
                raise ArtifactBundleError(
                    "Model bundle must contain exactly the model, metadata, and reference features."
                )
            if sum(member.size for member in members) > MAX_BUNDLE_BYTES:
                raise ArtifactBundleError("Uncompressed model bundle is too large.")
            for member in members:
                _copy_member(member, bundle, staging_dir)
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, ArtifactBundleError):
            raise
        raise ArtifactBundleError("Could not read the model artifact bundle.") from exc

    try:
        metadata = json.loads(
            (staging_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        features = metadata.get("features")
        if not isinstance(features, list) or not features or not all(
            isinstance(feature, str) for feature in features
        ):
            raise ArtifactBundleError("Model metadata has no feature list.")
        reference = pd.read_csv(staging_dir / "reference_features.csv", nrows=1)
        missing_features = set(features) - set(reference.columns)
        if missing_features:
            raise ArtifactBundleError(
                f"Reference baseline is missing model features: {sorted(missing_features)}"
            )
        if reference.empty:
            raise ArtifactBundleError("Reference baseline must contain at least one row.")
    except (json.JSONDecodeError, OSError, pd.errors.ParserError) as exc:
        if isinstance(exc, ArtifactBundleError):
            raise
        raise ArtifactBundleError("Model bundle metadata or reference CSV is invalid.") from exc


def download_artifact_bundle(
    bundle_url: str,
    expected_sha256: str,
    destination: Path,
    bearer_token: str | None = None,
) -> None:
    """Download, checksum, validate, and atomically install a model bundle."""
    if not bundle_url.lower().startswith("https://"):
        raise ArtifactBundleError("MODEL_BUNDLE_URL must use HTTPS.")
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ArtifactBundleError("MODEL_BUNDLE_SHA256 must be a 64-character SHA-256 hex digest.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = destination.parent / f".{destination.name}-{uuid.uuid4().hex}"
    archive_path = destination.parent / f".{destination.name}-{uuid.uuid4().hex}.tar.gz"
    headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
    try:
        with requests.get(
            bundle_url,
            headers=headers,
            stream=True,
            timeout=(10, 60),
        ) as response:
            response.raise_for_status()
            if not response.url.lower().startswith("https://"):
                raise ArtifactBundleError("Model bundle download redirected away from HTTPS.")
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    announced_size = int(content_length)
                except ValueError as exc:
                    raise ArtifactBundleError(
                        "Model bundle response has an invalid Content-Length."
                    ) from exc
                if announced_size < 0 or announced_size > MAX_BUNDLE_BYTES:
                    raise ArtifactBundleError(
                        "Model artifact bundle exceeds the download size limit."
                    )

            digest = hashlib.sha256()
            total_bytes = 0
            with archive_path.open("xb") as archive:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total_bytes += len(chunk)
                    if total_bytes > MAX_BUNDLE_BYTES:
                        raise ArtifactBundleError(
                            "Model artifact bundle exceeds the download size limit."
                        )
                    digest.update(chunk)
                    archive.write(chunk)

        if digest.hexdigest().lower() != expected_sha256.lower():
            raise ArtifactBundleError("Downloaded model bundle SHA-256 does not match.")

        staging_dir.mkdir()
        _extract_validated_bundle(archive_path, staging_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name in REQUIRED_ARTIFACTS:
            os.replace(staging_dir / name, destination / name)
    except requests.RequestException as exc:
        raise ArtifactBundleError("Could not download the configured model artifact bundle.") from exc
    finally:
        archive_path.unlink(missing_ok=True)
        if staging_dir.exists():
            for path in staging_dir.iterdir():
                path.unlink(missing_ok=True)
            staging_dir.rmdir()


def prepare_runtime_artifacts() -> None:
    """Provision remote artifacts when configured; otherwise require local artifacts."""
    bundle_url = os.getenv("MODEL_BUNDLE_URL", "").strip()
    artifact_dir = Path(
        os.getenv("MODEL_ARTIFACT_DIR", str(PROJECT_ROOT / "models"))
    )
    if bundle_url:
        checksum = os.getenv("MODEL_BUNDLE_SHA256", "").strip()
        token = os.getenv("MODEL_BUNDLE_TOKEN", "").strip() or None
        download_artifact_bundle(bundle_url, checksum, artifact_dir, token)
        return

    model_path = artifact_dir / "churn_model.joblib"
    metadata_path = artifact_dir / "model_metadata.json"
    reference_path = artifact_dir / "reference_features.csv"
    if model_path.is_file() and metadata_path.is_file() and reference_path.is_file():
        return
    if (
        model_path.is_file()
        and metadata_path.is_file()
        and (PROJECT_ROOT / "data" / "telco_customer_churn.csv").is_file()
    ):
        return
    raise ArtifactBundleError(
        "Model assets are not available. Configure MODEL_BUNDLE_URL and "
        "MODEL_BUNDLE_SHA256, or provide a locally trained model and reference dataset."
    )
