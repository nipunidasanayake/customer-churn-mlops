import hashlib
import io
import json
import tarfile

import pandas as pd
import pytest

from src.artifact_bundle import (
    ArtifactBundleError,
    create_artifact_bundle,
    download_artifact_bundle,
)


def _artifact_inputs(tmp_path):
    dataset_path = tmp_path / "training.csv"
    model_path = tmp_path / "churn_model.joblib"
    metadata_path = tmp_path / "model_metadata.json"
    pd.DataFrame(
        {
            "customerID": ["private-customer-1", "private-customer-2"],
            "Churn": ["No", "Yes"],
            "tenure": [12, 24],
            "TotalCharges": ["100.50", ""],
        }
    ).to_csv(dataset_path, index=False)
    model_path.write_bytes(b"trusted test model artifact")
    metadata_path.write_text(
        json.dumps({"features": ["tenure", "TotalCharges"]}),
        encoding="utf-8",
    )
    return dataset_path, model_path, metadata_path


class _BundleResponse:
    def __init__(self, content):
        self.content = content
        self.url = "https://artifacts.example.test/model-bundle.tar.gz"
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.content), chunk_size):
            yield self.content[offset : offset + chunk_size]


def test_artifact_bundle_contains_feature_baseline_without_identifiers(tmp_path):
    inputs = _artifact_inputs(tmp_path)
    bundle_path = tmp_path / "model-bundle.tar.gz"

    checksum = create_artifact_bundle(*inputs, bundle_path)

    with tarfile.open(bundle_path, "r:gz") as bundle:
        assert set(bundle.getnames()) == {
            "churn_model.joblib",
            "model_metadata.json",
            "reference_features.csv",
        }
        reference = pd.read_csv(bundle.extractfile("reference_features.csv"))
    assert "customerID" not in reference.columns
    assert "Churn" not in reference.columns
    assert reference["TotalCharges"].tolist() == [100.5, 0.0]
    assert checksum == hashlib.sha256(bundle_path.read_bytes()).hexdigest()


def test_download_checks_checksum_and_installs_bundle(tmp_path, monkeypatch):
    inputs = _artifact_inputs(tmp_path)
    bundle_path = tmp_path / "model-bundle.tar.gz"
    checksum = create_artifact_bundle(*inputs, bundle_path)
    content = bundle_path.read_bytes()
    requested = {}

    def fake_get(url, headers, stream, timeout):
        requested.update(url=url, headers=headers, stream=stream, timeout=timeout)
        return _BundleResponse(content)

    monkeypatch.setattr("src.artifact_bundle.requests.get", fake_get)
    destination = tmp_path / "installed"
    download_artifact_bundle(
        "https://artifacts.example.test/model-bundle.tar.gz",
        checksum,
        destination,
        "test-token",
    )

    assert requested["headers"] == {"Authorization": "Bearer test-token"}
    assert requested["stream"] is True
    assert (destination / "churn_model.joblib").read_bytes() == inputs[1].read_bytes()
    installed_reference = pd.read_csv(destination / "reference_features.csv")
    assert "customerID" not in installed_reference.columns


def test_download_rejects_http_and_checksum_mismatch(tmp_path, monkeypatch):
    with pytest.raises(ArtifactBundleError, match="HTTPS"):
        download_artifact_bundle(
            "http://artifacts.example.test/model-bundle.tar.gz",
            "a" * 64,
            tmp_path / "installed",
        )

    inputs = _artifact_inputs(tmp_path)
    bundle_path = tmp_path / "model-bundle.tar.gz"
    create_artifact_bundle(*inputs, bundle_path)
    content = bundle_path.read_bytes()
    monkeypatch.setattr(
        "src.artifact_bundle.requests.get",
        lambda *_args, **_kwargs: _BundleResponse(content),
    )
    with pytest.raises(ArtifactBundleError, match="SHA-256"):
        download_artifact_bundle(
            "https://artifacts.example.test/model-bundle.tar.gz",
            "a" * 64,
            tmp_path / "installed",
        )


def test_download_rejects_bundle_with_unexpected_archive_paths(tmp_path, monkeypatch):
    malicious_bundle = io.BytesIO()
    with tarfile.open(fileobj=malicious_bundle, mode="w:gz") as archive:
        member = tarfile.TarInfo("../outside.txt")
        member.size = 4
        archive.addfile(member, io.BytesIO(b"evil"))
    content = malicious_bundle.getvalue()
    checksum = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        "src.artifact_bundle.requests.get",
        lambda *_args, **_kwargs: _BundleResponse(content),
    )

    destination = tmp_path / "installed"
    with pytest.raises(ArtifactBundleError, match="exactly"):
        download_artifact_bundle(
            "https://artifacts.example.test/model-bundle.tar.gz",
            checksum,
            destination,
        )
    assert not (tmp_path / "outside.txt").exists()
