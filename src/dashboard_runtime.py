"""In-process prediction and telemetry access for the Streamlit Cloud demo."""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs, urlsplit

import joblib
import pandas as pd
import requests
import sklearn
from pydantic import ValidationError

from app.schemas import CustomerFeatures
from src.artifact_bundle import ArtifactBundleError, PROJECT_ROOT, prepare_runtime_artifacts
from src.monitoring import (
    check_store,
    get_api_metrics,
    get_drift_history,
    get_monitoring_snapshot,
    get_prediction_history,
    initialize_store,
    record_api_request,
    record_prediction,
)

logger = logging.getLogger(__name__)
EMBEDDED_API_URL = "embedded://local"


class DashboardRuntimeError(RuntimeError):
    """Raised when the embedded prediction runtime cannot serve a dashboard request."""


@dataclass
class DashboardRuntime:
    model: Any
    model_metadata: dict[str, Any]
    reference_features: pd.DataFrame
    database_path: Path | str


def embedded_mode_enabled() -> bool:
    return os.getenv("CHURN_DASHBOARD_MODE", "").strip().lower() == "embedded"


def load_dashboard_runtime() -> DashboardRuntime:
    """Load and validate the model and initialize ephemeral Streamlit telemetry."""
    artifact_dir = Path(
        os.getenv(
            "MODEL_ARTIFACT_DIR",
            str(Path(tempfile.gettempdir()) / "churn-model-artifacts"),
        )
    )
    os.environ.setdefault("MODEL_ARTIFACT_DIR", str(artifact_dir))
    if not os.getenv("MODEL_BUNDLE_URL") and not (artifact_dir / "churn_model.joblib").is_file():
        local_model_dir = PROJECT_ROOT / "models"
        if (local_model_dir / "churn_model.joblib").is_file():
            artifact_dir = local_model_dir
            os.environ["MODEL_ARTIFACT_DIR"] = str(artifact_dir)

    try:
        prepare_runtime_artifacts()
        model_path = artifact_dir / "churn_model.joblib"
        metadata_path = artifact_dir / "model_metadata.json"
        reference_path = artifact_dir / "reference_features.csv"
        model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trained_sklearn_version = model_metadata.get("scikit_learn_version")
        if trained_sklearn_version and trained_sklearn_version != sklearn.__version__:
            raise DashboardRuntimeError(
                "Model scikit-learn version does not match runtime: "
                f"trained with {trained_sklearn_version}, running {sklearn.__version__}"
            )
        if reference_path.is_file():
            reference_features = pd.read_csv(reference_path)
        else:
            dataset_path = PROJECT_ROOT / "data" / "telco_customer_churn.csv"
            if not dataset_path.is_file():
                raise DashboardRuntimeError(
                    "Drift reference features are missing from the artifact bundle."
                )
            reference_features = pd.read_csv(dataset_path)
            reference_features["TotalCharges"] = pd.to_numeric(
                reference_features["TotalCharges"], errors="coerce"
            ).fillna(0)
            reference_features = reference_features.drop(
                columns=["Churn", "customerID"], errors="ignore"
            )

        features = model_metadata.get("features", [])
        if not features:
            raise DashboardRuntimeError("Model metadata does not define its features.")
        missing_features = set(features) - set(reference_features.columns)
        if missing_features:
            raise DashboardRuntimeError(
                f"Drift reference baseline is missing model features: {sorted(missing_features)}"
            )
        reference_features = reference_features.loc[:, features]
        database_path = Path(
            os.getenv(
                "PREDICTION_DB_PATH",
                str(Path(tempfile.gettempdir()) / "churn-dashboard.db"),
            )
        )
        initialize_store(database_path)
        return DashboardRuntime(
            model=joblib.load(model_path),
            model_metadata=model_metadata,
            reference_features=reference_features,
            database_path=database_path,
        )
    except (
        ArtifactBundleError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        pd.errors.ParserError,
        requests.RequestException,
    ) as exc:
        raise DashboardRuntimeError(
            "Could not initialize the embedded prediction app. Check the private model "
            "bundle URL, SHA-256, token, and artifact files."
        ) from exc


def model_info(runtime: DashboardRuntime) -> dict[str, Any]:
    model_metadata = runtime.model_metadata
    trained_at = model_metadata.get("trained_at")
    if not trained_at:
        version_parts = model_metadata.get("model_version", "").split(".")
        if len(version_parts) == 3 and len(version_parts[2]) == 14 and version_parts[2].isdigit():
            trained_at = datetime.strptime(
                version_parts[2], "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc).isoformat()
    return {
        "model_name": model_metadata.get("model_name", "unknown"),
        "model_version": model_metadata.get("model_version", "unknown"),
        "trained_at": trained_at,
        "scikit_learn_version": model_metadata.get(
            "scikit_learn_version", "unknown"
        ),
        "target": model_metadata.get("target", "Churn"),
        "features": model_metadata.get("features", []),
        "test_metrics": model_metadata.get("test_results", {}),
    }


def predict_customer(
    runtime: DashboardRuntime,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate, score, and record one privacy-safe prediction."""
    started_at = perf_counter()
    status_code = 200
    try:
        customer = CustomerFeatures.model_validate(payload)
        clean_payload = customer.model_dump()
        expected_features = runtime.model_metadata["features"]
        row = pd.DataFrame([clean_payload], columns=expected_features)
        probability = float(runtime.model.predict_proba(row)[0][1])
        predicted_class = int(probability >= 0.5)
        prediction_id = record_prediction(
            runtime.database_path,
            clean_payload,
            probability,
            runtime.model_metadata.get("model_version", "unknown"),
        )
        return {
            "prediction_id": prediction_id,
            "prediction": predicted_class,
            "prediction_label": (
                "Likely to Churn" if predicted_class == 1 else "Not Likely to Churn"
            ),
            "churn_probability": round(probability, 4),
            "decision_threshold": 0.5,
            "model_version": runtime.model_metadata.get("model_version", "unknown"),
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except ValidationError as exc:
        status_code = 422
        raise DashboardRuntimeError(f"Could not process customer features: {exc}") from exc
    except (ValueError, KeyError) as exc:
        status_code = 422
        logger.exception("Embedded prediction failed validation")
        raise DashboardRuntimeError(
            f"Could not process customer features: {exc}"
        ) from exc
    finally:
        record_api_request(
            runtime.database_path,
            "POST",
            "/predict",
            status_code,
            (perf_counter() - started_at) * 1000,
        )


def get_embedded_json(runtime: DashboardRuntime, path: str) -> dict[str, Any]:
    """Return the data shapes used by the existing dashboard pages."""
    parsed_path = urlsplit(path)
    query = parse_qs(parsed_path.query)
    limit = int(query.get("limit", ["100"])[0])

    if parsed_path.path == "/health":
        check_store(runtime.database_path)
        return {
            "status": "healthy",
            "model_loaded": True,
            "database_connected": True,
        }
    if parsed_path.path == "/model-info":
        return model_info(runtime)
    if parsed_path.path == "/monitoring":
        snapshot = get_monitoring_snapshot(
            runtime.database_path,
            runtime.reference_features,
        )
        snapshot["api_metrics"] = get_api_metrics(runtime.database_path)
        return snapshot
    if parsed_path.path == "/prediction-history":
        return get_prediction_history(runtime.database_path, limit)
    if parsed_path.path == "/drift-history":
        return get_drift_history(runtime.database_path, limit)
    raise DashboardRuntimeError(f"Unsupported embedded dashboard path: {parsed_path.path}")
