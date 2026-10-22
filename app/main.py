import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from time import perf_counter

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Query, Request
from sqlalchemy.exc import SQLAlchemyError
from app.schemas import CustomerFeatures
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

ROOT = Path(__file__).resolve().parents[1]
MODEL_ARTIFACT_DIR = Path(
    os.getenv("MODEL_ARTIFACT_DIR", str(ROOT / "models"))
)
MODEL_PATH = MODEL_ARTIFACT_DIR / "churn_model.joblib"
METADATA_PATH = MODEL_ARTIFACT_DIR / "model_metadata.json"
REFERENCE_FEATURES_PATH = MODEL_ARTIFACT_DIR / "reference_features.csv"
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
PREDICTION_DB_PATH = Path(
    os.getenv("PREDICTION_DB_PATH", str(ROOT / "runtime" / "predictions.db"))
)
PREDICTION_DATABASE_URL = os.getenv("PREDICTION_DATABASE_URL") or os.getenv("DATABASE_URL")


def prediction_store() -> Path | str:
    return PREDICTION_DATABASE_URL or PREDICTION_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("customer-churn-api")

model = None
metadata = {}
reference_features = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, metadata, reference_features
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    trained_sklearn_version = metadata.get("scikit_learn_version")
    if trained_sklearn_version and trained_sklearn_version != sklearn.__version__:
        raise RuntimeError(
            "Model scikit-learn version does not match runtime: "
            f"trained with {trained_sklearn_version}, running {sklearn.__version__}"
        )
    if REFERENCE_FEATURES_PATH.exists():
        reference_features = pd.read_csv(REFERENCE_FEATURES_PATH)
    elif DATA_PATH.exists():
        reference = pd.read_csv(DATA_PATH)
        reference["TotalCharges"] = pd.to_numeric(
            reference["TotalCharges"], errors="coerce"
        ).fillna(0)
        reference_features = reference.drop(columns=["Churn", "customerID"])
    else:
        raise RuntimeError(
            "Drift reference features not found. Provide model artifacts and a "
            f"reference baseline at {REFERENCE_FEATURES_PATH}."
        )
    expected_features = metadata.get("features", [])
    missing_features = set(expected_features) - set(reference_features.columns)
    if missing_features:
        raise RuntimeError(
            f"Drift reference baseline is missing model features: {sorted(missing_features)}"
        )
    reference_features = reference_features.loc[:, expected_features]
    initialize_store(prediction_store())
    logger.info("Loaded model version %s", metadata.get("model_version", "unknown"))
    yield

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn predictions for the Telco customer churn model.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def collect_api_metrics(request: Request, call_next):
    excluded_paths = {
        "/",
        "/health",
        "/monitoring",
        "/prediction-history",
        "/drift-history",
        "/docs",
        "/openapi.json",
    }
    if request.url.path in excluded_paths or request.url.path.startswith("/docs/"):
        return await call_next(request)

    started_at = perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        response_time_ms = (perf_counter() - started_at) * 1000
        try:
            record_api_request(
                prediction_store(),
                request.method,
                request.url.path,
                status_code,
                response_time_ms,
            )
        except Exception:
            logger.exception("Failed to persist API request metrics")

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        check_store(prediction_store())
    except SQLAlchemyError as exc:
        logger.exception("Telemetry database health check failed")
        raise HTTPException(
            status_code=503,
            detail="Telemetry database is unavailable",
        ) from exc
    return {"status": "healthy", "model_loaded": True, "database_connected": True}

@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    trained_at = metadata.get("trained_at")
    if not trained_at:
        version_parts = metadata.get("model_version", "").split(".")
        if len(version_parts) == 3 and len(version_parts[2]) == 14 and version_parts[2].isdigit():
            trained_at = datetime.strptime(
                version_parts[2], "%Y%m%d%H%M%S"
            ).replace(tzinfo=timezone.utc).isoformat()
    return {
        "model_name": metadata.get("model_name", "unknown"),
        "model_version": metadata.get("model_version", "unknown"),
        "trained_at": trained_at,
        "scikit_learn_version": metadata.get("scikit_learn_version", "unknown"),
        "target": metadata.get("target", "Churn"),
        "features": metadata.get("features", []),
        "test_metrics": metadata.get("test_results", {}),
    }

@app.get("/monitoring")
def monitoring():
    if model is None or reference_features is None:
        raise HTTPException(status_code=503, detail="Model monitoring is not initialized")
    snapshot = get_monitoring_snapshot(prediction_store(), reference_features)
    snapshot["api_metrics"] = get_api_metrics(prediction_store())
    return snapshot

@app.get("/prediction-history")
def prediction_history(limit: int = Query(default=100, ge=1, le=500)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return get_prediction_history(prediction_store(), limit)

@app.get("/drift-history")
def drift_history(limit: int = Query(default=100, ge=1, le=500)):
    return get_drift_history(prediction_store(), limit)

@app.post("/predict")
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    try:
        payload = customer.model_dump()
        expected_features = metadata.get("features", list(payload.keys()))
        row = pd.DataFrame([payload], columns=expected_features)
        probability = float(model.predict_proba(row)[0][1])
        predicted_class = int(probability >= 0.5)
        prediction_id = record_prediction(
            prediction_store(),
            payload,
            probability,
            metadata.get("model_version", "unknown"),
        )
        return {
            "prediction_id": prediction_id,
            "prediction": predicted_class,
            "prediction_label": "Likely to Churn" if predicted_class == 1 else "Not Likely to Churn",
            "churn_probability": round(probability, 4),
            "decision_threshold": 0.5,
            "model_version": metadata.get("model_version", "unknown"),
            "prediction_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except (ValueError, KeyError) as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=422, detail=f"Could not process customer features: {exc}") from exc
    except Exception as exc:
        logger.exception("Unexpected prediction error")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
