import json

import joblib
import pandas as pd
from fastapi.testclient import TestClient
import pytest
import sklearn
from sqlalchemy.exc import OperationalError
from sklearn.dummy import DummyClassifier

from app.main import app

sample_customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 900.0,
}

@pytest.fixture(autouse=True)
def use_temporary_prediction_store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.main.PREDICTION_DB_PATH", tmp_path / "predictions.db")
    monkeypatch.setattr("app.main.PREDICTION_DATABASE_URL", None)

    features = list(sample_customer)
    training_rows = pd.DataFrame(
        [sample_customer, {**sample_customer, "tenure": 24}],
        columns=features,
    )
    model_path = tmp_path / "churn_model.joblib"
    classifier = DummyClassifier(strategy="prior").fit(training_rows, [0, 1])
    joblib.dump(classifier, model_path)

    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": "TestClassifier",
                "model_version": "test-version",
                "trained_at": "2026-01-01T00:00:00+00:00",
                "scikit_learn_version": sklearn.__version__,
                "target": "Churn",
                "features": features,
                "test_results": {"f1": 0.5},
            }
        ),
        encoding="utf-8",
    )
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {**sample_customer, "customerID": "fixture-1", "Churn": "No"},
            {
                **sample_customer,
                "tenure": 24,
                "customerID": "fixture-2",
                "Churn": "Yes",
            },
        ]
    ).to_csv(reference_path, index=False)

    monkeypatch.setattr("app.main.MODEL_PATH", model_path)
    monkeypatch.setattr("app.main.METADATA_PATH", metadata_path)
    monkeypatch.setattr("app.main.DATA_PATH", reference_path)

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["database_connected"] is True

def test_health_reports_database_unavailable(monkeypatch):
    def fail_health_check(_database):
        raise OperationalError("SELECT 1", {}, Exception("database unavailable"))

    monkeypatch.setattr("app.main.check_store", fail_health_check)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Telemetry database is unavailable"

def test_model_info():
    with TestClient(app) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_version"]
    assert response.json()["trained_at"]

def test_predict():
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["prediction_id"]
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"]

def test_predict_rejects_extra_fields():
    payload = {**sample_customer, "customerID": "not-a-model-feature"}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_monitoring_snapshot():
    with TestClient(app) as client:
        client.post("/predict", json=sample_customer)
        response = client.get("/monitoring")
    assert response.status_code == 200
    body = response.json()
    assert body["observations_total"] >= 1
    assert body["window_size"] >= 1
    assert body["mean_churn_probability"] is not None
    assert body["api_metrics"]["request_count"] >= 1
    assert body["api_metrics"]["average_response_time_ms"] is not None

def test_prediction_history_excludes_customer_features():
    with TestClient(app) as client:
        client.post("/predict", json=sample_customer)
        response = client.get("/prediction-history")
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 1
    assert body["predicted_churn_count"] + body["predicted_non_churn_count"] == 1
    assert len(body["predictions"]) == 1
    assert "features" not in body["predictions"][0]
    assert "customerID" not in body["predictions"][0]

def test_api_metrics_count_validation_errors():
    payload = {**sample_customer, "customerID": "not-a-model-feature"}
    with TestClient(app) as client:
        prediction_response = client.post("/predict", json=payload)
        metrics_response = client.get("/monitoring")
    assert prediction_response.status_code == 422
    metrics = metrics_response.json()["api_metrics"]
    assert metrics["request_count"] >= 1
    assert metrics["error_count"] >= 1
