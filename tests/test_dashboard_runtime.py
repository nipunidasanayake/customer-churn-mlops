import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.dashboard_runtime import (
    DashboardRuntime,
    DashboardRuntimeError,
    get_embedded_json,
    predict_customer,
)
from src.monitoring import initialize_store


SAMPLE_CUSTOMER = {
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


@pytest.fixture
def runtime(tmp_path):
    features = list(SAMPLE_CUSTOMER)
    training = pd.DataFrame([SAMPLE_CUSTOMER, {**SAMPLE_CUSTOMER, "tenure": 24}])
    model = DummyClassifier(strategy="prior").fit(training, [0, 1])
    database_path = tmp_path / "predictions.db"
    initialize_store(database_path)
    return DashboardRuntime(
        model=model,
        model_metadata={
            "model_name": "TestClassifier",
            "model_version": "test-version",
            "features": features,
            "test_results": {"f1": 0.5},
        },
        reference_features=training,
        database_path=database_path,
    )


def test_embedded_prediction_uses_shared_history_and_metrics(runtime):
    result = predict_customer(runtime, SAMPLE_CUSTOMER)

    assert result["prediction"] in (0, 1)
    assert result["prediction_id"]
    assert 0 <= result["churn_probability"] <= 1

    history = get_embedded_json(runtime, "/prediction-history?limit=10")
    metrics = get_embedded_json(runtime, "/monitoring")["api_metrics"]
    assert history["total_predictions"] == 1
    assert len(history["predictions"]) == 1
    assert "features" not in history["predictions"][0]
    assert metrics["request_count"] == 1
    assert metrics["average_response_time_ms"] is not None


def test_embedded_prediction_rejects_extra_customer_identifiers(runtime):
    with pytest.raises(DashboardRuntimeError, match="customerID"):
        predict_customer(runtime, {**SAMPLE_CUSTOMER, "customerID": "private-id"})

    metrics = get_embedded_json(runtime, "/monitoring")["api_metrics"]
    assert metrics["error_count"] == 1
