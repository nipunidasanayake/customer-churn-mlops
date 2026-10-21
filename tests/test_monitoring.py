import os

import pandas as pd
import pytest

from src.train import compare_promotion_f1, split_customer_cohorts
from src.monitoring import (
    _engine,
    get_api_metrics,
    get_drift_history,
    get_monitoring_snapshot,
    get_prediction_history,
    initialize_store,
    record_api_request,
    record_drift_report,
    record_prediction,
)
from sqlalchemy import text


def test_customer_cohorts_are_stable_and_disjoint():
    customer_ids = pd.Series([f"customer-{index}" for index in range(500)])
    development, promotion, test = split_customer_cohorts(customer_ids)
    assert not (development & promotion).any()
    assert not (development & test).any()
    assert not (promotion & test).any()
    assert (development | promotion | test).all()
    extended = pd.concat([customer_ids, pd.Series(["new-customer-1", "new-customer-2"])])
    assert split_customer_cohorts(extended)[1].iloc[: len(customer_ids)].equals(promotion)


def test_promotion_gate_rejects_regression_and_accepts_tie():
    compare_promotion_f1(0.7, 0.7)
    with pytest.raises(RuntimeError, match="promotion F1"):
        compare_promotion_f1(0.69, 0.7)


def test_feature_drift_is_reported_for_shifted_inputs(tmp_path):
    reference = pd.DataFrame(
        {
            "tenure": [1, 6, 12, 24, 36, 48] * 5,
            "Contract": ["Month-to-month", "One year", "Two year"] * 10,
        }
    )
    database_path = tmp_path / "predictions.db"
    features = reference.iloc[0].to_dict()
    features["tenure"] = 1000
    features["Contract"] = "new-contract"
    initialize_store(database_path)

    for _ in range(30):
        record_prediction(database_path, features, 0.8, "test")

    snapshot = get_monitoring_snapshot(database_path, reference)
    drift = {item["feature"]: item for item in snapshot["feature_drift"]}
    assert snapshot["observations_total"] == 30
    assert drift["tenure"]["status"] == "drift"
    assert drift["Contract"]["status"] == "drift"
    report_history = get_drift_history(database_path)
    assert report_history["total_reports"] == 1
    assert report_history["reports"][0]["drifted_features"] >= 2


def test_drift_waits_for_minimum_sample(tmp_path):
    reference = pd.DataFrame({"tenure": [1, 2, 3], "Contract": ["Month-to-month", "One year", "Two year"]})
    database_path = tmp_path / "predictions.db"
    initialize_store(database_path)
    record_prediction(
        database_path,
        {"tenure": 1000, "Contract": "unknown"},
        0.9,
        "test",
    )

    snapshot = get_monitoring_snapshot(tmp_path / "predictions.db", reference)
    assert snapshot["feature_drift"] == [
        {"feature": "tenure", "psi": None, "status": "insufficient_data"},
        {"feature": "Contract", "psi": None, "status": "insufficient_data"},
    ]


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_DATABASE_URL"),
    reason="TEST_POSTGRES_DATABASE_URL is not configured",
)
def test_postgres_prediction_store_round_trip():
    database_url = os.environ["TEST_POSTGRES_DATABASE_URL"]
    marker = f"postgres-test-{os.urandom(8).hex()}"
    initialize_store(database_url)
    prediction_id = None
    try:
        prediction_id = record_prediction(
            database_url,
            {"tenure": 12},
            0.75,
            marker,
        )
        history = get_prediction_history(database_url)
        prediction = next(
            row for row in history["predictions"] if row["prediction_id"] == prediction_id
        )
        assert prediction["churn_probability"] == 0.75

        record_api_request(database_url, "GET", f"/{marker}", 200, 12.0)
        assert get_api_metrics(database_url)["request_count"] >= 1

        drift_result = [{"feature": "tenure", "psi": 0.3, "status": "drift"}]
        record_drift_report(database_url, marker, marker, 30, drift_result)
        record_drift_report(database_url, marker, marker, 30, drift_result)
        reports = get_drift_history(database_url)["reports"]
        assert sum(report["model_version"] == marker for report in reports) == 1
    finally:
        with _engine(database_url).begin() as connection:
            if prediction_id:
                connection.execute(
                    text("DELETE FROM prediction_events WHERE prediction_id = :prediction_id"),
                    {"prediction_id": prediction_id},
                )
            connection.execute(
                text("DELETE FROM api_request_events WHERE path = :path"),
                {"path": f"/{marker}"},
            )
            connection.execute(
                text("DELETE FROM drift_report_events WHERE report_hash = :report_hash"),
                {"report_hash": marker},
            )
