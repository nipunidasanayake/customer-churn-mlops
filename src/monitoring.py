"""Persist prediction telemetry and compare recent inputs with the training baseline."""

import json
import hashlib
import uuid
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine, URL

MAX_EVENTS = 10_000
RETENTION_DAYS = 30
MIN_DRIFT_WINDOW = 30
DRIFT_THRESHOLD = 0.2


metadata = MetaData()
prediction_events = Table(
    "prediction_events",
    metadata,
    Column("prediction_id", String, primary_key=True),
    Column("created_at", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("probability", Float, nullable=False),
    Column("features_json", Text, nullable=False),
)
api_request_events = Table(
    "api_request_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", String, nullable=False),
    Column("method", String, nullable=False),
    Column("path", String, nullable=False),
    Column("status_code", Integer, nullable=False),
    Column("response_time_ms", Float, nullable=False),
)
drift_report_events = Table(
    "drift_report_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("report_hash", String, nullable=False, unique=True),
    Column("created_at", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("window_size", Integer, nullable=False),
    Column("drifted_features", Integer, nullable=False),
    Column("drift_percentage", Float, nullable=False),
    Column("feature_drift_json", Text, nullable=False),
)
Index("idx_prediction_events_created_at", prediction_events.c.created_at)
Index("idx_api_request_events_created_at", api_request_events.c.created_at)
Index("idx_drift_report_events_created_at", drift_report_events.c.created_at)


@lru_cache(maxsize=8)
def _engine(database: str) -> Engine:
    if database.startswith(("postgres://", "postgresql://")):
        database = database.replace("postgres://", "postgresql+psycopg://", 1)
        if database.startswith("postgresql://"):
            database = database.replace("postgresql://", "postgresql+psycopg://", 1)
    if "://" not in database:
        path = Path(database)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = URL.create("sqlite", database=str(path.resolve()))
    else:
        url = database
    connect_args = {"check_same_thread": False} if str(url).startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args, pool_pre_ping=True)


def _database_key(database_path: Path | str) -> str:
    return str(database_path)


def initialize_store(database_path: Path | str) -> None:
    engine = _engine(_database_key(database_path))
    metadata.create_all(engine)


def check_store(database_path: Path | str) -> None:
    with _engine(_database_key(database_path)).connect() as connection:
        connection.execute(text("SELECT 1"))


def record_prediction(
    database_path: Path | str,
    features: dict[str, Any],
    probability: float,
    model_version: str,
) -> str:
    prediction_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _engine(_database_key(database_path)).begin() as connection:
        connection.execute(
            text("""
            INSERT INTO prediction_events
                (prediction_id, created_at, model_version, probability, features_json)
            VALUES (:prediction_id, :created_at, :model_version, :probability, :features_json)
            """),
            {
                "prediction_id": prediction_id,
                "created_at": created_at,
                "model_version": model_version,
                "probability": probability,
                "features_json": json.dumps(features),
            },
        )
        retention_cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
        connection.execute(
            text("DELETE FROM prediction_events WHERE created_at < :cutoff"),
            {"cutoff": retention_cutoff},
        )
        connection.execute(
            text("""
            DELETE FROM prediction_events
            WHERE prediction_id NOT IN (
                SELECT prediction_id FROM prediction_events
                ORDER BY created_at DESC LIMIT :max_events
            )
            """),
            {"max_events": MAX_EVENTS},
        )
    return prediction_id


def record_api_request(
    database_path: Path | str,
    method: str,
    path: str,
    status_code: int,
    response_time_ms: float,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    retention_cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
    with _engine(_database_key(database_path)).begin() as connection:
        connection.execute(
            text("""
            INSERT INTO api_request_events
                (created_at, method, path, status_code, response_time_ms)
            VALUES (:created_at, :method, :path, :status_code, :response_time_ms)
            """),
            {
                "created_at": created_at,
                "method": method,
                "path": path,
                "status_code": status_code,
                "response_time_ms": response_time_ms,
            },
        )
        connection.execute(
            text("DELETE FROM api_request_events WHERE created_at < :cutoff"),
            {"cutoff": retention_cutoff},
        )
        connection.execute(
            text("""
            DELETE FROM api_request_events
            WHERE id NOT IN (
                SELECT id FROM api_request_events
                ORDER BY created_at DESC LIMIT :max_events
            )
            """),
            {"max_events": MAX_EVENTS},
        )


def get_api_metrics(database_path: Path | str) -> dict[str, Any]:
    initialize_store(database_path)
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with _engine(_database_key(database_path)).connect() as connection:
        rows = connection.execute(
            text("""
            SELECT created_at, status_code, response_time_ms
            FROM api_request_events
            WHERE created_at >= :cutoff
            ORDER BY created_at
            """),
            {"cutoff": cutoff},
        ).fetchall()

    response_times = sorted(float(row[2]) for row in rows)
    error_count = sum(int(row[1]) >= 400 for row in rows)
    hourly: dict[str, dict[str, int]] = {}
    for created_at, status_code, _ in rows:
        hour = created_at[:13] + ":00Z"
        bucket = hourly.setdefault(hour, {"requests": 0, "errors": 0})
        bucket["requests"] += 1
        bucket["errors"] += int(int(status_code) >= 400)

    if response_times:
        p95_index = max(0, int(np.ceil(len(response_times) * 0.95)) - 1)
        average_response_time = round(float(np.mean(response_times)), 2)
        p95_response_time = round(response_times[p95_index], 2)
    else:
        average_response_time = None
        p95_response_time = None

    return {
        "window_hours": 24,
        "request_count": len(rows),
        "error_count": error_count,
        "error_rate": round(error_count / len(rows), 6) if rows else None,
        "average_response_time_ms": average_response_time,
        "p95_response_time_ms": p95_response_time,
        "requests_by_hour": [
            {"hour": hour, **counts} for hour, counts in sorted(hourly.items())
        ],
    }


def get_prediction_history(
    database_path: Path | str,
    limit: int = 100,
) -> dict[str, Any]:
    initialize_store(database_path)
    with _engine(_database_key(database_path)).connect() as connection:
        total = connection.execute(text("SELECT COUNT(*) FROM prediction_events")).fetchone()[0]
        churn_count = connection.execute(
            text("SELECT COUNT(*) FROM prediction_events WHERE probability >= 0.5")
        ).fetchone()[0]
        rows = connection.execute(
            text("""
            SELECT prediction_id, created_at, model_version, probability
            FROM prediction_events
            ORDER BY created_at DESC LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()

    predictions = [
        {
            "prediction_id": row[0],
            "timestamp": row[1],
            "model_version": row[2],
            "churn_probability": float(row[3]),
            "prediction": int(float(row[3]) >= 0.5),
        }
        for row in rows
    ]
    return {
        "total_predictions": total,
        "predicted_churn_count": churn_count,
        "predicted_non_churn_count": total - churn_count,
        "predictions": predictions,
    }


def record_drift_report(
    database_path: Path | str,
    report_hash: str,
    model_version: str,
    window_size: int,
    feature_drift: list[dict[str, Any]],
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    drifted_features = sum(item["status"] == "drift" for item in feature_drift)
    drift_percentage = drifted_features / len(feature_drift) if feature_drift else 0.0
    retention_cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
    with _engine(_database_key(database_path)).begin() as connection:
        connection.execute(
            text("""
            INSERT INTO drift_report_events
                (report_hash, created_at, model_version, window_size,
                 drifted_features, drift_percentage, feature_drift_json)
            VALUES (:report_hash, :created_at, :model_version, :window_size,
                    :drifted_features, :drift_percentage, :feature_drift_json)
            ON CONFLICT (report_hash) DO NOTHING
            """),
            {
                "report_hash": report_hash,
                "created_at": created_at,
                "model_version": model_version,
                "window_size": window_size,
                "drifted_features": drifted_features,
                "drift_percentage": drift_percentage,
                "feature_drift_json": json.dumps(feature_drift),
            },
        )
        connection.execute(
            text("DELETE FROM drift_report_events WHERE created_at < :cutoff"),
            {"cutoff": retention_cutoff},
        )
        connection.execute(
            text("""
            DELETE FROM drift_report_events
            WHERE id NOT IN (
                SELECT id FROM drift_report_events
                ORDER BY created_at DESC LIMIT :max_events
            )
            """),
            {"max_events": MAX_EVENTS},
        )


def get_drift_history(database_path: Path | str, limit: int = 100) -> dict[str, Any]:
    initialize_store(database_path)
    with _engine(_database_key(database_path)).connect() as connection:
        total = connection.execute(
            text("SELECT COUNT(*) FROM drift_report_events")
        ).fetchone()[0]
        rows = connection.execute(
            text("""
            SELECT created_at, model_version, window_size, drifted_features,
                   drift_percentage, feature_drift_json
            FROM drift_report_events
            ORDER BY created_at DESC LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()

    reports = [
        {
            "timestamp": row[0],
            "model_version": row[1],
            "window_size": int(row[2]),
            "drifted_features": int(row[3]),
            "drift_percentage": float(row[4]),
            "feature_drift": json.loads(row[5]),
        }
        for row in rows
    ]
    reports.reverse()
    return {"total_reports": total, "reports": reports}


def _population_stability_index(
    reference: pd.Series,
    observed: pd.Series,
    numeric: bool,
) -> float:
    if numeric:
        reference_values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
        observed_values = pd.to_numeric(observed, errors="coerce").dropna().to_numpy()
        edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([-np.inf, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        reference_counts = np.histogram(reference_values, bins=edges)[0].astype(float)
        observed_counts = np.histogram(observed_values, bins=edges)[0].astype(float)
    else:
        categories = sorted(set(reference.astype(str)))
        reference_counts = (
            reference.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        )
        observed_counts = (
            observed.astype(str).value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
        )
        if "__other__" not in categories:
            categories.append("__other__")
            reference_counts = np.append(reference_counts, 0.0)
            observed_counts = np.append(
                observed_counts,
                observed.astype(str).isin(set(observed.astype(str)) - set(categories)).sum(),
            )
        else:
            other_values = set(observed.astype(str)) - set(categories)
            observed_counts[categories.index("__other__")] += observed.astype(str).isin(other_values).sum()

    epsilon = 1e-6
    reference_share = (reference_counts + epsilon) / (reference_counts.sum() + epsilon * len(reference_counts))
    observed_share = (observed_counts + epsilon) / (observed_counts.sum() + epsilon * len(observed_counts))
    return float(np.sum((observed_share - reference_share) * np.log(observed_share / reference_share)))


def get_monitoring_snapshot(
    database_path: Path | str,
    reference: pd.DataFrame,
    window_size: int = 1000,
) -> dict[str, Any]:
    initialize_store(database_path)
    with _engine(_database_key(database_path)).connect() as connection:
        total = connection.execute(
            text("SELECT COUNT(*) FROM prediction_events")
        ).fetchone()[0]
        recent_rows = connection.execute(
            text("""
            SELECT prediction_id, created_at, model_version, probability, features_json
            FROM prediction_events
            ORDER BY created_at DESC LIMIT :window_size
            """),
            {"window_size": window_size},
        ).fetchall()

    recent_rows.reverse()
    events = [
        {
            "prediction_id": row[0],
            "created_at": row[1],
            "model_version": row[2],
            "probability": float(row[3]),
            "features": json.loads(row[4]),
        }
        for row in recent_rows
    ]
    probabilities = [event["probability"] for event in events]

    feature_drift = []
    if len(events) < MIN_DRIFT_WINDOW:
        feature_drift = [
            {"feature": feature, "psi": None, "status": "insufficient_data"}
            for feature in reference.columns
        ]
    else:
        observed = pd.DataFrame([event["features"] for event in events])
        for feature in reference.columns:
            is_numeric = pd.api.types.is_numeric_dtype(reference[feature])
            psi = _population_stability_index(reference[feature], observed[feature], is_numeric)
            feature_drift.append(
                {
                    "feature": feature,
                    "psi": round(psi, 6),
                    "status": "drift" if psi >= DRIFT_THRESHOLD else "stable",
                }
            )

    drifted_features = [
        entry["feature"] for entry in feature_drift if entry["status"] == "drift"
    ]
    if len(events) >= MIN_DRIFT_WINDOW:
        window_signature = json.dumps(
            [event["prediction_id"] for event in events],
            separators=(",", ":"),
        )
        report_hash = hashlib.sha256(window_signature.encode("utf-8")).hexdigest()
        record_drift_report(
            database_path,
            report_hash,
            events[-1]["model_version"],
            len(events),
            feature_drift,
        )

    return {
        "observations_total": total,
        "window_size": len(events),
        "minimum_window_size": MIN_DRIFT_WINDOW,
        "mean_churn_probability": (
            round(float(np.mean(probabilities)), 6) if probabilities else None
        ),
        "predicted_high_risk_rate": (
            round(float(np.mean(np.array(probabilities) >= 0.5)), 6) if probabilities else None
        ),
        "drift_threshold": DRIFT_THRESHOLD,
        "drifted_features": drifted_features,
        "drift_history_count": get_drift_history(database_path, limit=1)["total_reports"],
        "feature_drift": feature_drift,
        "recent_predictions": [
            {
                "timestamp": event["created_at"],
                "churn_probability": event["probability"],
                "model_version": event["model_version"],
            }
            for event in events[-100:]
        ],
    }
