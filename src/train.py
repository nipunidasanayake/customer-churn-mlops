"""Train, compare, and track churn classifiers with MLflow.

Model selection uses validation F1. The held-out test set is evaluated only
once after selecting the best candidate.
"""
import hashlib
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("CHURN_DATA_PATH", str(ROOT / "data" / "telco_customer_churn.csv")))
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
REPORTS = ROOT / "reports"
PROMOTION_SPLIT_SCHEME = "customer-id-sha256-buckets-v1"


def metrics_for(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
    }


def log_candidate_params(name, classifier):
    mlflow.log_param("model_name", name)
    # Log only the main, interpretable settings rather than every pipeline object.
    for key, value in classifier.get_params().items():
        if key in {"max_iter", "class_weight", "n_estimators", "max_depth", "learning_rate", "random_state"}:
            mlflow.log_param(key, str(value))


def split_customer_cohorts(
    customer_ids: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keep customer cohorts stable as new rows are appended to the dataset."""
    bucket = customer_ids.map(
        lambda customer_id: int.from_bytes(
            hashlib.sha256(str(customer_id).encode("utf-8")).digest()[:4],
            byteorder="big",
        ) % 100
    )
    test_mask = bucket < 15
    promotion_mask = (bucket >= 15) & (bucket < 35)
    development_mask = bucket >= 35
    if min(test_mask.sum(), promotion_mask.sum(), development_mask.sum()) == 0:
        raise ValueError("Customer ID split produced an empty train, promotion, or test cohort.")
    return development_mask, promotion_mask, test_mask


def compare_promotion_f1(
    candidate_f1: float,
    champion_f1: float,
) -> None:
    if candidate_f1 < champion_f1:
        raise RuntimeError(
            "Candidate rejected by non-regression gate: "
            f"candidate promotion F1={candidate_f1:.6f}, "
            f"champion promotion F1={champion_f1:.6f}"
        )


def main():
    # Use a configured tracking server or default to local SQLite.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db")

    # Create or select the experiment.
    mlflow.set_experiment("customer-churn-prediction")

    df = pd.read_csv(DATA_PATH)
    required = {"Churn", "customerID", "TotalCharges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.drop_duplicates().copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    y = df["Churn"].map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected values found in Churn target.")
    y = y.astype(int)
    X = df.drop(columns=["Churn", "customerID"])

    development_mask, promotion_mask, test_mask = split_customer_cohorts(df["customerID"])
    X_development, y_development = X.loc[development_mask], y.loc[development_mask]
    X_promotion, y_promotion = X.loc[promotion_mask], y.loc[promotion_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    if any(values.nunique() != 2 for values in (y_development, y_promotion, y_test)):
        raise ValueError(
            "Stable customer cohorts must each contain both target classes; "
            "check the dataset size and churn label distribution."
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X_development, y_development, test_size=0.20, random_state=42, stratify=y_development
    )

    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1500, random_state=42, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250, max_depth=12, random_state=42,
            n_jobs=-1, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42
        ),
    }

    validation_results = []
    for name, classifier in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
        with mlflow.start_run(run_name=f"validation-{name}"):
            log_candidate_params(name, classifier)
            pipeline.fit(X_train, y_train)
            result = {"model": name, **metrics_for(pipeline, X_val, y_val)}
            mlflow.log_metrics({f"validation_{k}": v for k, v in result.items() if k != "model"})
            mlflow.set_tag("stage", "validation")
            validation_results.append(result)

    validation_results.sort(key=lambda item: item["f1"], reverse=True)
    best_name = validation_results[0]["model"]

    # Fit on the development cohort; promotion and test cohorts stay out of training.
    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clone(candidates[best_name])),
    ])
    final_model.fit(X_development, y_development)
    promotion_metrics = {"model": best_name, **metrics_for(final_model, X_promotion, y_promotion)}

    if os.getenv("CHURN_REQUIRE_NON_REGRESSION") == "1" and MODEL_PATH.exists():
        existing_metadata = (
            json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            if METADATA_PATH.exists()
            else {}
        )
        if existing_metadata.get("promotion_split_scheme") != PROMOTION_SPLIT_SCHEME:
            raise RuntimeError(
                "The champion was not trained with the current reserved promotion split. "
                "Run a reviewed bootstrap training with CHURN_REQUIRE_NON_REGRESSION unset "
                "before enabling automated promotion."
            )
        champion_model = joblib.load(MODEL_PATH)
        champion_promotion_metrics = metrics_for(champion_model, X_promotion, y_promotion)
        compare_promotion_f1(
            promotion_metrics["f1"],
            champion_promotion_metrics["f1"],
        )

    # Final test metrics are calculated once, after model selection and promotion.
    test_metrics = {"model": best_name, **metrics_for(final_model, X_test, y_test)}

    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(validation_results).to_csv(REPORTS / "validation_model_comparison.csv", index=False)
    (REPORTS / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    joblib.dump(final_model, MODEL_PATH)

    metadata = {
        "model_name": best_name,
        "model_version": f"1.1.{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d%H%M%S')}",
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "target": "Churn",
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "promotion_split_scheme": PROMOTION_SPLIT_SCHEME,
        "promotion_results": promotion_metrics,
        "features": X.columns.tolist(),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "validation_results": validation_results,
        "test_results": test_metrics,
        "training_rows": int(len(X_development)),
        "validation_rows": int(len(X_val)),
        "promotion_rows": int(len(X_promotion)),
        "test_rows": int(len(X_test)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"final-{best_name}"):
        mlflow.set_tag("stage", "final-test-evaluation")
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("model_version", metadata["model_version"])
        mlflow.log_param("training_rows", int(len(X_development)))
        mlflow.log_param("promotion_rows", int(len(X_promotion)))
        mlflow.log_param("test_rows", int(len(X_test)))
        mlflow.log_metrics(
            {
                f"promotion_{key}": value
                for key, value in promotion_metrics.items()
                if key != "model"
            }
        )
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if k != "model"})
        mlflow.log_artifact(str(REPORTS / "validation_model_comparison.csv"))
        mlflow.log_artifact(str(REPORTS / "test_metrics.json"))
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            serialization_format="pickle",
        )

    print(f"Selected model by validation F1: {best_name}")
    print("Held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print("Start the local MLflow UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    main()
