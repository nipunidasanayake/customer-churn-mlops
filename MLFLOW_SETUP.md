# MLflow experiment tracking

Training runs are tracked in a local SQLite database (`mlflow.db`) by default:

```powershell
python src\train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and select the `customer-churn-prediction` experiment. Each candidate has a validation run; a final run records the selected model's held-out test metrics and artifacts.

To use a remote tracking server, set `MLFLOW_TRACKING_URI` before running training. Keep tracking credentials in a secret manager or environment variables, not in source control. MLflow's local database is excluded from Git.
