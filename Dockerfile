FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

ENV MODEL_ARTIFACT_DIR=/tmp/churn-model-artifacts

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY src ./src

EXPOSE 8000
CMD ["sh", "-c", "python -m src.prepare_model_artifacts && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
