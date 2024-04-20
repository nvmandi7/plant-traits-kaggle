FROM python:3.11-slim

WORKDIR /mlflow

COPY services/mlflow/requirements.txt .

RUN pip install -r requirements.txt

CMD mlflow server \
    --host 0.0.0.0 \
    --port 50005 \
    --default-artifact-root ${MLFLOW_ARTIFACT_ROOT} \
    --backend-store-uri ${MLFLOW_BACKEND_STORE_URI}