import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "BankInsight-Campaign-Subscription"
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "candidate"
)


def load_registered_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # Consultar versión asociada al alias
    version_info = client.get_model_version_by_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS
    )

    # Cargar modelo desde MLflow Registry
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    model = mlflow.pyfunc.load_model(model_uri)

    return {
        "model": model,
        "name": MODEL_NAME,
        "version": str(version_info.version),
        "alias": MODEL_ALIAS
    }