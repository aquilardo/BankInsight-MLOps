
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from src.api.model_loader import load_registered_model
from src.api.prediction_store import save_prediction_event
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse
)


logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed"
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Cargando modelo desde MLflow Registry...")

    try:
        bundle = load_registered_model()

        app.state.model_bundle = bundle

        print("Modelo cargado correctamente.")
        print("Nombre:", bundle["name"])
        print("Versión:", bundle["version"])
        print("Alias:", bundle["alias"])

    except Exception as exc:
        print("ERROR AL CARGAR EL MODELO:", str(exc))
        raise RuntimeError(
            "No fue posible cargar el modelo registrado."
        ) from exc

    yield


app = FastAPI(
    title="BankInsight MLOps API",
    description=(
        "API de inferencia para predecir la suscripción "
        "a depósitos a plazo en campañas bancarias."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root():
    return {
        "application": "BankInsight MLOps API",
        "status": "running",
        "documentation": "/docs"
    }


@app.get("/health")
def health(request: Request):
    bundle = getattr(
        request.app.state,
        "model_bundle",
        None
    )

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": bundle["name"],
        "model_version": bundle["version"]
    }


@app.get("/model-info")
def model_info(request: Request):
    bundle = getattr(
        request.app.state,
        "model_bundle",
        None
    )

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )

    return {
        "model_name": bundle["name"],
        "model_version": bundle["version"],
        "model_alias": bundle["alias"],
        "tracking_uri": "configured"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    payload: PredictionRequest,
    request: Request
):
    bundle = getattr(
        request.app.state,
        "model_bundle",
        None
    )

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible."
        )

    try:
        record = payload.model_dump(by_alias=True)

        dataframe = pd.DataFrame(
            [record],
            columns=FEATURE_COLUMNS
        )

        prediction = str(
            bundle["model"].predict(dataframe)[0]
        )

    except Exception as exc:
        logger.exception("Error durante la predicción")

        raise HTTPException(
            status_code=500,
            detail="Error durante la predicción."
        ) from exc

    # La persistencia es independiente de la inferencia.
    # Un fallo de telemetría no debe interrumpir la API.
    try:
        save_prediction_event(
            features=record,
            prediction=prediction,
            model_name=bundle["name"],
            model_version=bundle["version"],
            model_alias=bundle["alias"]
        )

    except Exception:
        logger.exception(
            "No fue posible registrar el evento de predicción "
            "en PostgreSQL."
        )

    return PredictionResponse(
        prediction=prediction,
        model_name=bundle["name"],
        model_version=bundle["version"],
        model_alias=bundle["alias"]
    )
