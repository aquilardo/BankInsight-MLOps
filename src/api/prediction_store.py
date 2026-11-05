import os
from typing import Any

import psycopg
from psycopg.types.json import Jsonb


def get_database_config() -> dict[str, Any]:
    """
    Obtiene la configuración de PostgreSQL desde variables de entorno.
    No utiliza valores secretos predeterminados.
    """

    required_variables = [
        "POSTGRES_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD"
    ]

    missing = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing:
        raise RuntimeError(
            "Faltan variables de configuración de PostgreSQL: "
            + ", ".join(missing)
        )

    return {
        "host": os.environ["POSTGRES_HOST"],
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.environ["POSTGRES_DB"],
        "user": os.environ["POSTGRES_USER"],
        "password": os.environ["POSTGRES_PASSWORD"],
        "connect_timeout": 5
    }


def initialize_prediction_store() -> None:
    """
    Crea el esquema y la tabla de telemetría si no existen.
    La operación es idempotente.
    """

    config = get_database_config()

    with psycopg.connect(**config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE SCHEMA IF NOT EXISTS monitoring;
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS monitoring.prediction_events (
                    event_id BIGSERIAL PRIMARY KEY,
                    event_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    features JSONB NOT NULL,
                    prediction TEXT NOT NULL
                        CHECK (prediction IN ('yes', 'no')),
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    model_alias TEXT NOT NULL
                );
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_prediction_events_timestamp
                ON monitoring.prediction_events (event_timestamp);
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_prediction_events_model_version
                ON monitoring.prediction_events (model_name, model_version);
                """
            )

        connection.commit()


def save_prediction_event(
    features: dict[str, Any],
    prediction: str,
    model_name: str,
    model_version: str,
    model_alias: str
) -> None:
    """
    Persiste un evento de inferencia en PostgreSQL.
    Cada llamada utiliza una conexión independiente.
    """

    if prediction not in {"yes", "no"}:
        raise ValueError("La predicción debe ser 'yes' o 'no'.")

    config = get_database_config()

    with psycopg.connect(**config) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO monitoring.prediction_events (
                    features,
                    prediction,
                    model_name,
                    model_version,
                    model_alias
                )
                VALUES (%s, %s, %s, %s, %s);
                """,
                (
                    Jsonb(features),
                    prediction,
                    model_name,
                    model_version,
                    model_alias
                )
            )

        connection.commit()
