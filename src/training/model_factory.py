
"""
Fábrica de modelos de clasificación para BankInsight MLOps.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.preprocessing import build_preprocessor


RANDOM_STATE = 42


def build_model_pipeline(X, model_name, model_params=None):
    """Construye un pipeline completo de clasificación."""

    model_params = model_params or {}

    if model_name == "logistic_regression":
        params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        params.update(model_params)
        estimator = LogisticRegression(**params)

    elif model_name == "random_forest":
        params = {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        params.update(model_params)
        estimator = RandomForestClassifier(**params)

    elif model_name == "gradient_boosting":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": RANDOM_STATE,
        }
        params.update(model_params)
        estimator = GradientBoostingClassifier(**params)

    else:
        raise ValueError(
            f"Modelo no reconocido: {model_name}"
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("classifier", estimator)
        ]
    )


def get_available_models():
    return [
        "logistic_regression",
        "random_forest",
        "gradient_boosting"
    ]
