
"""
BankInsight MLOps - Entrenamiento experimental.

Ejecuta cinco experimentos de clasificación:
- Regresión logística con dos valores de C.
- Random Forest con dos configuraciones.
- Gradient Boosting con ponderación de clases.

MLflow registra parámetros, métricas, modelos y artefactos.
El conjunto test.csv no se utiliza en esta fase.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "evidence"
RESULTS_PATH = EVIDENCE_DIR / "experiment_results.csv"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
EXPERIMENT_NAME = "BankInsight-Classification"

TARGET_COLUMN = "y"
POSITIVE_CLASS = "yes"

sys.path.insert(0, str(PROJECT_ROOT))

from src.training.model_factory import build_model_pipeline


# ---------------------------------------------------------
# CONFIGURACIÓN DE LOS CINCO EXPERIMENTOS
# ---------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "logistic_regression_c_0.1",
        "model": "logistic_regression",
        "params": {
            "C": 0.1
        }
    },
    {
        "name": "logistic_regression_c_1.0",
        "model": "logistic_regression",
        "params": {
            "C": 1.0
        }
    },
    {
        "name": "random_forest_200_depth_12",
        "model": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }
    },
    {
        "name": "random_forest_300_depth_20",
        "model": "random_forest",
        "params": {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2
        }
    },
    {
        "name": "gradient_boosting_100_lr_005",
        "model": "gradient_boosting",
        "params": {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 3
        }
    }
]


def load_training_data():
    """Carga exclusivamente el conjunto de entrenamiento."""

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {TRAIN_PATH}"
        )

    df = pd.read_csv(TRAIN_PATH, sep=";")

    if TARGET_COLUMN not in df.columns:
        raise ValueError("No se encontró la variable objetivo 'y'.")

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Se esperaba una clasificación binaria.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    if "duration" in X.columns:
        raise ValueError(
            "La variable duration no debe utilizarse en el modelo."
        )

    return X, y


def calculate_metrics(y_true, y_pred, y_probability):
    """Calcula las métricas principales de clasificación."""

    return {
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            (y_true == POSITIVE_CLASS).astype(int),
            y_probability
        )
    }


def run_experiment(
    experiment,
    X_fit,
    X_validation,
    y_fit,
    y_validation
):
    """Entrena, evalúa y registra una ejecución en MLflow."""

    model_name = experiment["model"]
    model_params = experiment["params"]

    pipeline = build_model_pipeline(
        X_fit,
        model_name,
        model_params
    )

    with mlflow.start_run(
    run_name=f"{experiment['name']}_grouped_validation"
    ):

        # Parámetros generales.
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("validation_strategy", "StratifiedGroupKFold")
        mlflow.log_param("validation_splits", 5)
        mlflow.log_param("positive_class", POSITIVE_CLASS)
        mlflow.log_param("training_rows", len(X_fit))
        mlflow.log_param("validation_rows", len(X_validation))

        # Hiperparámetros específicos.
        for key, value in model_params.items():
            mlflow.log_param(key, value)

        # Gradient Boosting no dispone de class_weight.
        # Se utiliza ponderación de muestras durante el ajuste.
        if model_name == "gradient_boosting":
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_fit
            )

            pipeline.fit(
                X_fit,
                y_fit,
                classifier__sample_weight=sample_weights
            )

            mlflow.log_param("class_weight_strategy", "balanced_sample_weight")

        else:
            pipeline.fit(X_fit, y_fit)

            mlflow.log_param("class_weight_strategy", "balanced")

        # Predicciones de validación.
        y_pred = pipeline.predict(X_validation)

        positive_index = list(
            pipeline.named_steps["classifier"].classes_
        ).index(POSITIVE_CLASS)

        y_probability = pipeline.predict_proba(
            X_validation
        )[:, positive_index]

        metrics = calculate_metrics(
            y_validation,
            y_pred,
            y_probability
        )

        # Registro de métricas.
        mlflow.log_metrics(metrics)

        # Artefactos de evaluación.
        report = classification_report(
            y_validation,
            y_pred,
            output_dict=True,
            zero_division=0
        )

        mlflow.log_dict(
            report,
            "evaluation/classification_report.json"
        )

        matrix = confusion_matrix(
            y_validation,
            y_pred,
            labels=["no", "yes"]
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=["actual_no", "actual_yes"],
            columns=["predicted_no", "predicted_yes"]
        )

        mlflow.log_dict(
            matrix_df.to_dict(),
            "evaluation/confusion_matrix.json"
        )

        # Registrar el pipeline completo.
        """
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_fit.head(3)
        )"""
        model_input_example = X_fit.head(3).copy()

        integer_columns = model_input_example.select_dtypes(
            include=["int"]
        ).columns

        model_input_example[integer_columns] = (
            model_input_example[integer_columns].astype("float64")
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=model_input_example
        )
        
        result = {
            "run_name": f"{experiment['name']}_grouped_validation",
            "model": model_name,
            **model_params,
            **metrics
        }

        print("\n" + "=" * 65)
        print(f"EXPERIMENTO: {experiment['name']}")
        print("=" * 65)

        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.upper():12}: {metric_value:.4f}")

        return result

def create_profile_groups(X):
    """
    Agrupa registros que tienen exactamente los mismos predictores.
    """

    return X.groupby(
        list(X.columns),
        dropna=False,
        sort=False
    ).ngroup()


def main():
    """Ejecuta la matriz experimental."""

    print("=" * 65)
    print("BANKINSIGHT MLOPS - EXPERIMENTOS DE CLASIFICACION")
    print("=" * 65)

    X, y = load_training_data()

    # Separación interna: nunca utiliza test.csv.
    # Separación interna agrupada y estratificada.
    # Evita que perfiles idénticos aparezcan en ambos conjuntos.

    groups = create_profile_groups(X)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fit_indices, validation_indices = next(
        splitter.split(X, y, groups)
    )

    X_fit = X.iloc[fit_indices].copy()
    X_validation = X.iloc[validation_indices].copy()

    y_fit = y.iloc[fit_indices].copy()
    y_validation = y.iloc[validation_indices].copy()

    fit_groups = set(groups.iloc[fit_indices])
    validation_groups = set(groups.iloc[validation_indices])

    assert fit_groups.isdisjoint(validation_groups), (
        "ERROR: existen perfiles compartidos entre entrenamiento "
        "interno y validación."
    )

    print(
        f"Perfiles compartidos entre entrenamiento y validación: "
        f"{len(fit_groups.intersection(validation_groups))}"
    )

    print(f"\nDatos disponibles: {len(X)}")
    print(f"Entrenamiento interno: {len(X_fit)}")
    print(f"Validación interna: {len(X_validation)}")
    print(f"Clase positiva: {POSITIVE_CLASS}")
    print("\nDistribución de entrenamiento interno:")
    print(y_fit.value_counts(normalize=True).round(4))

    # Tracking local de MLflow.
    mlflow.set_tracking_uri(
        (PROJECT_ROOT / "mlruns").as_uri()
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    results = []

    for experiment in EXPERIMENTS:
        result = run_experiment(
            experiment,
            X_fit,
            X_validation,
            y_fit,
            y_validation
        )

        results.append(result)

    # Guardar resumen de resultados.
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1",
        ascending=False
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("RESUMEN DE EXPERIMENTOS")
    print("=" * 65)

    print(
        results_df[
            ["run_name", "f1", "precision", "recall", "roc_auc"]
        ].to_string(index=False)
    )

    print(f"\nResultados guardados en: {RESULTS_PATH}")
    print("\nEXPERIMENTOS COMPLETADOS")


if __name__ == "__main__":
    main()
