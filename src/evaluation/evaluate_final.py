
from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------
# CONFIGURACIÓN
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"
EXPERIMENT_NAME = "BankInsight-Classification"

CANDIDATE_RUN_NAME = (
    "random_forest_300_depth_20_grouped_validation"
)

TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
EVIDENCE_DIR = PROJECT_ROOT / "docs" / "evidence"

POSITIVE_CLASS = "yes"
NEGATIVE_CLASS = "no"


def main():

    print("=" * 65)
    print("BANKINSIGHT MLOPS - EVALUACIÓN FINAL")
    print("=" * 65)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. CONECTAR CON MLFLOW Y LOCALIZAR EL MODELO
    # -----------------------------------------------------

    mlflow.set_tracking_uri(TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError("No se encontró el experimento de MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    candidate = runs[
        runs["tags.mlflow.runName"] == CANDIDATE_RUN_NAME
    ]

    if len(candidate) != 1:
        raise RuntimeError(
            f"Se esperaba un único modelo candidato, "
            f"pero se encontraron {len(candidate)}."
        )

    run_id = candidate.iloc[0]["run_id"]

    model_uri = f"runs:/{run_id}/model"

    print("\nMODELO CANDIDATO")
    print("Run ID:", run_id)
    print("Modelo:", CANDIDATE_RUN_NAME)
    print("URI:", model_uri)

    # -----------------------------------------------------
    # 2. CARGAR EL MODELO REGISTRADO
    # -----------------------------------------------------

    model = mlflow.sklearn.load_model(model_uri)

    print("\nModelo cargado correctamente.")

    # -----------------------------------------------------
    # 3. CARGAR EL CONJUNTO DE PRUEBA
    # -----------------------------------------------------

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el conjunto de prueba: {TEST_DATA_PATH}"
        )

    test_data = pd.read_csv(TEST_DATA_PATH, sep=";")

    if "y" not in test_data.columns:
        raise ValueError("No se encontró la variable objetivo 'y'.")

    if "duration" in test_data.columns:
        raise ValueError(
            "Se detectó duration en el conjunto de prueba."
        )

    X_test = test_data.drop(columns=["y"])
    y_test = test_data["y"]

    if set(y_test.unique()) != {NEGATIVE_CLASS, POSITIVE_CLASS}:
        raise ValueError(
            "El conjunto de prueba no contiene las dos clases esperadas."
        )

    print("\nCONJUNTO DE PRUEBA")
    print("Registros:", len(test_data))
    print("Variables predictoras:", X_test.shape[1])
    print("\nDistribución de clases:")
    print(y_test.value_counts())

    # -----------------------------------------------------
    # 4. GENERAR PREDICCIONES
    # -----------------------------------------------------

    y_pred = model.predict(X_test)

    classifier_classes = list(
        model.named_steps["classifier"].classes_
    )

    positive_index = classifier_classes.index(POSITIVE_CLASS)

    y_probability = model.predict_proba(X_test)[:, positive_index]

    # -----------------------------------------------------
    # 5. CALCULAR MÉTRICAS
    # -----------------------------------------------------

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            pos_label=POSITIVE_CLASS,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            (y_test == POSITIVE_CLASS).astype(int),
            y_probability,
        ),
    }

    print("\n" + "=" * 65)
    print("RESULTADOS DE LA EVALUACIÓN FINAL")
    print("=" * 65)

    for metric_name, value in metrics.items():
        print(f"{metric_name.upper():15}: {value:.6f}")

    # -----------------------------------------------------
    # 6. MATRIZ DE CONFUSIÓN
    # -----------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=["Real_no", "Real_yes"],
        columns=["Predicho_no", "Predicho_yes"],
    )

    print("\nMATRIZ DE CONFUSIÓN")
    print(confusion_df)

    # -----------------------------------------------------
    # 7. REPORTE DE CLASIFICACIÓN
    # -----------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=[NEGATIVE_CLASS, POSITIVE_CLASS],
        target_names=["no", "yes"],
        zero_division=0,
    )

    print("\nREPORTE DE CLASIFICACIÓN")
    print(report)

    # -----------------------------------------------------
    # 8. GUARDAR EVIDENCIAS
    # -----------------------------------------------------

    metrics_df = pd.DataFrame([
        {
            "model": CANDIDATE_RUN_NAME,
            "candidate_run_id": run_id,
            "test_samples": len(test_data),
            **metrics,
        }
    ])

    metrics_path = EVIDENCE_DIR / "final_test_metrics.csv"
    confusion_path = EVIDENCE_DIR / "final_test_confusion_matrix.csv"
    report_path = EVIDENCE_DIR / "final_test_classification_report.txt"

    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("BANKINSIGHT MLOPS - EVALUACIÓN FINAL\n")
        file.write("=" * 55 + "\n\n")
        file.write(f"Modelo: {CANDIDATE_RUN_NAME}\n")
        file.write(f"Run ID: {run_id}\n")
        file.write(f"Registros de prueba: {len(test_data)}\n\n")

        file.write("MÉTRICAS\n")
        for metric_name, value in metrics.items():
            file.write(f"{metric_name}: {value:.6f}\n")

        file.write("\nMATRIZ DE CONFUSIÓN\n")
        file.write(confusion_df.to_string())

        file.write("\n\nREPORTE DE CLASIFICACIÓN\n")
        file.write(report)

    # -----------------------------------------------------
    # 9. REGISTRAR LA EVALUACIÓN EN MLFLOW
    # -----------------------------------------------------

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="final_test_evaluation_random_forest_300",
    ):

        mlflow.log_param("candidate_run_id", run_id)
        mlflow.log_param("evaluation_dataset", "test.csv")
        mlflow.log_param("test_samples", len(test_data))
        mlflow.log_param("positive_class", POSITIVE_CLASS)

        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(confusion_path))
        mlflow.log_artifact(str(report_path))

    print("\nEVIDENCIAS GUARDADAS")
    print(metrics_path)
    print(confusion_path)
    print(report_path)

    print("\nEvaluación final registrada en MLflow.")
    print("\nEVALUACIÓN COMPLETADA")


if __name__ == "__main__":
    main()
