from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from src.monitoring.data_drift import compare_datasets


DEFAULT_REFERENCE = (
    PROJECT_ROOT / "data" / "processed" / "train.csv"
)

DEFAULT_CURRENT = None

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "monitoring" / "reports"
)


def load_dataset(path: Path) -> pd.DataFrame:
    """Carga un conjunto de datos CSV separado por punto y coma."""

    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de datos: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"La ruta no corresponde a un archivo: {path}"
        )

    dataset = pd.read_csv(path, sep=";")

    if dataset.empty:
        raise ValueError(
            f"El conjunto de datos está vacío: {path}"
        )

    return dataset


def generate_report(
    reference_path: Path,
    current_path: Path,
    output_dir: Path
) -> dict:
    """Compara dos conjuntos y guarda informes JSON y CSV."""

    reference = load_dataset(reference_path)
    current = load_dataset(current_path)

    # La variable objetivo no forma parte del análisis de drift.
    reference = reference.drop(columns=["y"], errors="ignore")
    current = current.drop(columns=["y"], errors="ignore")

    report = compare_datasets(reference, current)

    execution_time = datetime.now(timezone.utc)

    report["metadata"] = {
        "execution_timestamp_utc": execution_time.isoformat(),
        "reference_file": str(reference_path),
        "current_file": str(current_path),
        "reference_dataset": "training_data",
        "target_excluded": True,
        "method": "PSI_KS_JENSEN_SHANNON"
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = execution_time.strftime("%Y%m%dT%H%M%SZ")

    json_path = output_dir / f"data_drift_{timestamp}.json"
    csv_path = output_dir / f"data_drift_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )

    results = pd.DataFrame(report["results"])
    results.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig"
    )

    report["metadata"]["json_report"] = str(json_path)
    report["metadata"]["csv_report"] = str(csv_path)

    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Monitoreo de Data Drift para BankInsight MLOps."
        )
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="CSV de referencia. Por defecto utiliza data/processed/train.csv."
    )

    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT,
        help="CSV del lote actual que se desea analizar."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde se guardarán los informes."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.current is None:
        print(
            "ERROR: debes indicar el conjunto actual con --current."
        )
        return 2

    try:
        report = generate_report(
            reference_path=args.reference,
            current_path=args.current,
            output_dir=args.output_dir
        )

        summary = report["summary"]

        print("\n=== BANKINSIGHT DATA DRIFT MONITORING ===")
        print("Estado:", summary["status"])
        print("Filas de referencia:", summary["reference_rows"])
        print("Filas actuales:", summary["current_rows"])
        print("Variables analizadas:", summary["features_analyzed"])
        print("Variables numéricas:", summary["numeric_features"])
        print("Variables categóricas:", summary["categorical_features"])
        print("Alertas:", summary["features_with_alerts"])
        print("Severidad alta:", summary["high_severity"])
        print("Severidad moderada:", summary["moderate_severity"])
        print(
            "Categorías nuevas:",
            summary["new_categories_detected"]
        )
        print(
            "Alertas por valores faltantes:",
            summary["features_with_missing_alerts"]
        )

        print("\n=== INFORMES GENERADOS ===")
        print("JSON:", report["metadata"]["json_report"])
        print("CSV:", report["metadata"]["csv_report"])

        return 1 if summary["features_with_alerts"] > 0 else 0

    except (FileNotFoundError, ValueError, KeyError) as error:
        print("\nERROR EN MONITOREO:", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
