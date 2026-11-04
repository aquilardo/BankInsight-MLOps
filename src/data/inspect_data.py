
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuración de rutas
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-additional-full.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "evidence"
    / "data_quality_report.txt"
)


# --------------------------------------------------
# 2. Carga de datos originales
# --------------------------------------------------

def load_data() -> pd.DataFrame:
    """Carga el conjunto de datos original sin modificarlo."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, sep=";")


# --------------------------------------------------
# 3. Inspección y calidad de datos
# --------------------------------------------------

def generate_quality_report(df: pd.DataFrame) -> str:
    """Genera un informe de inspección del dataset."""

    lines = []

    lines.append("=" * 65)
    lines.append("BANKINSIGHT MLOPS - INFORME DE CALIDAD DE DATOS")
    lines.append("=" * 65)

    # Dimensiones
    lines.append("\n1. DIMENSIONES")
    lines.append(f"Filas: {df.shape[0]}")
    lines.append(f"Columnas: {df.shape[1]}")

    # Tipos de datos
    lines.append("\n2. TIPOS DE DATOS")
    lines.append(df.dtypes.to_string())

    # Valores nulos
    lines.append("\n3. VALORES NULOS")
    null_counts = df.isnull().sum()
    lines.append(null_counts.to_string())
    lines.append(
        f"Total de valores nulos: {null_counts.sum()}"
    )

    # Valores desconocidos
    lines.append("\n4. VALORES 'UNKNOWN'")

    unknown_counts = {}

    for column in df.select_dtypes(include="object").columns:
        count = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("unknown")
            .sum()
        )

        if count > 0:
            unknown_counts[column] = int(count)

    if unknown_counts:
        for column, count in unknown_counts.items():
            percentage = count / len(df) * 100
            lines.append(
                f"{column}: {count} ({percentage:.2f}%)"
            )
    else:
        lines.append("No se encontraron valores unknown.")

    # Registros duplicados
    lines.append("\n5. REGISTROS DUPLICADOS")
    duplicates = int(df.duplicated().sum())
    lines.append(f"Duplicados exactos: {duplicates}")

    # Distribución de la variable objetivo
    lines.append("\n6. DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")

    target_counts = df["y"].value_counts()
    target_percentages = (
        df["y"].value_counts(normalize=True) * 100
    )

    for category in target_counts.index:
        lines.append(
            f"{category}: {target_counts[category]} "
            f"({target_percentages[category]:.2f}%)"
        )

    # Estadísticas numéricas
    lines.append("\n7. ESTADÍSTICAS DESCRIPTIVAS")

    numeric_stats = df.describe().round(2)
    lines.append(numeric_stats.to_string())

    # Advertencia metodológica
    lines.append("\n8. OBSERVACIONES METODOLÓGICAS")
    lines.append(
        "- La variable objetivo es 'y'."
    )
    lines.append(
        "- La variable 'duration' se excluirá del entrenamiento."
    )
    lines.append(
        "- Los valores 'unknown' requieren análisis específico."
    )
    lines.append(
        "- Se debe considerar el desbalance de clases."
    )

    lines.append("\n" + "=" * 65)

    return "\n".join(lines)


# --------------------------------------------------
# 4. Ejecución principal
# --------------------------------------------------

def main():

    print("Cargando conjunto de datos...")

    df = load_data()

    # Validación estructural inicial
    assert df.shape == (41188, 21), (
        f"Dimensiones inesperadas: {df.shape}"
    )

    assert "y" in df.columns, (
        "No se encontró la variable objetivo 'y'."
    )

    print("Dataset cargado y validado.")

    report = generate_quality_report(df)

    analyze_duplicates(df)

    # Guardar informe
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + report)

    print(
        f"\nInforme guardado en: {REPORT_PATH}"
    )



# --------------------------------------------------
# 4. Análisis detallado de registros duplicados
# --------------------------------------------------

def analyze_duplicates(df: pd.DataFrame) -> None:
    """Analiza los registros duplicados sin eliminarlos."""

    duplicate_groups = (
        df.groupby(list(df.columns), dropna=False)
        .size()
    )

    duplicate_groups = duplicate_groups[
        duplicate_groups > 1
    ]

    duplicated_rows = df[
        df.duplicated(keep=False)
    ]

    print("\n" + "=" * 65)
    print("ANÁLISIS DETALLADO DE DUPLICADOS")
    print("=" * 65)

    print(
        f"\nGrupos de registros duplicados: "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Filas involucradas en duplicaciones: "
        f"{len(duplicated_rows)}"
    )

    print(
        f"Copias adicionales: "
        f"{df.duplicated().sum()}"
    )

    if not duplicated_rows.empty:

        print("\nDistribución de la variable objetivo:")
        print(
            duplicated_rows["y"]
            .value_counts()
            .to_string()
        )

        print("\nEjemplos de registros duplicados:")
        print(
            duplicated_rows.head(10)
            .to_string(index=False)
        )

    print("\nNo se ha eliminado ningún registro.")



if __name__ == "__main__":
    main()
