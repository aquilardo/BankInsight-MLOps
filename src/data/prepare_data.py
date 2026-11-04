
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Configuración de rutas
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "bank-additional-full.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)


# --------------------------------------------------
# 2. Carga de datos
# --------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    """Carga el conjunto de datos original."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {RAW_DATA_PATH}"
        )

    return pd.read_csv(RAW_DATA_PATH, sep=";")


# --------------------------------------------------
# 3. Limpieza de datos
# --------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y la variable duration."""

    print("\nINICIANDO PREPARACIÓN DE DATOS")

    original_rows = len(df)

    # Eliminar duplicados exactos
    df_clean = df.drop_duplicates(
        keep="first"
    ).copy()

    removed_duplicates = original_rows - len(df_clean)

    print("\n1. ELIMINACIÓN DE DUPLICADOS")
    print(f"Filas originales: {original_rows}")
    print(f"Copias eliminadas: {removed_duplicates}")
    print(f"Filas restantes: {len(df_clean)}")

    # Comprobar que no queden duplicados exactos
    assert not df_clean.duplicated().any(), (
        "Todavía existen registros duplicados."
    )

    # Excluir duration para evitar fuga de información
    if "duration" in df_clean.columns:
        df_clean = df_clean.drop(
            columns=["duration"]
        )

    print("\n2. EXCLUSIÓN DE VARIABLE")
    print("Variable excluida: duration")
    print(f"Columnas restantes: {len(df_clean.columns)}")

    # Conservar los valores unknown
    unknown_total = (
        df_clean
        .select_dtypes(include="object")
        .apply(
            lambda column: (
                column.astype(str)
                .str.strip()
                .str.lower()
                .eq("unknown")
                .sum()
            )
        )
        .sum()
    )

    print("\n3. VALORES DESCONOCIDOS")
    print(f"Valores unknown conservados: {unknown_total}")

    # Validaciones del resultado esperado
    assert len(df_clean) == 41176, (
        f"Número inesperado de filas: {len(df_clean)}"
    )

    assert "y" in df_clean.columns
    assert "duration" not in df_clean.columns

    expected_target_counts = {
        "no": 36537,
        "yes": 4639
    }

    actual_target_counts = (
        df_clean["y"].value_counts().to_dict()
    )

    assert actual_target_counts == expected_target_counts, (
        f"Distribución inesperada: {actual_target_counts}"
    )

    return df_clean


# --------------------------------------------------
# 4. Guardar conjunto preparado
# --------------------------------------------------

def save_processed_data(df: pd.DataFrame) -> None:
    """Guarda el conjunto preparado en formato CSV."""

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
        sep=";"
    )

    print("\n4. ARCHIVO GENERADO")
    print(f"Ubicación: {PROCESSED_DATA_PATH}")
    print(f"Dimensiones finales: {df.shape}")


# --------------------------------------------------
# 5. Ejecución principal
# --------------------------------------------------

def main():

    print("BANKINSIGHT MLOPS")
    print("Preparación del conjunto de datos")

    df = load_raw_data()

    df_clean = clean_data(df)

    save_processed_data(df_clean)

    print("\nPROCESO COMPLETADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
