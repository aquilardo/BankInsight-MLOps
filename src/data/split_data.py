
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# 1. Configuración
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bank_marketing_clean.csv"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "docs"
    / "evidence"
    / "data_split_report.txt"
)

RANDOM_STATE = 42
N_SPLITS = 5


# --------------------------------------------------
# 2. Cargar datos
# --------------------------------------------------

def load_data() -> pd.DataFrame:

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH, sep=";")


# --------------------------------------------------
# 3. Crear grupos de perfiles idénticos
# --------------------------------------------------

def create_groups(df: pd.DataFrame):

    feature_columns = [
        column for column in df.columns
        if column != "y"
    ]

    groups = (
        df.groupby(
            feature_columns,
            dropna=False,
            sort=False
        )
        .ngroup()
    )

    return groups, feature_columns


# --------------------------------------------------
# 4. División agrupada y estratificada
# --------------------------------------------------

def split_dataset(df: pd.DataFrame):

    groups, feature_columns = create_groups(df)

    X = df[feature_columns]
    y = df["y"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    # Utilizamos la primera partición como conjunto de prueba
    train_indices, test_indices = next(
        splitter.split(X, y, groups)
    )

    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    # Comprobar que no existen grupos compartidos
    assert train_groups.isdisjoint(test_groups), (
        "Error: existen perfiles compartidos "
        "entre entrenamiento y prueba."
    )

    return train_df, test_df, feature_columns


# --------------------------------------------------
# 5. Generar informe
# --------------------------------------------------

def generate_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> str:

    lines = []

    lines.append("=" * 65)
    lines.append("BANKINSIGHT MLOPS - INFORME DE DIVISION")
    lines.append("=" * 65)

    lines.append("\n1. DIMENSIONES")

    total = len(train_df) + len(test_df)

    lines.append(f"Total: {total}")
    lines.append(
        f"Entrenamiento: {len(train_df)} "
        f"({len(train_df) / total * 100:.2f}%)"
    )
    lines.append(
        f"Prueba: {len(test_df)} "
        f"({len(test_df) / total * 100:.2f}%)"
    )

    lines.append("\n2. DISTRIBUCIÓN DE CLASES")

    for name, dataset in [
        ("ENTRENAMIENTO", train_df),
        ("PRUEBA", test_df)
    ]:

        lines.append(f"\n{name}")

        distribution = (
            dataset["y"]
            .value_counts(normalize=True)
            .mul(100)
        )

        counts = dataset["y"].value_counts()

        for category in ["no", "yes"]:
            lines.append(
                f"{category}: {counts.get(category, 0)} "
                f"({distribution.get(category, 0):.2f}%)"
            )

    lines.append("\n3. VALIDACIONES")

    lines.append(
        f"Columnas: {len(train_df.columns)}"
    )

    lines.append(
        f"Duplicados exactos en entrenamiento: "
        f"{train_df.duplicated().sum()}"
    )

    lines.append(
        f"Duplicados exactos en prueba: "
        f"{test_df.duplicated().sum()}"
    )

    lines.append(
        "Perfiles compartidos entre conjuntos: 0"
    )

    lines.append(
        "Variable duration presente: "
        f"{'duration' in train_df.columns}"
    )

    lines.append("\n4. CONFIGURACIÓN")

    lines.append(
        f"Estrategia: StratifiedGroupKFold"
    )
    lines.append(f"N_SPLITS: {N_SPLITS}")
    lines.append(f"RANDOM_STATE: {RANDOM_STATE}")

    lines.append("\n" + "=" * 65)

    return "\n".join(lines)


# --------------------------------------------------
# 6. Guardar resultados
# --------------------------------------------------

def save_results(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: str
):

    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
        sep=";"
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
        sep=";"
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8"
    )


# --------------------------------------------------
# 7. Ejecución principal
# --------------------------------------------------

def main():

    print("BANKINSIGHT MLOPS")
    print("Separación de entrenamiento y prueba")

    df = load_data()

    print(f"\nDataset cargado: {df.shape}")

    train_df, test_df, _ = split_dataset(df)

    report = generate_report(
        train_df,
        test_df
    )

    save_results(
        train_df,
        test_df,
        report
    )

    print("\n" + report)

    print("\nArchivos generados:")
    print(TRAIN_PATH)
    print(TEST_PATH)
    print(REPORT_PATH)

    print("\nDIVISIÓN COMPLETADA CORRECTAMENTE")


if __name__ == "__main__":
    main()
