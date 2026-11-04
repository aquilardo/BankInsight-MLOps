import sys
from pathlib import Path

import pandas as pd

# Incorporar la raíz del proyecto a la ruta de módulos.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocessing import (
    build_preprocessor,
    get_feature_groups
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

df = pd.read_csv(DATA_PATH, sep=";")

X_train = df.drop(columns=["y"])

groups = get_feature_groups(X_train)
preprocessor = build_preprocessor(X_train)

print("BANKINSIGHT MLOPS")
print("VERIFICACION DEL PREPROCESAMIENTO")
print("=" * 50)

print(f"Registros: {len(X_train)}")
print(f"Variables numericas: {len(groups['numeric'])}")
print(f"Variables categoricas: {len(groups['categorical'])}")

print("\nVariables numericas:")
print(groups["numeric"])

print("\nVariables categoricas:")
print(groups["categorical"])

print("\nPreprocesador:")
print(preprocessor)

print("\nVERIFICACION COMPLETADA")
