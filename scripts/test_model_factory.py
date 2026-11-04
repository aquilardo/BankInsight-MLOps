import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.training.model_factory import (
    build_model_pipeline,
    get_available_models
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

df = pd.read_csv(DATA_PATH, sep=";")
X_train = df.drop(columns=["y"])

print("=" * 60)
print("BANKINSIGHT MLOPS - VALIDACION DE FABRICA DE MODELOS")
print("=" * 60)

for model_name in get_available_models():
    pipeline = build_model_pipeline(X_train, model_name)

    print(f"\nModelo: {model_name}")
    print(f"Tipo de pipeline: {type(pipeline).__name__}")
    print(f"Pasos: {list(pipeline.named_steps.keys())}")
    print(
        "Clasificador:",
        type(pipeline.named_steps["classifier"]).__name__
    )

    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

print("\nMODELOS DISPONIBLES:")
print(get_available_models())

print("\nVALIDACION COMPLETADA CORRECTAMENTE")
