import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import issparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.preprocessing import build_preprocessor

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

# Cargar exclusivamente entrenamiento.
df = pd.read_csv(DATA_PATH, sep=";")
X_train = df.drop(columns=["y"])

# Construir y ajustar el preprocesador.
preprocessor = build_preprocessor(X_train)
X_transformed = preprocessor.fit_transform(X_train)

# Obtener nombres de las características transformadas.
feature_names = preprocessor.get_feature_names_out()

# Validar valores numéricos.
values = X_transformed.data if issparse(X_transformed) else X_transformed
all_finite = np.isfinite(values).all()

# Comprobar que una categoría nueva no provoca errores.
X_new = X_train.iloc[[0]].copy()
X_new["job"] = "categoria_no_observada"
X_new_transformed = preprocessor.transform(X_new)

print("=" * 60)
print("BANKINSIGHT MLOPS - VALIDACION REAL DEL PREPROCESAMIENTO")
print("=" * 60)

print(f"Registros originales: {X_train.shape[0]}")
print(f"Variables originales: {X_train.shape[1]}")
print(f"Dimensiones transformadas: {X_transformed.shape}")
print(f"Cantidad de características generadas: {len(feature_names)}")
print(f"Formato disperso (sparse): {issparse(X_transformed)}")
print(f"Todos los valores son finitos: {all_finite}")
print(f"Categoría nueva procesada: {X_new_transformed.shape[1] == X_transformed.shape[1]}")

print("\nPrimeras 15 características generadas:")
for name in feature_names[:15]:
    print(f" - {name}")

assert X_transformed.shape[0] == len(X_train)
assert X_transformed.shape[1] == len(feature_names)
assert all_finite
assert X_new_transformed.shape[1] == X_transformed.shape[1]

print("\nVALIDACION COMPLETADA CORRECTAMENTE")
