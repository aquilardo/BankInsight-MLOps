
"""
Módulo de preprocesamiento de características para BankInsight MLOps.

Responsabilidades:
- Identificar variables numéricas y categóricas.
- Imputar valores faltantes.
- Escalar variables numéricas.
- Codificar variables categóricas.
- Construir un preprocesador reutilizable para distintos modelos.

IMPORTANTE:
El preprocesador debe ajustarse únicamente con los datos de entrenamiento.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X):
    """
    Construye un pipeline de preprocesamiento a partir de X.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables predictoras, sin la columna objetivo.

    Returns
    -------
    ColumnTransformer
        Preprocesador para variables numéricas y categóricas.
    """

    if X.empty:
        raise ValueError("El conjunto de características está vacío.")

    if X.columns.duplicated().any():
        raise ValueError("Existen nombres de columnas duplicados.")

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    if not numeric_features and not categorical_features:
        raise ValueError("No se identificaron variables para procesar.")

    # Pipeline para variables numéricas.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Pipeline para variables categóricas.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    # Integración de ambos pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )

    return preprocessor


def get_feature_groups(X):
    """
    Devuelve los nombres de las variables numéricas y categóricas.
    """

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    return {
        "numeric": numeric_features,
        "categorical": categorical_features
    }
