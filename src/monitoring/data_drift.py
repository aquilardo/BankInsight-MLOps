"""
Detector de Data Drift para BankInsight MLOps.

Compara distribuciones de referencia y datos actuales.

Métricas:
- PSI para variables numéricas y categóricas.
- Kolmogorov-Smirnov para variables numéricas.
- Jensen-Shannon divergence para variables categóricas.
- Detección de categorías nuevas.
- Monitorización de valores faltantes.

La variable objetivo y no debe incluirse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp


EPSILON = 1e-6

PSI_MODERATE_THRESHOLD = 0.10
PSI_HIGH_THRESHOLD = 0.20

KS_MODERATE_THRESHOLD = 0.10
KS_HIGH_THRESHOLD = 0.20

JS_MODERATE_THRESHOLD = 0.05
JS_HIGH_THRESHOLD = 0.10

MISSING_MODERATE_THRESHOLD = 0.05
MISSING_HIGH_THRESHOLD = 0.10

MISSING_CATEGORY = "__BANKINSIGHT_MISSING__"


def _severity(value: float | None) -> str:
    """Clasifica una métrica continua."""
    if value is None or not np.isfinite(value):
        return "insufficient_data"

    if value >= PSI_HIGH_THRESHOLD:
        return "high"

    if value >= PSI_MODERATE_THRESHOLD:
        return "moderate"

    return "low"


def _severity_from_thresholds(
    value: float,
    moderate: float,
    high: float
) -> str:
    """Clasifica una métrica con umbrales propios."""
    if not np.isfinite(value):
        return "insufficient_data"

    if value >= high:
        return "high"

    if value >= moderate:
        return "moderate"

    return "low"


def _combine_severities(*values: str) -> str:
    """Devuelve la mayor severidad entre varias métricas."""
    priority = {
        "insufficient_data": -1,
        "low": 0,
        "moderate": 1,
        "high": 2
    }

    valid = [
        value for value in values
        if value in priority and value != "insufficient_data"
    ]

    if not valid:
        return "insufficient_data"

    return max(valid, key=lambda value: priority[value])


def _calculate_psi(
    reference_counts: np.ndarray,
    current_counts: np.ndarray
) -> float:
    """Calcula Population Stability Index con suavizado."""
    reference = reference_counts.astype(float) + EPSILON
    current = current_counts.astype(float) + EPSILON

    reference /= reference.sum()
    current /= current.sum()

    psi = np.sum(
        (current - reference) * np.log(current / reference)
    )

    return float(psi)


def _numeric_drift(
    reference: pd.Series,
    current: pd.Series,
    feature: str
) -> dict[str, Any]:
    """Calcula drift de una variable numérica."""

    reference_numeric = pd.to_numeric(
        reference, errors="coerce"
    )

    current_numeric = pd.to_numeric(
        current, errors="coerce"
    )

    reference_invalid = (
        reference.notna() & reference_numeric.isna()
    )

    current_invalid = (
        current.notna() & current_numeric.isna()
    )

    reference_finite_mask = np.isfinite(
        reference_numeric.fillna(0).to_numpy(dtype=float)
    )

    current_finite_mask = np.isfinite(
        current_numeric.fillna(0).to_numpy(dtype=float)
    )

    reference_infinite = (
        reference_numeric.notna().to_numpy()
        & ~reference_finite_mask
    )

    current_infinite = (
        current_numeric.notna().to_numpy()
        & ~current_finite_mask
    )

    if reference_infinite.any():
        raise ValueError(
            f"La referencia contiene valores infinitos en {feature}."
        )

    if current_infinite.any():
        raise ValueError(
            f"El conjunto actual contiene valores infinitos en {feature}."
        )

    reference_missing_rate = float(
        reference_numeric.isna().mean()
    )

    current_missing_rate = float(
        current_numeric.isna().mean()
    )

    missing_rate_change = (
        current_missing_rate - reference_missing_rate
    )

    reference_valid = reference_numeric.dropna().to_numpy(
        dtype=float
    )

    current_valid = current_numeric.dropna().to_numpy(
        dtype=float
    )

    result = {
        "feature": feature,
        "type": "numeric",
        "reference_missing_rate": reference_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_change": float(missing_rate_change),
        "reference_invalid_rate": float(reference_invalid.mean()),
        "current_invalid_rate": float(current_invalid.mean()),
        "reference_mean": None,
        "current_mean": None,
        "psi": None,
        "ks_statistic": None,
        "ks_p_value": None,
        "ks_significant": None,
        "psi_severity": "insufficient_data",
        "ks_severity": "insufficient_data",
        "missing_severity": _severity_from_thresholds(
            max(0.0, missing_rate_change),
            MISSING_MODERATE_THRESHOLD,
            MISSING_HIGH_THRESHOLD
        ),
        "severity": "insufficient_data",
        "alert": False
    }

    if len(reference_valid) == 0 or len(current_valid) == 0:
        result["alert"] = (
            missing_rate_change >= MISSING_MODERATE_THRESHOLD
            or reference_invalid.any()
            or current_invalid.any()
        )
        return result

    result["reference_mean"] = float(reference_valid.mean())
    result["current_mean"] = float(current_valid.mean())

    # Los intervalos se calculan únicamente con la referencia.
    quantiles = np.linspace(0, 1, 11)

    internal_edges = np.unique(
        np.quantile(reference_valid, quantiles)
    )

    if len(internal_edges) > 1:
        bin_edges = np.concatenate((
            [-np.inf],
            internal_edges[1:-1],
            [np.inf]
        ))
    else:
        # Variable constante en la referencia.
        constant_value = internal_edges[0]
        bin_edges = np.array([
            -np.inf,
            constant_value,
            np.inf
        ])

    reference_counts, _ = np.histogram(
        reference_valid,
        bins=bin_edges
    )

    current_counts, _ = np.histogram(
        current_valid,
        bins=bin_edges
    )

    # Se incorpora un intervalo adicional para los valores faltantes.
    reference_counts = np.append(
        reference_counts,
        int(reference_numeric.isna().sum())
    )

    current_counts = np.append(
        current_counts,
        int(current_numeric.isna().sum())
    )

    psi = _calculate_psi(
        reference_counts,
        current_counts
    )

    ks_result = ks_2samp(
        reference_valid,
        current_valid,
        alternative="two-sided",
        method="auto"
    )

    psi_severity = _severity(psi)

    ks_severity = _severity_from_thresholds(
        float(ks_result.statistic),
        KS_MODERATE_THRESHOLD,
        KS_HIGH_THRESHOLD
    )

    severity = _combine_severities(
        psi_severity,
        ks_severity,
        result["missing_severity"]
    )

    invalid_alert = (
        bool(reference_invalid.any())
        or bool(current_invalid.any())
    )

    result.update({
        "psi": psi,
        "ks_statistic": float(ks_result.statistic),
        "ks_p_value": float(ks_result.pvalue),
        "ks_significant": bool(ks_result.pvalue < 0.05),
        "psi_severity": psi_severity,
        "ks_severity": ks_severity,
        "severity": severity,
        "alert": (
            severity in ("moderate", "high")
            or invalid_alert
        )
    })

    return result


def _normalize_categories(series: pd.Series) -> pd.Series:
    """Normaliza categorías conservando unknown como categoría válida."""
    values = series.astype("string").str.strip()

    collision = (
        values == MISSING_CATEGORY
    ).fillna(False).any()

    if collision:
        raise ValueError(
            f"El valor reservado {MISSING_CATEGORY} "
            "aparece como categoría real."
        )

    return values.fillna(MISSING_CATEGORY).astype(str)


def _categorical_drift(
    reference: pd.Series,
    current: pd.Series,
    feature: str
) -> dict[str, Any]:
    """Calcula drift de una variable categórica."""

    reference_values = _normalize_categories(reference)
    current_values = _normalize_categories(current)

    reference_counts = reference_values.value_counts()
    current_counts = current_values.value_counts()

    categories = sorted(
        set(reference_counts.index)
        | set(current_counts.index)
    )

    reference_aligned = np.array([
        reference_counts.get(category, 0)
        for category in categories
    ], dtype=float)

    current_aligned = np.array([
        current_counts.get(category, 0)
        for category in categories
    ], dtype=float)

    psi = _calculate_psi(
        reference_aligned,
        current_aligned
    )

    reference_distribution = (
        reference_aligned + EPSILON
    )

    current_distribution = (
        current_aligned + EPSILON
    )

    reference_distribution /= reference_distribution.sum()
    current_distribution /= current_distribution.sum()

    js_distance = jensenshannon(
        reference_distribution,
        current_distribution,
        base=2
    )

    js_divergence = float(js_distance ** 2)

    reference_categories = set(reference_values.unique())
    current_categories = set(current_values.unique())

    new_categories = sorted(
        current_categories - reference_categories
    )

    reference_missing_rate = float(
        (reference_values == MISSING_CATEGORY).mean()
    )

    current_missing_rate = float(
        (current_values == MISSING_CATEGORY).mean()
    )

    missing_rate_change = (
        current_missing_rate - reference_missing_rate
    )

    psi_severity = _severity(psi)

    js_severity = _severity_from_thresholds(
        js_divergence,
        JS_MODERATE_THRESHOLD,
        JS_HIGH_THRESHOLD
    )

    missing_severity = _severity_from_thresholds(
        max(0.0, missing_rate_change),
        MISSING_MODERATE_THRESHOLD,
        MISSING_HIGH_THRESHOLD
    )

    severity = _combine_severities(
        psi_severity,
        js_severity,
        missing_severity
    )

    return {
        "feature": feature,
        "type": "categorical",
        "reference_missing_rate": reference_missing_rate,
        "current_missing_rate": current_missing_rate,
        "missing_rate_change": float(missing_rate_change),
        "psi": psi,
        "js_divergence": js_divergence,
        "psi_severity": psi_severity,
        "js_severity": js_severity,
        "missing_severity": missing_severity,
        "new_categories": new_categories,
        "severity": severity,
        "alert": (
            severity in ("moderate", "high")
            or len(new_categories) > 0
        )
    }


def compare_datasets(
    reference: pd.DataFrame,
    current: pd.DataFrame
) -> dict[str, Any]:
    """
    Compara las variables predictoras entre referencia y datos actuales.

    Ambos DataFrames deben contener las mismas columnas.
    La variable objetivo y está prohibida.
    """

    if not isinstance(reference, pd.DataFrame):
        raise TypeError("La referencia debe ser un DataFrame.")

    if not isinstance(current, pd.DataFrame):
        raise TypeError("El conjunto actual debe ser un DataFrame.")

    if reference.empty:
        raise ValueError("El conjunto de referencia está vacío.")

    if current.empty:
        raise ValueError("El conjunto actual está vacío.")

    if "y" in reference.columns or "y" in current.columns:
        raise ValueError(
            "La variable objetivo 'y' no debe incluirse en Data Drift."
        )

    if reference.columns.duplicated().any():
        raise ValueError(
            "La referencia contiene columnas duplicadas."
        )

    if current.columns.duplicated().any():
        raise ValueError(
            "El conjunto actual contiene columnas duplicadas."
        )

    reference_columns = set(reference.columns)
    current_columns = set(current.columns)

    missing_columns = sorted(
        reference_columns - current_columns
    )

    extra_columns = sorted(
        current_columns - reference_columns
    )

    if missing_columns or extra_columns:
        raise ValueError(
            "Las columnas no coinciden. "
            f"Faltantes: {missing_columns}. "
            f"Adicionales: {extra_columns}."
        )

    current = current[reference.columns]

    numeric_features = reference.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = [
        column
        for column in reference.columns
        if column not in numeric_features
    ]

    results = []

    for feature in numeric_features:
        results.append(
            _numeric_drift(
                reference[feature],
                current[feature],
                feature
            )
        )

    for feature in categorical_features:
        results.append(
            _categorical_drift(
                reference[feature],
                current[feature],
                feature
            )
        )

    results_df = pd.DataFrame(results)

    alert_count = int(results_df["alert"].sum())

    summary = {
        "status": (
            "drift_detected"
            if alert_count > 0
            else "no_significant_drift"
        ),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "features_analyzed": int(len(results_df)),
        "numeric_features": int(len(numeric_features)),
        "categorical_features": int(len(categorical_features)),
        "features_with_alerts": alert_count,
        "high_severity": int(
            (results_df["severity"] == "high").sum()
        ),
        "moderate_severity": int(
            (results_df["severity"] == "moderate").sum()
        ),
        "new_categories_detected": int(
            sum(
                len(item.get("new_categories", []))
                for item in results
            )
        ),
        "features_with_missing_alerts": int(
            sum(
                item.get("missing_severity")
                in ("moderate", "high")
                for item in results
            )
        )
    }

    return {
        "summary": summary,
        "results": results
    }
