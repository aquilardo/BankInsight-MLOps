import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_drift import compare_datasets


def test_identical_datasets_have_no_drift():
    reference = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "job": ["admin", "technician", "admin", "services", "blue-collar"]
    })

    report = compare_datasets(reference, reference.copy())

    assert report["summary"]["status"] == "no_significant_drift"
    assert report["summary"]["features_with_alerts"] == 0


def test_numeric_drift_is_detected():
    reference = pd.DataFrame({
        "age": np.arange(20, 120, dtype=float)
    })

    current = pd.DataFrame({
        "age": np.arange(35, 135, dtype=float)
    })

    report = compare_datasets(reference, current)
    result = report["results"][0]

    assert result["alert"] is True
    assert result["severity"] == "high"
    assert result["ks_statistic"] >= 0.10


def test_new_categories_are_detected():
    reference = pd.DataFrame({
        "job": ["admin"] * 50 + ["technician"] * 50
    })

    current = reference.copy()
    current.loc[:29, "job"] = "new_category"

    report = compare_datasets(reference, current)
    result = report["results"][0]

    assert "new_category" in result["new_categories"]
    assert result["alert"] is True


def test_missing_values_generate_alert():
    reference = pd.DataFrame({
        "age": np.arange(100, dtype=float)
    })

    current = reference.copy()
    current.loc[:19, "age"] = np.nan

    report = compare_datasets(reference, current)
    result = report["results"][0]

    assert result["current_missing_rate"] == pytest.approx(0.20)
    assert result["missing_severity"] == "high"
    assert result["alert"] is True


def test_constant_variable_change_is_detected():
    reference = pd.DataFrame({
        "constant_feature": [5.0] * 100
    })

    current = pd.DataFrame({
        "constant_feature": [6.0] * 100
    })

    report = compare_datasets(reference, current)
    result = report["results"][0]

    assert result["ks_statistic"] == pytest.approx(1.0)
    assert result["severity"] == "high"
    assert result["alert"] is True


def test_invalid_numeric_values_are_detected():
    reference = pd.DataFrame({
        "numeric_feature": [1.0] * 100
    })

    current = pd.DataFrame({
        "numeric_feature": [1.0] * 90 + ["ERROR"] * 10
    })

    report = compare_datasets(reference, current)
    result = report["results"][0]

    assert result["current_invalid_rate"] == pytest.approx(0.10)
    assert result["alert"] is True


def test_target_variable_is_rejected():
    reference = pd.DataFrame({
        "age": [20, 30, 40],
        "y": ["no", "yes", "no"]
    })

    with pytest.raises(ValueError, match="variable objetivo"):
        compare_datasets(reference, reference.copy())


def test_incompatible_columns_are_rejected():
    reference = pd.DataFrame({
        "age": [20, 30]
    })

    current = pd.DataFrame({
        "income": [1000, 2000]
    })

    with pytest.raises(ValueError, match="columnas no coinciden"):
        compare_datasets(reference, current)


def test_empty_datasets_are_rejected():
    reference = pd.DataFrame({
        "age": pd.Series(dtype=float)
    })

    current = pd.DataFrame({
        "age": pd.Series(dtype=float)
    })

    with pytest.raises(ValueError, match="referencia está vacío"):
        compare_datasets(reference, current)


def test_infinite_values_are_rejected():
    reference = pd.DataFrame({
        "age": [20.0, 30.0, 40.0]
    })

    current = pd.DataFrame({
        "age": [20.0, np.inf, 40.0]
    })

    with pytest.raises(ValueError, match="valores infinitos"):
        compare_datasets(reference, current)
