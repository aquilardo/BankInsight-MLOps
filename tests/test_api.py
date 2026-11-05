import pytest
from fastapi.testclient import TestClient

from src.api import main as api_module


class FakeModel:
    def predict(self, dataframe):
        assert len(dataframe) == 1
        return ["yes"]


@pytest.fixture
def client(monkeypatch):
    fake_bundle = {
        "model": FakeModel(),
        "name": "BankInsight-Campaign-Subscription",
        "version": "1",
        "alias": "candidate"
    }

    monkeypatch.setattr(
        api_module,
        "load_registered_model",
        lambda: fake_bundle
    )

    with TestClient(api_module.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    return {
        "age": 45,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": -1.8,
        "cons.price.idx": 92.893,
        "cons.conf.idx": -46.2,
        "euribor3m": 1.313,
        "nr.employed": 5099.1
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == (
        "BankInsight-Campaign-Subscription"
    )
    assert response.json()["model_version"] == "1"
    assert response.json()["model_alias"] == "candidate"


def test_successful_prediction(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == "yes"
    assert response.json()["model_version"] == "1"


def test_prediction_rejects_missing_field(client, valid_payload):
    valid_payload.pop("nr.employed")

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_prediction_rejects_extra_field(client, valid_payload):
    valid_payload["unexpected_field"] = "invalid"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_prediction_rejects_invalid_age(client, valid_payload):
    valid_payload["age"] = 16

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422
