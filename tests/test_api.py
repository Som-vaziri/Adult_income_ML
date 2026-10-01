from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api import app


def test_api():
    client = TestClient(app)

    # Test /
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Adult Income API is running"
    }

    # Test /status
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }

def test_model_info():
    client = TestClient(app)

    # Test /model-info
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
            "model" : "Gradient Boosting" ,
            "threshold" : 0.39
    }

def test_predict():
    client = TestClient(app)

    # Test /predict
    response = client.post(
        "/predict",
        json= {
             "age": 39,
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        } 
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert "threshold" in response.json()

def test_predict_invalid_input():
    client = TestClient(app)

    # Test /predict with invalid input
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422  

def test_predict_model_error():
    client = TestClient(app)

    with patch("src.api.predict_income", side_effect=Exception("Model error")):
        response = client.post(
            "/predict",
            json={
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        )

    assert response.status_code == 500


def test_predict_invalid_data_type():
    client = TestClient(app)

    response = client.post(
        "/predict",
        json={
            "age": "hello",
            "workclass": "State-gov",
            "fnlwgt": 77516,
            "education": "Bachelors",
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )

    assert response.status_code == 422