from urllib import response

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

