from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/adult_income_model.joblib")


def test_saved_model():
    model = joblib.load(MODEL_PATH)

    person = {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 150000,
        "education": "Bachelors",
        "marital.status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital.gain": 0,
        "capital.loss": 0,
        "hours.per.week": 40,
        "native.country": "United-States"
    }

    X_new = pd.DataFrame([person])

    probability = model.predict_proba(X_new)[0, 1]

    assert 0 <= probability <= 1