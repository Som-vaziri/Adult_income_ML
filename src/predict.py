from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/adult_income_model.joblib")
THRESHOLD_PATH = Path("models/adult_income_threshold.joblib")


def predict_income(person: dict):
    """Predict whether a person's income is >50K or <=50K."""

    # Load saved model and threshold
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    # Convert the person's data into a DataFrame
    X_new = pd.DataFrame([person])

    # Get probability of income >50K
    probability = model.predict_proba(X_new)[0, 1]

    # Apply the selected threshold
    prediction = (
        ">50K"
        if probability >= threshold
        else "<=50K"
    )

    return prediction, probability


if __name__ == "__main__":

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

    prediction, probability = predict_income(person)

    print(f"Probability of >50K: {probability:.4f}")
    print(f"Prediction: {prediction}")