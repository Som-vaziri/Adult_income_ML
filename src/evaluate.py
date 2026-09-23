from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data import load_data, clean_data, split_data
from features import prepare_features


DATA_PATH = Path("data/adult.csv")
MODEL_PATH = Path("models/adult_income_model.joblib")
THRESHOLD_PATH = Path("models/adult_income_threshold.joblib")


def evaluate_model():
    """Evaluate the saved model on the test dataset."""

    # 1. Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # 2. Prepare features
    X, y, _, _ = prepare_features(df)

    # 3. Load the trained pipeline
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    # 4. Recreate the same train/validation/test split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Get probabilities for the positive class
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # 6. Apply the selected threshold
    y_pred = (
        y_test_proba >= threshold
    ).astype(int)

    # 7. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Evaluation")
    print("================")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()