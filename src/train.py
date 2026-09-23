from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from data import load_data, clean_data, split_data
from features import prepare_features, build_preprocessor


DATA_PATH = Path("data/adult.csv")
MODEL_PATH = Path("models/adult_income_model.joblib")
THRESHOLD_PATH = Path("models/adult_income_threshold.joblib")


def train_model():
    """Train the final Gradient Boosting model on the training set."""

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Clean data
    df = clean_data(df)

    # 3. Prepare features
    X, y, numerical_cols, categorical_cols = prepare_features(df)

    # 4. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 5. Build preprocessing
    preprocessor = build_preprocessor(
        numerical_cols,
        categorical_cols
    )

    # 6. Define final Gradient Boosting model
    classifier = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.10,
        max_depth=4,
        random_state=42
    )

    # 7. Build pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier)
        ]
    )

    # 8. Train on TRAIN only
    print("Training final Gradient Boosting model...")
    pipeline.fit(X_train, y_train)

    # 9. Save the complete pipeline
    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    # 10. Save selected classification threshold
    threshold = 0.39

    joblib.dump(
        threshold,
        THRESHOLD_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nThreshold saved to:")
    print(THRESHOLD_PATH)


if __name__ == "__main__":
    train_model()