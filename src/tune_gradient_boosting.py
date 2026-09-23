from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from data import load_data, clean_data, split_data
from features import prepare_features, build_preprocessor


DATA_PATH = Path("data/adult.csv")


def tune_gradient_boosting():

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y, numerical_cols, categorical_cols = prepare_features(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    parameter_combinations = [
        {
            "n_estimators": 200,
            "learning_rate": 0.10,
            "max_depth": 3,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.10,
            "max_depth": 3,
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.10,
            "max_depth": 3,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.10,
            "max_depth": 4,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.10,
            "max_depth": 4,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
        },
    ]

    
    for params in parameter_combinations:

        print("\nTraining:", params)

        preprocessor = build_preprocessor(
            numerical_cols,
            categorical_cols
        )

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        **params,
                        random_state=42
                    )
                )
            ]
        )

        model.fit(X_train, y_train)

        y_val_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_val_pred)
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")


if __name__ == "__main__":
    tune_gradient_boosting()