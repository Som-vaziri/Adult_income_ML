from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline

from data import load_data, clean_data, split_data
from features import prepare_features, build_preprocessor
from sklearn.model_selection import cross_validate

DATA_PATH = Path("data/adult.csv")


def tune_gradient_boosting():

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y, numerical_cols, categorical_cols = prepare_features(df)
    X_train, _, X_test, y_train, _, y_test = split_data(X, y)

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

        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=5,
            scoring=["accuracy", "precision", "recall", "f1"],
        )

        f1_scores = scores["test_f1"]

        print(f"Accuracy mean : {scores['test_accuracy'].mean():.4f}")
        print(f"Precision mean: {scores['test_precision'].mean():.4f}")
        print(f"Recall mean   : {scores['test_recall'].mean():.4f}")
        print(f"F1 folds      : {[round(score, 4) for score in f1_scores]}")
        print(f"F1 mean       : {f1_scores.mean():.4f}")
        print(f"F1 std        : {f1_scores.std():.4f}")
        

if __name__ == "__main__":
    tune_gradient_boosting()