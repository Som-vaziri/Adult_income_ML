from pathlib import Path

import time

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data import load_data, clean_data, split_data
from features import prepare_features, build_preprocessor


DATA_PATH = Path("data/adult.csv")


def compare_models():

    # 1. Load and clean data
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # 2. Prepare features
    X, y, numerical_cols, categorical_cols = prepare_features(df)

    # 3. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 4. Define models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=100,
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            max_iter=2000,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
    }

    results = []

    # 5. Train and evaluate each model
    for name, classifier in models.items():

        print(f"\nTraining: {name}")

        preprocessor = build_preprocessor(
            numerical_cols,
            categorical_cols
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", classifier)
            ]
        )

        start_time = time.time()

        try:
            pipeline.fit(X_train, y_train)

            y_val_pred = pipeline.predict(X_val)

            training_time = time.time() - start_time

            accuracy = accuracy_score(
                y_val,
                y_val_pred
            )

            precision = precision_score(
                y_val,
                y_val_pred,
                zero_division=0
            )

            recall = recall_score(
                y_val,
                y_val_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_val,
                y_val_pred,
                zero_division=0
            )

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "Training Time (s)": training_time
            })

            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1       : {f1:.4f}")
            print(f"Time     : {training_time:.2f}s")

        except Exception as e:
            print(f"FAILED: {e}")

    # 6. Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    results.sort(
        key=lambda x: x["F1"],
        reverse=True
    )

    for result in results:

        print(
            f"{result['Model']:25s} "
            f"Accuracy={result['Accuracy']:.4f}  "
            f"Precision={result['Precision']:.4f}  "
            f"Recall={result['Recall']:.4f}  "
            f"F1={result['F1']:.4f}  "
            f"Time={result['Training Time (s)']:.2f}s"
        )


if __name__ == "__main__":
    compare_models()