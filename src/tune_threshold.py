from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from data import load_data, clean_data, split_data
from features import prepare_features, build_preprocessor


DATA_PATH = Path("data/adult.csv")


def tune_threshold():
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
    
    # 6. Build Gradient Boosting pipeline
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.10,
                    max_depth=4,
                    random_state=42
                )
            )
        ]
    )

    # 7. Train on TRAIN only
    print("Training Gradient Boosting...")
    model.fit(X_train, y_train)

    # 8. Get probability of class 1 (>50K)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # 9. Test different thresholds
    results = []

    for threshold in [i / 100 for i in range(20, 51)]:
    
        y_val_pred = (
            y_val_proba >= threshold
        ).astype(int)

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
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    # 10. Display results
    print("\nThreshold Results")
    print("=" * 70)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['threshold']:<12.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
        )

    # 11. Find threshold with best F1
    best_result = max(
        results,
        key=lambda x: x["f1"]
    )

    print("\nBest Threshold Based on F1")
    print("=" * 70)

    print(
        f"Threshold : {best_result['threshold']:.2f}"
    )
    print(
        f"Precision : {best_result['precision']:.4f}"
    )
    print(
        f"Recall    : {best_result['recall']:.4f}"
    )
    print(
        f"F1        : {best_result['f1']:.4f}"
    )


if __name__ == "__main__":
    tune_threshold()