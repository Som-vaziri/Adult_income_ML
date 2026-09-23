import pandas as pd

from src.features import (
    prepare_features,
    build_preprocessor
)

def test_prepare_features():
    df = pd.DataFrame({
        "age": [25, 40],
        "education.num": [10, 13],
        "education": ["Bachelors", "Masters"],
        "workclass": ["Private", "Government"],
        "income": ["<=50K", ">50K"]
    })

    X, y, numerical_cols, categorical_cols = prepare_features(df)

    # Target should be separated from the features
    assert "income" not in X.columns

    # education.num should be excluded
    assert "education.num" not in X.columns

    # Target should be converted to 0/1
    assert list(y) == [0, 1]

    # Check feature types
    assert "age" in numerical_cols
    assert "education" in categorical_cols
    assert "workclass" in categorical_cols

def test_build_preprocessor():
    numerical_cols = ["age"]
    categorical_cols = ["education"]

    preprocessor = build_preprocessor(
        numerical_cols,
        categorical_cols
    )

    
    transformer_names = [
    name
    for name, _, _ in preprocessor.transformers
    ]

    assert "num" in transformer_names
    assert "cat" in transformer_names