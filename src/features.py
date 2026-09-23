import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "income"

EXCLUDED_COLUMNS = [
    "education.num",
]


def prepare_features(df: pd.DataFrame):
    """Separate features and target and identify feature types."""

    df = df.copy()

    # Separate target
    y = df[TARGET_COLUMN].map({
        "<=50K": 0,
        ">50K": 1
    })

    # Remove target and excluded columns
    X = df.drop(
        columns=[TARGET_COLUMN] + EXCLUDED_COLUMNS,
        errors="ignore"
    )

    # Automatically detect numerical features
    numerical_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Automatically detect categorical features
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    return X, y, numerical_cols, categorical_cols


def build_preprocessor(
    numerical_cols,
    categorical_cols
):
    """Build preprocessing based on detected feature types."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_cols
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols
            ),
        ]
    )

    return preprocessor