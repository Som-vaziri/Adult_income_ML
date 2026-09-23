import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split


def load_data(path: Path) -> pd.DataFrame:
    """Load the dataset from a CSV file."""

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean whitespace and handle missing values."""

    # Remove leading/trailing whitespace from string values
    df = df.map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

    # Replace '?' with missing values
    df = df.replace("?", pd.NA)

    # Remove rows containing missing values
    df = df.dropna()

    return df


def split_data(X, y):
    """Split data into training, validation, and test sets."""

    # First: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Second: split remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test