from pathlib import Path

import pandas as pd

from src.data import load_data, clean_data


DATA_PATH = Path("data/adult.csv")


def test_load_data():
    df = load_data(DATA_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_clean_data():
    df = pd.DataFrame({
        "name": [" Alice ", "Bob", "?"],
        "age": [25, 30, 40]
    })

    cleaned_df = clean_data(df)

    assert len(cleaned_df) == 2
    assert cleaned_df.iloc[0]["name"] == "Alice"
    assert "?" not in cleaned_df["name"].values