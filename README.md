# Adult Income ML

An end-to-end machine learning project for predicting whether a person's annual income is above or below $50K using the Adult Census Income dataset.

The project demonstrates a practical machine learning workflow including data cleaning, feature preprocessing, model comparison, hyperparameter tuning, decision-threshold optimization, model persistence, prediction, and automated testing.

## Project Goals

- Build a reproducible machine learning pipeline.
- Clean and preprocess tabular data.
- Compare multiple classification models.
- Tune the selected model.
- Optimize the classification threshold using validation data.
- Save and reuse the trained model.
- Add automated tests for important components.

## Dataset

The project uses the Adult Census Income dataset.

The dataset contains:

- 32,561 rows
- 15 columns

After cleaning:

- 30,162 rows
- 15 columns

The target variable is `income`:

- `<=50K` → 0
- `>50K` → 1

## Project Structure

```text
Adult_income_ML/
├── data/
│   └── adult.csv
├── models/
│   ├── adult_income_model.joblib
│   └── adult_income_threshold.joblib
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── compare_models.py
│   ├── tune_gradient_boosting.py
│   └── tune_threshold.py
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_model.py
│   └── test_prediction.py
├── requirements.txt
└── README.md