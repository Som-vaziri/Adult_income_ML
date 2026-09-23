from src.predict import predict_income


def test_predict_income():
    person = {
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 150000,
        "education": "Bachelors",
        "marital.status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital.gain": 0,
        "capital.loss": 0,
        "hours.per.week": 40,
        "native.country": "United-States"
    }

    prediction, probability = predict_income(person)

    assert prediction in ["<=50K", ">50K"]
    assert 0 <= probability <= 1