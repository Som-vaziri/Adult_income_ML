from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict_income


app = FastAPI()
  
class Person(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str 
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str





@app.post("/predict")
def predict(person: Person):

    person_data = {
        "age": person.age,
        "workclass": person.workclass,
        "fnlwgt": person.fnlwgt,
        "education": person.education,
        "marital.status": person.marital_status,
        "occupation": person.occupation,
        "relationship": person.relationship,
        "race": person.race,
        "sex": person.sex,
        "capital.gain": person.capital_gain,
        "capital.loss": person.capital_loss,
        "hours.per.week": person.hours_per_week,
        "native.country": person.native_country,
    }

    try:
        prediction, probability = predict_income(person_data)
    except Exception:
       raise HTTPException(status_code=500, detail="Error occurred while making prediction")

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": 0.39
    }

@app.get("/")
def home():
    return {"message": "Adult Income API is running"}

@app.get("/model-info")
def model_info():
    return {
        "model": "Gradient Boosting",
        "threshold": 0.39
}

@app.get("/greet")
def greet(name: str):
    return {"message": f"Hello, {name}!"}


@app.get("/person/{person_id}")
def person_id(person_id: int):
    return {"person_id": person_id}

@app.get("/status")
def status():
    return {
        "status": "healthy",}
        