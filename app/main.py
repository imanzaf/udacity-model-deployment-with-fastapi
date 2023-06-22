'''
This file contains the code for the API
'''

import sys
sys.path.append("..")

import joblib  # noqa: E402
import src.data as dt  # noqa: E402
import src.model as ml  # noqa: E402
import pandas as pd  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from fastapi import FastAPI  # noqa: E402


# Instantiate app
app = FastAPI()


# Define GET on root domain
@app.get("/")
async def say_hello():
    return {"greeting": "Hello World! Welcome to my app!"}


# Declare data object for model input
class Census(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example='State-gov')
    fnlgt: int = Field(example='77516')
    education: str = Field(example='Bachelors')
    education_num: int = Field(example=13)
    marital_status: str = Field(example='Never-married')
    occupation: str = Field(example='Adm-clerical')
    relationship: str = Field(example='Not-in-family')
    race: str = Field(example='White')
    sex: str = Field(example='Male')
    capital_gain: int = Field(example=2174)
    capital_loss: int = Field(example=0)
    hours_per_week: int = Field(example=40)
    native_country: str = Field(example='United-States')


# Define POST method for model inference
@app.post("/predict")
async def get_prediction(data: Census):
    # import model and encoder
    try:
        model = joblib.load('training_output/model.pkl')
        encoder = joblib.load('training_output/encoder.pkl')
    except FileNotFoundError:
        model = joblib.load('../training_output/model.pkl')
        encoder = joblib.load('../training_output/encoder.pkl')

    # Create dataframe
    try:
        df = pd.DataFrame([data.dict()])
    except AttributeError:
        df = pd.DataFrame([data])

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country"]

    # Transform data
    X = dt.process_data(df,
                        categorical_features=cat_features,
                        training=False,
                        encoder=encoder)[0]

    # Generate prediction
    pred = ml.inference(X, model)

    # Get non-binarized prediction
    if pred == 1:
        salary_pred = '>50K'
    else:
        salary_pred = '<=50K'

    return {"prediction": salary_pred}
