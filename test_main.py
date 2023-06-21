'''
Test cases for FastAPI app
'''

from fastapi.testclient import TestClient
from app.main import app
import json
import pytest


client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World! Welcome to my app!"}


@pytest.fixture(scope='module')
def post_request_low_data():
    data = {"age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 2174,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
            }
    return data


def test_post_low_salary(post_request_low_data):
    data = json.dumps(post_request_low_data)
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}


@pytest.fixture(scope='module')
def post_request_high_data():
    data = {"age": 29,
            "workclass": "Private",
            "fnlgt": 185908,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "Black",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 55,
            "native_country": "United-States"
            }
    return data


def test_post_high_salary(post_request_high_data):
    data = json.dumps(post_request_high_data)
    r = client.post("/predict", data=data)
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}
