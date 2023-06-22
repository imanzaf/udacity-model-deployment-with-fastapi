'''
POST request to API
'''

import requests

# set up data for request
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

r = requests.post('https://imanzaf-udacity-app.onrender.com/predict',
                  json=data)
print(f"Request Status Code: {r.status_code}")
print(f"Request Response (Model Inference Result): {r.json()}")
