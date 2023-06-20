"""
Uses slice_metrics function from model.py to evaluate model on data slices
"""

import src.data as dt
import src.model as ml
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


# Import and Clean data
df = pd.read_csv('data/census.csv')
df = dt.clean_data(df)

# Import model training outputs
model = joblib.load('training_output/model.pkl')
encoder = joblib.load('training_output/encoder.pkl')
binarizer = joblib.load('training_output/binarizer.pkl')

# Get test data
test = train_test_split(df, test_size=0.20, random_state=42)[1]

# Transform data
categorical_cols = [x for x in test.columns if x not in test.select_dtypes(
    include=np.number).columns.tolist() and x not in ['salary']]

X_test, y_test = dt.process_data(test,
                                 categorical_features=categorical_cols,
                                 label='salary',
                                 encoder=encoder,
                                 lb=binarizer,
                                 training=False)[0:2]

# Generate predictions
preds = ml.inference(X_test, model)

# Get metrics for each
for col in test.drop('salary', axis=1).select_dtypes('object').columns:
    metrics = ml.slice_metrics(col, test, y_test, preds)

    with open(f'slice_output/{col}.txt', 'a') as f:
        string = metrics.to_string(header=True, index=True)
        f.write(string)
