'''
Functions to clean and preprocess data
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

def clean_data(df):
    """
    Cleans data by removing trailing whitespaces

    :param data: (pandas dataframe) Raw data

    :returns data: (pandas dataframe) Clean data
    """
    # remove whitespaces from column names
    for column in df.columns:
        df.rename({column: column.strip()}, axis=1, inplace=True)

    # remove whitespaces from text column values
    for column in df.columns:
        if column not in df.select_dtypes(include=np.number).columns.tolist():
            df[column] = df[column].apply(lambda x: x.strip())

    return df


# incomplete
def transform_data(X, label=None):
    """
    Transforms data using One Hot Encoding for categorical variables.
    If label provided, splits data into X, y, before encoding

    :param df: Clean data
    :param label: Name of label column

    :returns X: encoded features
    :returns y: label
    :returns encoder: encoder object fit on provided data
    """
    if label != None:
        y = X[label]
        X = X.drop(label, axis=1, inplace=True)
    else:
        y = "Label not provided"

    # get list of categorical and numeric columns
    categorical_cols = [x for x in X.columns if x not in
                            X.select_dtypes(include=np.number).columns.tolist()]
    numeric_cols = [x for x in X.columns if x not in categorical_cols]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = encoder.fit_transform(X[categorical_cols])
    X = pd.concat([df_categorical, X[numeric_cols]], axis=1)

    return X, y, encoder