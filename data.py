'''
Functions to clean and preprocess data
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def clean_data(df):
    """
    Cleans data by removing trailing whitespaces

    :param df: (pandas dataframe) Raw data

    :returns df: (pandas dataframe) Clean data
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

    :param X: Clean data
    :param label: Name of label column

    :returns X: (pandas dataframe) encoded features
    :returns y: (array) label
    :returns encoder: encoder object fit on provided data
    """
    if label is not None:
        y = pd.DataFrame(X[label])
        X = X.drop(label, axis=1, inplace=True)
    else:
        y = "Label not provided"

    # get list of categorical and numeric columns
    categorical_cols = [x for x in X.columns if x not in
                        X.select_dtypes(include=np.number).columns.tolist()]
    numeric_cols = [x for x in X.columns if x not in categorical_cols]

    # one hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = encoder.fit_transform(X[categorical_cols])
    X = pd.concat([df_categorical, X[numeric_cols]], axis=1)

    # binarize label
    if y != "Label not provided":
        lb = LabelBinarizer()
        y = lb.fit_transform(y.values).ravel()

    return X, y, encoder
