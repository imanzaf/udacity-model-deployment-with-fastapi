'''
Functions to clean and preprocess data
'''

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


def process_data(
        df,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """
    Process the data used in the machine learning pipeline.
    (function taken from udacity starter code)

    Processes the data using one hot encoding for the categorical
    features and a label binarizer for the labels. This can be used
    in either training or inference/validation.

    Note: depending on the type of model used, you may want to add
    in functionality that scales the continuous data.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label.
        Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array
        will be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True,
        otherwise returns the encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True,
        otherwise returns the binarizer passed in.
    """

    if label is not None:
        y = df[label]
        X = df.drop(label, axis=1)
    else:
        X = df
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
