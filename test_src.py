'''
Test model and data functions located at src/
'''

from sklearn.model_selection import train_test_split
import src.model as ml
import src.data as dt
import pytest
import pandas as pd


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv('data/census.csv')
    df = dt.clean_data(df)
    return df


@pytest.fixture(scope='module')
def cat_cols():
    return ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country']


def test_transform_data(df, cat_cols, label='salary'):
    '''
    Test transform_data

    :param df: Clean data
    :param label: Name of label column
    '''
    try:
        X, y, encoder, lb = dt.process_data(df, cat_cols, label)
        assert X.shape[0] != 0
        assert y.shape[0] != 0
    except AssertionError as err:
        print('transform_data failed: no data returned')
        raise err

    pytest.X = X
    pytest.y = y


@pytest.fixture(scope='module')
def transform_data():
    # Split data
    X = pytest.X
    y = pytest.y

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


@pytest.fixture(scope='module')
def X_train(transform_data):
    return transform_data[0]


@pytest.fixture(scope='module')
def X_test(transform_data):
    return transform_data[1]


@pytest.fixture(scope='module')
def y_train(transform_data):
    return transform_data[2]


@pytest.fixture(scope='module')
def y_test(transform_data):
    return transform_data[3]


def test_train_model(X_train, y_train):
    '''
    Test train_model

    :param X_train: Training data
    :param y_train: Training labels
    '''
    try:
        model = ml.train_model(X_train, y_train)
        assert model is not None
    except AssertionError as err:
        print('train_model failed: returned model object is empty')
        raise err

    pytest.lgbm = model


def test_inference(X_test):
    '''
    Test inference function

    :param X_test: Test data
    '''
    lgbm = pytest.lgbm
    try:
        preds = ml.inference(X_test, lgbm)
        assert preds.shape[0] != 0
    except AssertionError as err:
        print('inference failed: No predictions generated')
        raise err

    pytest.preds = preds


@pytest.fixture(scope='module')
def feature():
    return 'education'


@pytest.fixture(scope='module')
def test(df):
    test = train_test_split(df, test_size=0.20, random_state=42)[1]
    return test


def test_slice_metrics(feature, test, y_test):
    '''
    Test slice_metrics function

    :param feature: feature to slice on
    :param df: features and label
    :param y_test: ground truth colums
    '''
    preds = pytest.preds
    try:
        metrics = ml.slice_metrics(feature, test, y_test, preds)
        assert metrics.empty is False
    except AssertionError as err:
        print('slice_metrics failed: no metrics generated')
        raise err
