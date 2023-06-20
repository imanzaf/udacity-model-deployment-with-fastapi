'''
Functions to train, evaluate, and run a Machine Learning model
'''

from sklearn.metrics import fbeta_score, precision_score,\
    recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
import pandas as pd


def train_model(X_train, y_train):
    """
    Trains a machine learning model (LGBM Classifier) and returns it.
    Implements hyperparameter tuning using Grid Search CV

    :param X_train: (np.array) Training data
    :param y_train: (np.array) Labels

    :returns best_estimator: Trained machine learning model
    """
    # Set parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100, 150],
        'num_leaves': [2 ** 4, 2 ** 6, 2 ** 8],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.9]
    }

    # Create estimator
    estimator = LGBMClassifier()

    # Create and fit model
    model = GridSearchCV(estimator=estimator,
                         param_grid=param_grid,
                         cv=6,
                         n_jobs=-1,
                         scoring='accuracy')
    model.fit(X_train, y_train)

    # Return model object
    return model


def inference(X, model, encoder=None):
    """
    Run model inferences and return the predictions.

    :param X: (np.array) Data used for prediction.
    :param model: Trained machine learning model.
    :param encoder: trained encoder object for categorical features.

    :returns preds: (np.array) Predictions from the model.
    """
    if encoder is not None:
        # Run encoder on data if not transformed
        X = encoder.transform(X)

    # Run inferences on best estimator
    preds = model.best_estimator_.predict(X)

    return preds


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using:
     precision, recall, F1, and accuracy

    :param y: (np.array) Known labels, binarized
    :param preds: (np.array) Predicted labels, binarized.

    :returns precision: (float)
    :returns recall: (float)
    :returns fbeta: (float)
    :returns accuracy: (float)
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    accuracy = accuracy_score(y, preds)

    return precision, recall, fbeta, accuracy


def slice_metrics(feature, X, y, preds):
    """
    Get evaluation metrics on slice of data

    :param feature: (str) categorical feature to create slices for
    :param X: (df) features
    :param y: ground truth data
    :param preds: model predictions

    :returns metrics: df with metrics for each slice
    """
    # initialize empty df for metrics
    metrics = pd.DataFrame(columns=['precision', 'recall', 'fbeta', 'accuracy'],
                           index=X[feature].unique().tolist())

    # join features and label / preds
    df = X.copy()
    df['ground_truth'], df['predictions'] = y, preds

    #
    for slice in df[feature].unique():

        # get labels and predictions
        y_true = df[df[feature] == slice]['ground_truth']
        y_pred = df[df[feature] == slice]['predictions']

        # compute metrics
        # print(f'Classification Metrics for {slice_feature}:{slice}')
        fbeta = fbeta_score(y_true, y_pred, beta=1, zero_division=1)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        accuracy = accuracy_score(y_true, y_pred)

        # add metrics to df
        metrics.loc[slice] = pd.Series({'precision': precision, 'recall': recall, 'fbeta': fbeta, 'accuracy': accuracy})

    return metrics
