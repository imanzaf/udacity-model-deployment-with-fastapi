"""
Train LGBM Classifier on census data, Save model, encoder, binarizer
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import src.model as ml
import src.data as dt
import joblib


if __name__ == '__main__':

    # Import & Clean data
    df = pd.read_csv('data/census.csv')
    df = dt.clean_data(df)

    # Split data
    train, test = train_test_split(df,
                                   test_size=0.20,
                                   random_state=42)

    # Get list of categorical columns
    categorical_cols = [x for x in train.columns if x not in
                        train.select_dtypes(include=np.number).columns.tolist()
                        and x not in ['salary']]

    # Transform data
    X_train, y_train, encoder, lb = dt.process_data(train,
                                                    categorical_features=categorical_cols,
                                                    label='salary',
                                                    training=True)

    X_test, y_test, encoder, lb = dt.process_data(test,
                                                  categorical_features=categorical_cols,
                                                  label='salary',
                                                  encoder=encoder,
                                                  lb=lb,
                                                  training=False)

    # Train model
    lgbm = ml.train_model(X_train, y_train)

    # Generate predictions
    preds = ml.inference(X_test, lgbm)

    # Print classification scores
    precision, recall, fbeta, accuracy = \
        ml.compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"fbeta: {fbeta}")
    print(f"Accuracy: {accuracy}")

    # save model and encoder
    joblib.dump(lgbm, "training_output/model.pkl")
    joblib.dump(encoder, "training_output/encoder.pkl")
    joblib.dump(lb, "training_output/binarizer.pkl")
