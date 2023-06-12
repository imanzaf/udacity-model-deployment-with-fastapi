"""
Train LGBM Classifier on census data, Save model and encoder
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import src.model as model
import src.data as data
import joblib


if __name__ == '__main__':

    # Import data
    df = pd.read_csv('data/census.csv')

    # Clean & Transform data
    df = data.clean_data(df)
    X, y, encoder, lb = data.transform_data(df, label='salary')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20,
                                                        random_state=42)

    # Train model
    lgbm = model.train_model(X_train, y_train)

    # Generate predictions
    preds = model.inference(X_test, lgbm)

    # Print classification scores
    precision, recall, fbeta, accuracy = \
        model.compute_model_metrics(y_test, preds)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"fbeta: {fbeta}")
    print(f"Accuracy: {accuracy}")

    # save model and encoder
    joblib.dump(lgbm, "training_output/model.pkl")
    joblib.dump(encoder, "training_output/encoder.pkl")
    joblib.dump(lb, "training_output/binarizer.pkl")
