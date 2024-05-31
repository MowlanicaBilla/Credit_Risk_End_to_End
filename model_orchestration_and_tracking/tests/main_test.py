import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import joblib
from prefect import flow
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


from model_orchestration_and_tracking.main import train_and_tune_model
from model_orchestration_and_tracking.scripts.data_preparation import load_train_data, prepare_data
from model_orchestration_and_tracking.scripts.model_predictions import make_predictions

TRAIN_PATH = "/Users/mowlanica.billa/Desktop/Desktop/Data_Science/Projects/MLOps/credit_risk_end_to_end/model_orchestration_and_tracking/tests/data_for_tests/train/credit_train.csv"
MODEL_PATH = "/Users/mowlanica.billa/Desktop/Desktop/Data_Science/Projects/MLOps/credit_risk_end_to_end/model_orchestration_and_tracking/tests/data_for_tests/data_for_tests/model_for_test/model_test.pkl"
TEST_PATH = "/Users/mowlanica.billa/Desktop/Desktop/Data_Science/Projects/MLOps/credit_risk_end_to_end/model_orchestration_and_tracking/tests/data_for_tests/data_for_tests/test/credit_test.csv"
PREDICTION_PATH = "/Users/mowlanica.billa/Desktop/Desktop/Data_Science/Projects/MLOps/credit_risk_end_to_end/model_orchestration_and_tracking/tests/data_for_tests/data_for_tests/predictions/predictions.csv"

def test_load_train_data():
    data = flow(load_train_data.fn)(TRAIN_PATH)
    assert data.shape == (800, 21)

def test_train_and_tune_model():
    """ Train, tune and 5-fold cross-validate a model on the train set.
        Test whether the cross-validated AUC is at least 77%.
    """
    # Load the data
    data = pd.read_csv(TRAIN_PATH)
    print(f"Original data shape: {data.shape}")
    print(data.head())  # Print first few rows to inspect

    # Make simple data preparations
    X, Y = prepare_data(data)
    print(f"Prepared data shape: X={X.shape}, Y={Y.shape}")

    # Verify the unique values in Y before mapping
    print("Unique values in Y before mapping:", Y.unique())

    # Map labels to expected values
    Y = Y.map({'bad': 0, 'good': 1})

    # Check for NaN values in Y after mapping
    if Y.isna().sum() > 0:
        raise ValueError(f"NaN values found in Y after mapping: {Y.isna().sum()}")

    # Define the model
    model = RandomForestClassifier()

    # Define the cross-validation method to use
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and calculate AUC
    auc_scores = cross_val_score(model, X, Y, cv=cv, scoring='roc_auc')

    # Check the mean AUC score
    mean_auc = auc_scores.mean()
    print(f"Mean AUC score: {mean_auc}")

    # Assert the mean AUC score is at least 77%
    assert mean_auc >= 0.77, f"Mean AUC score is below 77%: {mean_auc}"


def test_make_predictions():
    """ Test that the model AUC is at least 75% on the test set.
    """
    # Load the recently saved model
    model_pipeline = joblib.load(MODEL_PATH)

    # Load data
    data = pd.read_csv(TEST_PATH)
    print(f"Test data shape: {data.shape}")
    print(data.head())  # Print first few rows to inspect

    # Map labels to expected values for the test set
    data['class'] = data['class'].map({'bad': 0, 'good': 1})

    predictions = make_predictions(model_pipeline, data, PREDICTION_PATH)
    auc = roc_auc_score(predictions['class'], predictions['prediction_probs']) * 100

    print(f"AUC: {auc}")  # Debug statement
    assert auc >= 75


# print(test_load_train_data())
print(test_train_and_tune_model())
print(test_make_predictions())