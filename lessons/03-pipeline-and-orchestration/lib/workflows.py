import os
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from prefect import flow, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from helpers import load_pickle, save_pickle
from modeling import train_model, predict, evaluate_model
from preprocessing import prepare_features


@task(name="Process Training Data", retries=2, retry_delay_seconds=30)
def process_train_data(filepath: str):
    """Process training data to prepare for model training"""
    logger.info(f"Processing training data from {filepath}...")
    df = pd.read_csv(filepath)
    X, dv, scaler = prepare_features(
        df,
        fit=True
    )
    y = df['price'].values
    return X, y, dv, scaler


@task(name="Process Test Data", retries=2, retry_delay_seconds=30)
def process_test_data(filepath: str, dv: DictVectorizer, scaler: StandardScaler):
    """Process test data using fitted preprocessors"""
    logger.info(f"Processing test data from {filepath}...")
    df = pd.read_csv(filepath)
    X = prepare_features(
        df,
        dv=dv,
        scaler=scaler,
        fit=False
    )
    y = df['price'].values
    return X, y


@task(name="Train Model", retries=1)
def train_model_task(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Task wrapper for training a model"""
    logger.info("Training model...")
    return train_model(X, y)


@task(name="Make Predictions", retries=1)
def predict_task(X: np.ndarray, model: LinearRegression) -> np.ndarray:
    """Task wrapper for making predictions"""
    logger.info("Making predictions...")
    return predict(X, model)


@task(name="Evaluate Model", retries=1)
def evaluate_model_task(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Task wrapper for evaluating model performance"""
    logger.info("Evaluating model...")
    metrics = evaluate_model(y_true, y_pred)
    logger.info(f"Model metrics: {metrics}")
    return metrics


@task(name="Save Artifacts", retries=2)
def save_artifacts(model: LinearRegression, dv: DictVectorizer, scaler: StandardScaler, artifacts_filepath: str):
    """Save model and preprocessors to disk"""
    logger.info(f"Saving artifacts to {artifacts_filepath}...")
    os.makedirs(artifacts_filepath, exist_ok=True)
    save_pickle(os.path.join(artifacts_filepath, "diamond_model.pkl"), model)
    save_pickle(os.path.join(artifacts_filepath, "diamond_dv.pkl"), dv)
    save_pickle(os.path.join(artifacts_filepath, "diamond_scaler.pkl"), scaler)
    logger.info("Artifacts saved successfully")


@task(name="Load Artifacts", retries=2)
def load_artifacts(artifacts_filepath: str):
    """Load model and preprocessors from disk"""
    logger.info(f"Loading artifacts from {artifacts_filepath}...")
    model = load_pickle(os.path.join(artifacts_filepath, "diamond_model.pkl"))
    dv = load_pickle(os.path.join(artifacts_filepath, "diamond_dv.pkl"))
    scaler = load_pickle(os.path.join(artifacts_filepath, "diamond_scaler.pkl"))
    logger.info("Artifacts loaded successfully")
    return model, dv, scaler


@flow(name="Diamond Price Prediction Training Flow", log_prints=True)
def train_model_workflow(
    train_filepath: str,
    test_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    """Workflow for training and evaluating a diamond price prediction model"""
    # Process data
    X_train, y_train, dv, scaler = process_train_data(train_filepath)
    X_test, y_test = process_test_data(test_filepath, dv, scaler)

    # Train model
    model = train_model_task(X_train, y_train)

    # Evaluate model
    train_pred = predict_task(X_train, model)
    test_pred = predict_task(X_test, model)
    train_metrics = evaluate_model_task(y_train, train_pred)
    test_metrics = evaluate_model_task(y_test, test_pred)

    # Save artifacts if path provided
    if artifacts_filepath:
        save_artifacts(model, dv, scaler, artifacts_filepath)

    # Return metrics
    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }


@flow(name="Diamond Price Prediction Batch Flow", log_prints=True)
def batch_predict_workflow(
    input_filepath: str,
    artifacts_filepath: str,
    output_filepath: Optional[str] = None
) -> np.ndarray:
    """Workflow for batch predictions on new data"""
    # Load artifacts
    model, dv, scaler = load_artifacts(artifacts_filepath)

    # Process data without target
    df = pd.read_csv(input_filepath)
    X = prepare_features(df, dv=dv, scaler=scaler, fit=False)

    # Make predictions
    predictions = predict_task(X, model)

    # Save predictions if output path is provided
    if output_filepath:
        logger.info(f"Saving predictions to {output_filepath}")
        pd.DataFrame({"price_prediction": predictions}).to_csv(output_filepath, index=False)

    return predictions


if __name__ == "__main__":
    # Example usage
    train_path = "../../data/diamonds_train.csv"
    test_path = "../../data/diamonds_test.csv"
    artifacts_path = "../../models"

    # Run training workflow
    metrics = train_model_workflow(train_path, test_path, artifacts_path)
    print(f"Training complete with metrics: {metrics}")

    # Run prediction workflow
    batch_predict_workflow(test_path, artifacts_path, "../../data/predictions.csv")
    print("Batch prediction complete")
