# To complete
from typing import Optional, Dict, Any

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from prefect import flow, task, get_run_logger

from config import DATA_DIRPATH, MODELS_DIRPATH
from modeling import train_model, predict, evaluate_model
from preprocessing import prepare_features, load_data
from helpers import task_save_pickle, task_load_pickle


@flow(name="Diamond Price Prediction Training Workflow",
      description="Trains a diamond price prediction model and evaluates its performance")
def train_model_workflow(
    train_filepath: str,
    test_filepath: str,
    artifacts_filepath: Optional[str] = None,
) -> dict:
    """Complete end-to-end workflow for training a diamond price model

    Args:
        train_filepath: Path to the training data CSV
        test_filepath: Path to the test data CSV
        artifacts_filepath: Directory to save model artifacts

    Returns:
        Dictionary with training results
    """
    logger = get_run_logger()

    if artifacts_filepath is None:
        artifacts_filepath = MODELS_DIRPATH

    # Ensure directory exists
    os.makedirs(artifacts_filepath, exist_ok=True)

    # Load and process training data
    logger.info(f"Loading training data from {train_filepath}")
    train_df = load_data(train_filepath)

    logger.info("Preparing features for training data")
    X_train, y_train, dv, scaler = prepare_data_task(train_df, fit=True)

    # Train model
    logger.info("Training model")
    model = train_model_task(X_train, y_train)

    # Evaluate on training data
    logger.info("Evaluating model on training data")
    train_predictions = predict_task(X_train, model)
    train_metrics = evaluate_model_task(y_train, train_predictions)

    # Load and process test data
    logger.info(f"Loading test data from {test_filepath}")
    test_df = load_data(test_filepath)

    logger.info("Preparing features for test data")
    X_test, y_test, _, _ = prepare_data_task(test_df, dv=dv, scaler=scaler, fit=False)

    # Evaluate on test data
    logger.info("Evaluating model on test data")
    test_predictions = predict_task(X_test, model)
    test_metrics = evaluate_model_task(y_test, test_predictions)

    # Save artifacts
    logger.info(f"Saving model artifacts to {artifacts_filepath}")
    save_model_artifacts_task(
        artifacts_filepath,
        model=model,
        dv=dv,
        scaler=scaler
    )

    # Compile and return results
    results = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "model_path": os.path.join(artifacts_filepath, "diamond_model.pkl"),
        "dv_path": os.path.join(artifacts_filepath, "diamond_dv.pkl"),
        "scaler_path": os.path.join(artifacts_filepath, "diamond_scaler.pkl")
    }

    logger.info(f"Training results: {results}")
    return results


@flow(name="Diamond Price Prediction Batch Workflow",
      description="Makes batch predictions using a trained diamond price model")
def batch_predict_workflow(
    input_filepath: str,
    model: Optional[LinearRegression] = None,
    dv: Optional[DictVectorizer] = None,
    scaler: Optional[StandardScaler] = None,
    artifacts_filepath: Optional[str] = None,
) -> np.ndarray:
    """Complete end-to-end workflow for batch prediction

    Args:
        input_filepath: Path to the input data CSV
        model: Optional pre-loaded model
        dv: Optional pre-loaded DictVectorizer
        scaler: Optional pre-loaded StandardScaler
        artifacts_filepath: Directory where model artifacts are saved

    Returns:
        Array of predictions
    """
    logger = get_run_logger()

    if artifacts_filepath is None:
        artifacts_filepath = MODELS_DIRPATH

    # Load artifacts if not provided
    if model is None or dv is None or scaler is None:
        logger.info(f"Loading model artifacts from {artifacts_filepath}")
        model, dv, scaler = load_model_artifacts_task(artifacts_filepath)

    # Load and process input data
    logger.info(f"Loading input data from {input_filepath}")
    input_df = load_data(input_filepath)

    logger.info("Preparing features for prediction")
    X_input = prepare_features_task(input_df, dv=dv, scaler=scaler)

    # Make predictions
    logger.info("Making predictions")
    predictions = predict_task(X_input, model)

    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


@task(name="Prepare Data", retries=2)
def prepare_data_task(
    df,
    dv=None,
    scaler=None,
    fit=True
):
    """Task to prepare data for training or prediction"""
    if fit:
        X, dv, scaler = prepare_features(df, fit=True)
        y = df['price'].values
        return X, y, dv, scaler
    else:
        X = prepare_features(df, dv=dv, scaler=scaler, fit=False)
        y = df['price'].values
        return X, y, dv, scaler


@task(name="Prepare Features", retries=2)
def prepare_features_task(df, dv, scaler):
    """Task to prepare features for prediction only (no target)"""
    X = prepare_features(df, dv=dv, scaler=scaler, fit=False)
    return X


@task(name="Train Model", retries=2)
def train_model_task(X, y):
    """Task to train a model"""
    return train_model(X, y)


@task(name="Predict", retries=2)
def predict_task(X, model):
    """Task to make predictions"""
    return predict(X, model)


@task(name="Evaluate Model", retries=2)
def evaluate_model_task(y_true, y_pred):
    """Task to evaluate model performance"""
    return evaluate_model(y_true, y_pred)


@task(name="Save Model Artifacts", retries=3)
def save_model_artifacts_task(artifacts_dir, model, dv, scaler):
    """Task to save model artifacts"""
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "diamond_model.pkl")
    dv_path = os.path.join(artifacts_dir, "diamond_dv.pkl")
    scaler_path = os.path.join(artifacts_dir, "diamond_scaler.pkl")

    task_save_pickle(model_path, model)
    task_save_pickle(dv_path, dv)
    task_save_pickle(scaler_path, scaler)

    return {
        "model_path": model_path,
        "dv_path": dv_path,
        "scaler_path": scaler_path
    }


@task(name="Load Model Artifacts", retries=3)
def load_model_artifacts_task(artifacts_dir):
    """Task to load model artifacts"""
    model_path = os.path.join(artifacts_dir, "diamond_model.pkl")
    dv_path = os.path.join(artifacts_dir, "diamond_dv.pkl")
    scaler_path = os.path.join(artifacts_dir, "diamond_scaler.pkl")

    model = task_load_pickle(model_path)
    dv = task_load_pickle(dv_path)
    scaler = task_load_pickle(scaler_path)

    return model, dv, scaler
