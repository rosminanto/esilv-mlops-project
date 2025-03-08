# To complete
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from loguru import logger


def train_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Train a linear regression model for diamond price prediction

    Args:
        X: Feature matrix
        y: Target values (prices)

    Returns:
        Trained LinearRegression model
    """
    logger.info(f"Training linear regression model on {X.shape[0]} samples")
    model = LinearRegression()
    model.fit(X, y)
    logger.info("Model training complete")
    return model


def predict(X: np.ndarray, model: LinearRegression) -> np.ndarray:
    """Make predictions using a trained model

    Args:
        X: Feature matrix
        model: Trained model

    Returns:
        Array of predictions
    """
    logger.info(f"Making predictions for {X.shape[0]} samples")
    return model.predict(X)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate model performance

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary of evaluation metrics
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "rmse": float(rmse),
        "r2_score": float(r2)
    }

    logger.info(f"Model evaluation results: {metrics}")
    return metrics
