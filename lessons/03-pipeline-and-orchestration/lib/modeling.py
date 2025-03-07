# lib/modeling.py
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def train_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    """Train a linear regression model for diamond price prediction

    Args:
        X: Feature matrix
        y: Target values (diamond prices)

    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def predict(X: np.ndarray, model: LinearRegression) -> np.ndarray:
    """Make predictions using the trained model

    Args:
        X: Feature matrix
        model: Trained model

    Returns:
        Array of predictions
    """
    return model.predict(X)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate model performance metrics

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary of performance metrics
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {
        'rmse': rmse,
        'r2_score': r2
    }
