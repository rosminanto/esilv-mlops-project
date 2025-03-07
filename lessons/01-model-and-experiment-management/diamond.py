# %%
import pickle
from typing import Any, List
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
from mlflow.tracking import MlflowClient

# %%
print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

# Create an MLflow client
client = MlflowClient()

# Search for experiments
experiments = client.search_experiments()
print("Available Experiments:")
for experiment in experiments:
    print(f"- Name: {experiment.name}")


# %%
def load_data(path: str) -> pd.DataFrame:
    """Load the diamonds dataset from CSV file and clean column names"""
    df = pd.read_csv(path)
    # Clean up column names - remove extra quotation marks
    df.columns = [col.strip("'") for col in df.columns]
    return df


# Load the data
diamonds_df = load_data('../../data/diamonds.csv')
print("Dataset shape:", diamonds_df.shape)
print("\nFirst few rows:")
print(diamonds_df.head())
print("\nData info:")
print(diamonds_df.info())

# %%


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split the data into training and testing sets"""
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )


train_df, test_df = prepare_train_test_data(diamonds_df)
print("Train set shape:", train_df.shape)
print("Test set shape:", test_df.shape)


# %%
CATEGORICAL_COLS = ['cut', 'color', 'clarity']
NUMERICAL_COLS = ['carat', 'depth', 'table', 'x', 'y', 'z']
TARGET = 'price'


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """Encode categorical columns"""
    df = df.copy()
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df[categorical_cols] = df[categorical_cols].astype(str)
    return df


def prepare_features(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None,
    dv: DictVectorizer = None,
    scaler: StandardScaler = None,
    fit: bool = False
) -> tuple:
    """Prepare features for modeling"""

    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    # Prepare categorical features
    df = encode_categorical_cols(df, categorical_cols)
    cat_dicts = df[categorical_cols].to_dict(orient='records')

    if fit:
        dv = DictVectorizer(sparse=False)
        cat_features = dv.fit_transform(cat_dicts)
    else:
        cat_features = dv.transform(cat_dicts)

    # Prepare numerical features
    if fit:
        scaler = StandardScaler()
        num_features = scaler.fit_transform(df[numerical_cols])
    else:
        num_features = scaler.transform(df[numerical_cols])

    # Combine features
    X = np.hstack([num_features, cat_features])

    if fit:
        return X, dv, scaler
    return X


# Prepare training data
X_train, dv, scaler = prepare_features(
    train_df,
    categorical_cols=CATEGORICAL_COLS,
    numerical_cols=NUMERICAL_COLS,
    fit=True
)
y_train = train_df[TARGET].values

# Prepare test data
X_test = prepare_features(
    test_df,
    categorical_cols=CATEGORICAL_COLS,
    numerical_cols=NUMERICAL_COLS,
    dv=dv,
    scaler=scaler,
    fit=False
)
y_test = test_df[TARGET].values


# %%
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **model_params
) -> LinearRegression:
    """Train a linear regression model"""
    model = LinearRegression(**model_params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate model performance metrics"""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {
        'rmse': rmse,
        'r2_score': r2
    }


# %%
mlflow.set_experiment("diamond-price-prediction")

# Start a run
with mlflow.start_run() as run:
    run_id = run.info.run_id

    # Set tags for the run
    mlflow.set_tag("model_type", "linear_regression")
    mlflow.set_tag("dataset", "diamonds")

    # Log data info
    mlflow.log_params({
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "categorical_features": CATEGORICAL_COLS,
        "numerical_features": NUMERICAL_COLS
    })

    # Train model
    model = train_model(X_train, y_train)

    # Log model parameters
    mlflow.log_params(model.get_params())

    # Make predictions and evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = evaluate_model(y_train, train_pred)
    test_metrics = evaluate_model(y_test, test_pred)

    # Log metrics
    mlflow.log_metrics({
        "train_rmse": train_metrics['rmse'],
        "train_r2": train_metrics['r2_score'],
        "test_rmse": test_metrics['rmse'],
        "test_r2": test_metrics['r2_score']
    })

    # Log the model
    mlflow.sklearn.log_model(model, "diamond_price_model")

    print(f"Run ID: {run_id}")
    print(f"Train RMSE: {train_metrics['rmse']:.2f}")
    print(f"Test RMSE: {test_metrics['rmse']:.2f}")
    print(f"Train R2: {train_metrics['r2_score']:.4f}")
    print(f"Test R2: {test_metrics['r2_score']:.4f}")


# %%
client = MlflowClient()

# Register the model
model_name = "diamond_price_predictor"
model_uri = f"runs:/{run_id}/diamond_price_model"
registered_model = mlflow.register_model(model_uri, model_name)

# Transition the model to production
client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Production"
)

print(f"Model '{model_name}' version {registered_model.version} transitioned to Production stage")


# %%
def predict_price(df: pd.DataFrame, model, dv: DictVectorizer, scaler: StandardScaler) -> np.ndarray:
    """Make predictions using the trained model"""
    X = prepare_features(
        df,
        categorical_cols=CATEGORICAL_COLS,
        numerical_cols=NUMERICAL_COLS,
        dv=dv,
        scaler=scaler,
        fit=False
    )
    return model.predict(X)


# Load production model
production_model = mlflow.sklearn.load_model(f"models:/{model_name}/production")

# Create sample data for prediction
sample_data = pd.DataFrame({
    'carat': [0.5, 1.0, 1.5],
    'cut': ['Ideal', 'Premium', 'Very Good'],
    'color': ['E', 'F', 'G'],
    'clarity': ['VS1', 'VS2', 'SI1'],
    'depth': [61.5, 62.0, 62.5],
    'table': [55.0, 56.0, 57.0],
    'x': [5.15, 6.3, 7.2],
    'y': [5.2, 6.35, 7.25],
    'z': [3.2, 3.9, 4.5]
})

# Make predictions
predictions = predict_price(sample_data, production_model, dv, scaler)
print("\nSample Predictions:")
for i, pred in enumerate(predictions):
    print(f"Diamond {i+1}: ${pred:,.2f}")


# %%
def save_pickle(path: str, obj: Any):
    """Save object to pickle file"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# Save preprocessors and model
MODEL_VERSION = "0.0.1"
save_pickle("../../models/diamond_dv.pkl", dv)
save_pickle(f"../02-model-deployment/web_service/local_models/diamond_dv__v{MODEL_VERSION}.pkl", dv)
save_pickle("../../models/diamond_scaler.pkl", scaler)
save_pickle(f"../02-model-deployment/web_service/local_models/diamond_scaler__v{MODEL_VERSION}.pkl", scaler)
save_pickle("../../models/diamond_model.pkl", model)
save_pickle(f"../02-model-deployment/web_service/local_models/diamond_model__v{MODEL_VERSION}.pkl", model)

# %%
