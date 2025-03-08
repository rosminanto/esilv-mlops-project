# To complete
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from loguru import logger

# Define constants
CATEGORICAL_COLS = ['cut', 'color', 'clarity']
NUMERICAL_COLS = ['carat', 'depth', 'table', 'x', 'y', 'z']


def load_data(filepath: str) -> pd.DataFrame:
    """Load diamond data from CSV file and clean column names

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with cleaned column names
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    # Clean up column names - remove extra quotation marks
    df.columns = [col.strip("'") for col in df.columns]
    logger.info(f"Loaded {len(df)} rows")
    return df


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """Encode categorical columns as strings

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names, defaults to CATEGORICAL_COLS

    Returns:
        DataFrame with encoded categorical columns
    """
    df = df.copy()
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS

    logger.info(f"Encoding categorical columns: {categorical_cols}")
    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


def prepare_features(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None,
    dv: Optional[DictVectorizer] = None,
    scaler: Optional[StandardScaler] = None,
    fit: bool = False
) -> Tuple[np.ndarray, Optional[DictVectorizer], Optional[StandardScaler]]:
    """Prepare features for modeling

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        dv: Optional DictVectorizer for categorical features
        scaler: Optional StandardScaler for numerical features
        fit: Whether to fit or transform the data

    Returns:
        Feature matrix and optionally the fitted vectorizer and scaler
    """
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    logger.info(f"Preparing features (fit={fit})")

    # Prepare categorical features
    df = encode_categorical_cols(df, categorical_cols)
    cat_dicts = df[categorical_cols].to_dict(orient='records')

    if fit:
        logger.info("Fitting DictVectorizer")
        dv = DictVectorizer(sparse=False)
        cat_features = dv.fit_transform(cat_dicts)
    else:
        logger.info("Transforming with DictVectorizer")
        cat_features = dv.transform(cat_dicts)

    # Prepare numerical features
    if fit:
        logger.info("Fitting StandardScaler")
        scaler = StandardScaler()
        num_features = scaler.fit_transform(df[numerical_cols])
    else:
        logger.info("Transforming with StandardScaler")
        num_features = scaler.transform(df[numerical_cols])

    # Combine features
    X = np.hstack([num_features, cat_features])
    logger.info(f"Feature matrix shape: {X.shape}")

    if fit:
        return X, dv, scaler
    return X
