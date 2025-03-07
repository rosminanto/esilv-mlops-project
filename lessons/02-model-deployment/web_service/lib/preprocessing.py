from typing import List

import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

CATEGORICAL_COLS = ["cut", "color", "clarity"]
NUMERICAL_COLS = ["carat", "depth", "table", "x", "y", "z"]


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    df[categorical_cols] = df[categorical_cols].astype("str")
    return df


def prepare_features(
    df: pd.DataFrame,
    categorical_cols: List[str] = None,
    numerical_cols: List[str] = None,
    dv: DictVectorizer = None,
    scaler: StandardScaler = None,
) -> np.ndarray:
    """Prepare features for prediction"""
    if categorical_cols is None:
        categorical_cols = CATEGORICAL_COLS
    if numerical_cols is None:
        numerical_cols = NUMERICAL_COLS

    # Prepare categorical features
    df = encode_categorical_cols(df, categorical_cols)
    cat_dicts = df[categorical_cols].to_dict(orient='records')
    cat_features = dv.transform(cat_dicts)

    # Prepare numerical features
    num_features = scaler.transform(df[numerical_cols])

    # Combine features
    if isinstance(cat_features, np.ndarray):
        return np.hstack([num_features, cat_features])
    return np.hstack([num_features, cat_features.toarray()])
