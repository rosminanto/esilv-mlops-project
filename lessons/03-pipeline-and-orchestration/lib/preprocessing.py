# To complete
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

# Constants from the diamond price prediction project
CATEGORICAL_COLS = ["cut", "color", "clarity"]
NUMERICAL_COLS = ["carat", "depth", "table", "x", "y", "z"]
TARGET = 'price'


def encode_categorical_cols(df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
    """Encode categorical columns"""
    df = df.copy()
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
