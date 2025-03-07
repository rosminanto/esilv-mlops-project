from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from lib.models import DiamondData
from lib.preprocessing import prepare_features


def run_inference(
    input_data: List[DiamondData],
    dv: DictVectorizer,
    scaler: StandardScaler,
    model: BaseEstimator
) -> np.ndarray:
    """Run inference on a list of input data.

    Args:
        input_data (List[DiamondData]): list of diamond characteristics
        dv (DictVectorizer): fitted DictVectorizer for categorical features
        scaler (StandardScaler): fitted StandardScaler for numerical features
        model (BaseEstimator): fitted price prediction model

    Returns:
        np.ndarray: predicted diamond prices in USD

    Example payload:
        {
        "carat": 0.9,
        "cut": "Good",
        "color": "H",
        "clarity": "VS2",
        "depth": 63.6,
        "table": 61,
        "x": 6.03,
        "y": 5.97,
        "z": 3.8
        }
    """
    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])
    X = prepare_features(df, dv=dv, scaler=scaler)
    predictions = model.predict(X)
    logger.info(f"Predicted prices:\n{predictions}")
    return predictions
