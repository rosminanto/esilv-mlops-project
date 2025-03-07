import os
import pickle
from functools import lru_cache

from loguru import logger


@lru_cache
def load_preprocessor_dv(filepath: os.PathLike):
    logger.info(f"Loading DictVectorizer from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


@lru_cache
def load_preprocessor_scaler(filepath: os.PathLike):
    logger.info(f"Loading StandardScaler from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)


@lru_cache
def load_model(filepath: os.PathLike):
    logger.info(f"Loading model from {filepath}")
    with open(filepath, "rb") as f:
        return pickle.load(f)
