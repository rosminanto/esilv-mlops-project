# MODELS
MODEL_VERSION = "0.0.1"
PATH_TO_PREPROCESSOR_DV = f"local_models/diamond_dv__v{MODEL_VERSION}.pkl"
PATH_TO_PREPROCESSOR_SCALER = f"local_models/diamond_scaler__v{MODEL_VERSION}.pkl"
PATH_TO_MODEL = f"local_models/diamond_model__v{MODEL_VERSION}.pkl"

CATEGORICAL_VARS = ["cut", "color", "clarity"]
NUMERICAL_VARS = ["carat", "depth", "table", "x", "y", "z"]

# MISC
APP_TITLE = "DiamondPricePredictionApp"
APP_DESCRIPTION = (
    "A simple API to predict diamond prices in USD "
    "given characteristics such as carat weight, cut, "
    "color, clarity, and dimensions."
)
APP_VERSION = "0.0.1"
