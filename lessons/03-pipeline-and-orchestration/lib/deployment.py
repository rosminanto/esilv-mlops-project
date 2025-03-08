from prefect import flow, serve
from datetime import timedelta

from workflows import train_model_workflow, batch_predict_workflow
from config import DATA_DIRPATH, MODELS_DIRPATH

# Define deployments using the flow.to_deployment() method
if __name__ == "__main__":
    # Define flow deployments
    training_deployment = train_model_workflow.to_deployment(
        name="Diamond Price Model Training",
        version="0.1.0",
        tags=["diamond", "training"],
        description="Weekly training of the diamond price prediction model",
        interval=604800,  # 1 week in seconds
        parameters={
            "train_filepath": f"{DATA_DIRPATH}/diamonds_train.csv",
            "test_filepath": f"{DATA_DIRPATH}/diamonds_test.csv",
            "artifacts_filepath": MODELS_DIRPATH
        }
    )

    prediction_deployment = batch_predict_workflow.to_deployment(
        name="Diamond Price Batch Prediction",
        version="0.1.0",
        tags=["diamond", "prediction"],
        description="Hourly batch prediction of diamond prices",
        interval=3600,  # 1 hour in seconds
        parameters={
            "input_filepath": f"{DATA_DIRPATH}/diamonds_new.csv",
            "artifacts_filepath": MODELS_DIRPATH
        }
    )

    # Serve the deployments
    serve(
        training_deployment,
        prediction_deployment,
    )
    print("Deployments are being served!")
