# lib/deployment.py
from prefect import serve
from datetime import timedelta

from workflows import train_model_workflow, batch_predict_workflow

# Define paths
train_data_path = "../../data/diamonds_train.csv"
test_data_path = "../../data/diamonds_test.csv"
batch_data_path = "../../data/diamonds_new.csv"  # Path to new data for prediction
artifacts_path = "../../models"
predictions_path = "../../data/predictions.csv"

# Create a weekly training deployment
training_deployment = train_model_workflow.to_deployment(
    name="Diamond Price Weekly Training",
    version="1.0",
    tags=["training", "diamonds"],
    interval=timedelta(weeks=1).total_seconds(),
    parameters={
        "train_filepath": train_data_path,
        "test_filepath": test_data_path,
        "artifacts_filepath": artifacts_path
    }
)

# Create an hourly prediction deployment
prediction_deployment = batch_predict_workflow.to_deployment(
    name="Diamond Price Hourly Prediction",
    version="1.0",
    tags=["prediction", "diamonds"],
    interval=timedelta(hours=1).total_seconds(),
    parameters={
        "input_filepath": batch_data_path,
        "artifacts_filepath": artifacts_path,
        "output_filepath": predictions_path
    }
)

if __name__ == "__main__":
    # Serve the deployments
    serve(
        training_deployment,
        prediction_deployment
    )
