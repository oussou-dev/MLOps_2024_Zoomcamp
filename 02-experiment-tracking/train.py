import os
import pickle

import click
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Change directory to the location of the script
os.chdir("/workspaces/MLOps_2024_Zoomcamp/02-experiment-tracking")

# Set MLflow tracking URI and experiment
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("mlops-zoomcamp-week-2")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved",
)
def run_train(data_path: str):
    # Start MLflow run
    with mlflow.start_run():
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        # Set MLFlow tags
        mlflow.set_tag("stage", "dev")
        mlflow.set_tag("dev", "dev-1")
        mlflow.set_tag("project_name", "mlops-zoomcamp")

        mlflow.sklearn.autolog()
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Log model parameters
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("random_state", 0)
        mlflow.log_param("Regressor", "RandomForestRegressor")
        mlflow.log_param("min_samples_split", rf.min_samples_split)

        print(f"Min samples split: {rf.min_samples_split}")
        # Min samples split: 2

        # Log model metrics
        mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":
    run_train()
