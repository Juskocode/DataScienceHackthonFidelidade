from pathlib import Path

import mlflow
from azureml.core import Workspace

from ..utils.config_reader import read_config

# Set the tracking URI to your AzureML workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


def get_last_run_id(experiment_name):
    # Get the experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id

        # Get the last run
        runs = mlflow.search_runs(experiment_ids=experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs.empty:
            last_run_id = runs.iloc[0]["run_id"]
            return last_run_id
        else:
            print(f"No runs found in experiment '{experiment_name}'.")
            return None
    else:
        print(f"Experiment '{experiment_name}' not found.")
        return None


if __name__ == "__main__":

    PROJECT_PATH = Path(__file__).parents[1]
    MODEL_PATH = PROJECT_PATH / "models"
    CONFIG_PATH = PROJECT_PATH / "configs"

    config = read_config(CONFIG_PATH / "register_model_config.yml")

    version = config.version
    model_name = f"{PROJECT_PATH.name}_{version.replace('.','-')}"
    experiment_name = PROJECT_PATH.name
    run_id = config.run_id

    if run_id is None or run_id == "":
        run_id = get_last_run_id(experiment_name)

    model_uri = "runs:/{}/{}".format(run_id, model_name)

    model_details = mlflow.register_model(model_uri, name=model_name)
