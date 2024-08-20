import subprocess
import os

import mlflow
import mlflow_oauth_keycloak_auth.authentication as mlflow_auth

from prefect import task

from prefect_utils import generate_run_name


@task(name="Setup MLflow",
      description="Setup MLflow for tracking.",
      task_run_name=generate_run_name("setup-mlflow"))
def setup_mlflow(tracking_uri: str, experiment: str, autologging: bool = False):
    print("Setting up MLflow...")
    mlflow_auth.init()
    mlflow_auth.authenticate()
    print("Authentication successful.")
    print("Settings tracking URI and experiment...")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment)
    print("Starting MLflow run...")
    run = mlflow.start_run()
    print(f"Started MLflow run with ID: {run.info.run_id}")

    print(f"Activate autologging: {autologging}")
    if (autologging):
        mlflow.autolog()
    print("MLflow setup complete")


@task(name="Log parameters.",
      description="Log parameters to MLflow.",
      task_run_name=generate_run_name("log-parameters"))
def log_parameters():
    print("Logging parameters to MLflow...")
    mlflow.log_param("git_commit", get_current_git_commit())
    print("Parameters logged.")


def get_current_git_commit():
    process = subprocess.Popen(
        ["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE)
    stdout, _ = process.communicate()
    last_commit = stdout.decode('ascii').strip()
    os.environ["LATEST_GIT_COMMIT"] = last_commit
    return last_commit


def end_mlflow_run(status: str = "FINISHED"):
    mlflow.end_run(status)
    print("MLflow run ended.")
