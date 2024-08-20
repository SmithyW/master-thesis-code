import mlflow
from mlflow_oauth_keycloak_auth import authentication as mlflow_auth
import pandas as pd

if __name__ == "__main__":
    print("Setting up MLflow...")
    mlflow_auth.init()
    mlflow_auth.authenticate()
    print("Authentication successful.")
    print("Settings tracking URI...")
    mlflow.set_tracking_uri("https://mlflow.mlops.smiwit.de")

    data = {
        "recency": [10, 160, 749],
        "frequency": [15, 2, 4],
        "monetary": [5500, 450, 7000]
    }

    df = pd.DataFrame(data)
    print(df.head())

    print("Getting model for inference...")
    model_name = "RFM"
    model_version = "1"
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{model_version}")

    pred = model.predict(df)
    print(pred)
