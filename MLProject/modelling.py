"""
modelling.py (Workflow-CI / MLflow Project)
Melatih RandomForest dengan MLflow autolog. Dijalankan via `mlflow run MLProject`.

- Di CI (mlflow run): tracking URI & run ID diatur otomatis oleh MLflow lewat environment variable.
- Lokal (python modelling.py): memakai MLFLOW_TRACKING_URI jika ada, default http://127.0.0.1:5000/
"""
import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="telco_preprocessing")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir

    # Jika dijalankan oleh `mlflow run`, run & experiment sudah ditentukan MLflow
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/"))
        mlflow.set_experiment("Telco Churn - CI")

    mlflow.sklearn.autolog()

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    X_train, y_train = train.drop(columns=["Churn"]), train["Churn"]
    X_test, y_test = test.drop(columns=["Churn"]), test["Churn"]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print(f"Akurasi data uji: {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
