from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd

from pulso_transmi import PulsoTransmiClient


TRACKING_URI = "file:///home/matti/code/ProyectoVisual2/mlruns"


def wape_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    true = y_true.to_numpy(dtype=float)
    pred = y_pred.to_numpy(dtype=float)
    denom = abs(true).sum()
    if denom == 0:
        return 100.0
    wape = abs(true - pred).sum() / denom
    return float(max(0.0, 100.0 * (1.0 - wape)))


def build_station_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["station_id", "observed_at"]).copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["hour"] = frame["observed_at"].dt.hour
    frame["minute"] = frame["observed_at"].dt.minute
    frame["dayofweek"] = frame["observed_at"].dt.dayofweek
    frame["is_weekend"] = frame["dayofweek"].ge(5).astype(int)
    frame["month"] = frame["observed_at"].dt.month

    for lag in [1, 2, 3, 4, 6, 12, 24, 48, 96]:
        frame[f"lag_{lag}"] = frame.groupby("station_id")["demand"].transform(lambda s: s.shift(lag))

    for window in [3, 6, 12, 24, 48, 96]:
        frame[f"rolling_mean_{window}"] = frame.groupby("station_id")["demand"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    return frame


def train_and_evaluate(stations: pd.DataFrame, observations: pd.DataFrame, context: pd.DataFrame):
    full = observations.merge(stations[["station_id", "station_name", "corridor", "latitude", "longitude"]], on="station_id", how="left")
    full = full.merge(context, on="observed_at", how="left")
    full = build_station_features(full)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_6", "lag_12", "lag_24", "lag_48", "lag_96",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_mean_24", "rolling_mean_48", "rolling_mean_96",
        "hour", "minute", "dayofweek", "is_weekend", "month",
        "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity",
        "latitude", "longitude",
    ]
    full = full.dropna(subset=["demand", *feature_cols]).copy()

    station_scores = []
    trained_models = {}

    for station_id, station_frame in full.groupby("station_id"):
        n = len(station_frame)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        train = station_frame.iloc[:train_end]
        val = station_frame.iloc[train_end:val_end]
        test = station_frame.iloc[val_end:]

        if len(train) < 10 or len(val) < 5 or len(test) < 5:
            continue

        X_train = train[feature_cols]
        y_train = train["demand"]
        X_val = val[feature_cols]
        y_val = val["demand"]
        X_test = test[feature_cols]
        y_test = test["demand"]

        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_accuracy = wape_accuracy(y_val, pd.Series(val_pred, index=y_val.index))
        test_accuracy = wape_accuracy(y_test, pd.Series(test_pred, index=y_test.index))

        station_scores.append(
            {
                "station_id": station_id,
                "station_name": station_frame["station_name"].iloc[0],
                "corridor": station_frame["corridor"].iloc[0],
                "val_accuracy": round(val_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
            }
        )

        trained_models[station_id] = {
            "model": model,
            "feature_cols": feature_cols,
            "train_rows": len(train),
            "val_rows": len(val),
            "test_rows": len(test),
            "station_name": station_frame["station_name"].iloc[0],
            "corridor": station_frame["corridor"].iloc[0],
        }

    if not station_scores:
        raise RuntimeError("No fue posible entrenar ningún modelo para ninguna estación.")

    station_scores_df = pd.DataFrame(station_scores)
    avg_test_accuracy = station_scores_df["test_accuracy"].mean()
    avg_val_accuracy = station_scores_df["val_accuracy"].mean()

    return station_scores_df, trained_models, avg_val_accuracy, avg_test_accuracy


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("PulsoTransMi-Experiments")

    client = PulsoTransmiClient()
    stations = client.stations()
    observations = client.observations_dataframe(page_size=5000)
    context = client.context_dataframe(page_size=5000)

    stations["station_id"] = stations["station_id"].astype(str)
    observations["station_id"] = observations["station_id"].astype(str)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)

    dataset_snapshot = observations.merge(stations[["station_id", "station_name", "corridor", "latitude", "longitude"]], on="station_id", how="left")
    dataset_snapshot = dataset_snapshot.merge(context, on="observed_at", how="left")
    dataset_snapshot = dataset_snapshot.sort_values(["station_id", "observed_at"]).copy()

    dataset_path = Path("artifacts") / "exp_01_dataset_snapshot.csv"
    dataset_path.parent.mkdir(exist_ok=True)
    dataset_snapshot.to_csv(dataset_path, index=False)

    score_df, trained_models, avg_val_accuracy, avg_test_accuracy = train_and_evaluate(stations, observations, context)

    with mlflow.start_run(run_name="exp_01_linear_regression") as run:
        mlflow.set_tag("experiment_name", "EXP_01")
        mlflow.set_tag("model_name", "linear_regression")
        mlflow.set_tag("project", "PulsoTransMi")
        mlflow.set_tag("dataset_source", "pulso-transmi-sdk-api")

        mlflow.log_params(
            {
                "model_type": "LinearRegression",
                "station_count": len(stations),
                "train_ratio": 0.6,
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
                "lag_values": "[1,2,3,4,6,12,24,48,96]",
                "rolling_windows": "[3,6,12,24,48,96]",
            }
        )

        mlflow.log_metric("avg_validation_accuracy", avg_val_accuracy)
        mlflow.log_metric("avg_test_accuracy", avg_test_accuracy)

        per_station = score_df[["station_id", "corridor", "test_accuracy"]].to_dict(orient="records")
        mlflow.log_dict({"per_station_accuracy": per_station}, artifact_file="per_station_accuracy.json")

        mlflow.log_artifact(str(dataset_path), artifact_path="dataset")
        mlflow.log_artifact(__file__, artifact_path="source")

        model_path = Path("artifacts") / "exp_01_linear_regression_models.joblib"
        model_path.parent.mkdir(exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(trained_models, f)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Average validation accuracy: {avg_val_accuracy:.4f}%")
        print(f"Average test accuracy: {avg_test_accuracy:.4f}%")
        print(f"Per-station metrics logged to: {dataset_path}")
        print(f"Model artifact logged to: {model_path}")


if __name__ == "__main__":
    main()
