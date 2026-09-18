from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from pulso_transmi import PulsoTransmiClient


def wape_accuracy(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return 100.0
    wape = np.sum(np.abs(y_true - y_pred)) / denominator
    return float(max(0.0, 100.0 * (1.0 - wape)))


def build_station_dataset(stations: pd.DataFrame, observations: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    merged = observations.merge(stations[["station_id", "station_name", "corridor", "latitude", "longitude"]], on="station_id", how="left")
    merged = merged.merge(context, on="observed_at", how="left")
    merged = merged.sort_values(["station_id", "observed_at"]).copy()
    merged["observed_at"] = pd.to_datetime(merged["observed_at"], utc=True)
    merged["hour"] = merged["observed_at"].dt.hour
    merged["minute"] = merged["observed_at"].dt.minute
    merged["dayofweek"] = merged["observed_at"].dt.dayofweek
    merged["is_weekend"] = merged["dayofweek"].ge(5).astype(int)
    merged["month"] = merged["observed_at"].dt.month

    for lag in [1, 2, 3, 4, 6, 12, 24, 48, 96]:
        merged[f"lag_{lag}"] = merged.groupby("station_id")["demand"].transform(lambda s: s.shift(lag))

    for window in [3, 6, 12, 24, 48, 96]:
        merged[f"rolling_mean_{window}"] = merged.groupby("station_id")["demand"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )

    for col in ["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"]:
        merged[col] = merged[col].astype(float)

    feature_columns = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_6", "lag_12", "lag_24", "lag_48", "lag_96",
        "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_mean_24", "rolling_mean_48", "rolling_mean_96",
        "hour", "minute", "dayofweek", "is_weekend", "month",
        "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity",
        "latitude", "longitude",
    ]
    merged = merged.dropna(subset=["demand", *feature_columns])
    return merged, feature_columns


def split_station_frame(frame: pd.DataFrame, train_ratio: float = 0.6, val_ratio: float = 0.2):
    total = len(frame)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train = frame.iloc[:train_end]
    val = frame.iloc[train_end:val_end]
    test = frame.iloc[val_end:]
    return train, val, test


def main() -> None:
    client = PulsoTransmiClient()
    stations = client.stations()
    observations = client.observations_dataframe(page_size=5000)
    context = client.context_dataframe(page_size=5000)

    stations["station_id"] = stations["station_id"].astype(str)
    observations["station_id"] = observations["station_id"].astype(str)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)

    merged, feature_columns = build_station_dataset(stations, observations, context)

    results = []
    for station_id, frame in merged.groupby("station_id"):
        train, val, test = split_station_frame(frame)
        if len(train) < 20 or len(val) < 10 or len(test) < 10:
            continue

        X_train = train[feature_columns]
        y_train = train["demand"]
        X_val = val[feature_columns]
        y_val = val["demand"]
        X_test = test[feature_columns]
        y_test = test["demand"]

        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_accuracy = wape_accuracy(y_val, val_pred)
        test_accuracy = wape_accuracy(y_test, test_pred)

        results.append(
            {
                "station_id": station_id,
                "station_name": frame["station_name"].iloc[0],
                "corridor": frame["corridor"].iloc[0],
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "val_accuracy": round(val_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
            }
        )

    if not results:
        raise RuntimeError("No se pudo entrenar ningún modelo para ninguna estación.")

    df_results = pd.DataFrame(results).sort_values("test_accuracy", ascending=False)
    print("\nAccuracy por estación (test set)")
    print(df_results[["station_id", "station_name", "corridor", "test_accuracy"]].to_string(index=False))
    mean_accuracy = df_results["test_accuracy"].mean()
    print(f"\nAccuracy promedio global (test): {mean_accuracy:.4f}%")

    # save simple csv
    out_dir = Path("model_results")
    out_dir.mkdir(exist_ok=True)
    df_results.to_csv(out_dir / "linear_regression_station_accuracy.csv", index=False)
    print(f"\nResultados guardados en: {out_dir / 'linear_regression_station_accuracy.csv'}")


if __name__ == "__main__":
    main()
