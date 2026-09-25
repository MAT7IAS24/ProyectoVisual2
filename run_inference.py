#!/usr/bin/env python3
"""Select, retrain and submit the best model for the current forecast cycle."""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd
import psycopg
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from pulso_transmi import PulsoTransmiClient

ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_6", "lag_12", "lag_24", "lag_48", "lag_96",
    "rolling_mean_3", "rolling_mean_6", "rolling_mean_12", "rolling_mean_24", "rolling_mean_48", "rolling_mean_96",
    "hour", "minute", "dayofweek", "is_weekend", "month",
    "rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity",
    "latitude", "longitude",
]


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 100.0
    return float(max(0.0, 100.0 * (1 - np.abs(y_true - y_pred).sum() / denominator)))


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["station_id", "observed_at"]).copy()
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    frame["hour"] = frame["observed_at"].dt.hour
    frame["minute"] = frame["observed_at"].dt.minute
    frame["dayofweek"] = frame["observed_at"].dt.dayofweek
    frame["is_weekend"] = frame["dayofweek"].ge(5).astype(int)
    frame["month"] = frame["observed_at"].dt.month
    grouped = frame.groupby("station_id")["demand"]
    for lag in [1, 2, 3, 4, 6, 12, 24, 48, 96]:
        frame[f"lag_{lag}"] = grouped.transform(lambda s, n=lag: s.shift(n))
    for window in [3, 6, 12, 24, 48, 96]:
        frame[f"rolling_mean_{window}"] = grouped.transform(
            lambda s, n=window: s.shift(1).rolling(n, min_periods=1).mean()
        )
    for col in ["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame.dropna(subset=["demand", *FEATURES]).copy()


def candidates():
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, random_state=42),
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def save_metrics(scores, model_version, cycle, commit):
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        print("SUPABASE_DB_URL no está configurada; se omite el guardado de métricas.")
        return
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            for score in scores:
                for model_name in ["linear_regression", "ridge", "random_forest", "hist_gradient_boosting"]:
                    cur.execute(
                        """
                        INSERT INTO public.model_evaluation_metrics (
                            model_version, model_name, station_id, station_name, corridor,
                            accuracy, cycle_id, data_cutoff, train_rows, evaluation_rows,
                            git_commit, metadata_json
                        )
                        SELECT %s, %s, %s, s.station_name, s.corridor, %s, %s, %s,
                               %s, %s, %s, %s
                        FROM public.stations s
                        WHERE s.station_id = %s
                        """,
                        (
                            model_version, model_name, score["station_id"], score[model_name],
                            cycle.get("cycle_id"), cycle.get("data_cutoff"), score.get("train_rows"),
                            score.get("evaluation_rows"), commit, json.dumps({"selected": score["model"]}),
                            score["station_id"],
                        ),
                    )


def main():
    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise RuntimeError("PULSO_API_KEY es obligatorio para enviar la submission")

    client = PulsoTransmiClient()
    stations = client.stations()
    observations = client.observations_dataframe(page_size=5000)
    context = client.context_dataframe(page_size=5000)
    for frame in (stations, observations):
        frame["station_id"] = frame["station_id"].astype(str)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)

    full = observations.merge(stations[["station_id", "station_name", "corridor", "latitude", "longitude"]], on="station_id")
    full = add_features(full.merge(context, on="observed_at", how="left"))
    selected, bundles, scores = {}, {}, []

    for station_id, frame in full.groupby("station_id"):
        split = max(10, int(len(frame) * 0.8))
        train, recent = frame.iloc[:split], frame.iloc[split:]
        if len(train) < 30 or len(recent) < 10:
            continue
        station_candidates = candidates()
        station_scores = {}
        for name, model in station_candidates.items():
            model.fit(train[FEATURES], train["demand"])
            station_scores[name] = accuracy(recent["demand"], model.predict(recent[FEATURES]))
        best_name = max(station_scores, key=station_scores.get)
        best_model = candidates()[best_name]
        best_model.fit(frame[FEATURES], frame["demand"])
        latest = frame.iloc[-1][FEATURES].to_dict()
        bundles[station_id] = {"model": best_model, "feature_cols": FEATURES, "last_known": latest}
        selected[station_id] = best_name
        scores.append({"station_id": station_id, "model": best_name, "accuracy": station_scores[best_name], "train_rows": len(train), "evaluation_rows": len(recent), **station_scores})

    if not bundles:
        raise RuntimeError("No se pudo entrenar ningún modelo")

    cycle_response = httpx.get(f"{API_URL}/v1/forecast-cycles/current", timeout=30)
    if cycle_response.status_code == 404:
        try:
            detail = cycle_response.json().get("detail", {})
        except ValueError:
            detail = {}
        if detail.get("code") == "no_open_cycle":
            print("No hay un ciclo de pronóstico abierto; no se envía submission en esta ejecución.")
            return
    cycle_response.raise_for_status()
    cycle = cycle_response.json()
    predictions = []
    for target in cycle.get("targets", []):
        station_id = str(target["station_id"])
        bundle = bundles[station_id]
        target_dt = datetime.fromisoformat(target["target_at"].replace("Z", "+00:00"))
        row = bundle["last_known"].copy()
        row.update(hour=target_dt.hour, minute=target_dt.minute, dayofweek=target_dt.weekday(), is_weekend=int(target_dt.weekday() >= 5), month=target_dt.month)
        value = max(0.0, float(bundle["model"].predict(pd.DataFrame([row], columns=FEATURES))[0]))
        predictions.append({"station_id": station_id, "target_at": target["target_at"], "value": round(value, 4)})

    model_names = sorted(set(selected.values()))
    version = "adaptive_" + "_".join(model_names)
    save_metrics(scores, version, cycle, git_commit())
    payload = {
        "schema_version": "1.0", "cycle_id": cycle["cycle_id"],
        "client_run_id": f"adaptive_{uuid.uuid4().hex[:12]}", "data_cutoff": cycle["data_cutoff"],
        "model": {"version": version, "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "training_data_end": cycle["data_cutoff"], "git_commit": git_commit()},
        "predictions": predictions,
    }
    response = httpx.post(f"{API_URL}/v1/submissions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Idempotency-Key": uuid.uuid4().hex}, json=payload, timeout=60)
    print(json.dumps({"status_code": response.status_code, "response": response.json(), "selected_models": scores}, indent=2, default=str))
    response.raise_for_status()
    joblib.dump(bundles, ROOT / "model_artifacts" / "adaptive_models.joblib")


if __name__ == "__main__":
    main()
