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
from pulso_stream import stream_observations_dataframe

from pulso_transmi import PulsoTransmiClient

ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "3.0"))
MODEL_WEIGHT = float(os.getenv("MODEL_WEIGHT", "0.6"))
ENSEMBLE_WEIGHTS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
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


def recursive_predictions(bundle, targets):
    """Predict future horizons sequentially, feeding predictions into future lags."""
    history = list(bundle["history"])
    static = bundle["static"]
    output = []
    for target in sorted(targets, key=lambda item: item["target_at"]):
        target_dt = datetime.fromisoformat(target["target_at"].replace("Z", "+00:00"))
        row = dict(static)
        row.update(hour=target_dt.hour, minute=target_dt.minute, dayofweek=target_dt.weekday(), is_weekend=int(target_dt.weekday() >= 5), month=target_dt.month)
        for lag in [1, 2, 3, 4, 6, 12, 24, 48, 96]:
            row[f"lag_{lag}"] = history[-lag] if len(history) >= lag else history[0]
        for window in [3, 6, 12, 24, 48, 96]:
            row[f"rolling_mean_{window}"] = float(np.mean(history[-window:]))
        model_value = float(bundle["model"].predict(pd.DataFrame([row], columns=FEATURES))[0])
        seasonal_value = float(row["lag_96"])
        weight = bundle.get("model_weight", MODEL_WEIGHT)
        value = max(0.0, weight * model_value + (1.0 - weight) * seasonal_value)
        history.append(value)
        output.append({"station_id": str(target["station_id"]), "target_at": target["target_at"], "value": round(value, 4)})
    return output


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
                            score.get("evaluation_rows"), commit, json.dumps({"selected": score["model"], "model_weight": score.get("model_weight")}),
                            score["station_id"],
                        ),
                    )


def merge_context_asof(observations, context):
    """Attach the latest known context at or before each observation timestamp."""
    left = observations.sort_values("observed_at").copy()
    right = context.sort_values("observed_at").copy()
    return pd.merge_asof(left, right, on="observed_at", direction="backward")


def champion_state():
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        return None
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT model_name, model_version, last_accuracy FROM public.model_registry WHERE is_champion = TRUE LIMIT 1")
            row = cur.fetchone()
    return {"model_name": row[0], "model_version": row[1], "last_accuracy": float(row[2]) if row[2] is not None else None} if row else None


def already_submitted(cycle_id):
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        return False
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM public.forecast_submissions WHERE cycle_id = %s LIMIT 1", (cycle_id,))
            return cur.fetchone() is not None


def save_submission(cycle, payload, response, predictions, commit):
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        print("SUPABASE_DB_URL no está configurada; se omite la trazabilidad de submission.")
        return
    cutoff = datetime.fromisoformat(cycle["data_cutoff"].replace("Z", "+00:00"))
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO public.forecast_submissions
                (cycle_id, client_run_id, model_version, api_status, api_response, git_commit)
                VALUES (%s,%s,%s,%s,%s,%s) RETURNING submission_id""",
                (cycle["cycle_id"], payload["client_run_id"], payload["model"]["version"], response.status_code, json.dumps(response.json()), commit))
            submission_id = cur.fetchone()[0]
            for prediction in predictions:
                target_at = datetime.fromisoformat(prediction["target_at"].replace("Z", "+00:00"))
                horizon = round((target_at - cutoff).total_seconds() / 60)
                cur.execute("""INSERT INTO public.forecast_predictions
                    (submission_id, cycle_id, station_id, target_at, horizon_minutes, predicted_demand)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (cycle_id, station_id, target_at) DO UPDATE SET
                        predicted_demand = EXCLUDED.predicted_demand""",
                    (submission_id, cycle["cycle_id"], prediction["station_id"], target_at, horizon, prediction["value"]))


def save_drift_and_champion(champion, model_name, model_version, baseline, recent, drift, cycle, commit):
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        return
    drop = (baseline - recent) if baseline is not None else None
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO public.drift_events
                (model_name, model_version, baseline_accuracy, recent_accuracy, accuracy_drop, threshold, drift_detected, action_taken, cycle_id, git_commit)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (champion["model_name"] if champion else model_name, champion["model_version"] if champion else model_version,
                 baseline, recent, drop, DRIFT_THRESHOLD, drift, "evaluate_challengers" if drift else "retrain_champion",
                 cycle.get("cycle_id"), commit))
            cur.execute("UPDATE public.model_registry SET is_champion = FALSE WHERE is_champion = TRUE")
            cur.execute("""INSERT INTO public.model_registry (model_name, model_version, is_champion, last_retrained_at, last_accuracy, git_commit)
                VALUES (%s,%s,TRUE,NOW(),%s,%s)
                ON CONFLICT (model_name, model_version) DO UPDATE SET is_champion=TRUE, last_retrained_at=NOW(), last_accuracy=EXCLUDED.last_accuracy, git_commit=EXCLUDED.git_commit""",
                (model_name, model_version, recent, commit))


def main():
    api_key = os.getenv("PULSO_API_KEY")
    if not api_key:
        raise RuntimeError("PULSO_API_KEY es obligatorio para enviar la submission")

    client = PulsoTransmiClient()
    stations = client.stations()
    base_observations = client.observations_dataframe(page_size=5000)
    stream_observations = stream_observations_dataframe()
    if "released_at" not in base_observations:
        base_observations["released_at"] = pd.NaT
    observations = pd.concat([base_observations, stream_observations], ignore_index=True)
    observations = observations.drop_duplicates(subset=["station_id", "observed_at"], keep="last")
    context = client.context_dataframe(page_size=5000)
    for frame in (stations, observations):
        frame["station_id"] = frame["station_id"].astype(str)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)

    full = observations.merge(stations[["station_id", "station_name", "corridor", "latitude", "longitude"]], on="station_id")
    full = add_features(merge_context_asof(full, context))
    champion = champion_state()
    selected, bundles, scores = {}, {}, []

    for station_id, frame in full.groupby("station_id"):
        split = max(10, int(len(frame) * 0.8))
        train, recent = frame.iloc[:split], frame.iloc[split:]
        if len(train) < 30 or len(recent) < 10:
            continue
        station_candidates = candidates()
        station_scores = {}
        best_name, best_weight, best_score = None, None, -float("inf")
        for name, model in station_candidates.items():
            model.fit(train[FEATURES], train["demand"])
            model_pred = model.predict(recent[FEATURES])
            station_scores[name] = accuracy(recent["demand"], model_pred)
            for weight in ENSEMBLE_WEIGHTS:
                blended = weight * model_pred + (1.0 - weight) * recent["lag_96"].to_numpy(dtype=float)
                score = accuracy(recent["demand"], blended)
                if score > best_score:
                    best_name, best_weight, best_score = name, weight, score
        best_model = candidates()[best_name]
        best_model.fit(frame[FEATURES], frame["demand"])
        latest = frame.iloc[-1][FEATURES].to_dict()
        bundles[station_id] = {"model": best_model, "feature_cols": FEATURES, "last_known": latest,
                               "history": frame["demand"].astype(float).tail(96).tolist(),
                               "static": {key: latest[key] for key in ["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity", "latitude", "longitude"]},
                               "model_weight": best_weight}
        selected[station_id] = best_name
        scores.append({"station_id": station_id, "model": best_name, "model_weight": best_weight, "accuracy": best_score, "train_rows": len(train), "evaluation_rows": len(recent), **station_scores})

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
    if already_submitted(cycle["cycle_id"]):
        print(f"El ciclo {cycle['cycle_id']} ya tiene una submission registrada; se omite el duplicado.")
        return
    champion_recent = float(np.mean([s[champion["model_name"]] for s in scores])) if champion and champion["model_name"] in candidates() else None
    drift = bool(champion and champion["last_accuracy"] is not None and champion_recent is not None and champion["last_accuracy"] - champion_recent >= DRIFT_THRESHOLD)
    if champion and not drift:
        selected = {station_id: champion["model_name"] for station_id in bundles}
        for station_id, frame in full.groupby("station_id"):
            model = candidates()[champion["model_name"]]
            model.fit(frame[FEATURES], frame["demand"])
            latest = frame.iloc[-1]
            bundles[station_id] = {"model": model, "feature_cols": FEATURES, "last_known": latest[FEATURES].to_dict(), "model_weight": MODEL_WEIGHT,
                                   "history": frame["demand"].astype(float).tail(96).tolist(),
                                   "static": {key: latest[key] for key in ["rain_mm", "rain_forecast", "temperature_c", "temperature_forecast", "event_intensity", "latitude", "longitude"]}}
    model_names = sorted(set(selected.values()))
    version = "adaptive_" + "_".join(model_names)
    recent_accuracy = float(np.mean([s[selected[s["station_id"]]] for s in scores]))
    save_drift_and_champion(champion, model_names[0], version, champion["last_accuracy"] if champion else None, recent_accuracy, drift, cycle, git_commit())
    save_metrics(scores, version, cycle, git_commit())
    targets_by_station = {}
    for target in cycle.get("targets", []):
        targets_by_station.setdefault(str(target["station_id"]), []).append(target)
    predictions = []
    for station_id, targets in targets_by_station.items():
        predictions.extend(recursive_predictions(bundles[station_id], targets))
    values = [item["value"] for item in predictions]
    print(json.dumps({"prediction_count": len(values), "prediction_min": min(values) if values else None, "prediction_max": max(values) if values else None, "prediction_mean": float(np.mean(values)) if values else None}, indent=2))
    payload = {
        "schema_version": "1.0", "cycle_id": cycle["cycle_id"],
        "client_run_id": f"adaptive_{uuid.uuid4().hex[:12]}", "data_cutoff": cycle["data_cutoff"],
        "model": {"version": version, "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "training_data_end": cycle["data_cutoff"], "git_commit": git_commit()},
        "predictions": predictions,
    }
    response = httpx.post(f"{API_URL}/v1/submissions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Idempotency-Key": uuid.uuid4().hex}, json=payload, timeout=60)
    print(json.dumps({"status_code": response.status_code, "response": response.json(), "selected_models": scores}, indent=2, default=str))
    response.raise_for_status()
    if response.status_code in (200, 201):
        save_submission(cycle, payload, response, predictions, git_commit())
    joblib.dump(bundles, ROOT / "model_artifacts" / "adaptive_models.joblib")


if __name__ == "__main__":
    main()
