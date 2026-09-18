#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
MODEL_PATH = ROOT / "model_artifacts" / "linear_regression_by_station.joblib"


def load_api_key() -> str:
    api_key = os.getenv("PULSO_API_KEY")
    if api_key and api_key.strip():
        return api_key.strip()

    candidates = [
        ROOT / ".env",
        ROOT / "pulso-transmi-sdk" / ".env",
        ROOT / "pulso-transmi-sdk" / ".env.example",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        text = candidate.read_text(encoding="utf-8")
        m = re.search(r"^\s*PULSO_API_KEY\s*=\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if m:
            value = m.group(1).strip().strip('"\'')
            if value and value != "se-habilitara-para-submissions":
                return value

    raise SystemExit(
        "No se encontró una API key. Define PULSO_API_KEY o crea un .env con la variable PULSO_API_KEY."
    )


def current_cycle(base_url: str) -> dict:
    response = httpx.get(f"{base_url}/v1/forecast-cycles/current", timeout=30)
    response.raise_for_status()
    return response.json()


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()[:40]
    except Exception:
        return None


def build_payload(cycle: dict, model_bundle: dict, model_version: str) -> dict:
    cycle_id = cycle["cycle_id"]
    cutoff = cycle["data_cutoff"]
    predictions = []

    for target in cycle.get("targets", []):
        station_id = str(target["station_id"])
        target_at = target["target_at"]
        station_meta = model_bundle.get(station_id)
        if station_meta is None:
            raise ValueError(f"No existe modelo para la estación {station_id}")

        last_known = station_meta["last_known"].copy()
        target_dt = __import__("datetime").datetime.fromisoformat(target_at.replace("Z", "+00:00"))
        last_known["hour"] = target_dt.hour
        last_known["minute"] = target_dt.minute
        last_known["dayofweek"] = target_dt.weekday()
        last_known["is_weekend"] = 1 if target_dt.weekday() >= 5 else 0
        last_known["month"] = target_dt.month

        feature_cols = station_meta["feature_cols"]
        feature_row = []
        for name in feature_cols:
            if name in last_known:
                value = last_known[name]
            else:
                value = 0.0
            feature_row.append(float(value))

        model = station_meta["model"]
        pred = float(model.predict(np.asarray([feature_row], dtype=float))[0])
        pred = max(0.0, float(pred))

        predictions.append(
            {
                "station_id": station_id,
                "target_at": target_at,
                "value": round(pred, 4),
            }
        )

    payload = {
        "schema_version": "1.0",
        "cycle_id": cycle_id,
        "client_run_id": f"exp_01_{uuid.uuid4().hex[:12]}",
        "data_cutoff": cutoff,
        "model": {
            "version": model_version,
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "training_data_end": cutoff,
            "git_commit": git_commit(),
        },
        "predictions": predictions,
    }
    return payload


def submit(base_url: str, api_key: str, payload: dict):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Idempotency-Key": uuid.uuid4().hex,
    }
    response = httpx.post(f"{base_url}/v1/submissions", headers=headers, json=payload, timeout=60)
    print(f"status={response.status_code}")
    try:
        print(response.json())
    except Exception:
        print(response.text)
    response.raise_for_status()
    return response.json()


def main() -> None:
    api_key = load_api_key()
    base_url = os.getenv("PULSO_API_URL", DEFAULT_URL)
    cycle = current_cycle(base_url)

    if not MODEL_PATH.exists():
        raise SystemExit(f"No existe el modelo: {MODEL_PATH}")

    model_bundle = joblib.load(MODEL_PATH)
    payload = build_payload(cycle, model_bundle, "exp_01_linear_regression")

    print("Cycle:", cycle.get("cycle_id"))
    print("Cutoff:", cycle.get("data_cutoff"))
    print("Predicciones a enviar:", len(payload["predictions"]))
    print(json.dumps(payload["predictions"][:3], indent=2))

    submit(base_url, api_key, payload)
    print("Submission enviada correctamente.")


if __name__ == "__main__":
    main()
