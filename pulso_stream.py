from __future__ import annotations

import os
from typing import Any

import httpx
import pandas as pd


API_URL = os.getenv("PULSO_API_URL", "https://pulso-transmi.72-60-245-2.sslip.io").rstrip("/")


def stream_observations_dataframe(limit: int = 5000) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor = None
    with httpx.Client(base_url=API_URL, timeout=60.0) as client:
        while True:
            params = {"limit": limit}
            if cursor:
                params["cursor"] = cursor
            response = client.get("/v1/stream/observations", params=params)
            response.raise_for_status()
            payload = response.json()
            rows.extend(payload.get("data", []))
            cursor = payload.get("next_cursor")
            if not cursor:
                break
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["station_id"] = frame["station_id"].astype(str)
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        frame["released_at"] = pd.to_datetime(frame["released_at"], utc=True)
        # API schema v2 may leave demand null and expose the value under
        # measurement.value. Normalize both schemas to the internal demand field.
        if "measurement" in frame.columns:
            measured_values = frame["measurement"].map(
                lambda item: item.get("value") if isinstance(item, dict) else None
            )
            frame["demand"] = frame["demand"].where(frame["demand"].notna(), measured_values)
        frame["demand"] = pd.to_numeric(frame["demand"], errors="coerce")
        invalid = frame["demand"].isna()
        if invalid.any():
            print(f"Advertencia: se descartaron {int(invalid.sum())} observaciones del stream sin demanda válida.")
            frame = frame.loc[~invalid].copy()
        frame["demand"] = frame["demand"].round().astype(int)
    return frame
