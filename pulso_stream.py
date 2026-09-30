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
    return frame
