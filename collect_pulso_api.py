#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import psycopg

ROOT = Path(__file__).resolve().parent
SDK_SRC = ROOT / "pulso-transmi-sdk" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from pulso_transmi import PulsoTransmiClient


def get_db_url() -> str:
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL or DATABASE_URL is required")
    return db_url


def upsert_stations(conn: psycopg.Connection, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0

    rows = [
        (
            str(row["station_id"]),
            str(row["station_name"]),
            str(row["corridor"]),
            float(row["latitude"]),
            float(row["longitude"]),
        )
        for _, row in frame.iterrows()
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO stations (station_id, station_name, corridor, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (station_id) DO UPDATE SET
                station_name = EXCLUDED.station_name,
                corridor = EXCLUDED.corridor,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude
            """,
            rows,
        )
    return len(rows)


def upsert_observations(conn: psycopg.Connection, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0

    rows = [
        (
            str(row["station_id"]),
            row["observed_at"].to_pydatetime(),
            int(row["demand"]),
        )
        for _, row in frame.iterrows()
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO demand_observations (station_id, observed_at, demand)
            VALUES (%s, %s, %s)
            ON CONFLICT (station_id, observed_at) DO UPDATE SET
                demand = EXCLUDED.demand
            """,
            rows,
        )
    return len(rows)


def upsert_context(conn: psycopg.Connection, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0

    rows = [
        (
            row["observed_at"].to_pydatetime(),
            float(row["rain_mm"]),
            float(row["rain_forecast"]),
            float(row["temperature_c"]),
            float(row["temperature_forecast"]),
            float(row["event_intensity"]),
        )
        for _, row in frame.iterrows()
    ]

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO context_observations (
                observed_at,
                rain_mm,
                rain_forecast,
                temperature_c,
                temperature_forecast,
                event_intensity
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (observed_at) DO UPDATE SET
                rain_mm = EXCLUDED.rain_mm,
                rain_forecast = EXCLUDED.rain_forecast,
                temperature_c = EXCLUDED.temperature_c,
                temperature_forecast = EXCLUDED.temperature_forecast,
                event_intensity = EXCLUDED.event_intensity
            """,
            rows,
        )
    return len(rows)


def main() -> None:
    client = PulsoTransmiClient()

    stations = client.stations()
    observations = client.observations_dataframe(page_size=5000)
    context = client.context_dataframe(page_size=5000)

    db_url = get_db_url()
    with psycopg.connect(db_url) as conn:
        station_count = upsert_stations(conn, stations)
        observation_count = upsert_observations(conn, observations)
        context_count = upsert_context(conn, context)

    print(
        {
            "stations_upserted": station_count,
            "observations_upserted": observation_count,
            "context_upserted": context_count,
        }
    )


if __name__ == "__main__":
    main()
