from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shapely.geometry import Point

matplotlib.use("Agg")

from pulso_transmi import PulsoTransmiClient

OUT_DIR = Path("eda_outputs")
OUT_DIR.mkdir(exist_ok=True)


def _save_plot(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    client = PulsoTransmiClient()

    stations = client.stations()
    observations = client.observations_dataframe(page_size=5000)
    context = client.context_dataframe(page_size=5000)

    if stations.empty or observations.empty or context.empty:
        raise RuntimeError("No se recibieron datos de la API. Revisa la conexión o la URL base.")

    stations = stations.copy()
    observations = observations.copy()
    context = context.copy()

    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)

    merged = observations.merge(stations[["station_id", "station_name", "corridor", "latitude", "longitude"]], on="station_id", how="left")
    full_df = merged.merge(context, on="observed_at", how="left")

    full_df["hour"] = full_df["observed_at"].dt.hour
    full_df["date"] = full_df["observed_at"].dt.date
    full_df["day_of_week"] = full_df["observed_at"].dt.day_name()
    full_df["month"] = full_df["observed_at"].dt.to_period("M").astype(str)

    print("\nResumen general")
    print(f"Estaciones: {stations['station_id'].nunique()}")
    print(f"Observaciones: {len(observations):,}")
    print(f"Fechas: {observations['observed_at'].min()} a {observations['observed_at'].max()}")
    print(f"Demanda media: {observations['demand'].mean():.2f}")
    print(f"Demanda máxima: {observations['demand'].max()}")
    print(f"Demanda mínima: {observations['demand'].min()}")

    print("\nTop 5 estaciones por demanda promedio:")
    station_summary = (
        full_df.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            corridor=("corridor", "first"),
            avg_demand=("demand", "mean"),
            p95_demand=("demand", lambda s: s.quantile(0.95)),
            max_demand=("demand", "max"),
            total_demand=("demand", "sum"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .sort_values("avg_demand", ascending=False)
        .head(10)
    )
    print(station_summary.to_string(index=False))

    # 1) Serie temporal total por timestamp
    total_by_time = full_df.groupby("observed_at")["demand"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.lineplot(data=total_by_time, x="observed_at", y="demand", ax=ax, color="#1f77b4")
    ax.set_title("Demanda total agregada por timestamp")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Demanda total")
    _save_plot(fig, "demand_total_timeline.png")

    # 2) Patrón horario promedio por día de la semana
    heat = (
        full_df.groupby(["day_of_week", "hour"], as_index=False)["demand"]
        .mean()
        .pivot(index="day_of_week", columns="hour", values="demand")
    )
    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = heat.reindex(index=ordered_days)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heat, cmap="YlOrRd", cbar_kws={"label": "Demanda promedio"}, ax=ax)
    ax.set_title("Patrón horario promedio por día de la semana")
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Día")
    _save_plot(fig, "hourly_demand_heatmap.png")

    # 3) Demanda por corredor
    corridor_demand = (
        full_df.groupby("corridor")["demand"].sum().sort_values(ascending=False).reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=corridor_demand, x="corridor", y="demand", palette="viridis", ax=ax)
    ax.set_title("Demanda total por corredor")
    ax.set_xlabel("Corredor")
    ax.set_ylabel("Demanda total")
    ax.tick_params(axis="x", rotation=35)
    _save_plot(fig, "demand_by_corridor.png")

    # 4) Boxplot por corredor
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=full_df, x="corridor", y="demand", palette="Set2", ax=ax)
    ax.set_title("Distribución de demanda por corredor")
    ax.set_xlabel("Corredor")
    ax.set_ylabel("Demanda")
    ax.tick_params(axis="x", rotation=35)
    _save_plot(fig, "boxplot_demand_by_corridor.png")

    # 5) Relación con clima
    climate_cols = [
        "rain_mm",
        "rain_forecast",
        "temperature_c",
        "temperature_forecast",
        "event_intensity",
    ]
    climate_sample = full_df[["demand", *climate_cols]].dropna()
    corr = climate_sample.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlación entre demanda y variables climáticas")
    _save_plot(fig, "correlation_climate.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.flatten(), climate_cols):
        sns.scatterplot(data=climate_sample, x=col, y="demand", alpha=0.6, ax=ax)
        ax.set_title(f"Demanda vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Demanda")
    _save_plot(fig, "demand_vs_climate.png")

    # 6) Georeferenciación: mapa de estaciones con geopandas
    geo_df = gpd.GeoDataFrame(
        station_summary,
        geometry=gpd.points_from_xy(station_summary["longitude"], station_summary["latitude"]),
        crs="EPSG:4326",
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    geo_df.plot(
        ax=ax,
        column="avg_demand",
        cmap="viridis",
        markersize=180,
        legend=True,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    for _, row in geo_df.iterrows():
        ax.text(row.geometry.x + 0.003, row.geometry.y + 0.003, row["station_id"], fontsize=7, color="black")
    ax.set_title("Estaciones georeferenciadas por demanda promedio")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    _save_plot(fig, "station_geo_map.png")

    # 7) Mapa interactivo con folium
    import folium

    bogota_center = [4.65, -74.09]
    map_obj = folium.Map(location=bogota_center, zoom_start=10.5, tiles="OpenStreetMap")
    for _, row in station_summary.iterrows():
        popup_html = (
            f"<b>{row['station_name']}</b><br>"
            f"Corredor: {row['corridor']}<br>"
            f"Demanda avg: {row['avg_demand']:.1f}<br>"
            f"Max: {row['max_demand']}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(8, min(24, math.sqrt(row["total_demand"]) / 3)),
            color="#d62728",
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=220),
        ).add_to(map_obj)
    map_obj.save(OUT_DIR / "stations_interactive_map.html")

    print("\nArchivos generados en:", OUT_DIR)
    for p in sorted(OUT_DIR.iterdir()):
        print("-", p.name)


if __name__ == "__main__":
    main()
