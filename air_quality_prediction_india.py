#!/usr/bin/env python3
"""
Air Quality Prediction System (Indian Cities) - end-to-end script.

What it does:
- Uses a predefined dictionary of Indian cities grouped by region with lat/lon
- Fetches OpenWeather Air Pollution + Weather data for a selected city
- Appends a row to a CSV dataset
- Loads dataset, preprocesses, creates AQI target + emission estimate
- Trains & compares: RandomForestRegressor, XGBoost Regressor, ARIMA on PM2.5 series
- Saves model metrics to model_comparison.csv
- Saves models: rf_model.pkl, xgb_model.pkl, arima_model.pkl
- Saves final model (Random Forest) as model.pkl

Usage:
  export OPENWEATHER_API_KEY="your_key"
  python air_quality_prediction_india.py --city "Delhi"

Notes:
- You can run this script repeatedly to grow the dataset CSV over time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


DATASET_CSV = "air_quality_dataset.csv"
MODEL_COMPARISON_CSV = "model_comparison.csv"


@dataclass(frozen=True)
class CityInfo:
    region: str
    lat: float
    lon: float


def build_city_catalog() -> Dict[str, CityInfo]:
    """
    Predefined dictionary of 20 Indian cities grouped by region.
    Regions: North, South, East, West, Central
    """
    return {
        # North
        "Delhi": CityInfo("North", 28.6139, 77.2090),
        "Chandigarh": CityInfo("North", 30.7333, 76.7794),
        "Jaipur": CityInfo("North", 26.9124, 75.7873),
        "Lucknow": CityInfo("North", 26.8467, 80.9462),
        # South
        "Bengaluru": CityInfo("South", 12.9716, 77.5946),
        "Chennai": CityInfo("South", 13.0827, 80.2707),
        "Hyderabad": CityInfo("South", 17.3850, 78.4867),
        "Kochi": CityInfo("South", 9.9312, 76.2673),
        # East
        "Kolkata": CityInfo("East", 22.5726, 88.3639),
        "Bhubaneswar": CityInfo("East", 20.2961, 85.8245),
        "Patna": CityInfo("East", 25.5941, 85.1376),
        "Guwahati": CityInfo("East", 26.1445, 91.7362),
        # West
        "Mumbai": CityInfo("West", 19.0760, 72.8777),
        "Pune": CityInfo("West", 18.5204, 73.8567),
        "Ahmedabad": CityInfo("West", 23.0225, 72.5714),
        "Surat": CityInfo("West", 21.1702, 72.8311),
        # Central
        "Bhopal": CityInfo("Central", 23.2599, 77.4126),
        "Indore": CityInfo("Central", 22.7196, 75.8577),
        "Nagpur": CityInfo("Central", 21.1458, 79.0882),
        "Raipur": CityInfo("Central", 21.2514, 81.6296),
    }


def require_api_key() -> str:
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENWEATHER_API_KEY. Set it in your environment, e.g.\n"
            "  export OPENWEATHER_API_KEY='your_key_here'\n"
            "Then re-run the script."
        )
    return api_key


def fetch_openweather_data(api_key: str, lat: float, lon: float, timeout_s: int = 20) -> Tuple[dict, dict]:
    """
    Returns (air_pollution_json, weather_json).
    OpenWeather endpoints:
      - Air pollution: /data/2.5/air_pollution
      - Current weather: /data/2.5/weather
    """
    air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
    weather_url = "https://api.openweathermap.org/data/2.5/weather"

    air_params = {"lat": lat, "lon": lon, "appid": api_key}
    weather_params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

    air_resp = requests.get(air_url, params=air_params, timeout=timeout_s)
    air_resp.raise_for_status()
    weather_resp = requests.get(weather_url, params=weather_params, timeout=timeout_s)
    weather_resp.raise_for_status()

    return air_resp.json(), weather_resp.json()


def parse_to_row(city: str, city_info: CityInfo, air_json: dict, weather_json: dict) -> dict:
    """
    Extracts required fields into a normalized row.
    Columns:
      city, region, datetime, PM2.5, NO2, CO, temperature, humidity, wind_speed
    """
    # Air pollution response: list[0].components
    air_item = (air_json.get("list") or [{}])[0]
    components = air_item.get("components") or {}
    unix_ts = air_item.get("dt")
    if unix_ts is None:
        # Fallback to current UTC time if dt is missing
        timestamp = dt.datetime.now(dt.timezone.utc)
    else:
        timestamp = dt.datetime.fromtimestamp(int(unix_ts), tz=dt.timezone.utc)

    # Weather response: main, wind
    main = weather_json.get("main") or {}
    wind = weather_json.get("wind") or {}

    row = {
        "city": city,
        "region": city_info.region,
        "datetime": timestamp.isoformat(),
        "PM2.5": components.get("pm2_5", np.nan),
        "NO2": components.get("no2", np.nan),
        "CO": components.get("co", np.nan),
        "temperature": main.get("temp", np.nan),
        "humidity": main.get("humidity", np.nan),
        "wind_speed": wind.get("speed", np.nan),
    }
    return row


def append_row_to_csv(row: dict, csv_path: str = DATASET_CSV) -> None:
    cols = [
        "city",
        "region",
        "datetime",
        "PM2.5",
        "NO2",
        "CO",
        "temperature",
        "humidity",
        "wind_speed",
    ]
    df_row = pd.DataFrame([row], columns=cols)

    if os.path.exists(csv_path):
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)


def load_dataset(csv_path: str = DATASET_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}\n"
            "Run the script once with --fetch to create it, e.g.\n"
            "  python air_quality_prediction_india.py --city Delhi --fetch"
        )
    df = pd.read_csv(csv_path)
    return df


def compute_aqi_from_pm25(pm25: pd.Series) -> pd.Series:
    """
    Simple AQI proxy based on PM2.5.
    Not an official AQI standard; designed for demonstration and modeling target creation.
    """
    # Linear proxy with cap
    aqi = (pm25.astype(float) * 4.0).clip(lower=0, upper=500)
    return aqi


def compute_emission_estimate(df: pd.DataFrame) -> pd.Series:
    """
    Simple emission estimate for climate connection.
    Uses a weighted combination of CO and NO2 (both commonly linked to combustion sources).
    """
    co = pd.to_numeric(df["CO"], errors="coerce")
    no2 = pd.to_numeric(df["NO2"], errors="coerce")
    # Scale to keep values in a reasonable range for learning
    emission = (0.001 * co) + (0.05 * no2)
    return emission


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse datetime and extract time features
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day

    # Ensure numeric columns
    numeric_cols = ["PM2.5", "NO2", "CO", "temperature", "humidity", "wind_speed", "hour", "day"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Missing values: time features -> mode, measurements -> median
    for c in ["hour", "day"]:
        if df[c].isna().any():
            if df[c].dropna().empty:
                df[c] = df[c].fillna(0)
            else:
                df[c] = df[c].fillna(int(df[c].mode().iloc[0]))

    for c in ["PM2.5", "NO2", "CO", "temperature", "humidity", "wind_speed"]:
        if df[c].isna().any():
            med = df[c].median()
            if np.isnan(med):
                df[c] = df[c].fillna(0.0)
            else:
                df[c] = df[c].fillna(med)

    # AQI target + emission estimate
    df["AQI"] = compute_aqi_from_pm25(df["PM2.5"])
    df["emission_estimate"] = compute_emission_estimate(df)

    # Drop rows where datetime is missing (can’t use for ARIMA split/order reliably)
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)
    return df


def make_supervised_matrices(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns X, y for supervised models (RF, XGB) predicting AQI.
    Includes one-hot encodings for city and region.
    """
    feature_cols = [
        "PM2.5",
        "NO2",
        "CO",
        "temperature",
        "humidity",
        "wind_speed",
        "hour",
        "day",
        "emission_estimate",
        "city",
        "region",
    ]
    X = df[feature_cols].copy()
    y = df["AQI"].astype(float)
    X = pd.get_dummies(X, columns=["city", "region"], drop_first=False)
    return X, y


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return float(r2), rmse


def train_random_forest(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    print("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2, rmse = evaluate_regression(y_test.to_numpy(), preds)
    return model, r2, rmse


def train_xgboost(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    print("Training XGBoost...")
    try:
        from xgboost import XGBRegressor
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "xgboost is not installed. Install it with:\n"
            "  pip install xgboost\n"
            "Then re-run the script."
        ) from e

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2, rmse = evaluate_regression(y_test.to_numpy(), preds)
    return model, r2, rmse


def train_arima_on_pm25(df: pd.DataFrame, city: str):
    """
    Trains ARIMA on PM2.5 series for the selected city.
    Evaluates by forecasting the last 20% of observations.
    """
    print("Training ARIMA...")
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "statsmodels is not installed. Install it with:\n"
            "  pip install statsmodels\n"
            "Then re-run the script."
        ) from e

    ts = df[df["city"] == city].sort_values("datetime").reset_index(drop=True)
    series = ts["PM2.5"].astype(float).to_numpy()

    if len(series) < 12:
        raise RuntimeError(
            f"Not enough time-series points for ARIMA for city={city}. "
            f"Need at least ~12, found {len(series)}. Keep fetching more rows by re-running with --fetch."
        )

    split = max(1, int(0.8 * len(series)))
    train, test = series[:split], series[split:]

    # Conservative order for stability on small datasets
    order = (1, 0, 1)
    model = ARIMA(train, order=order).fit()
    forecast = model.forecast(steps=len(test))

    r2, rmse = evaluate_regression(test, np.asarray(forecast))
    return model, r2, rmse


def save_model(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_arima_model(obj, path: str) -> None:
    """
    Saves ARIMA model (statsmodels results) via pickle for demonstration purposes.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def write_comparison(results: list[dict], path: str = MODEL_COMPARISON_CSV) -> None:
    pd.DataFrame(results).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    catalog = build_city_catalog()
    parser = argparse.ArgumentParser(description="Air quality prediction system for Indian cities (OpenWeather).")
    parser.add_argument("--city", type=str, default="Delhi", help="City name (must be in predefined catalog).")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=f"Fetch latest OpenWeather data and append to {DATASET_CSV}.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DATASET_CSV,
        help="Path to dataset CSV (default: air_quality_dataset.csv).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--list-cities",
        action="store_true",
        help="Print available cities and exit.",
    )

    args = parser.parse_args()
    if args.list_cities:
        regions: Dict[str, list[str]] = {}
        for name, info in catalog.items():
            regions.setdefault(info.region, []).append(name)
        for region in ["North", "South", "East", "West", "Central"]:
            names = sorted(regions.get(region, []))
            print(f"{region}: {', '.join(names)}")
        sys.exit(0)

    if args.city not in catalog:
        raise SystemExit(
            f"Unknown city '{args.city}'. Use --list-cities to see available options."
        )
    return args


def main() -> None:
    args = parse_args()
    catalog = build_city_catalog()
    city = args.city
    info = catalog[city]

    if args.fetch:
        api_key = require_api_key()
        air_json, weather_json = fetch_openweather_data(api_key, info.lat, info.lon)
        row = parse_to_row(city, info, air_json, weather_json)
        append_row_to_csv(row, csv_path=args.csv)
        print(f"Appended 1 row to {args.csv} for {city} ({info.region}).")

    df = load_dataset(args.csv)
    df = preprocess(df)

    # Supervised models (AQI regression)
    X, y = make_supervised_matrices(df)

    if len(df) < 10:
        print(
            f"Warning: dataset has only {len(df)} rows. "
            "Model metrics may be unstable. Re-run with --fetch multiple times to grow the dataset."
        )

    comparison: list[dict] = []

    rf_model, rf_r2, rf_rmse = train_random_forest(X, y, random_state=args.random_state)
    comparison.append({"model": "RandomForestRegressor", "r2": rf_r2, "rmse": rf_rmse})
    save_model(rf_model, "rf_model.pkl")

    try:
        xgb_model, xgb_r2, xgb_rmse = train_xgboost(X, y, random_state=args.random_state)
        comparison.append({"model": "XGBoostRegressor", "r2": xgb_r2, "rmse": xgb_rmse})
        save_model(xgb_model, "xgb_model.pkl")
    except Exception as e:
        print(f"Training XGBoost... (skipped) Reason: {e}")
        comparison.append({"model": "XGBoostRegressor", "r2": np.nan, "rmse": np.nan})
        save_model(None, "xgb_model.pkl")

    # ARIMA on PM2.5 time series (selected city)
    try:
        arima_model, arima_r2, arima_rmse = train_arima_on_pm25(df, city=city)
        comparison.append({"model": "ARIMA(PM2.5)", "r2": arima_r2, "rmse": arima_rmse})
        save_arima_model(arima_model, "arima_model.pkl")
    except Exception as e:
        print(f"Training ARIMA... (skipped) Reason: {e}")
        comparison.append({"model": "ARIMA(PM2.5)", "r2": np.nan, "rmse": np.nan})
        save_arima_model(None, "arima_model.pkl")

    # Print metrics
    for r in comparison:
        model_name = r["model"]
        r2 = r["r2"]
        rmse = r["rmse"]
        print(f"{model_name}: R2={r2:.4f} RMSE={rmse:.4f}" if pd.notna(r2) else f"{model_name}: R2=NA RMSE=NA")

    # Save comparison CSV
    write_comparison(comparison, path=MODEL_COMPARISON_CSV)
    print(f"Saved model comparison to {MODEL_COMPARISON_CSV}.")

    # Select Random Forest as final model
    save_model(rf_model, "model.pkl")
    print("Saved final model (Random Forest) to model.pkl.")


if __name__ == "__main__":
    main()

