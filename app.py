#!/usr/bin/env python3
"""
Flask backend for the Air Quality Prediction System (Indian cities).

Endpoints:
  POST /predict            { "city": "Delhi" }
  GET  /history?city=Delhi
  GET  /top-polluted
  GET  /model-comparison

Env:
  OPENWEATHER_API_KEY   required for /predict and /top-polluted
"""

from __future__ import annotations

import datetime as dt
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple
from flask_cors import CORS
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request


DATASET_CSV = "air_quality_dataset.csv"
MODEL_PATH = "model.pkl"
MODEL_COMPARISON_CSV = "model_comparison.csv"


@dataclass(frozen=True)
class CityInfo:
    region: str
    lat: float
    lon: float


def build_city_catalog() -> Dict[str, CityInfo]:
    # 20 cities grouped by regions (North, South, East, West, Central)
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
        raise RuntimeError("Missing OPENWEATHER_API_KEY environment variable.")
    return api_key


def fetch_openweather_data(api_key: str, lat: float, lon: float, timeout_s: int = 20) -> Tuple[dict, dict]:
    air_url = "https://api.openweathermap.org/data/2.5/air_pollution"
    weather_url = "https://api.openweathermap.org/data/2.5/weather"

    air_params = {"lat": lat, "lon": lon, "appid": api_key}
    weather_params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}

    air_resp = requests.get(air_url, params=air_params, timeout=timeout_s)
    air_resp.raise_for_status()
    weather_resp = requests.get(weather_url, params=weather_params, timeout=timeout_s)
    weather_resp.raise_for_status()
    return air_resp.json(), weather_resp.json()


def compute_emission_estimate(co: float, no2: float) -> float:
    # Same proxy as training script
    return float((0.001 * float(co)) + (0.05 * float(no2)))


def climate_impact_label(emission_estimate: float) -> str:
    # Simple buckets; tuned for interpretability
    if emission_estimate < 5:
        return "Low"
    if emission_estimate < 15:
        return "Moderate"
    return "High"


def aqi_health_advisory(aqi: float) -> dict:
    aqi = float(aqi)
    if aqi < 50:
        return {"level": "Good", "actions": ["Safe for outdoor activity"]}
    if 50 <= aqi < 100:
        return {"level": "Moderate", "actions": ["Sensitive groups should reduce exposure"]}
    if 100 <= aqi <= 150:
        return {"level": "Unhealthy for Sensitive Groups", "actions": ["Wear mask, limit outdoor activity"]}
    return {
        "level": "Unhealthy",
        "actions": ["N95 mask recommended, avoid going out, keep windows closed"],
    }


def load_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}. Train first to create model.pkl.")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not hasattr(obj, "predict"):
        raise RuntimeError(
            f"{path} was loaded but does not look like a trained model (missing .predict()). "
            "Re-train using air_quality_prediction_india.py to generate a valid model.pkl."
        )
    return obj


def expected_feature_names(model) -> List[str]:
    # sklearn models trained on DataFrame typically carry feature_names_in_
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise RuntimeError(
            "Loaded model does not expose feature_names_in_. Re-train using the provided training script."
        )
    return list(names)


def build_feature_row(
    model,
    city: str,
    region: str,
    pm25: float,
    no2: float,
    co: float,
    temperature: float,
    humidity: float,
    wind_speed: float,
    timestamp_utc: dt.datetime,
) -> pd.DataFrame:
    hour = int(timestamp_utc.hour)
    day = int(timestamp_utc.day)
    emission_est = compute_emission_estimate(co=co, no2=no2)

    base = {
        "PM2.5": float(pm25),
        "NO2": float(no2),
        "CO": float(co),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "hour": float(hour),
        "day": float(day),
        "emission_estimate": float(emission_est),
        "city": city,
        "region": region,
    }

    X = pd.DataFrame([base])
    X = pd.get_dummies(X, columns=["city", "region"], drop_first=False)

    # Align with model training columns
    cols = expected_feature_names(model)
    for c in cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[cols]
    return X


def parse_realtime_features(air_json: dict, weather_json: dict) -> dict:
    air_item = (air_json.get("list") or [{}])[0]
    components = air_item.get("components") or {}

    unix_ts = air_item.get("dt")
    if unix_ts is None:
        timestamp = dt.datetime.now(dt.timezone.utc)
    else:
        timestamp = dt.datetime.fromtimestamp(int(unix_ts), tz=dt.timezone.utc)

    main = weather_json.get("main") or {}
    wind = weather_json.get("wind") or {}

    pm25 = components.get("pm2_5", np.nan)
    no2 = components.get("no2", np.nan)
    co = components.get("co", np.nan)

    temperature = main.get("temp", np.nan)
    humidity = main.get("humidity", np.nan)
    wind_speed = wind.get("speed", np.nan)

    # Coerce NaNs to 0 for robustness (model was trained with missing-value handling)
    def nz(x):
        try:
            x = float(x)
            return 0.0 if np.isnan(x) else x
        except Exception:
            return 0.0

    return {
        "timestamp": timestamp,
        "PM2.5": nz(pm25),
        "NO2": nz(no2),
        "CO": nz(co),
        "temperature": nz(temperature),
        "humidity": nz(humidity),
        "wind_speed": nz(wind_speed),
    }


app = Flask(__name__)
CORS(app)
MODEL = None
CITIES = build_city_catalog()

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    city = (payload.get("city") or "").strip()
    if not city:
        return jsonify({"error": "Missing required field: city"}), 400
    if city not in CITIES:
        return jsonify({"error": f"Unknown city '{city}'. Must be one of predefined cities."}), 400

    info = CITIES[city]
    try:
        api_key = require_api_key()
        model = get_model()
        air_json, weather_json = fetch_openweather_data(api_key, info.lat, info.lon)
        feats = parse_realtime_features(air_json, weather_json)
        X = build_feature_row(
            model,
            city=city,
            region=info.region,
            pm25=feats["PM2.5"],
            no2=feats["NO2"],
            co=feats["CO"],
            temperature=feats["temperature"],
            humidity=feats["humidity"],
            wind_speed=feats["wind_speed"],
            timestamp_utc=feats["timestamp"],
        )

        aqi_pred = float(model.predict(X)[0])
        emission_est = compute_emission_estimate(co=feats["CO"], no2=feats["NO2"])
        climate = climate_impact_label(emission_est)
        advice = aqi_health_advisory(aqi_pred)

        return jsonify(
            {
                "AQI": aqi_pred,
                "city": city,
                "region": info.region,
                "health_advice": advice,
                "climate_impact": climate,
                "emission_estimate": emission_est,
                "coordinates": {"lat": info.lat, "lon": info.lon},
                "realtime": {
                    "datetime_utc": feats["timestamp"].isoformat(),
                    "PM2.5": feats["PM2.5"],
                    "NO2": feats["NO2"],
                    "CO": feats["CO"],
                    "temperature": feats["temperature"],
                    "humidity": feats["humidity"],
                    "wind_speed": feats["wind_speed"],
                },
            }
        )
    except requests.HTTPError as e:
        return jsonify({"error": "OpenWeather request failed", "details": str(e)}), 502
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


@app.get("/history")
def history():
    city = (request.args.get("city") or "").strip()
    if not city:
        return jsonify({"error": "Missing required query param: city"}), 400
    if not os.path.exists(DATASET_CSV):
        return jsonify({"error": f"Dataset CSV not found: {DATASET_CSV}"}), 404

    df = pd.read_csv(DATASET_CSV)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df_city = df[df["city"] == city].copy()
    if df_city.empty:
        return jsonify({"city": city, "records": []})
    df_city = df_city.sort_values("datetime", ascending=False).head(20)

    # Return raw recorded fields only (as requested)
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
    for c in cols:
        if c not in df_city.columns:
            df_city[c] = None
    records = df_city[cols].to_dict(orient="records")
    # Make datetime JSON-friendly
    for r in records:
        if isinstance(r.get("datetime"), (pd.Timestamp,)):
            r["datetime"] = r["datetime"].isoformat()
    return jsonify({"city": city, "records": records})


@app.get("/top-polluted")
def top_polluted():
    try:
        api_key = require_api_key()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        model = get_model()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = []
    for city, info in CITIES.items():
        try:
            air_json, weather_json = fetch_openweather_data(api_key, info.lat, info.lon)
            feats = parse_realtime_features(air_json, weather_json)
            X = build_feature_row(
                model,
                city=city,
                region=info.region,
                pm25=feats["PM2.5"],
                no2=feats["NO2"],
                co=feats["CO"],
                temperature=feats["temperature"],
                humidity=feats["humidity"],
                wind_speed=feats["wind_speed"],
                timestamp_utc=feats["timestamp"],
            )
            aqi_pred = float(model.predict(X)[0])
            results.append({"city": city, "region": info.region, "AQI": aqi_pred})
        except Exception:
            continue

    results = sorted(results, key=lambda x: x["AQI"], reverse=True)[:5]
    return jsonify({"top_5": results})


@app.get("/model-comparison")
def model_comparison():
    if not os.path.exists(MODEL_COMPARISON_CSV):
        return jsonify({"error": f"File not found: {MODEL_COMPARISON_CSV}"}), 404
    df = pd.read_csv(MODEL_COMPARISON_CSV)
    return jsonify({"metrics": df.to_dict(orient="records")})


if __name__ == "__main__":
    # For local dev:
    #   export OPENWEATHER_API_KEY=...
    #   python app.py
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)

