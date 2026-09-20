"""
forecasting.py
--------------
Multi-step forward forecasting engine and rule-based engineering insights generator.
Supports horizons: 1 Hour (4 steps), 6 Hours (24 steps), 24 Hours (96 steps).
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from utils.data_processing import (
    PV_FEATURE_COLS,
    WIND_FEATURE_COLS,
    PV_RATED_CAPACITY_KW,
    WIND_RATED_CAPACITY_KW,
    PV_TEMP_COEFF,
    INVERTER_EFFICIENCY,
)


def load_trained_models(models_dir: str = "models") -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load serialized PV and Wind models along with metadata.
    """
    from pathlib import Path
    import json

    m_dir = Path(models_dir)
    pv_path = m_dir / "pv_model.pkl"
    wind_path = m_dir / "wind_model.pkl"
    meta_path = m_dir / "model_metadata.json"

    if not pv_path.exists() or not wind_path.exists():
        raise FileNotFoundError(
            f"Trained models not found in '{models_dir}'. Please run train_model.py first."
        )

    pv_model = joblib.load(pv_path)
    wind_model = joblib.load(wind_path)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return pv_model, wind_model, metadata


def generate_forecast(
    df_history: pd.DataFrame,
    pv_model: Any,
    wind_model: Any,
    horizon_hours: int = 6,
    interval_minutes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perform multi-step recursive forecasting for PV, Wind, and Combined generation.

    Parameters:
    - df_history: Historical DataFrame up to current reference time (must contain all engineered columns)
    - pv_model: Trained PV regressor
    - wind_model: Trained Wind regressor
    - horizon_hours: Forecast horizon in hours (e.g., 1, 6, 24)
    - interval_minutes: Telemetry interval (auto-detected from data if None)

    Returns:
    - DataFrame containing timestamp, forecasted environmental conditions,
      and predicted pv_power, wind_power, and total_power.
    """
    if interval_minutes is None:
        if len(df_history) > 1 and "timestamp" in df_history.columns:
            t0 = pd.to_datetime(df_history["timestamp"].iloc[-2])
            t1 = pd.to_datetime(df_history["timestamp"].iloc[-1])
            dt_secs = abs((t1 - t0).total_seconds())
            interval_minutes = max(1, int(round(dt_secs / 60.0)))
        else:
            interval_minutes = 60

    steps = int(horizon_hours * 60 / interval_minutes)

    # Reference point: last available row
    last_row = df_history.iloc[-1]
    last_time = pd.to_datetime(last_row["timestamp"])

    # 24-hour diurnal lag step count (24 for 1h data, 96 for 15m data)
    diurnal_steps = int(round(24 * 60 / interval_minutes))

    # Recent history ring-buffers for autoregressive lag calculations
    pv_history = list(df_history["pv_power"].values[-max(120, diurnal_steps + 10):])
    wind_history = list(df_history["wind_power"].values[-max(120, diurnal_steps + 10):])

    last_cloud = float(last_row["cloud_cover"])
    last_wind_speed = float(last_row["wind_speed"])
    last_wind_dir = float(last_row["wind_direction"])
    last_temp = float(last_row["ambient_temperature"])

    forecast_rows = []

    for step in range(1, steps + 1):
        future_time = last_time + timedelta(minutes=step * interval_minutes)
        hr_float = future_time.hour + future_time.minute / 60.0
        day_of_year = future_time.timetuple().tm_yday
        month = future_time.month

        # 1. Project environmental variables using diurnal physics + persistence
        # Sun position
        solar_noon = 12.5
        daylight_half_width = 6.25
        time_diff = abs(hr_float - solar_noon)
        is_daylight = time_diff < daylight_half_width

        # Extrapolate cloud cover with mean reversion to 25%
        projected_cloud = np.clip(last_cloud * 0.95 + 25.0 * 0.05 + np.random.normal(0, 1.5), 0.0, 95.0)

        # Solar irradiance projection
        if is_daylight:
            clear_sky = 1000.0 * np.cos((time_diff / daylight_half_width) * (np.pi / 2.0))
            attenuation = 1.0 - (projected_cloud / 100.0) * 0.78
            projected_irr = np.clip(clear_sky * attenuation, 0.0, 1150.0)
        else:
            projected_irr = 0.0

        # Ambient and module temp projection
        diurnal_temp = 7.0 * np.sin(2 * np.pi * (hr_float - 9) / 24.0)
        seasonal_temp = 22.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        projected_amb_temp = np.clip(seasonal_temp + diurnal_temp, 5.0, 45.0)
        projected_mod_temp = projected_amb_temp + (projected_irr / 800.0) * 25.0

        # Projected humidity (inversely proportional to temp)
        projected_humidity = np.clip(65.0 - (diurnal_temp * 2.2), 20.0, 95.0)

        # Projected wind speed: diurnal thermal wind pattern + persistence
        thermal_wind = 1.8 * np.sin(2 * np.pi * (hr_float - 12) / 24.0)
        projected_wind_speed = np.clip(last_wind_speed * 0.92 + (5.5 + thermal_wind) * 0.08, 0.5, 24.0)
        projected_wind_dir = (last_wind_dir + np.random.normal(0, 1.0)) % 360.0

        # 2. Assemble features for PV model
        pv_lag_1 = pv_history[-1]
        pv_lag_2 = pv_history[-2] if len(pv_history) >= 2 else pv_lag_1
        pv_lag_4 = pv_history[-4] if len(pv_history) >= 4 else pv_lag_1
        pv_lag_diurnal = pv_history[-diurnal_steps] if len(pv_history) >= diurnal_steps else pv_lag_1
        pv_roll_4_mean = np.mean(pv_history[-4:])
        pv_roll_4_std = np.std(pv_history[-4:])

        pv_feature_dict = {
            "hour": future_time.hour,
            "minute": future_time.minute,
            "day_of_year": day_of_year,
            "month": month,
            "sin_hour": np.sin(2 * np.pi * hr_float / 24.0),
            "cos_hour": np.cos(2 * np.pi * hr_float / 24.0),
            "solar_irradiance": projected_irr,
            "ambient_temperature": projected_amb_temp,
            "module_temperature": projected_mod_temp,
            "humidity": projected_humidity,
            "cloud_cover": projected_cloud,
            "pv_lag_1": pv_lag_1,
            "pv_lag_2": pv_lag_2,
            "pv_lag_4": pv_lag_4,
            "pv_lag_96": pv_lag_diurnal,
            "pv_rolling_mean_4": pv_roll_4_mean,
            "pv_rolling_std_4": pv_roll_4_std,
        }

        # 3. Assemble features for Wind model
        wind_lag_1 = wind_history[-1]
        wind_lag_2 = wind_history[-2] if len(wind_history) >= 2 else wind_lag_1
        wind_lag_4 = wind_history[-4] if len(wind_history) >= 4 else wind_lag_1
        wind_lag_diurnal = wind_history[-diurnal_steps] if len(wind_history) >= diurnal_steps else wind_lag_1
        wind_roll_4_mean = np.mean(wind_history[-4:])
        wind_roll_4_std = np.std(wind_history[-4:])

        wind_rad = np.radians(projected_wind_dir)
        wind_feature_dict = {
            "hour": future_time.hour,
            "minute": future_time.minute,
            "day_of_year": day_of_year,
            "month": month,
            "wind_speed": projected_wind_speed,
            "wind_speed_cubed": projected_wind_speed**3,
            "sin_wind_dir": np.sin(wind_rad),
            "cos_wind_dir": np.cos(wind_rad),
            "ambient_temperature": projected_amb_temp,
            "humidity": projected_humidity,
            "wind_lag_1": wind_lag_1,
            "wind_lag_2": wind_lag_2,
            "wind_lag_4": wind_lag_4,
            "wind_lag_96": wind_lag_diurnal,
            "wind_rolling_mean_4": wind_roll_4_mean,
            "wind_rolling_std_4": wind_roll_4_std,
        }

        # 4. Predict forward step
        X_pv = pd.DataFrame([pv_feature_dict])[PV_FEATURE_COLS]
        pred_pv = float(pv_model.predict(X_pv)[0])
        # Solar cannot generate without irradiance
        if projected_irr < 5.0 or not is_daylight:
            pred_pv = 0.0
        else:
            pred_pv = float(np.clip(pred_pv, 0.0, PV_RATED_CAPACITY_KW * 1.05))

        X_wind = pd.DataFrame([wind_feature_dict])[WIND_FEATURE_COLS]
        pred_wind = float(wind_model.predict(X_wind)[0])
        if projected_wind_speed < 3.0 or projected_wind_speed >= 25.0:
            pred_wind = 0.0
        else:
            pred_wind = float(np.clip(pred_wind, 0.0, WIND_RATED_CAPACITY_KW * 1.05))

        pred_total = pred_pv + pred_wind

        # Append to recursive history buffers
        pv_history.append(pred_pv)
        wind_history.append(pred_wind)
        last_cloud = projected_cloud
        last_wind_speed = projected_wind_speed
        last_wind_dir = projected_wind_dir

        forecast_rows.append(
            {
                "timestamp": future_time,
                "solar_irradiance": round(projected_irr, 1),
                "ambient_temperature": round(projected_amb_temp, 1),
                "module_temperature": round(projected_mod_temp, 1),
                "humidity": round(projected_humidity, 1),
                "cloud_cover": round(projected_cloud, 1),
                "wind_speed": round(projected_wind_speed, 2),
                "wind_direction": round(projected_wind_dir, 1),
                "pv_power_forecast": round(pred_pv, 3),
                "wind_power_forecast": round(pred_wind, 3),
                "total_power_forecast": round(pred_total, 3),
            }
        )

    return pd.DataFrame(forecast_rows)


def generate_ai_insights(
    df_history: pd.DataFrame, df_forecast: pd.DataFrame
) -> List[Dict[str, str]]:
    """
    Generate physics-based engineering insights from historical telemetry
    and forward predictions without hallucinations or fake AI statements.
    """
    insights = []
    if df_forecast.empty:
        return insights

    # Current telemetry
    current_row = df_history.iloc[-1]
    curr_pv = float(current_row["pv_power"])
    curr_wind = float(current_row["wind_power"])
    curr_total = float(current_row["total_power"])
    curr_irr = float(current_row["solar_irradiance"])

    # Forecast statistics
    avg_pv_fc = float(df_forecast["pv_power_forecast"].mean())
    max_pv_fc = float(df_forecast["pv_power_forecast"].max())
    max_pv_time = df_forecast.loc[df_forecast["pv_power_forecast"].idxmax()]["timestamp"]

    avg_wind_fc = float(df_forecast["wind_power_forecast"].mean())
    max_wind_fc = float(df_forecast["wind_power_forecast"].max())
    min_wind_fc = float(df_forecast["wind_power_forecast"].min())
    wind_volatility = float(df_forecast["wind_power_forecast"].std())

    avg_total_fc = float(df_forecast["total_power_forecast"].mean())
    max_total_fc = float(df_forecast["total_power_forecast"].max())
    max_total_time = df_forecast.loc[df_forecast["total_power_forecast"].idxmax()]["timestamp"]

    # Total expected energy yield over horizon (kWh = kW * hours)
    if len(df_forecast) > 1:
        step_hours = (pd.to_datetime(df_forecast["timestamp"].iloc[1]) - pd.to_datetime(df_forecast["timestamp"].iloc[0])).total_seconds() / 3600.0
        if step_hours <= 0:
            step_hours = 1.0
    else:
        step_hours = 1.0
    duration_hours = len(df_forecast) * step_hours
    energy_kwh = float(df_forecast["total_power_forecast"].sum() * step_hours)
    pv_energy_kwh = float(df_forecast["pv_power_forecast"].sum() * step_hours)
    wind_energy_kwh = float(df_forecast["wind_power_forecast"].sum() * step_hours)

    # 1. Solar Photovoltaic Dynamics
    if max_pv_fc > 0.5:
        if avg_pv_fc > curr_pv:
            insights.append(
                {
                    "category": "Solar PV Ramp-Up",
                    "type": "positive",
                    "title": "Solar Generation Increasing",
                    "text": f"PV generation is projected to rise, peaking at {max_pv_fc:.2f} kW around {max_pv_time.strftime('%H:%M')} due to increasing solar irradiance.",
                }
            )
        else:
            insights.append(
                {
                    "category": "Solar PV Ramp-Down",
                    "type": "warning",
                    "title": "Post-Peak Solar Attenuation",
                    "text": f"Solar output will trend downward from current {curr_pv:.2f} kW as solar zenith angles drop, reaching minimum levels near sunset.",
                }
            )
    else:
        insights.append(
            {
                "category": "Solar PV Night Regime",
                "type": "neutral",
                "title": "No Solar Irradiance Expected",
                "text": "Solar irradiance will remain at 0 W/m² during nighttime hours. The plant will rely entirely on wind generation and grid storage reserves.",
            }
        )

    # 2. Wind Turbine Stability & Fluctuations
    if wind_volatility > 0.6:
        insights.append(
            {
                "category": "Wind Volatility",
                "type": "warning",
                "title": "High Wind Generation Variability",
                "text": f"Wind turbine generation exhibits high standard deviation (±{wind_volatility:.2f} kW) ranging between {min_wind_fc:.2f} kW and {max_wind_fc:.2f} kW.",
            }
        )
    elif avg_wind_fc > 2.0:
        insights.append(
            {
                "category": "Wind Baseload",
                "type": "positive",
                "title": "Sustained High Wind Generation",
                "text": f"Wind generation remains strong and steady, averaging {avg_wind_fc:.2f} kW (rated capacity factor ~{avg_wind_fc/WIND_RATED_CAPACITY_KW*100:.1f}%).",
            }
        )
    else:
        insights.append(
            {
                "category": "Wind Generation",
                "type": "neutral",
                "title": "Moderate Wind Resource",
                "text": f"Expected average wind output is {avg_wind_fc:.2f} kW with typical atmospheric thermal circulation.",
            }
        )

    # 3. Combined Renewable Dispatch & Storage Opportunity
    if max_total_fc > 5.5:
        insights.append(
            {
                "category": "Dispatch & Storage",
                "type": "positive",
                "title": "High Renewable Surplus Window",
                "text": f"Total generation peaks at {max_total_fc:.2f} kW at {max_total_time.strftime('%H:%M')}. Favorable conditions for battery storage charging or high-load scheduling.",
            }
        )
    elif avg_total_fc < 1.5:
        insights.append(
            {
                "category": "Grid Reliability",
                "type": "alert",
                "title": "Low Combined Generation Advisory",
                "text": f"Combined output across the next {duration_hours:.0f}h averages only {avg_total_fc:.2f} kW. Recommend dispatching energy storage or enabling grid backup.",
            }
        )

    # 4. Energy Yield Prediction
    insights.append(
        {
            "category": "Forecasted Yield",
            "type": "info",
            "title": f"{duration_hours:.0f}-Hour Energy Production Estimate",
            "text": f"Projected cumulative generation is {energy_kwh:.2f} kWh (Solar: {pv_energy_kwh:.2f} kWh, Wind: {wind_energy_kwh:.2f} kWh).",
        }
    )

    return insights
