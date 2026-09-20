"""
data_processing.py
------------------
Data generation, validation, cleaning, and feature engineering for
AI-Powered Renewable Energy Monitoring & Forecasting Dashboard.

Follows physics-informed models for Solar Photovoltaic (PV) and Wind turbine generation.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Physical constants and plant ratings (1 kW rated matching real datasets)
PV_RATED_CAPACITY_KW = 1.0  # 1 kW peak solar array (from NREL PVWatts 1 kW DC specification)
WIND_RATED_CAPACITY_KW = 1.0  # 1 kW small-scale wind turbine (from Renewables.ninja 1 kW capacity)
PV_TEMP_COEFF = 0.004  # -0.4% efficiency per deg C above 25 deg C (STC)
INVERTER_EFFICIENCY = 0.96

WIND_CUT_IN = 3.0  # m/s
WIND_RATED = 12.0  # m/s
WIND_CUT_OUT = 25.0  # m/s


def generate_synthetic_renewable_data(
    days: int = 60,
    interval_minutes: int = 15,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate realistic 15-minute resolution synthetic PV and wind generation telemetry.

    Features:
    - Diurnal solar irradiance curve with morning ramp-up, midday peak, evening ramp-down,
      and zero generation at night.
    - Cloud cover events attenuating solar radiation.
    - PV module heating model and negative thermal efficiency derating.
    - Weibull-distributed wind speed with diurnal thermal breeze patterns.
    - Non-linear aerodynamic wind turbine power curve (cut-in, cubic rated ramp, plateau, cut-out).
    """
    np.random.seed(seed)

    # Base timestamp starting 60 days prior to current date
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(
        start=start_time, end=end_time, freq=f"{interval_minutes}min"
    )
    n_points = len(timestamps)

    # 1. Temporal variables as pure numpy arrays
    hours = np.array(timestamps.hour + timestamps.minute / 60.0, dtype=np.float64)
    day_of_year = np.array(timestamps.dayofyear, dtype=np.float64)

    # 2. Weather & Atmospheric Conditions
    # Cloud cover: Autoregressive smoothed random walk bounded between 0% and 95%
    cloud_raw = np.zeros(n_points, dtype=np.float64)
    current_cloud = 20.0
    for i in range(n_points):
        step = np.random.normal(0, 4.0)
        current_cloud = np.clip(current_cloud * 0.96 + step + 1.2, 0.0, 95.0)
        cloud_raw[i] = current_cloud
    cloud_cover = np.round(cloud_raw, 1)

    # Ambient Temperature: Diurnal cycle peaking around 14:00-15:00, seasonal trend, noise
    seasonal_temp = 22.0 + 6.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    diurnal_temp = 7.0 * np.sin(2 * np.pi * (hours - 9) / 24.0)
    ambient_temp = (
        seasonal_temp
        + diurnal_temp
        - (cloud_cover * 0.04)
        + np.random.normal(0, 0.8, n_points)
    )
    ambient_temp = np.round(np.clip(ambient_temp, 5.0, 45.0), 2)

    # Relative Humidity: Inversely correlated with temperature
    humidity_base = 65.0 - (diurnal_temp * 2.5) + (cloud_cover * 0.25)
    humidity = np.round(
        np.clip(humidity_base + np.random.normal(0, 3.0, n_points), 15.0, 98.0),
        1,
    )

    # 3. Solar Irradiance (W/m^2)
    # Sun elevation angle proxy: peak at 12:30 PM (solar noon approx)
    # Sunrise ~ 06:00, Sunset ~ 18:30
    solar_noon = 12.5
    daylight_half_width = 6.25  # hours from sunrise to noon
    time_diff = np.abs(hours - solar_noon)
    is_daylight = (time_diff < daylight_half_width)

    clear_sky_irradiance = np.zeros(n_points, dtype=np.float64)
    # Cosine bell curve for daylight hours
    clear_sky_irradiance[is_daylight] = 1000.0 * np.cos(
        (time_diff[is_daylight] / daylight_half_width) * (np.pi / 2.0)
    )
    # Apply cloud attenuation: Beer-Lambert proxy
    cloud_attenuation = 1.0 - (cloud_cover / 100.0) * 0.78
    solar_irradiance = clear_sky_irradiance * cloud_attenuation

    # Add realistic atmospheric micro-variations during daylight
    daylight_noise = np.random.normal(1.0, 0.035, n_points)
    solar_irradiance[is_daylight] *= daylight_noise[is_daylight]
    solar_irradiance = np.array(np.round(np.clip(solar_irradiance, 0.0, 1150.0), 1), dtype=np.float64)
    solar_irradiance[~is_daylight] = 0.0

    # 4. PV Module Temperature (deg C)
    # Module warms significantly under irradiance: T_module = T_ambient + (G / 800) * (NOCT - 20)
    # Nominal Operating Cell Temperature (NOCT) ~ 45 deg C
    noct = 45.0
    module_temp = ambient_temp + (solar_irradiance / 800.0) * (
        noct - 20.0
    ) + np.random.normal(0, 0.5, n_points)
    module_temp = np.array(np.round(np.clip(module_temp, ambient_temp, 75.0), 2), dtype=np.float64)

    # 5. PV Power Generation (kW)
    # Thermal derating: efficiency decreases if module_temp > 25 deg C
    thermal_derating = 1.0 - PV_TEMP_COEFF * np.maximum(0.0, module_temp - 25.0)
    pv_power_raw = (
        PV_RATED_CAPACITY_KW
        * (solar_irradiance / 1000.0)
        * thermal_derating
        * INVERTER_EFFICIENCY
    )
    pv_power = np.array(np.round(np.clip(pv_power_raw, 0.0, PV_RATED_CAPACITY_KW * 1.05), 3), dtype=np.float64)
    pv_power[solar_irradiance < 5.0] = 0.0

    # 6. Wind Speed & Direction
    # Weibull-distributed baseline wind + afternoon thermal gustiness
    weibull_wind = np.random.weibull(2.1, n_points) * 4.8
    thermal_wind = 1.8 * np.sin(2 * np.pi * (hours - 12) / 24.0)
    wind_speed = np.array(
        np.round(
            np.clip(weibull_wind + thermal_wind + np.random.normal(0, 0.5, n_points), 0.2, 26.0),
            2,
        ),
        dtype=np.float64,
    )

    # Wind direction: slow drift with random fluctuations (0 - 360 deg)
    wind_dir_raw = np.zeros(n_points)
    curr_dir = 210.0  # South-Southwest prevailing
    for i in range(n_points):
        curr_dir = (curr_dir + np.random.normal(0, 3.5)) % 360.0
        wind_dir_raw[i] = curr_dir
    wind_direction = np.round(wind_dir_raw, 1)

    # 7. Wind Power Generation (kW)
    # Realistic wind turbine power curve
    wind_power = np.zeros(n_points)
    for i in range(n_points):
        v = wind_speed[i]
        if v < WIND_CUT_IN or v >= WIND_CUT_OUT:
            wind_power[i] = 0.0
        elif WIND_CUT_IN <= v < WIND_RATED:
            # Cubic power ramp between cut-in and rated speed
            fraction = (v - WIND_CUT_IN) / (WIND_RATED - WIND_CUT_IN)
            wind_power[i] = WIND_RATED_CAPACITY_KW * (fraction**3)
        else:  # WIND_RATED <= v < WIND_CUT_OUT
            # Rated power with slight aerodynamic flutter
            wind_power[i] = WIND_RATED_CAPACITY_KW * (1.0 + np.random.normal(0, 0.015))

    # Add small mechanical turbulence jitter
    wind_power = np.round(np.clip(wind_power, 0.0, WIND_RATED_CAPACITY_KW * 1.05), 3)

    # 8. Total Renewable Power (kW)
    total_power = np.round(pv_power + wind_power, 3)

    df = pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "solar_irradiance": solar_irradiance,
            "ambient_temperature": ambient_temp,
            "module_temperature": module_temp,
            "humidity": humidity,
            "cloud_cover": cloud_cover,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "pv_power": pv_power,
            "wind_power": wind_power,
            "total_power": total_power,
        }
    )

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_file, index=False)
        print(f"Generated {len(df)} telemetry rows saved to {out_file.resolve()}")

    return df


# Known column aliases for plug-and-play third-party datasets
COLUMN_ALIASES = {
    "timestamp": ["time", "datetime", "date_time", "date", "Timestamp", "DateTime", "Date", "Time"],
    "solar_irradiance": [
        "irradiance", "ghi", "GHI", "poa", "solar_rad", "radiation", "SolarIrradiance",
        "Solar_Radiation", "Solar", "Plane of Array Irradiance (W/m2)", "Beam Irradiance (W/m2)"
    ],
    "ambient_temperature": [
        "ambient_temp", "temp", "temperature", "Temperature", "AirTemp", "Ambient_Temp",
        "T_amb", "Ambient Temperature (C)"
    ],
    "module_temperature": [
        "module_temp", "cell_temp", "panel_temp", "ModuleTemp", "Cell_Temp", "Module_Temperature",
        "T_mod", "Cell Temperature (C)"
    ],
    "humidity": ["rh", "relative_humidity", "Humidity", "RH", "Relative_Humidity"],
    "cloud_cover": ["clouds", "cloudiness", "CloudCover", "Cloud_Cover", "cloud"],
    "wind_speed": ["speed", "wind_velocity", "WindSpeed", "Wind_Speed", "velocity", "WS", "Wind Speed (m/s)"],
    "wind_direction": ["direction", "wind_dir", "WindDirection", "Wind_Direction", "heading", "WD"],
    "pv_power": [
        "pv", "solar_power", "solar_generation", "PVPower", "PV_Power", "pv_generation",
        "solar_output", "P_pv", "AC System Output (W)", "DC Array Output (W)"
    ],
    "wind_power": ["wind", "wind_generation", "WindPower", "Wind_Power", "wind_output", "P_wind", "electricity"],
    "total_power": ["total", "total_generation", "TotalPower", "Total_Power", "combined_power", "P_total"],
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize user CSV column names to match system schema using known aliases.
    Automatically synthesizes missing secondary variables if possible.
    """
    df_clean = df.copy()

    # 1. Map column aliases
    existing_cols = {str(c).strip(): c for c in df_clean.columns}
    lower_map = {str(c).strip().lower(): c for c in df_clean.columns}

    rename_dict = {}
    for target_name, aliases in COLUMN_ALIASES.items():
        if target_name in existing_cols:
            continue  # Already exact match
        for alias in [target_name] + aliases:
            if alias in existing_cols:
                rename_dict[existing_cols[alias]] = target_name
                break
            elif alias.lower() in lower_map:
                rename_dict[lower_map[alias.lower()]] = target_name
                break

    if rename_dict:
        df_clean = df_clean.rename(columns=rename_dict)

    # 1b. Convert PV power from Watts to kW if needed
    if "pv_power" in df_clean.columns and df_clean["pv_power"].max() > 50.0:
        df_clean["pv_power"] = df_clean["pv_power"] / 1000.0

    # 2. Synthesize total_power if missing but pv_power and wind_power exist
    if "total_power" not in df_clean.columns:
        if "pv_power" in df_clean.columns and "wind_power" in df_clean.columns:
            df_clean["total_power"] = df_clean["pv_power"] + df_clean["wind_power"]
        elif "pv_power" in df_clean.columns:
            df_clean["total_power"] = df_clean["pv_power"]
            df_clean["wind_power"] = 0.0
        elif "wind_power" in df_clean.columns:
            df_clean["total_power"] = df_clean["wind_power"]
            df_clean["pv_power"] = 0.0

    # 3. Synthesize module_temperature if missing from ambient_temp & irradiance
    if "module_temperature" not in df_clean.columns:
        amb = df_clean.get("ambient_temperature", pd.Series(25.0, index=df_clean.index))
        irr = df_clean.get("solar_irradiance", pd.Series(0.0, index=df_clean.index))
        df_clean["module_temperature"] = amb + (irr / 800.0) * 25.0
    else:
        # At night, PVWatts cell temp is 0.0; fallback to ambient temperature
        if "ambient_temperature" in df_clean.columns:
            df_clean["module_temperature"] = np.where(
                df_clean["module_temperature"] > 0,
                df_clean["module_temperature"],
                df_clean["ambient_temperature"],
            )

    # 4. Fill defaults for other auxiliary environmental variables if omitted
    if "ambient_temperature" not in df_clean.columns:
        df_clean["ambient_temperature"] = 25.0
    if "humidity" not in df_clean.columns:
        df_clean["humidity"] = 55.0
    if "cloud_cover" not in df_clean.columns:
        df_clean["cloud_cover"] = 20.0
    if "wind_direction" not in df_clean.columns:
        df_clean["wind_direction"] = 180.0
    if "wind_speed" not in df_clean.columns:
        df_clean["wind_speed"] = 5.0
    if "solar_irradiance" not in df_clean.columns:
        df_clean["solar_irradiance"] = 0.0

    return df_clean


def find_user_datasets(data_dir: str = "data") -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate the Solar (PVWatts) and Wind (Renewables.ninja) CSV files in data_dir.
    Handles flexible file names (e.g., 'pvwatts_hourly (1).csv', 'pvwatts_hourly.csv', etc.).
    """
    d_path = Path(data_dir)
    if not d_path.exists():
        return None, None

    solar_path = None
    wind_path = None

    for f in d_path.glob("*.csv"):
        name_lower = f.name.lower()
        if "renewable_data" in name_lower:
            continue

        if "pvwatts" in name_lower or "solar" in name_lower:
            solar_path = f
        elif "ninja_wind" in name_lower or "wind" in name_lower:
            wind_path = f
        else:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                    first_line = fp.readline()
                if "pvwatts" in first_line.lower():
                    solar_path = f
                elif "renewables.ninja" in first_line.lower():
                    wind_path = f
            except Exception:
                pass

    return solar_path, wind_path


def merge_user_datasets(
    data_dir: str = "data", output_path: Optional[str] = "data/renewable_data.csv"
) -> pd.DataFrame:
    """
    Parse and merge user's two raw datasets:
    1. PVWatts hourly solar performance CSV (8,760 hours)
    2. Renewables.ninja hourly wind power CSV (8,760 hours)

    Synchronizes timestamps, harmonizes units (both in kW), extracts atmospheric
    measurements, and outputs a complete unified telemetry dataset.
    """
    solar_file, wind_file = find_user_datasets(data_dir)
    if not solar_file or not wind_file:
        raise FileNotFoundError(
            f"Could not find both solar and wind CSV files in '{data_dir}'. Found: solar={solar_file}, wind={wind_file}"
        )

    print(f"Ingesting solar dataset: {solar_file.name}")
    print(f"Ingesting wind dataset : {wind_file.name}")

    # 1. Parse Renewables.ninja Wind CSV (skip metadata lines 1-3)
    df_wind = pd.read_csv(wind_file, skiprows=3)
    df_wind = df_wind.loc[:, ~df_wind.columns.str.contains("^Unnamed")]
    df_wind["electricity"] = pd.to_numeric(df_wind["electricity"], errors="coerce").fillna(0.0)

    # 2. Parse PVWatts Solar CSV (detect header line containing Month, Day, Hour)
    header_line = 31
    with open(solar_file, "r", encoding="utf-8", errors="ignore") as fp:
        for idx, line in enumerate(fp):
            if "Month" in line and "Day" in line and "Hour" in line:
                header_line = idx
                break
    df_pv = pd.read_csv(solar_file, skiprows=header_line)

    # 3. Align timestamps (2019-01-01 00:00:00 to 2019-12-31 23:00:00)
    if "time" in df_wind.columns:
        try:
            timestamps = pd.to_datetime(df_wind["time"], format="%d-%m-%Y %H:%M")
        except Exception:
            timestamps = pd.to_datetime(df_wind["time"])
    else:
        timestamps = pd.date_range(start="2019-01-01 00:00:00", periods=len(df_wind), freq="1h")

    # Ensure length consistency across both files
    min_len = min(len(df_pv), len(df_wind), len(timestamps))
    df_pv = df_pv.iloc[:min_len].reset_index(drop=True)
    df_wind = df_wind.iloc[:min_len].reset_index(drop=True)
    timestamps = timestamps.iloc[:min_len]

    # 4. Standardize Power outputs to kW (both normalized to 1 kW capacity)
    pv_power_kw = np.round(df_pv["AC System Output (W)"] / 1000.0, 4)
    wind_power_kw = np.round(df_wind["electricity"], 4)
    total_power_kw = np.round(pv_power_kw + wind_power_kw, 4)

    # 5. Extract atmospheric measurements from PVWatts
    solar_irr = df_pv.get("Plane of Array Irradiance (W/m2)", df_pv.get("Beam Irradiance (W/m2)", pd.Series(0.0, index=df_pv.index)))
    beam_irr = df_pv.get("Beam Irradiance (W/m2)", pd.Series(0.0, index=df_pv.index))
    diffuse_irr = df_pv.get("Diffuse Irradiance (W/m2)", pd.Series(0.0, index=df_pv.index))
    ambient_t = df_pv.get("Ambient Temperature (C)", pd.Series(25.0, index=df_pv.index))
    cell_t = df_pv.get("Cell Temperature (C)", ambient_t)
    mod_temp = np.where(cell_t > 0, cell_t, ambient_t)
    wind_spd = df_pv.get("Wind Speed (m/s)", pd.Series(2.0, index=df_pv.index))

    # Derive cloud cover from diffuse vs total irradiance
    tot_irr = beam_irr + diffuse_irr
    diffuse_ratio = np.where(tot_irr > 10, diffuse_irr / tot_irr, 0.2)
    cloud_cov = np.round(np.clip(diffuse_ratio * 100.0, 0.0, 100.0), 1)

    # Derive humidity: inverse correlation with temperature + Indian monsoon seasonal boost
    monsoon_boost = np.where(timestamps.dt.month.isin([6, 7, 8, 9]), 25.0, 0.0)
    hum = np.round(np.clip(70.0 - 0.8 * ambient_t + monsoon_boost, 15.0, 95.0), 1)

    # Derive wind direction with Indian seasonal prevailing winds (SW monsoon summer ~225 deg, NE winter ~45 deg)
    np.random.seed(42)
    day_of_yr = timestamps.dt.dayofyear
    base_wind_dir = 135.0 + 90.0 * np.sin(2 * np.pi * (day_of_yr - 60) / 365.25)
    wind_dir = np.round((base_wind_dir + np.random.normal(0, 10, len(timestamps))) % 360.0, 1)

    df_merged = pd.DataFrame(
        {
            "timestamp": timestamps.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "solar_irradiance": np.round(solar_irr, 2),
            "ambient_temperature": np.round(ambient_t, 2),
            "module_temperature": np.round(mod_temp, 2),
            "humidity": hum,
            "cloud_cover": cloud_cov,
            "wind_speed": np.round(wind_spd, 2),
            "wind_direction": wind_dir,
            "pv_power": pv_power_kw,
            "wind_power": wind_power_kw,
            "total_power": total_power_kw,
        }
    )

    if output_path:
        out_f = Path(output_path)
        out_f.parent.mkdir(parents=True, exist_ok=True)
        df_merged.to_csv(out_f, index=False)
        print(f"Successfully compiled {len(df_merged)} real-world telemetry rows to: {out_f.resolve()}")

    return df_merged


def load_dataset(file_path: str = "data/renewable_data.csv") -> pd.DataFrame:
    """
    Load dataset from CSV.
    If the unified dataset file is missing or contains fewer than 8,760 rows,
    automatically checks for user's uploaded raw CSV files in data/ and merges them.
    If raw CSVs are absent, falls back to generating synthetic telemetry.
    """
    csv_path = Path(file_path)
    solar_file, wind_file = find_user_datasets(csv_path.parent if csv_path.parent.exists() else "data")

    # If renewable_data.csv is missing or outdated (< 8760 records) and user datasets exist, merge them
    if solar_file and wind_file:
        needs_build = not csv_path.exists()
        if not needs_build and csv_path.exists():
            try:
                row_count = sum(1 for _ in open(csv_path)) - 1
                if row_count < 8760:
                    needs_build = True
                elif max(solar_file.stat().st_mtime, wind_file.stat().st_mtime) > csv_path.stat().st_mtime:
                    needs_build = True
            except Exception:
                needs_build = True

        if needs_build:
            print("Compiling unified dataset from user's uploaded solar & wind CSV files...")
            df = merge_user_datasets(data_dir=str(csv_path.parent), output_path=str(csv_path))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)

    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Dataset '{csv_path}' not found. Generating realistic synthetic dataset...")
        df = generate_synthetic_renewable_data(days=60, output_path=str(csv_path))
    else:
        df = pd.read_csv(csv_path)

    # Standardize column headers and auto-repair missing auxiliary channels
    df = standardize_columns(df)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    else:
        # Fallback: create timestamps if none provided
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="1h")

    return df


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-series, cyclical, environmental, lag, and rolling features.
    Adapts dynamically to data resolution (hourly 60m vs 15m intervals).
    Strictly prevents look-ahead leakage: all lags and rolling windows computed on t-1 and earlier.
    """
    df_feat = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_feat["timestamp"]):
        df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"])

    # Detect data step interval
    if len(df_feat) > 1:
        dt_secs = (df_feat["timestamp"].iloc[1] - df_feat["timestamp"].iloc[0]).total_seconds()
        interval_minutes = max(1, int(round(dt_secs / 60.0)))
    else:
        interval_minutes = 60

    # 24-hour diurnal lag step count (24 for 1h data, 96 for 15m data)
    diurnal_lag_steps = int(round(24 * 60 / interval_minutes))

    # Temporal & Cyclical features
    hours = df_feat["timestamp"].dt.hour + df_feat["timestamp"].dt.minute / 60.0
    day_of_year = df_feat["timestamp"].dt.dayofyear
    month = df_feat["timestamp"].dt.month

    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat["minute"] = df_feat["timestamp"].dt.minute
    df_feat["day_of_year"] = day_of_year
    df_feat["month"] = month

    df_feat["sin_hour"] = np.sin(2 * np.pi * hours / 24.0)
    df_feat["cos_hour"] = np.cos(2 * np.pi * hours / 24.0)
    df_feat["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df_feat["cos_month"] = np.cos(2 * np.pi * month / 12.0)

    # Wind physical transformations
    df_feat["wind_speed_cubed"] = df_feat["wind_speed"] ** 3
    wind_rad = np.radians(df_feat["wind_direction"])
    df_feat["sin_wind_dir"] = np.sin(wind_rad)
    df_feat["cos_wind_dir"] = np.cos(wind_rad)

    # PV Autoregressive Lags (t-1, t-2, t-4, and diurnal 24h lag)
    df_feat["pv_lag_1"] = df_feat["pv_power"].shift(1)
    df_feat["pv_lag_2"] = df_feat["pv_power"].shift(2)
    df_feat["pv_lag_4"] = df_feat["pv_power"].shift(min(4, len(df_feat) - 1))
    df_feat["pv_lag_24"] = df_feat["pv_power"].shift(diurnal_lag_steps)
    # Maintain pv_lag_96 as backwards-compatible alias for 24h diurnal lag
    df_feat["pv_lag_96"] = df_feat["pv_lag_24"]
    df_feat["pv_rolling_mean_4"] = df_feat["pv_power"].shift(1).rolling(4, min_periods=1).mean()
    df_feat["pv_rolling_std_4"] = df_feat["pv_power"].shift(1).rolling(4, min_periods=1).std().fillna(0)

    # Wind Autoregressive Lags (t-1, t-2, t-4, and diurnal 24h lag)
    df_feat["wind_lag_1"] = df_feat["wind_power"].shift(1)
    df_feat["wind_lag_2"] = df_feat["wind_power"].shift(2)
    df_feat["wind_lag_4"] = df_feat["wind_power"].shift(min(4, len(df_feat) - 1))
    df_feat["wind_lag_24"] = df_feat["wind_power"].shift(diurnal_lag_steps)
    # Maintain wind_lag_96 as backwards-compatible alias for 24h diurnal lag
    df_feat["wind_lag_96"] = df_feat["wind_lag_24"]
    df_feat["wind_rolling_mean_4"] = df_feat["wind_power"].shift(1).rolling(4, min_periods=1).mean()
    df_feat["wind_rolling_std_4"] = df_feat["wind_power"].shift(1).rolling(4, min_periods=1).std().fillna(0)

    # Drop the initial rows with NaN due to 24h lag
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat


# Column specifications for models
PV_FEATURE_COLS = [
    "hour",
    "minute",
    "day_of_year",
    "month",
    "sin_hour",
    "cos_hour",
    "solar_irradiance",
    "ambient_temperature",
    "module_temperature",
    "humidity",
    "cloud_cover",
    "pv_lag_1",
    "pv_lag_2",
    "pv_lag_4",
    "pv_lag_96",
    "pv_rolling_mean_4",
    "pv_rolling_std_4",
]

WIND_FEATURE_COLS = [
    "hour",
    "minute",
    "day_of_year",
    "month",
    "wind_speed",
    "wind_speed_cubed",
    "sin_wind_dir",
    "cos_wind_dir",
    "ambient_temperature",
    "humidity",
    "wind_lag_1",
    "wind_lag_2",
    "wind_lag_4",
    "wind_lag_96",
    "wind_rolling_mean_4",
    "wind_rolling_std_4",
]


def chronological_split(
    df_features: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time-series data chronologically (First 80% train, last 20% test).
    Never random shuffle to preserve sequential temporal structure.
    """
    split_idx = int(len(df_features) * train_ratio)
    train_df = df_features.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df_features.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df
