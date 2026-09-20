"""
train_model.py
--------------
Model training pipeline for AI-Powered Renewable Energy Forecasting.

1. Loads and validates dataset.
2. Engineers time-series, cyclical, environmental, and autoregressive lag features.
3. Chronologically splits data (80% Train, 20% Test) to prevent look-ahead bias.
4. Trains two specialized machine learning models:
   - Model 1: Solar Photovoltaic (PV) Generation Regressor
   - Model 2: Wind Turbine Generation Regressor
5. Evaluates model performance on the unseen chronological test set (MAE, RMSE, R²).
6. Serializes trained models to `models/` via Joblib and saves metadata JSON.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_processing import (
    load_dataset,
    build_engineered_features,
    chronological_split,
    PV_FEATURE_COLS,
    WIND_FEATURE_COLS,
)


def train_models(data_path: str = "data/renewable_data.csv", models_dir: str = "models") -> Dict[str, Any]:
    """
    Execute complete end-to-end model training, evaluation, and serialization.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("AI RENEWABLE FORECASTING: MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load data
    print(f"\n[1/6] Loading telemetry from: {data_path}")
    raw_df = load_dataset(data_path)
    print(f"      Total records loaded: {len(raw_df)} rows")
    print(f"      Date range: {raw_df['timestamp'].min()} to {raw_df['timestamp'].max()}")

    # 2. Validate data
    print("\n[2/6] Validating dataset integrity...")
    required_cols = [
        "timestamp",
        "solar_irradiance",
        "ambient_temperature",
        "module_temperature",
        "humidity",
        "cloud_cover",
        "wind_speed",
        "wind_direction",
        "pv_power",
        "wind_power",
        "total_power",
    ]
    missing_cols = [c for c in required_cols if c not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Clean any accidental NaNs in raw data
    raw_df = raw_df.ffill().bfill()
    print("      Data validation passed. Zero unhandled nulls.")

    # 3. Feature engineering
    print("\n[3/6] Computing engineered features (lags, rolling stats, cyclical time)...")
    df_feat = build_engineered_features(raw_df)
    print(f"      Engineered dataset size after lag alignment: {len(df_feat)} rows")

    # 4. Chronological Train/Test Split
    print("\n[4/6] Performing Chronological Split (80% Train, 20% Test)...")
    print("      Note: Random shuffling is strictly prohibited in time-series forecasting")
    print("      to avoid look-ahead data leakage and preserve temporal autocorrelation.")
    train_df, test_df = chronological_split(df_feat, train_ratio=0.8)
    print(f"      Training set : {len(train_df)} observations ({train_df['timestamp'].min()} -> {train_df['timestamp'].max()})")
    print(f"      Testing set  : {len(test_df)} observations ({test_df['timestamp'].min()} -> {test_df['timestamp'].max()})")

    # 5. Train PV Power Model
    print("\n[5/6] Training Model 1: Solar PV Power Regressor...")
    X_train_pv = train_df[PV_FEATURE_COLS]
    y_train_pv = train_df["pv_power"]
    X_test_pv = test_df[PV_FEATURE_COLS]
    y_test_pv = test_df["pv_power"]

    # XGBoost: gradient-boosted trees, handles non-linearities well
    pv_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )
    pv_model.fit(X_train_pv, y_train_pv)

    # Predictions & metrics
    y_pred_pv = pv_model.predict(X_test_pv)
    # Clip predictions to non-negative physical limits
    y_pred_pv = np.clip(y_pred_pv, 0.0, None)

    pv_mae = mean_absolute_error(y_test_pv, y_pred_pv)
    pv_rmse = np.sqrt(mean_squared_error(y_test_pv, y_pred_pv))
    pv_r2 = r2_score(y_test_pv, y_pred_pv)

    print(f"      PV Model Evaluation:")
    print(f"        MAE  : {pv_mae:.4f} kW")
    print(f"        RMSE : {pv_rmse:.4f} kW")
    print(f"        R²   : {pv_r2:.4f} ({pv_r2*100:.2f}% variance explained)")

    # 6. Train Wind Power Model
    print("\n[6/6] Training Model 2: Wind Turbine Power Regressor...")
    X_train_wind = train_df[WIND_FEATURE_COLS]
    y_train_wind = train_df["wind_power"]
    X_test_wind = test_df[WIND_FEATURE_COLS]
    y_test_wind = test_df["wind_power"]

    wind_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
    )
    wind_model.fit(X_train_wind, y_train_wind)

    # Predictions & metrics
    y_pred_wind = wind_model.predict(X_test_wind)
    y_pred_wind = np.clip(y_pred_wind, 0.0, None)

    wind_mae = mean_absolute_error(y_test_wind, y_pred_wind)
    wind_rmse = np.sqrt(mean_squared_error(y_test_wind, y_pred_wind))
    wind_r2 = r2_score(y_test_wind, y_pred_wind)

    print(f"      Wind Model Evaluation:")
    print(f"        MAE  : {wind_mae:.4f} kW")
    print(f"        RMSE : {wind_rmse:.4f} kW")
    print(f"        R²   : {wind_r2:.4f} ({wind_r2*100:.2f}% variance explained)")

    # Combined Total Renewable Evaluation
    y_test_total = y_test_pv + y_test_wind
    y_pred_total = y_pred_pv + y_pred_wind
    tot_mae = mean_absolute_error(y_test_total, y_pred_total)
    tot_rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    tot_r2 = r2_score(y_test_total, y_pred_total)

    print(f"\n      Combined Renewable Power Evaluation:")
    print(f"        MAE  : {tot_mae:.4f} kW")
    print(f"        RMSE : {tot_rmse:.4f} kW")
    print(f"        R²   : {tot_r2:.4f}")

    # Feature Importance computation via permutation importance on test set
    print("\nComputing feature importances for engineering explainability...")
    perm_pv = permutation_importance(pv_model, X_test_pv, y_test_pv, n_repeats=5, random_state=42)
    pv_importance = {
        col: float(perm_pv.importances_mean[i])
        for i, col in enumerate(PV_FEATURE_COLS)
    }

    perm_wind = permutation_importance(wind_model, X_test_wind, y_test_wind, n_repeats=5, random_state=42)
    wind_importance = {
        col: float(perm_wind.importances_mean[i])
        for i, col in enumerate(WIND_FEATURE_COLS)
    }

    # Save trained models
    pv_model_path = models_path / "pv_model.pkl"
    wind_model_path = models_path / "wind_model.pkl"

    joblib.dump(pv_model, pv_model_path)
    joblib.dump(wind_model, wind_model_path)
    print(f"\nSuccessfully serialized models to:")
    print(f"  -> {pv_model_path.resolve()}")
    print(f"  -> {wind_model_path.resolve()}")

    # Save comprehensive metadata
    metadata = {
        "training_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "XGBoost (XGBRegressor)",
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_date_range": [str(train_df["timestamp"].min()), str(train_df["timestamp"].max())],
        "test_date_range": [str(test_df["timestamp"].min()), str(test_df["timestamp"].max())],
        "pv_features": PV_FEATURE_COLS,
        "wind_features": WIND_FEATURE_COLS,
        "metrics": {
            "pv": {"mae": round(float(pv_mae), 4), "rmse": round(float(pv_rmse), 4), "r2": round(float(pv_r2), 4)},
            "wind": {"mae": round(float(wind_mae), 4), "rmse": round(float(wind_rmse), 4), "r2": round(float(wind_r2), 4)},
            "total": {"mae": round(float(tot_mae), 4), "rmse": round(float(tot_rmse), 4), "r2": round(float(tot_r2), 4)},
        },
        "feature_importances": {
            "pv": dict(sorted(pv_importance.items(), key=lambda item: item[1], reverse=True)),
            "wind": dict(sorted(wind_importance.items(), key=lambda item: item[1], reverse=True)),
        }
    }

    metadata_path = models_path / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  -> {metadata_path.resolve()}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE AND VERIFIED!")
    print("=" * 70)

    return metadata


if __name__ == "__main__":
    train_models()
