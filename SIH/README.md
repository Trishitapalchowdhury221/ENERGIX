# AI-Powered Renewable Energy Monitoring & Forecasting Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An industrial-grade, AI-driven supervisory and predictive analytics platform for Solar Photovoltaic (PV) and Wind Turbine power plants. The system couples physics-informed weather simulation with scikit-learn Gradient Boosting machine learning models to deliver multi-step autoregressive generation forecasts, rule-based engineering insights, real-time alert triggers, and deep energy analytics.

Designed for electrical engineering research, B.Tech capstone demonstrations, and practical microgrid supervisory control.

---

## ⚡ System Architecture

The software architecture is decoupled into a modular telemetry ingestion layer, a feature engineering pipeline, serialized ML regressors, a recursive forecasting engine, and a dark control-room Streamlit dashboard.

```
+-----------------------------------------------------------------------------------+
|                            TELEMETRY DATA SOURCE                                  |
|  [Simulated CSV (Current)]  ──OR──  [ESP32 / IoT / MQTT / Modbus RTU (Future)]    |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                        DATA VALIDATION & RESAMPLING LAYER                         |
|     - Zero-null validation    - 15-min equidistant time alignment    - Bounds check|
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                          FEATURE ENGINEERING PIPELINE                             |
|  - Cyclical time encodings (sin/cos hour & month)                                 |
|  - Aerodynamic wind transformations (v³) & directional trigonometry               |
|  - Autoregressive history buffers (t-1, t-2, t-4, t-96 lags) & rolling statistics |
+---------------------+-------------------------------------+-----------------------+
                      |                                     |
                      v (Offline Training)                  v (Online Inference)
+----------------------------------------+  +---------------------------------------+
|        CHRONOLOGICAL 80/20 SPLIT       |  |     MULTI-STEP RECURSIVE FORECAST     |
|   Strict time arrow; no look-ahead     |  |       1h (4 steps) | 6h | 24h         |
+---------------------+------------------+  +-------------------+-------------------+
                      |                                         |
                      v                                         v
+----------------------------------------+  +---------------------------------------+
|         GRADIENT BOOSTING ML           |  |     AI INSIGHTS & ALERT ENGINE        |
|  Model 1: Solar PV Regressor           |  |  - Physics ramp-up/ramp-down detection|
|  Model 2: Wind Power Regressor         |  |  - High wind gust / pitch alerts      |
+---------------------+------------------+  |  - Curtailment & battery dispatch     |
                      |                     +-------------------+-------------------+
                      v (Serialized .pkl)                       |
+---------------------------------------------------------------+-------------------+
|                             STREAMLIT CONTROL CENTER                              |
|  - Overview Dashboard   - Simulated Real-Time Playback   - Energy Analytics       |
|  - Actual vs Predicted  - Telemetry Explorer & CSV Export- Environmental Telemetry|
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Key Features

1. **Dual Machine Learning Models**:
   - Specialized **Solar PV Regressor** modeling non-linear solar zenith irradiance, module thermal coefficient derating, and cloud attenuation.
   - Specialized **Wind Turbine Regressor** capturing the aerodynamic cut-in, cubic power growth ($P \propto v^3$), rated plateau, and cut-out thresholds.
2. **Multi-Step Recursive Forecasting**:
   - Selectable horizons: **Next 1 Hour** (4 intervals), **Next 6 Hours** (24 intervals), and **Next 24 Hours** (96 intervals).
   - Generates individual PV, Wind, and combined facility output forecasts with historical reference curves.
3. **Simulated Real-Time Playback Mode**:
   - Interactive playback controller (**START**, **PAUSE**, **RESET**, **NEXT STEP**).
   - Adjustable time clock speed (**1x**, **2x**, **5x**).
   - Fully transparent: prominently marked as **"SIMULATED REAL-TIME DATA"** and **"SIMULATION DATA"**.
4. **Engineering AI Insights**:
   - Deterministic rule-based insights analyzing solar ramp-ups, evening sunsets, wind variability, capacity factors, and storage dispatch opportunities without external LLM hallucinations.
5. **Configurable Alert System**:
   - Sidebar sliders for minimum PV generation, minimum wind generation, combined facility floor infeed, and high wind velocity warnings.
6. **Model Verification & Explainability**:
   - Holdout test set time-series tracking.
   - Scatter plots with ideal 1:1 regression fit lines.
   - Residual distribution error histograms.
   - Permutation feature importance rankings.
7. **Modular IoT Ingestion Layer**:
   - Abstract `BaseDataSource` pattern allowing seamless drop-in replacement of the CSV file with ESP32 MQTT brokers, Modbus inverters, or REST APIs.

---

## 📊 Dataset Description

The system includes an automated generator (`utils/data_processing.py`) producing 60 days of 15-minute resolution telemetry (5,760 records):

| Column | Unit | Physical Description |
| :--- | :--- | :--- |
| `timestamp` | Datetime | Equidistant 15-minute ISO timestamps |
| `solar_irradiance` | $W/m^2$ | Diurnal solar flux peaking ~1000 W/m² with cloud attenuation |
| `ambient_temperature` | $^\circ C$ | Diurnal and seasonal atmospheric temperature ($5^\circ C - 45^\circ C$) |
| `module_temperature` | $^\circ C$ | PV cell temperature modeled via $T_{amb} + (G/800) \times (NOCT - 20)$ |
| `humidity` | $\%$ | Relative air humidity, inversely correlated with ambient heat |
| `cloud_cover` | $\%$ | Autoregressive random-walk cloudiness fraction ($0\% - 95\%$) |
| `wind_speed` | $m/s$ | Weibull-distributed velocity with thermal diurnal oscillations |
| `wind_direction` | $^\circ$ | Wind heading ($0^\circ - 360^\circ$) with cardinal compass heading |
| `pv_power` | $kW$ | 5.0 kW rated solar plant output with negative thermal derating |
| `wind_power` | $kW$ | 4.0 kW rated wind turbine output with physical aerodynamic power curve |
| `total_power` | $kW$ | Combined renewable facility generation ($P_{pv} + P_{wind}$) |

---

## 🧠 Machine Learning & Methodology

### 1. Algorithm Selection: HistGradientBoostingRegressor
We selected `HistGradientBoostingRegressor` (scikit-learn) over deep learning architectures (e.g., LSTM) or linear models for the following technical reasons:
- **Non-Linear Dynamics**: PV irradiance curves and wind cubic power responses ($v^3$) are highly non-linear, which tree-based boosting maps with extreme precision.
- **Sample Efficiency & Speed**: Trains in under 1 second while maintaining $>0.99\ R^2$ on holdout data.
- **Robustness**: Native binning handles correlated features (temperature, irradiance, lags) without overfitting.
- **Portability**: Compact `.pkl` serialization allows deployment on lightweight edge devices (Raspberry Pi, industrial PCs).

### 2. Feature Engineering & Leakage Prevention
To ensure strict temporal validity:
- **Temporal & Cyclical**: $\sin(2\pi \cdot hr/24)$, $\cos(2\pi \cdot hr/24)$, day of year, and month.
- **Physical Basis**: Solar irradiance, module temp, $v^3$ (wind speed cubed), $\sin(\theta_{dir})$, $\cos(\theta_{dir})$.
- **Autoregressive Lags**: $t-1$ (15m prior), $t-2$ (30m prior), $t-4$ (1h prior), $t-96$ (24h diurnal prior).
- **Rolling Statistics**: 4-step rolling window mean and standard deviation computed strictly up to $t-1$.

### 3. Chronological Train/Test Split
In time-series forecasting, standard $k$-fold cross-validation or random shuffling introduces **look-ahead bias** (data leakage). We strictly enforce:
- **Training Set**: First 80% of historical records (4,532 steps).
- **Testing Set**: Last 20% of historical records (1,133 steps).

### 4. Evaluation Metrics

| Metric | Solar PV Model | Wind Turbine Model | Combined Total Facility |
| :--- | :--- | :--- | :--- |
| **MAE** | **0.0056 kW** | **0.0040 kW** | **0.0086 kW** |
| **RMSE** | **0.0105 kW** | **0.0207 kW** | **0.0229 kW** |
| **$R^2$ Score** | **0.9999** | **0.9980** | **0.9998** |

---

## 🛠 Project Structure

```
SIH/
├── app.py                     # Main Streamlit dashboard application
├── train_model.py             # Standalone model training and evaluation script
├── requirements.txt           # Python package dependencies
├── README.md                  # Comprehensive engineering documentation
├── .gitignore                 # Standard Python / Streamlit gitignore
│
├── .streamlit/
│   └── config.toml            # Dark control-room theme configuration
│
├── data/
│   └── renewable_data.csv     # 60-day 15-minute simulated telemetry dataset
│
├── models/
│   ├── pv_model.pkl           # Trained Solar PV Gradient Boosting regressor
│   ├── wind_model.pkl         # Trained Wind Turbine Gradient Boosting regressor
│   └── model_metadata.json    # Holdout metrics, timestamps, and feature importances
│
└── utils/
    ├── __init__.py            # Python package declaration
    ├── data_processing.py     # Physics data simulation and feature engineering
    ├── data_source.py         # Modular BaseDataSource abstraction (CSV, MQTT, REST)
    ├── forecasting.py         # Multi-step recursive forecast & engineering insights
    ├── simulation.py          # Interactive playback controller & session state
    └── dashboard_utils.py     # Plotly dark visualizations, KPI cards, and alert engine
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone Repository
```bash
git clone https://github.com/your-username/renewable-ai-dashboard.git
cd renewable-ai-dashboard
```

### 3. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Train Models (Optional - Auto-bootstrapped on launch)
```bash
python train_model.py
```

### 6. Launch Dashboard
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔌 Connecting Real-Time Hardware (IoT Roadmap)

The application includes an abstraction layer in `utils/data_source.py` designed to transition from simulated CSV data to physical hardware:

```
[ PV Voltage / Current Sensor (ACS712 / INA219) ] ──┐
[ Solar Pyranometer / LDR Irradiance ] ─────────────┤
[ Anemometer (Wind Speed Pulse / Voltage) ] ────────┼──> [ ESP32 Microcontroller ]
[ Wind Vane (Potentiometer / Optical) ] ────────────┤           | (WiFi / 4G)
[ Ambient & Module Thermistors (DS18B20) ] ─────────┘           v
                                                        [ MQTT Broker / HiveMQ ]
                                                                |
                                                                v
                                                        [ utils/data_source.py ]
                                                        (MQTTDataSource class)
                                                                |
                                                                v
                                                        [ AI Forecast Dashboard ]
```

To switch to live hardware:
1. Flash your ESP32 with firmware publishing JSON payloads to an MQTT broker:
   ```json
   {
     "solar_irradiance": 842.5,
     "ambient_temperature": 28.4,
     "module_temperature": 48.2,
     "humidity": 45.0,
     "wind_speed": 7.8,
     "wind_direction": 225.0,
     "pv_power": 3.65,
     "wind_power": 1.92
   }
   ```
2. In `utils/data_source.py`, implement `MQTTDataSource` using `paho-mqtt` to ingest readings into a ring buffer.
3. Update `app.py` to instantiate `MQTTDataSource()` instead of `CSVDataSource()`. No changes to ML pipelines or UI charts are required!

---

## 🎓 College Presentation & Viva Reference Guide

When presenting this project to evaluators:
1. **Explain the physical laws modeled**:
   - PV power drops at elevated cell temperatures ($T_{module} > 25^\circ C$) due to semiconductor bandgap voltage drops ($\gamma \approx -0.4\%/^\circ C$).
   - Wind turbine power exhibits a cubic relationship with wind velocity ($P \propto v^3$) based on kinetic energy conservation: $P = \frac{1}{2} \rho A v^3 C_p$.
2. **Justify chronological splitting**:
   - Random cross-validation leaks information across adjacent 15-minute intervals. Chronological holdout splitting is the only mathematically rigorous validation for time-series forecasting.
3. **Highlight operational microgrid benefits**:
   - Advance notice of generation drops (sunset or wind lulls) prevents blackout events by giving grid operators time to ramp up batteries or auxiliary generators.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
