"""
dashboard_utils.py
-------------------
Visualization and UI styling utilities for AI Renewable Energy Control Center.
Includes Plotly chart generators, KPI card renderers, alert system, and telemetry gauges.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color Palette Constants
COLOR_PV = "#FFB300"  # Amber Sun
COLOR_WIND = "#00B4D8"  # Cyan Wind
COLOR_TOTAL = "#00D26A"  # Emerald Renewable Total
COLOR_BG_CARD = "#161B22"
COLOR_BG_CANVAS = "#0E1117"
COLOR_BORDER = "#30363D"
COLOR_TEXT = "#E6EDF3"
COLOR_MUTED = "#8B949E"


def apply_custom_css():
    """
    Inject professional dark control-center CSS styles.
    """
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
            
            html, body, [class*="css"] {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            }}
            
            /* Metric / KPI Card Styling */
            .kpi-container {{
                background-color: {COLOR_BG_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 16px 20px;
                margin-bottom: 14px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
                transition: transform 0.15s ease, border-color 0.15s ease;
            }}
            .kpi-container:hover {{
                border-color: #58A6FF;
            }}
            .kpi-title {{
                font-size: 0.75rem;
                font-weight: 700;
                color: {COLOR_MUTED};
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 4px;
            }}
            .kpi-value {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.85rem;
                font-weight: 700;
                color: {COLOR_TEXT};
                line-height: 1.2;
            }}
            .kpi-unit {{
                font-size: 0.9rem;
                font-weight: 500;
                color: {COLOR_MUTED};
                margin-left: 4px;
            }}
            .kpi-subtext {{
                font-size: 0.75rem;
                color: {COLOR_MUTED};
                margin-top: 6px;
            }}
            
            /* Status badges */
            .status-badge-online {{
                display: inline-flex;
                align-items: center;
                background: rgba(0, 210, 106, 0.12);
                border: 1px solid rgba(0, 210, 106, 0.4);
                color: #00D26A;
                font-size: 0.78rem;
                font-weight: 700;
                padding: 4px 12px;
                border-radius: 20px;
                letter-spacing: 0.05em;
            }}
            .pulse-dot {{
                width: 8px;
                height: 8px;
                background-color: #00D26A;
                border-radius: 50%;
                margin-right: 8px;
                box-shadow: 0 0 8px #00D26A;
            }}
            
            /* Alert boxes */
            .alert-banner {{
                padding: 10px 16px;
                border-radius: 6px;
                margin-bottom: 12px;
                font-size: 0.84rem;
                font-weight: 500;
                display: flex;
                align-items: center;
            }}
            .alert-danger {{
                background: rgba(248, 81, 73, 0.15);
                border-left: 4px solid #F85149;
                color: #FF7B72;
            }}
            .alert-warning {{
                background: rgba(210, 153, 34, 0.15);
                border-left: 4px solid #D29922;
                color: #E3B341;
            }}
            .alert-success {{
                background: rgba(46, 160, 67, 0.15);
                border-left: 4px solid #2EA043;
                color: #56D364;
            }}
            .alert-info {{
                background: rgba(56, 139, 253, 0.15);
                border-left: 4px solid #388BFD;
                color: #79C0FF;
            }}
            
            /* Clean Streamlit padding adjustment */
            .block-container {{
                padding-top: 1.8rem;
                padding-bottom: 2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_header():
    """
    Renders top header section according to Section 5:
    RENEWABLE AI ENERGY DASHBOARD
    AI-Based PV & Wind Power Monitoring and Forecasting
    ● SYSTEM ONLINE / Simulation Data
    """
    col1, col2 = st.columns([3.2, 1.2])

    with col1:
        st.markdown(
            """
            <div style="margin-bottom: 18px;">
                <h1 style="font-size: 1.95rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 4px; color: #FFFFFF;">
                    RENEWABLE AI ENERGY DASHBOARD
                </h1>
                <div style="font-size: 0.95rem; color: #8B949E; font-weight: 400;">
                    AI-Based PV & Wind Power Monitoring and Forecasting
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 6px;">
                <div class="status-badge-online">
                    <span class="pulse-dot"></span>SYSTEM ONLINE
                </div>
                <div style="font-size: 0.75rem; color: #8B949E; margin-top: 6px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                    Simulation Data
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def degrees_to_cardinal(d: float) -> str:
    """Convert wind direction in degrees to 8-point cardinal string."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    ix = int((d + 22.5) / 45.0) % 8
    return dirs[ix]


def render_kpi_cards(reading: Dict[str, Any]):
    """
    Renders primary power KPI cards and secondary environmental telemetry.
    """
    pv_val = float(reading.get("pv_power", 0.0))
    wind_val = float(reading.get("wind_power", 0.0))
    total_val = float(reading.get("total_power", 0.0))
    
    # Calculate renewable breakdown percentages
    if total_val > 0.05:
        pv_pct = (pv_val / total_val) * 100.0
        wind_pct = (wind_val / total_val) * 100.0
    else:
        pv_pct = 0.0
        wind_pct = 0.0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-container" style="border-top: 3px solid {COLOR_PV};">
                <div class="kpi-title">PV Power Generation</div>
                <div class="kpi-value" style="color: {COLOR_PV};">
                    {pv_val:.2f}<span class="kpi-unit">kW</span>
                </div>
                <div class="kpi-subtext">
                    Irradiance: <strong>{reading.get('solar_irradiance', 0):.0f} W/m²</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-container" style="border-top: 3px solid {COLOR_WIND};">
                <div class="kpi-title">Wind Power Generation</div>
                <div class="kpi-value" style="color: {COLOR_WIND};">
                    {wind_val:.2f}<span class="kpi-unit">kW</span>
                </div>
                <div class="kpi-subtext">
                    Wind Speed: <strong>{reading.get('wind_speed', 0):.1f} m/s</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi-container" style="border-top: 3px solid {COLOR_TOTAL};">
                <div class="kpi-title">Total Renewable Power</div>
                <div class="kpi-value" style="color: {COLOR_TOTAL};">
                    {total_val:.2f}<span class="kpi-unit">kW</span>
                </div>
                <div class="kpi-subtext">
                    Combined PV + Wind Infeed
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi-container" style="border-top: 3px solid #A371F7;">
                <div class="kpi-title">Renewable Contribution</div>
                <div class="kpi-value" style="color: #D2A8FF;">
                    {pv_pct:.0f}% <span style="font-size: 1.1rem; color: {COLOR_MUTED};">/</span> {wind_pct:.0f}%
                </div>
                <div class="kpi-subtext">
                    Solar share vs Wind share
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Environmental Telemetry Sub-strip
    wind_cardinal = degrees_to_cardinal(reading.get("wind_direction", 0.0))
    e1, e2, e3, e4, e5 = st.columns(5)

    with e1:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.7rem; color: {COLOR_MUTED}; text-transform: uppercase; font-weight: 600;">Solar Irradiance</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #F0F6FC;">
                    {reading.get('solar_irradiance', 0.0):.1f} <span style="font-size: 0.75rem; color: {COLOR_MUTED};">W/m²</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e2:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.7rem; color: {COLOR_MUTED}; text-transform: uppercase; font-weight: 600;">Ambient Temp</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #F0F6FC;">
                    {reading.get('ambient_temperature', 0.0):.1f} <span style="font-size: 0.75rem; color: {COLOR_MUTED};">°C</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e3:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.7rem; color: {COLOR_MUTED}; text-transform: uppercase; font-weight: 600;">Module Temp</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #F0F6FC;">
                    {reading.get('module_temperature', 0.0):.1f} <span style="font-size: 0.75rem; color: {COLOR_MUTED};">°C</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e4:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.7rem; color: {COLOR_MUTED}; text-transform: uppercase; font-weight: 600;">Wind Velocity</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #F0F6FC;">
                    {reading.get('wind_speed', 0.0):.2f} <span style="font-size: 0.75rem; color: {COLOR_MUTED};">m/s</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with e5:
        st.markdown(
            f"""
            <div style="background: {COLOR_BG_CARD}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; padding: 10px 14px;">
                <div style="font-size: 0.7rem; color: {COLOR_MUTED}; text-transform: uppercase; font-weight: 600;">Wind Heading</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700; color: #F0F6FC;">
                    {reading.get('wind_direction', 0.0):.0f}° <span style="font-size: 0.8rem; color: #58A6FF; font-weight: 600;">({wind_cardinal})</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_alert_system(reading: Dict[str, Any], thresholds: Dict[str, float]):
    """
    Evaluates telemetry against user-configured sidebar thresholds and renders active alerts.
    """
    pv_val = float(reading.get("pv_power", 0.0))
    wind_val = float(reading.get("wind_power", 0.0))
    total_val = float(reading.get("total_power", 0.0))
    wind_spd = float(reading.get("wind_speed", 0.0))
    irradiance = float(reading.get("solar_irradiance", 0.0))

    alerts = []

    # Threshold checks
    pv_thresh = thresholds.get("pv_low", 0.5)
    wind_thresh = thresholds.get("wind_low", 0.5)
    total_thresh = thresholds.get("total_low", 1.0)
    wind_var_thresh = thresholds.get("wind_gust", 14.0)

    # 1. Low PV during daylight hours
    if irradiance > 200.0 and pv_val < pv_thresh:
        alerts.append(
            {
                "level": "warning",
                "title": "LOW PV GENERATION",
                "text": f"Solar PV generation ({pv_val:.2f} kW) is below threshold ({pv_thresh:.2f} kW) despite adequate irradiance ({irradiance:.0f} W/m²). Inspect cloud shading or panel soiling.",
            }
        )

    # 2. Low Wind Generation
    if wind_spd >= 3.5 and wind_val < wind_thresh:
        alerts.append(
            {
                "level": "warning",
                "title": "LOW WIND GENERATION",
                "text": f"Wind turbine power ({wind_val:.2f} kW) is under-performing below {wind_thresh:.2f} kW with active wind speed of {wind_spd:.1f} m/s.",
            }
        )

    # 3. Low Total Output
    if total_val < total_thresh:
        alerts.append(
            {
                "level": "danger",
                "title": "LOW TOTAL RENEWABLE OUTPUT",
                "text": f"Combined renewable output ({total_val:.2f} kW) is below minimum facility threshold ({total_thresh:.2f} kW). System dispatching grid reserve.",
            }
        )

    # 4. High Wind / Turbulence Warning
    if wind_spd > wind_var_thresh:
        alerts.append(
            {
                "level": "warning",
                "title": "HIGH WIND / GUST ALERT",
                "text": f"High wind speeds detected ({wind_spd:.1f} m/s > {wind_var_thresh:.1f} m/s). Turbine pitch control active to protect drivetrain.",
            }
        )

    if alerts:
        for a in alerts:
            lvl_class = f"alert-{a['level']}"
            st.markdown(
                f"""
                <div class="alert-banner {lvl_class}">
                    <span style="font-weight: 700; margin-right: 8px;">⚠ {a['title']}:</span>
                    <span>{a['text']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="alert-banner alert-success">
                <span style="font-weight: 700; margin-right: 8px;">✓ ALL SYSTEMS NOMINAL:</span>
                <span>All generation metrics operating within expected physical design tolerances.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def create_forecast_chart(
    df_history: pd.DataFrame,
    df_forecast: pd.DataFrame,
    horizon_hours: int = 6,
) -> go.Figure:
    """
    Renders large interactive Plotly forecast chart according to Section 16:
    - X-axis: Time
    - Y-axis: Power (kW)
    - Traces: PV, Wind, Total
    - Solid lines for historical, Dashed lines for forecast
    - Hover info, zoom, pan, range selector, legends
    """
    fig = go.Figure()

    # Determine historical window to show (e.g. 24h prior)
    hist_sub = df_history.iloc[-96:].copy()

    # 1. Historical Actuals (Solid lines)
    fig.add_trace(
        go.Scatter(
            x=hist_sub["timestamp"],
            y=hist_sub["total_power"],
            mode="lines",
            name="Actual Total Renewable",
            line=dict(color=COLOR_TOTAL, width=2.5),
            hovertemplate="%{x}<br>Actual Total: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hist_sub["timestamp"],
            y=hist_sub["pv_power"],
            mode="lines",
            name="Actual Solar PV",
            line=dict(color=COLOR_PV, width=2.0),
            hovertemplate="%{x}<br>Actual PV: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=hist_sub["timestamp"],
            y=hist_sub["wind_power"],
            mode="lines",
            name="Actual Wind",
            line=dict(color=COLOR_WIND, width=2.0),
            hovertemplate="%{x}<br>Actual Wind: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    # Seamless bridge point from history to forecast
    last_hist_time = hist_sub["timestamp"].iloc[-1]
    bridge_time = [last_hist_time] + list(df_forecast["timestamp"])
    bridge_total = [hist_sub["total_power"].iloc[-1]] + list(df_forecast["total_power_forecast"])
    bridge_pv = [hist_sub["pv_power"].iloc[-1]] + list(df_forecast["pv_power_forecast"])
    bridge_wind = [hist_sub["wind_power"].iloc[-1]] + list(df_forecast["wind_power_forecast"])

    # 2. Forecast Traces (Dashed lines)
    fig.add_trace(
        go.Scatter(
            x=bridge_time,
            y=bridge_total,
            mode="lines",
            name=f"Forecast Total (+{horizon_hours}h)",
            line=dict(color=COLOR_TOTAL, width=2.5, dash="dash"),
            hovertemplate="%{x}<br>Forecast Total: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bridge_time,
            y=bridge_pv,
            mode="lines",
            name=f"Forecast Solar PV (+{horizon_hours}h)",
            line=dict(color=COLOR_PV, width=2.0, dash="dash"),
            hovertemplate="%{x}<br>Forecast PV: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bridge_time,
            y=bridge_wind,
            mode="lines",
            name=f"Forecast Wind (+{horizon_hours}h)",
            line=dict(color=COLOR_WIND, width=2.0, dash="dash"),
            hovertemplate="%{x}<br>Forecast Wind: <b>%{y:.2f} kW</b><extra></extra>",
        )
    )

    # Add vertical separator for "Forecast Origin"
    fig.add_vline(
        x=last_hist_time,
        line_width=1.5,
        line_dash="dot",
        line_color="#8B949E",
        annotation_text="← Historical | AI Forecast →",
        annotation_position="top left",
        annotation_font_color="#C9D1D9",
        annotation_font_size=11,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>Renewable Generation Forecast</b> — Next {horizon_hours} Hours",
            font=dict(size=16, color="#FFFFFF"),
        ),
        xaxis=dict(
            title="Timestamp",
            showgrid=True,
            gridcolor="#21262D",
            color="#C9D1D9",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Power Generation (kW)",
            showgrid=True,
            gridcolor="#21262D",
            color="#C9D1D9",
            zeroline=True,
            zerolinecolor="#30363D",
        ),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#C9D1D9"),
        ),
        margin=dict(l=40, r=30, t=60, b=40),
        height=480,
    )

    return fig


def create_overview_generation_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders 24-hour generation area plot for the overview page.
    """
    sub_df = df.iloc[-96:].copy()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sub_df["timestamp"],
            y=sub_df["pv_power"],
            mode="lines",
            name="Solar PV (kW)",
            line=dict(color=COLOR_PV, width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 179, 0, 0.12)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sub_df["timestamp"],
            y=sub_df["wind_power"],
            mode="lines",
            name="Wind Power (kW)",
            line=dict(color=COLOR_WIND, width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 180, 216, 0.12)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=sub_df["timestamp"],
            y=sub_df["total_power"],
            mode="lines",
            name="Total Renewable (kW)",
            line=dict(color=COLOR_TOTAL, width=2.5),
        )
    )

    fig.update_layout(
        title=dict(
            text="<b>24-Hour Power Generation Profile (Simulated Infeed)</b>",
            font=dict(size=14, color="#FFFFFF"),
        ),
        xaxis=dict(title="", showgrid=True, gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Power (kW)", showgrid=True, gridcolor="#21262D", color="#C9D1D9"),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=50, b=30),
        height=360,
    )
    return fig


def create_actual_vs_predicted_charts(
    test_df: pd.DataFrame,
    pv_model: Any,
    wind_model: Any,
    pv_features: List[str],
    wind_features: List[str],
) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """
    Renders Section 17 Actual vs Predicted plots:
    1. Time series comparison of test set (PV + Wind)
    2. Regression scatter plot with ideal 1:1 line
    3. Residual distribution histogram
    4. Total combined energy generation: Actual vs Predicted time series
    """
        # Use the full date-filtered range passed in (already scoped by the
    # date picker upstream). Cap only if extremely large, to keep rendering fast.
    MAX_RENDER_POINTS = 2000
    if len(test_df) > MAX_RENDER_POINTS:
        sample_df = test_df.iloc[-MAX_RENDER_POINTS:].copy()
    else:
        sample_df = test_df.copy()

    X_pv = sample_df[pv_features]
    y_test_pv = sample_df["pv_power"].values
    y_pred_pv = np.clip(pv_model.predict(X_pv), 0.0, None)

    X_wind = sample_df[wind_features]
    y_test_wind = sample_df["wind_power"].values
    y_pred_wind = np.clip(wind_model.predict(X_wind), 0.0, None)

    # 1. Overlay Time Series Chart
    fig_ts = go.Figure()
    fig_ts.add_trace(
        go.Scatter(
            x=sample_df["timestamp"],
            y=y_test_pv,
            name="Actual PV",
            line=dict(color=COLOR_PV, width=2),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=sample_df["timestamp"],
            y=y_pred_pv,
            name="Predicted PV",
            line=dict(color="#FFD54F", width=2, dash="dash"),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=sample_df["timestamp"],
            y=y_test_wind,
            name="Actual Wind",
            line=dict(color=COLOR_WIND, width=2),
        )
    )
    fig_ts.add_trace(
        go.Scatter(
            x=sample_df["timestamp"],
            y=y_pred_wind,
            name="Predicted Wind",
            line=dict(color="#80DEEA", width=2, dash="dash"),
        )
    )

    fig_ts.update_layout(
        title=dict(text="<b>Holdout Test Set: Actual vs Predicted Time-Series Tracking</b>", font=dict(color="#FFF")),
        xaxis=dict(gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Power (kW)", gridcolor="#21262D", color="#C9D1D9"),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        margin=dict(l=30, r=20, t=50, b=30),
        height=380,
    )

    # 2. Regression Scatter Plot (Actual vs Predicted with 1:1 reference line)
    fig_scatter = go.Figure()
    fig_scatter.add_trace(
        go.Scatter(
            x=y_test_pv,
            y=y_pred_pv,
            mode="markers",
            name="PV Predictions",
            marker=dict(color=COLOR_PV, size=5, opacity=0.6),
        )
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=y_test_wind,
            y=y_pred_wind,
            mode="markers",
            name="Wind Predictions",
            marker=dict(color=COLOR_WIND, size=5, opacity=0.6),
        )
    )
    actual_max = max(float(np.max(y_test_pv)), float(np.max(y_test_wind)), 0.5)
    max_val = round(actual_max * 1.08, 2)
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="Ideal 1:1 Fit",
            line=dict(color="#F0F6FC", dash="dot", width=1.5),
        )
    )
    fig_scatter.update_layout(
        title=dict(text="<b>Regression Scatter (Actual vs Predicted)</b>", font=dict(color="#FFF")),
        xaxis=dict(title="Actual Power (kW)", gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Predicted Power (kW)", gridcolor="#21262D", color="#C9D1D9"),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        margin=dict(l=30, r=20, t=50, b=30),
        height=380,
    )

    # 3. Residual Error Histogram
    pv_residuals = y_test_pv - y_pred_pv
    wind_residuals = y_test_wind - y_pred_wind

    fig_resid = go.Figure()
    fig_resid.add_trace(
        go.Histogram(
            x=pv_residuals,
            name="PV Residuals (kW)",
            marker_color="rgba(255, 179, 0, 0.7)",
            nbinsx=40,
        )
    )
    fig_resid.add_trace(
        go.Histogram(
            x=wind_residuals,
            name="Wind Residuals (kW)",
            marker_color="rgba(0, 180, 216, 0.7)",
            nbinsx=40,
        )
    )
    fig_resid.update_layout(
        title=dict(text="<b>Residual Error Distribution (Actual - Predicted)</b>", font=dict(color="#FFF")),
        xaxis=dict(title="Error (kW)", gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Frequency", gridcolor="#21262D", color="#C9D1D9"),
        barmode="overlay",
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        margin=dict(l=30, r=20, t=50, b=30),
        height=380,
    )

    fig_total = _build_total_actual_vs_predicted_chart(
        sample_df, y_test_pv, y_pred_pv, y_test_wind, y_pred_wind
    )

    return fig_ts, fig_scatter, fig_resid, fig_total


def _build_total_actual_vs_predicted_chart(
    sample_df: pd.DataFrame, y_test_pv: np.ndarray, y_pred_pv: np.ndarray,
    y_test_wind: np.ndarray, y_pred_wind: np.ndarray,
) -> go.Figure:
    y_test_total = y_test_pv + y_test_wind
    y_pred_total = y_pred_pv + y_pred_wind

    fig_total = go.Figure()
    fig_total.add_trace(
        go.Scatter(
            x=sample_df["timestamp"],
            y=y_test_total,
            name="Actual Total Generation",
            line=dict(color=COLOR_TOTAL, width=2.5),
        )
    )
    fig_total.add_trace(
        go.Scatter(
            x=sample_df["timestamp"],
            y=y_pred_total,
            name="Predicted Total Generation",
            line=dict(color="#F778BA", width=2.5, dash="dash"),
        )
    )
    fig_total.update_layout(
        title=dict(
            text="<b>Total Energy Generation: Actual vs Predicted</b>",
            font=dict(color="#FFF"),
        ),
        xaxis=dict(gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Total Power (kW)", gridcolor="#21262D", color="#C9D1D9"),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        margin=dict(l=30, r=20, t=50, b=30),
        height=380,
    )
    return fig_total


def create_environmental_gauges(reading: Dict[str, Any]) -> List[go.Figure]:
    """
    Renders environmental gauges for Irradiance, Temperatures, and Wind Speed.
    """
    figures = []

    # 1. Solar Irradiance Gauge
    fig_irr = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(reading.get("solar_irradiance", 0.0)),
            number={"suffix": " W/m²", "font": {"color": COLOR_PV, "size": 22}},
            title={"text": "Solar Irradiance", "font": {"color": "#C9D1D9", "size": 13}},
            gauge={
                "axis": {"range": [0, 1200], "tickcolor": "#8B949E"},
                "bar": {"color": COLOR_PV},
                "bgcolor": "#161B22",
                "borderwidth": 1,
                "bordercolor": "#30363D",
                "steps": [
                    {"range": [0, 200], "color": "#21262D"},
                    {"range": [200, 700], "color": "#30363D"},
                    {"range": [700, 1200], "color": "#484F58"},
                ],
            },
        )
    )
    fig_irr.update_layout(paper_bgcolor="#161B22", height=200, margin=dict(l=20, r=20, t=40, b=10))
    figures.append(fig_irr)

    # 2. Wind Speed Gauge
    fig_wind = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(reading.get("wind_speed", 0.0)),
            number={"suffix": " m/s", "font": {"color": COLOR_WIND, "size": 22}},
            title={"text": "Wind Velocity", "font": {"color": "#C9D1D9", "size": 13}},
            gauge={
                "axis": {"range": [0, 25], "tickcolor": "#8B949E"},
                "bar": {"color": COLOR_WIND},
                "bgcolor": "#161B22",
                "borderwidth": 1,
                "bordercolor": "#30363D",
                "steps": [
                    {"range": [0, 3], "color": "#21262D"},
                    {"range": [3, 12], "color": "#1F6FEB"},
                    {"range": [12, 25], "color": "#D29922"},
                ],
            },
        )
    )
    fig_wind.update_layout(paper_bgcolor="#161B22", height=200, margin=dict(l=20, r=20, t=40, b=10))
    figures.append(fig_wind)

    # 3. Ambient & Module Temp Gauge
    fig_temp = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(reading.get("module_temperature", 0.0)),
            number={"suffix": " °C", "font": {"color": "#FF7B72", "size": 22}},
            title={"text": "PV Module Temperature", "font": {"color": "#C9D1D9", "size": 13}},
            gauge={
                "axis": {"range": [0, 75], "tickcolor": "#8B949E"},
                "bar": {"color": "#F85149"},
                "bgcolor": "#161B22",
                "borderwidth": 1,
                "bordercolor": "#30363D",
                "steps": [
                    {"range": [0, 25], "color": "#238636"},
                    {"range": [25, 50], "color": "#D29922"},
                    {"range": [50, 75], "color": "#DA3633"},
                ],
            },
        )
    )
    fig_temp.update_layout(paper_bgcolor="#161B22", height=200, margin=dict(l=20, r=20, t=40, b=10))
    figures.append(fig_temp)

    return figures


def create_energy_analytics_charts(
    df: pd.DataFrame,
    pv_pred_col: str = None,
    wind_pred_col: str = None,
) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Renders Section 21 Energy Analytics:
    1. Daily energy generation bar chart (kWh), with optional predicted overlay
    2. PV vs Wind energy share donut chart
    3. 24-hour diurnal generation curves, with optional predicted overlay

    pv_pred_col / wind_pred_col: optional column names in df holding model-predicted
    PV / Wind power (kW). When provided, predicted lines are overlaid on the
    daily and diurnal charts alongside the actual data for visual comparison.
    """
    # 1. Compute Daily Energy (kWh = sum of kW * step_hours)
    daily_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(daily_df["timestamp"]):
        daily_df["timestamp"] = pd.to_datetime(daily_df["timestamp"])
    step_hours = 1.0
    if len(daily_df) > 1:
        step_hours = (daily_df["timestamp"].iloc[1] - daily_df["timestamp"].iloc[0]).total_seconds() / 3600.0
        if step_hours <= 0:
            step_hours = 1.0

    daily_df["date"] = daily_df["timestamp"].dt.date
    agg_map = {
        "pv_kwh": ("pv_power", lambda x: float(x.sum() * step_hours)),
        "wind_kwh": ("wind_power", lambda x: float(x.sum() * step_hours)),
        "total_kwh": ("total_power", lambda x: float(x.sum() * step_hours)),
    }
    if pv_pred_col and pv_pred_col in daily_df.columns:
        agg_map["pv_pred_kwh"] = (pv_pred_col, lambda x: float(x.sum() * step_hours))
    if wind_pred_col and wind_pred_col in daily_df.columns:
        agg_map["wind_pred_kwh"] = (wind_pred_col, lambda x: float(x.sum() * step_hours))
    daily_energy = daily_df.groupby("date").agg(**agg_map).reset_index()
    if "pv_pred_kwh" in daily_energy.columns or "wind_pred_kwh" in daily_energy.columns:
        daily_energy["total_pred_kwh"] = daily_energy.get("pv_pred_kwh", 0) + daily_energy.get("wind_pred_kwh", 0)

    # Daily Bar Chart (last 30 days)
    recent_daily = daily_energy.tail(30)
    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(
            x=recent_daily["date"],
            y=recent_daily["pv_kwh"],
            name="Solar PV (kWh)",
            marker_color=COLOR_PV,
        )
    )
    fig_daily.add_trace(
        go.Bar(
            x=recent_daily["date"],
            y=recent_daily["wind_kwh"],
            name="Wind (kWh)",
            marker_color=COLOR_WIND,
        )
    )
    if "total_pred_kwh" in recent_daily.columns:
        fig_daily.add_trace(
            go.Scatter(
                x=recent_daily["date"],
                y=recent_daily["total_pred_kwh"],
                name="XGBoost Predicted Total (kWh)",
                mode="lines+markers",
                line=dict(color="#F778BA", width=2.5, dash="dash"),
                marker=dict(size=6),
            )
        )
    fig_daily.update_layout(
        barmode="stack",
        title=dict(text="<b>Daily Renewable Energy Production (Last 30 Days)</b>", font=dict(color="#FFF")),
        xaxis=dict(title="Date", gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Energy Yield (kWh)", gridcolor="#21262D", color="#C9D1D9"),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        margin=dict(l=30, r=20, t=50, b=30),
        height=360,
    )

    # 2. PV vs Wind Energy Breakdown Donut Chart
    total_pv_kwh = float(daily_energy["pv_kwh"].sum())
    total_wind_kwh = float(daily_energy["wind_kwh"].sum())
    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=["Solar PV Energy", "Wind Energy"],
                values=[total_pv_kwh, total_wind_kwh],
                hole=0.6,
                marker_colors=[COLOR_PV, COLOR_WIND],
                textinfo="label+percent",
                insidetextorientation="radial",
            )
        ]
    )
    fig_donut.update_layout(
        title=dict(text="<b>Cumulative Lifetime Energy Share</b>", font=dict(color="#FFF")),
        paper_bgcolor="#161B22",
        margin=dict(l=20, r=20, t=50, b=20),
        height=360,
        showlegend=False,
    )

    # 3. 24-Hour Diurnal Generation Curve (Average kW by hour of day)
    hourly_df = df.copy()
    hourly_df["hour"] = hourly_df["timestamp"].dt.hour
    diurnal_agg_map = {
        "pv_avg": ("pv_power", "mean"),
        "wind_avg": ("wind_power", "mean"),
        "total_avg": ("total_power", "mean"),
    }
    if pv_pred_col and pv_pred_col in hourly_df.columns:
        diurnal_agg_map["pv_pred_avg"] = (pv_pred_col, "mean")
    if wind_pred_col and wind_pred_col in hourly_df.columns:
        diurnal_agg_map["wind_pred_avg"] = (wind_pred_col, "mean")
    diurnal_profile = hourly_df.groupby("hour").agg(**diurnal_agg_map).reset_index()

    fig_diurnal = go.Figure()
    fig_diurnal.add_trace(
        go.Scatter(
            x=diurnal_profile["hour"],
            y=diurnal_profile["pv_avg"],
            mode="lines+markers",
            name="Avg Solar PV",
            line=dict(color=COLOR_PV, width=2.5),
        )
    )
    fig_diurnal.add_trace(
        go.Scatter(
            x=diurnal_profile["hour"],
            y=diurnal_profile["wind_avg"],
            mode="lines+markers",
            name="Avg Wind Power",
            line=dict(color=COLOR_WIND, width=2.5),
        )
    )
    fig_diurnal.add_trace(
        go.Scatter(
            x=diurnal_profile["hour"],
            y=diurnal_profile["total_avg"],
            mode="lines+markers",
            name="Avg Combined Output",
            line=dict(color=COLOR_TOTAL, width=3),
        )
    )
    if "pv_pred_avg" in diurnal_profile.columns:
        fig_diurnal.add_trace(
            go.Scatter(
                x=diurnal_profile["hour"],
                y=diurnal_profile["pv_pred_avg"],
                mode="lines+markers",
                name="Predicted Avg Solar PV",
                line=dict(color=COLOR_PV, width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
    if "wind_pred_avg" in diurnal_profile.columns:
        fig_diurnal.add_trace(
            go.Scatter(
                x=diurnal_profile["hour"],
                y=diurnal_profile["wind_pred_avg"],
                mode="lines+markers",
                name="Predicted Avg Wind",
                line=dict(color=COLOR_WIND, width=2, dash="dash"),
                marker=dict(size=5, symbol="diamond"),
            )
        )
    fig_diurnal.update_layout(
        title=dict(text="<b>Average 24-Hour Diurnal Generation Profile</b>", font=dict(color="#FFF")),
        xaxis=dict(title="Hour of Day (00:00 - 23:00)", tickmode="linear", tick0=0, dtick=2, gridcolor="#21262D", color="#C9D1D9"),
        yaxis=dict(title="Average Power (kW)", gridcolor="#21262D", color="#C9D1D9"),
        plot_bgcolor="#0D1117",
        paper_bgcolor="#161B22",
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right"),
        margin=dict(l=30, r=20, t=50, b=30),
        height=360,
    )

    return fig_daily, fig_donut, fig_diurnal
