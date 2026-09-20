"""
simulation.py
-------------
Interactive time-series simulation engine for the dashboard.
Provides START, PAUSE, RESET controls and variable playback speeds (1x, 2x, 5x)
to realistically stream telemetry records without misrepresenting simulated data as physical sensors.
"""

from typing import Any, Dict, Tuple
import pandas as pd
import streamlit as st


def init_simulation_state(total_records: int, default_window: int = 96) -> None:
    """
    Initialize Streamlit session state variables for simulation playback.
    default_window = 96 steps = 24 hours of 15-minute intervals.
    """
    if "sim_running" not in st.session_state:
        st.session_state.sim_running = False

    if "sim_index" not in st.session_state:
        # Start at 85% through dataset so there is ample historical buffer
        st.session_state.sim_index = max(default_window, int(total_records * 0.85))

    if "sim_speed" not in st.session_state:
        st.session_state.sim_speed = "1x"

    if "sim_step_count" not in st.session_state:
        st.session_state.sim_step_count = 0


def advance_simulation(df: pd.DataFrame) -> None:
    """
    Advance simulation pointer according to speed multiplier if running.
    """
    if st.session_state.get("sim_running", False):
        speed_map = {"1x": 1, "2x": 2, "5x": 5}
        step_increment = speed_map.get(st.session_state.get("sim_speed", "1x"), 1)
        
        max_idx = len(df) - 1
        new_idx = st.session_state.sim_index + step_increment
        if new_idx >= max_idx:
            new_idx = max_idx
            st.session_state.sim_running = False  # End of dataset reached
        st.session_state.sim_index = new_idx
        st.session_state.sim_step_count += 1


def get_current_simulation_window(
    df: pd.DataFrame, window_steps: int = 96
) -> Tuple[Dict[str, Any], pd.DataFrame, float]:
    """
    Retrieve the current simulated telemetry record and historical buffer.

    Parameters:
    - df: Full telemetry DataFrame
    - window_steps: Number of prior steps to include in recent history (e.g. 96 = 24 hours)

    Returns:
    - current_reading: Dict of latest record
    - df_history: DataFrame containing records up to current simulated timestamp
    - progress_pct: Progress percentage through dataset (0.0 - 100.0)
    """
    curr_idx = st.session_state.get("sim_index", len(df) - 1)
    curr_idx = min(max(0, curr_idx), len(df) - 1)

    current_reading = df.iloc[curr_idx].to_dict()
    
    start_idx = max(0, curr_idx - window_steps + 1)
    df_history = df.iloc[start_idx : curr_idx + 1].copy().reset_index(drop=True)

    progress_pct = (curr_idx / (len(df) - 1)) * 100.0

    return current_reading, df_history, progress_pct


def render_simulation_controls(df: pd.DataFrame) -> None:
    """
    Render simulation controller widget bar with Start, Pause, Reset, and Speed buttons.
    """
    init_simulation_state(len(df))

    interval_min = 60
    if len(df) > 1 and "timestamp" in df.columns:
        dt = (pd.to_datetime(df["timestamp"].iloc[1]) - pd.to_datetime(df["timestamp"].iloc[0])).total_seconds()
        interval_min = max(1, int(round(dt / 60.0)))
    interval_label = f"{interval_min}-minute" if interval_min < 60 else f"{interval_min // 60}-hour"
    step_button_label = f"{interval_min} mins" if interval_min < 60 else f"{interval_min // 60} hr"

    st.markdown(
        f"""
        <div style="background: #161B22; border-left: 4px solid #00D26A; padding: 12px 18px; border-radius: 6px; margin-bottom: 16px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.82rem; font-weight: 700; color: #00D26A; letter-spacing: 0.08em; text-transform: uppercase;">
                        Simulated Real-Time Controller
                    </span>
                    <div style="font-size: 0.8rem; color: #8B949E; margin-top: 2px;">
                        Sequential playback across {interval_label} telemetry intervals. Physical sensor emulation mode.
                    </div>
                </div>
                <div style="background: rgba(0, 210, 106, 0.1); border: 1px solid rgba(0, 210, 106, 0.3); border-radius: 4px; padding: 4px 10px; font-size: 0.75rem; font-weight: 600; color: #00D26A;">
                    ● SIMULATED REAL-TIME
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1.4, 2])

    with c1:
        if st.session_state.sim_running:
            if st.button("⏸ Pause", use_container_width=True, help="Pause simulation playback"):
                st.session_state.sim_running = False
                st.rerun()
        else:
            if st.button("▶ Start", type="primary", use_container_width=True, help="Start real-time simulation"):
                st.session_state.sim_running = True
                st.rerun()

    with c2:
        if st.button("⏮ Reset", use_container_width=True, help="Reset simulation to default index"):
            st.session_state.sim_running = False
            st.session_state.sim_index = max(96, int(len(df) * 0.85))
            st.session_state.sim_step_count = 0
            st.rerun()

    with c3:
        if st.button("⏭ Next Step", use_container_width=True, help=f"Advance 1 interval ({step_button_label})"):
            st.session_state.sim_index = min(len(df) - 1, st.session_state.sim_index + 1)
            st.rerun()

    with c4:
        speed_choice = st.selectbox(
            "Playback Speed",
            options=["1x", "2x", "5x"],
            index=["1x", "2x", "5x"].index(st.session_state.sim_speed),
            label_visibility="collapsed",
            help="Simulation clock speed multiplier",
        )
        if speed_choice != st.session_state.sim_speed:
            st.session_state.sim_speed = speed_choice

    with c5:
        curr_idx = st.session_state.get("sim_index", 0)
        curr_time = df.iloc[curr_idx]["timestamp"]
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #C9D1D9; text-align: right; padding-top: 6px;">
                Current Sim Time: <strong style="color: #58A6FF;">{curr_time}</strong>
                <br>
                <span style="font-size: 0.72rem; color: #8B949E;">Step {curr_idx} of {len(df)-1}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
