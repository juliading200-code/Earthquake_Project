from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DEMO_DATA_FILE = "data/app_demo_data.npz"
MAX_ANIMATION_FRAMES = 260
SHAKEMAP_TABLE = [
    {"Shaking Category": "Moderate", "PGA Range (g)": "0.0276 - 0.115", "Description": "Moderate shaking; small non-structural effects can occur."},
    {"Shaking Category": "Strong", "PGA Range (g)": "0.115 - 0.215", "Description": "Strong shaking; non-structural damage becomes more likely."},
    {"Shaking Category": "Very Strong", "PGA Range (g)": "0.215 - 0.401", "Description": "Very strong shaking; structural damage may begin in vulnerable buildings."},
    {"Shaking Category": "Severe", "PGA Range (g)": "0.401 - 0.747", "Description": "Severe shaking; significant damage potential."},
    {"Shaking Category": "Violent", "PGA Range (g)": "0.747 - 1.39", "Description": "Violent shaking; heavy damage is likely."},
    {"Shaking Category": "Extreme", "PGA Range (g)": "> 1.39", "Description": "Extreme shaking; very destructive potential."},
]


@dataclass
class DemoData:
    time: np.ndarray
    ground_motion: np.ndarray
    actual_displacement: np.ndarray
    y_pred: np.ndarray
    pga_g: np.ndarray
    category: np.ndarray
    category_description: np.ndarray
    record_id: np.ndarray
    prediction_file: str
    selection_warnings: List[str]
    pga_note: str
    single_channel_floor_idx: int


@st.cache_data(show_spinner=False)
def load_demo_data(workspace: str) -> DemoData:
    root = Path(workspace)
    data_path = root / DEMO_DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Demo data file not found: {data_path}. Run make_demo_data.py first."
        )

    npz = np.load(str(data_path), allow_pickle=True)
    time = np.asarray(npz["time"], dtype=np.float64)
    ground_motion = np.asarray(npz["ground_motion"], dtype=np.float64)
    actual_displacement = np.asarray(npz["actual_displacement"], dtype=np.float64)
    y_pred = np.asarray(npz["y_pred"], dtype=np.float64)
    if y_pred.ndim == 2:
        y_pred = y_pred[:, :, None]

    pga_g = np.asarray(npz["pga_g"], dtype=np.float64)
    category = np.asarray(npz["category"]).astype(str)
    category_description = np.asarray(npz["category_description"]).astype(str)
    record_id = np.asarray(npz["record_id"], dtype=np.int64)

    prediction_file_arr = np.asarray(npz["prediction_file"]).astype(str)
    prediction_file = prediction_file_arr[0] if len(prediction_file_arr) else ""

    selection_warnings = [str(x) for x in np.asarray(npz.get("selection_warnings", []), dtype=str)]
    pga_note_arr = np.asarray(npz.get("pga_note", [""]), dtype=str)
    pga_note = pga_note_arr[0] if len(pga_note_arr) else ""

    floor_idx_arr = np.asarray(npz.get("single_channel_floor_idx", [0]), dtype=np.int64)
    single_channel_floor_idx = int(floor_idx_arr[0]) if len(floor_idx_arr) else 0

    return DemoData(
        time=time,
        ground_motion=ground_motion,
        actual_displacement=actual_displacement,
        y_pred=y_pred,
        pga_g=pga_g,
        category=category,
        category_description=category_description,
        record_id=record_id,
        prediction_file=prediction_file,
        selection_warnings=selection_warnings,
        pga_note=pga_note,
        single_channel_floor_idx=single_channel_floor_idx,
    )


def select_comparison_channels(
    actual_t_f: np.ndarray,
    pred_t_f: np.ndarray,
    single_channel_floor_idx: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], str | None, str]:
    pred_channels = pred_t_f.shape[1]

    if pred_channels == 6:
        labels = [f"Floor {i}" for i in range(1, 7)]
        return actual_t_f[:, :6], pred_t_f[:, :6], labels, None, "FNO prediction"

    if pred_channels == 1:
        idx = single_channel_floor_idx if 0 <= single_channel_floor_idx < 6 else 0
        warning = (
            "This FNO output file contains only one predicted response channel, "
            "so it cannot be used for a full six-floor comparison."
        )
        warning += f" The single channel aligns with Floor {idx + 1}."
        return actual_t_f[:, idx : idx + 1], pred_t_f[:, :1], [f"Floor {idx + 1}"], warning, "Single-channel FNO prediction"

    shared = min(actual_t_f.shape[1], pred_channels)
    warning = (
        f"This FNO output file contains {pred_channels} predicted response channels; "
        f"showing {shared} channel(s) that overlap with available ground truth."
    )
    labels = [f"Response channel {i}" for i in range(1, shared + 1)]
    return actual_t_f[:, :shared], pred_t_f[:, :shared], labels, warning, "FNO prediction"


def compute_metrics(actual_t_f: np.ndarray, pred_t_f: np.ndarray) -> pd.DataFrame:
    err = pred_t_f - actual_t_f
    mse_all = float(np.mean(err**2))
    rmse_all = float(np.sqrt(mse_all))
    maxabs_all = float(np.max(np.abs(err)))
    rel_all = float(np.linalg.norm(err) / max(np.linalg.norm(actual_t_f), 1e-12) * 100.0)

    return pd.DataFrame(
        {
            "Metric": [
                "Mean Squared Error (MSE)",
                "Root Mean Squared Error (RMSE)",
                "Max Absolute Error",
                "Relative Error %",
            ],
            "Value": [mse_all, rmse_all, maxabs_all, rel_all],
        }
    )


def building_animation_figure(
    t: np.ndarray,
    floor_disp_t_f: np.ndarray,
    title: str,
) -> go.Figure:
    n = len(t)
    step = max(1, n // MAX_ANIMATION_FRAMES)
    ids = np.arange(0, n, step)

    max_disp = float(np.max(np.abs(floor_disp_t_f)))
    x_axis_half_range = max(4.0, max_disp * 4.0)

    y_levels = np.arange(0, 7)

    def frame_xy(k: int) -> Tuple[np.ndarray, np.ndarray]:
        base_x = 0.0
        floor_x = floor_disp_t_f[k]
        x = np.concatenate(([base_x], floor_x))
        return x, y_levels

    x0, y0 = frame_xy(int(ids[0]))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.0, 0.0],
            y=[0.0, 6.0],
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            name="Undeformed reference",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x0,
            y=y0,
            mode="lines+markers+text",
            text=["Base", "1", "2", "3", "4", "5", "6"],
            textposition="middle right",
            marker=dict(size=9, color="#2d6cdf"),
            line=dict(color="#1f77b4", width=4),
            name="Deformed building",
        )
    )

    frames = []
    for k in ids:
        x, y = frame_xy(int(k))
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=[0.0, 0.0], y=[0.0, 6.0]),
                    go.Scatter(x=x, y=y),
                ],
                name=f"{t[int(k)]:.3f}",
            )
        )

    fig.frames = frames
    fig.update_layout(
        title=title,
        xaxis_title="Relative horizontal position",
        yaxis_title="Story level",
        yaxis=dict(range=[-0.4, 6.6], tickmode="array", tickvals=np.arange(0, 7)),
        xaxis=dict(range=[-x_axis_half_range, x_axis_half_range]),
        width=760,
        height=500,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 45, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.1,
                "y": -0.08,
                "len": 0.85,
                "currentvalue": {"prefix": "Time (s): "},
                "steps": [
                    {
                        "method": "animate",
                        "label": f"{f.name}",
                        "args": [[f.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                    }
                    for f in frames
                ],
            }
        ],
    )
    return fig


def make_comparison_figure(
    time_s: np.ndarray,
    actual_t: np.ndarray,
    pred_t: np.ndarray,
    panel_title: str,
    pred_legend_label: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=actual_t,
            mode="lines",
            line=dict(color="red", width=2),
            name="Actual response (Ground Truth)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_s,
            y=pred_t,
            mode="lines",
            line=dict(color="blue", width=2),
            name=pred_legend_label,
        )
    )
    fig.update_layout(
        title=panel_title,
        xaxis_title="Time (seconds)",
        yaxis_title="Displacement (dataset values)",
        legend=dict(orientation="h"),
        height=290,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Earthquake FNO Educational Simulator", layout="wide")
    st.title("Modeling Structural Response Under Seismic Loading: A Comparison of Fourier Neural Operator Predictions and Numerical Simulations")
    st.caption(
        "Interactive educational demo: choose a real earthquake record, watch a six-story building respond, "
        "and compare simulation ground truth against FNO predictions."
    )

    workspace = Path(__file__).resolve().parent
    try:
        data = load_demo_data(str(workspace))
    except Exception as exc:
        st.error(f"Could not load demo data. Details: {exc}")
        st.stop()

    n_records = data.ground_motion.shape[0]
    if n_records < 3:
        st.error("Demo dataset must contain at least three records.")
        st.stop()

    st.markdown("### Screen 1: Earthquake Selection")
    st.caption("Representative records selected using USGS ShakeMap Instrumental Intensity PGA categories.")
    st.dataframe(pd.DataFrame(SHAKEMAP_TABLE), width="stretch", hide_index=True)

    button_cols = st.columns(3)
    labels = ["Earthquake A", "Earthquake B", "Earthquake C"]
    for i in range(3):
        with button_cols[i]:
            if st.button(labels[i], width="stretch"):
                st.session_state["selected_idx"] = i
            st.caption(f"{data.category[i]} | PGA: {data.pga_g[i]:.4f} g")

    if "selected_idx" not in st.session_state:
        st.session_state["selected_idx"] = 1

    selected_idx = int(st.session_state["selected_idx"])
    selected_idx = max(0, min(selected_idx, 2))

    time_series = data.time[selected_idx]
    ground = data.ground_motion[selected_idx]
    actual = data.actual_displacement[selected_idx]
    pred_raw = data.y_pred[selected_idx]

    common_n = min(len(time_series), len(ground), actual.shape[0], pred_raw.shape[0])
    time_series = time_series[:common_n]
    ground = ground[:common_n]
    actual = actual[:common_n, :6]
    pred_raw = pred_raw[:common_n]

    compare_actual, compare_pred, compare_labels, compare_warning, pred_legend_label = select_comparison_channels(
        actual, pred_raw, data.single_channel_floor_idx
    )

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("Selected Ground Motion")
        gm_fig = go.Figure(
            data=[go.Scatter(x=time_series, y=ground, mode="lines", line=dict(color="#d35400", width=2))]
        )
        gm_fig.update_layout(
            title="Selected Ground Motion",
            xaxis_title="Time (seconds)",
            yaxis_title="Ground acceleration (g)",
            height=330,
            margin=dict(l=40, r=20, t=45, b=40),
        )
        st.plotly_chart(gm_fig, width="stretch")

    with col_right:
        st.subheader("Selected Record Info")
        st.write(f"Dataset record id: {int(data.record_id[selected_idx])}")
        st.write(f"Peak ground acceleration (|max|): {float(data.pga_g[selected_idx]):.4f} g")
        st.write(f"ShakeMap category: {data.category[selected_idx]}")
        st.write(f"Description: {data.category_description[selected_idx]}")
        st.write(f"Prediction source: {data.prediction_file}")
        if data.pga_note:
            st.caption(data.pga_note)
        for msg in data.selection_warnings:
            st.caption(msg)
        if compare_warning:
            st.caption(compare_warning)

    st.markdown("### Screen 2: Building Animation")
    anim_fig = building_animation_figure(
        t=time_series,
        floor_disp_t_f=actual,
        title="Six-Story Building Response Driven by Selected Earthquake",
    )
    st.plotly_chart(anim_fig, width="content")
    st.caption(
        "Animation note: plotted floor offset = displacement x 1.0 (no displacement scaling). "
        "Physical displacement units are not explicitly provided in the available dataset metadata. "
        "The horizontal axis range is expanded only for easier viewing."
    )

    st.markdown("### Screen 3: FNO vs Actual Comparison")
    for idx, label in enumerate(compare_labels):
        fig = make_comparison_figure(
            time_series,
            compare_actual[:, idx],
            compare_pred[:, idx],
            f"{label}: Actual vs {pred_legend_label}",
            pred_legend_label,
        )
        st.plotly_chart(fig, width="stretch")

    overall = compute_metrics(compare_actual, compare_pred)
    st.markdown("### Prediction Error Metrics")
    st.dataframe(overall, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
