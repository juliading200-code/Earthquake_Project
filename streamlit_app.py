from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.io
import streamlit as st


DATASET_FILE = "dataset_EQ.mat"
INDEX_FILE = "train_test_index.mat"
PREDICTION_CANDIDATES = [
    "Earthquake_responses_test.mat",
    "Earthquake_responses_test_fno.mat",
]
CROP_START = 21
MAX_ANIMATION_FRAMES = 260
SHAKEMAP_LEVELS = [
    {"name": "Below Moderate", "low": 0.0, "high": 0.0276, "description": "Usually weak or light shaking."},
    {"name": "Moderate", "low": 0.0276, "high": 0.115, "description": "Moderate shaking; small non-structural effects can occur."},
    {"name": "Strong", "low": 0.115, "high": 0.215, "description": "Strong shaking; non-structural damage becomes more likely."},
    {"name": "Very Strong", "low": 0.215, "high": 0.401, "description": "Very strong shaking; structural damage may begin in vulnerable buildings."},
    {"name": "Severe", "low": 0.401, "high": 0.747, "description": "Severe shaking; significant damage potential."},
    {"name": "Violent", "low": 0.747, "high": 1.39, "description": "Violent shaking; heavy damage is likely."},
    {"name": "Extreme", "low": 1.39, "high": np.inf, "description": "Extreme shaking; very destructive potential."},
]


@dataclass
class LoadedData:
    workspace: Path
    variable_report: Dict[str, List[Tuple[str, Tuple[int, ...], str]]]
    time_full: np.ndarray
    ground_full: np.ndarray
    displacement_full: np.ndarray
    test_indices_zero_based: np.ndarray
    prediction_file: str | None
    y_pred_test_order: np.ndarray | None
    y_test_test_order: np.ndarray | None


@st.cache_data(show_spinner=False)
def _load_mat(path: str) -> Dict[str, np.ndarray]:
    return scipy.io.loadmat(path)


@st.cache_data(show_spinner=False)
def load_all_data(workspace: str) -> LoadedData:
    root = Path(workspace)
    mat_files = sorted(root.glob("*.mat"))
    variable_report: Dict[str, List[Tuple[str, Tuple[int, ...], str]]] = {}

    for mat_path in mat_files:
        data = _load_mat(str(mat_path))
        rows: List[Tuple[str, Tuple[int, ...], str]] = []
        for key, value in data.items():
            if key.startswith("__"):
                continue
            arr = np.asarray(value)
            rows.append((key, tuple(arr.shape), str(arr.dtype)))
        variable_report[mat_path.name] = rows

    dataset = _load_mat(str(root / DATASET_FILE))
    indices = _load_mat(str(root / INDEX_FILE))

    time_full = np.asarray(dataset["time"], dtype=np.float64)
    ground_full = np.asarray(dataset["ground_motion"], dtype=np.float64)
    displacement_full = np.asarray(dataset["displacement"], dtype=np.float64)

    test_indices_zero_based = np.asarray(indices["test"]).astype(int).ravel() - 1

    pred_file = None
    y_pred = None
    y_test = None
    for candidate in PREDICTION_CANDIDATES:
        p = root / candidate
        if not p.exists():
            continue
        candidate_data = _load_mat(str(p))
        if "y_pred" not in candidate_data:
            continue

        pred_raw = np.asarray(candidate_data["y_pred"])
        if pred_raw.ndim == 2:
            pred_raw = pred_raw[:, :, None]
        if pred_raw.ndim != 3:
            continue

        y_test_raw = candidate_data.get("y_test")
        if y_test_raw is not None:
            y_test_raw = np.asarray(y_test_raw)
            if y_test_raw.ndim == 2:
                y_test_raw = y_test_raw[:, :, None]
            if y_test_raw.ndim == 3:
                y_test = y_test_raw.astype(np.float64)

        pred_file = candidate
        y_pred = pred_raw.astype(np.float64)
        break

    return LoadedData(
        workspace=root,
        variable_report=variable_report,
        time_full=time_full,
        ground_full=ground_full,
        displacement_full=displacement_full,
        test_indices_zero_based=test_indices_zero_based,
        prediction_file=pred_file,
        y_pred_test_order=y_pred,
        y_test_test_order=y_test,
    )


def _assign_shakemap_category(pga_g: float) -> Tuple[str, int, str]:
    for rank, level in enumerate(SHAKEMAP_LEVELS):
        if level["low"] <= pga_g < level["high"]:
            return str(level["name"]), rank, str(level["description"])
    # Defensive fallback; should never be reached.
    tail = SHAKEMAP_LEVELS[-1]
    return str(tail["name"]), len(SHAKEMAP_LEVELS) - 1, str(tail["description"])


def build_pga_summary(data: LoadedData) -> Tuple[pd.DataFrame, str]:
    n_records = data.ground_full.shape[0]
    ground_all = data.ground_full[:, CROP_START:]
    pga = np.max(np.abs(ground_all), axis=1)

    test_ids = data.test_indices_zero_based
    pred_local_lookup = {int(global_idx): i for i, global_idx in enumerate(test_ids)}

    rows = []
    for global_idx_zero in range(n_records):
        pga_val = float(pga[global_idx_zero])
        category_name, category_rank, category_description = _assign_shakemap_category(float(pga_val))
        pred_local_idx = pred_local_lookup.get(int(global_idx_zero), -1)
        rows.append(
            {
                "global_idx_zero": int(global_idx_zero),
                "global_record_id": int(global_idx_zero) + 1,
                "pga_g": float(pga_val),
                "category": category_name,
                "category_rank": int(category_rank),
                "category_description": category_description,
                "has_prediction": pred_local_idx >= 0,
                "pred_local_idx": int(pred_local_idx),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["category_rank", "pga_g"]).reset_index(drop=True)
    note = (
        "PGA/category assignment is computed for every record in dataset_EQ.mat; representative picks are chosen "
        "from records that also have FNO predictions for comparison."
    )
    return summary, note


def pick_representative_records(pga_summary: pd.DataFrame) -> Tuple[List[Dict[str, object]], List[str]]:
    warnings: List[str] = []

    prediction_ready = pga_summary[pga_summary["has_prediction"]].copy()
    if prediction_ready.empty:
        return [], ["No records with both ground truth and FNO predictions are available."]

    available_ranks = sorted(prediction_ready["category_rank"].unique().tolist())
    if not available_ranks:
        return [], ["No earthquake records were available to select."]

    target_ranks: List[int]
    if len(available_ranks) >= 3:
        target_ranks = [
            int(available_ranks[0]),
            int(available_ranks[len(available_ranks) // 2]),
            int(available_ranks[-1]),
        ]
    elif len(available_ranks) == 2:
        target_ranks = [int(available_ranks[0]), int(available_ranks[1]), int(available_ranks[1])]
        warnings.append(
            "Only two ShakeMap categories are present in this dataset split; one category is reused to provide three selections."
        )
    else:
        target_ranks = [int(available_ranks[0]), int(available_ranks[0]), int(available_ranks[0])]
        warnings.append(
            "Only one ShakeMap category is present in this dataset split; all three selections come from that category."
        )

    selected_rows: List[pd.Series] = []
    used_pred_local_idx: set[int] = set()
    for rank in target_ranks:
        in_cat = prediction_ready[prediction_ready["category_rank"] == rank].copy()
        in_cat = in_cat.sort_values("pga_g").reset_index(drop=True)

        # Use the category median PGA as the representative, with uniqueness fallback.
        candidate_order = [len(in_cat) // 2] + list(range(len(in_cat)))
        chosen = None
        for cand_i in candidate_order:
            row = in_cat.iloc[int(cand_i)]
            pred_local_idx = int(row["pred_local_idx"])
            if pred_local_idx not in used_pred_local_idx:
                chosen = row
                break

        if chosen is None:
            chosen = in_cat.iloc[len(in_cat) // 2]

        used_pred_local_idx.add(int(chosen["pred_local_idx"]))
        selected_rows.append(chosen)

    selected_rows = sorted(selected_rows[:3], key=lambda r: float(r["pga_g"]))
    letters = ["A", "B", "C"]
    selections: List[Dict[str, object]] = []
    for i, row in enumerate(selected_rows):
        selections.append(
            {
                "button_label": f"Earthquake {letters[i]}",
                "global_idx_zero": int(row["global_idx_zero"]),
                "pred_local_idx": int(row["pred_local_idx"]),
                "global_record_id": int(row["global_record_id"]),
                "pga_g": float(row["pga_g"]),
                "category": str(row["category"]),
                "category_description": str(row["category_description"]),
            }
        )

    return selections, warnings


def infer_single_channel_floor_from_y_test(data: LoadedData) -> int | None:
    if data.y_test_test_order is None:
        return None
    if data.y_test_test_order.ndim != 3 or data.y_test_test_order.shape[2] != 1:
        return None
    if data.y_test_test_order.shape[0] != len(data.test_indices_zero_based):
        return None

    disp_test = data.displacement_full[data.test_indices_zero_based, CROP_START:, :6]
    n_time = min(disp_test.shape[1], data.y_test_test_order.shape[1])
    if n_time < 1:
        return None

    y_test_single = data.y_test_test_order[:, :n_time, 0]
    mse_by_floor = [
        float(np.mean((y_test_single - disp_test[:, :n_time, floor_idx]) ** 2))
        for floor_idx in range(6)
    ]
    return int(np.argmin(mse_by_floor))


def select_comparison_channels(
    actual_t_f: np.ndarray,
    pred_t_f: np.ndarray,
    single_channel_floor_idx: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], str | None, str, str]:
    pred_channels = pred_t_f.shape[1]

    if pred_channels == 6:
        labels = [f"Floor {i}" for i in range(1, 7)]
        return (
            actual_t_f[:, :6],
            pred_t_f[:, :6],
            labels,
            None,
            "FNO prediction",
            "Floor",
        )

    if pred_channels == 1:
        resolved_floor_idx = single_channel_floor_idx if single_channel_floor_idx in range(6) else 0
        warning = (
            "This FNO output file contains only one predicted response channel, "
            "so it cannot be used for a full six-floor comparison."
        )
        if single_channel_floor_idx in range(6):
            warning += f" The single channel aligns with Floor {resolved_floor_idx + 1}."
        return (
            actual_t_f[:, resolved_floor_idx : resolved_floor_idx + 1],
            pred_t_f[:, :1],
            [f"Floor {resolved_floor_idx + 1}"],
            warning,
            "Single-channel FNO prediction",
            "Floor",
        )

    shared = min(actual_t_f.shape[1], pred_channels)
    warning = (
        f"This FNO output file contains {pred_channels} predicted response channels; "
        f"showing {shared} channel(s) that overlap with available ground truth."
    )
    labels = [f"Response channel {i}" for i in range(1, shared + 1)]
    return (
        actual_t_f[:, :shared],
        pred_t_f[:, :shared],
        labels,
        warning,
        "FNO prediction",
        "Response channel",
    )


def compute_metrics(actual_t_f: np.ndarray, pred_t_f: np.ndarray, channel_labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    err = pred_t_f - actual_t_f
    mse_floor = np.mean(err**2, axis=0)
    rmse_floor = np.sqrt(mse_floor)
    maxabs_floor = np.max(np.abs(err), axis=0)

    denom_floor = np.linalg.norm(actual_t_f, axis=0)
    denom_floor = np.where(denom_floor < 1e-12, 1e-12, denom_floor)
    rel_floor = np.linalg.norm(err, axis=0) / denom_floor * 100.0

    by_floor = pd.DataFrame(
        {
            "Channel": channel_labels,
            "MSE": mse_floor,
            "RMSE": rmse_floor,
            "Max Absolute Error": maxabs_floor,
            "Relative Error %": rel_floor,
        }
    )

    mse_all = float(np.mean(err**2))
    rmse_all = float(np.sqrt(mse_all))
    maxabs_all = float(np.max(np.abs(err)))
    rel_all = float(np.linalg.norm(err) / max(np.linalg.norm(actual_t_f), 1e-12) * 100.0)

    overall = pd.DataFrame(
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

    return overall, by_floor


def building_animation_figure(
    t: np.ndarray,
    ground: np.ndarray,
    floor_disp_t_f: np.ndarray,
    title: str,
) -> Tuple[go.Figure, float]:
    n = len(t)
    step = max(1, n // MAX_ANIMATION_FRAMES)
    ids = np.arange(0, n, step)

    # Scale both base and floor offsets into an easy-to-view x-range.
    max_disp = float(np.max(np.abs(floor_disp_t_f)))
    d_scale = 1.0
    x_axis_half_range = max(4.0, max_disp * 4.0)

    y_levels = np.arange(0, 7)

    def frame_xy(k: int) -> Tuple[np.ndarray, np.ndarray]:
        base_x = 0.0
        floor_x = floor_disp_t_f[k] * d_scale
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
    return fig, d_scale


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
        yaxis_title="Displacement (dataset units)",
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
        data = load_all_data(str(workspace))
    except Exception as exc:
        st.error(f"Could not load required data files. Details: {exc}")
        st.stop()

    if data.y_pred_test_order is None:
        st.error(
            "No prediction file with variable 'y_pred' was found. Add Earthquake_responses_test.mat "
            "or Earthquake_responses_test_fno.mat to continue."
        )
        st.stop()

    if data.y_pred_test_order.shape[0] != len(data.test_indices_zero_based):
        st.error(
            "Prediction sample count does not match test indices. "
            f"Predictions: {data.y_pred_test_order.shape[0]}, tests: {len(data.test_indices_zero_based)}"
        )
        st.stop()

    single_channel_floor_idx = infer_single_channel_floor_from_y_test(data)

    pga_summary, pga_note = build_pga_summary(data)
    picked, selection_warnings = pick_representative_records(pga_summary)

    if len(picked) < 3:
        st.error("Could not select three representative earthquake records from available categories.")
        st.stop()

    st.markdown("### Screen 1: Earthquake Selection")
    st.caption("Representative records selected using USGS ShakeMap Instrumental Intensity PGA categories.")
    intensity_table = pd.DataFrame(
        [
            {"Shaking Category": "Moderate", "PGA Range (g)": "0.0276 - 0.115", "Description": "Moderate shaking; small non-structural effects can occur."},
            {"Shaking Category": "Strong", "PGA Range (g)": "0.115 - 0.215", "Description": "Strong shaking; non-structural damage becomes more likely."},
            {"Shaking Category": "Very Strong", "PGA Range (g)": "0.215 - 0.401", "Description": "Very strong shaking; structural damage may begin in vulnerable buildings."},
            {"Shaking Category": "Severe", "PGA Range (g)": "0.401 - 0.747", "Description": "Severe shaking; significant damage potential."},
            {"Shaking Category": "Violent", "PGA Range (g)": "0.747 - 1.39", "Description": "Violent shaking; heavy damage is likely."},
            {"Shaking Category": "Extreme", "PGA Range (g)": "> 1.39", "Description": "Extreme shaking; very destructive potential."},
        ]
    )
    st.dataframe(intensity_table, width="stretch", hide_index=True)
    button_cols = st.columns(3)
    for i, record in enumerate(picked):
        with button_cols[i]:
            if st.button(str(record["button_label"]), width="stretch"):
                st.session_state["selected_local_idx"] = int(record["pred_local_idx"])
            st.caption(f"{record['category']} | PGA: {record['pga_g']:.4f} g")

    if "selected_local_idx" not in st.session_state:
        st.session_state["selected_local_idx"] = int(picked[1]["pred_local_idx"])

    selected_local = int(st.session_state["selected_local_idx"])
    selected_meta = next((r for r in picked if int(r["pred_local_idx"]) == selected_local), None)
    if selected_meta is None:
        selected_meta = picked[1]
        st.session_state["selected_local_idx"] = int(selected_meta["pred_local_idx"])

    selected_global_record = int(selected_meta["global_idx_zero"])

    # Crop to match the training/test convention used by main.py and prediction files.
    time_series = data.time_full[selected_global_record, CROP_START:]
    if len(time_series) < 2:
        time_series = np.arange(data.ground_full.shape[1] - CROP_START, dtype=np.float64)

    ground = data.ground_full[selected_global_record, CROP_START:]
    actual = data.displacement_full[selected_global_record, CROP_START:, :]
    pred_raw = data.y_pred_test_order[selected_local]

    common_n = min(len(time_series), len(ground), actual.shape[0], pred_raw.shape[0])
    time_series = time_series[:common_n]
    ground = ground[:common_n]
    actual = actual[:common_n, :6]
    pred_raw = pred_raw[:common_n]

    compare_actual, compare_pred, compare_labels, compare_warning, pred_legend_label, label_prefix = (
        select_comparison_channels(actual, pred_raw, single_channel_floor_idx=single_channel_floor_idx)
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
        pga_val = float(np.max(np.abs(ground)))
        st.write(f"Dataset record id: {selected_global_record + 1}")
        st.write(f"Peak ground acceleration (|max|): {pga_val:.4f} g")
        st.write(f"ShakeMap category: {selected_meta['category']}")
        st.write(f"Description: {selected_meta['category_description']}")
        st.write(f"Prediction source: {data.prediction_file}")
        st.caption(pga_note)
        for msg in selection_warnings:
            st.caption(msg)
        if compare_warning:
            st.caption(compare_warning)

    st.markdown("### Screen 2: Building Animation")
    anim_fig, disp_scale = building_animation_figure(
        t=time_series,
        ground=ground,
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

    overall, by_floor = compute_metrics(compare_actual, compare_pred, compare_labels)

    st.markdown("### Prediction Error Metrics")
    st.dataframe(overall, width="stretch", hide_index=True)

if __name__ == "__main__":
    main()
