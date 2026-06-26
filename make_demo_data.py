from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io


CROP_START = 21
PREDICTION_CANDIDATES = [
    "Earthquake_responses_test.mat",
    "Earthquake_responses_test_fno.mat",
]
SHAKEMAP_LEVELS = [
    {"name": "Below Moderate", "low": 0.0, "high": 0.0276, "description": "Usually weak or light shaking."},
    {"name": "Moderate", "low": 0.0276, "high": 0.115, "description": "Moderate shaking; small non-structural effects can occur."},
    {"name": "Strong", "low": 0.115, "high": 0.215, "description": "Strong shaking; non-structural damage becomes more likely."},
    {"name": "Very Strong", "low": 0.215, "high": 0.401, "description": "Very strong shaking; structural damage may begin in vulnerable buildings."},
    {"name": "Severe", "low": 0.401, "high": 0.747, "description": "Severe shaking; significant damage potential."},
    {"name": "Violent", "low": 0.747, "high": 1.39, "description": "Violent shaking; heavy damage is likely."},
    {"name": "Extreme", "low": 1.39, "high": np.inf, "description": "Extreme shaking; very destructive potential."},
]


def assign_shakemap_category(pga_g: float) -> Tuple[str, int, str]:
    for rank, level in enumerate(SHAKEMAP_LEVELS):
        if level["low"] <= pga_g < level["high"]:
            return str(level["name"]), rank, str(level["description"])
    tail = SHAKEMAP_LEVELS[-1]
    return str(tail["name"]), len(SHAKEMAP_LEVELS) - 1, str(tail["description"])


def load_prediction_data(root: Path) -> Tuple[str | None, np.ndarray | None, np.ndarray | None]:
    for candidate in PREDICTION_CANDIDATES:
        p = root / candidate
        if not p.exists():
            continue

        data = scipy.io.loadmat(str(p))
        if "y_pred" not in data:
            continue

        y_pred = np.asarray(data["y_pred"])
        if y_pred.ndim == 2:
            y_pred = y_pred[:, :, None]
        if y_pred.ndim != 3:
            continue

        y_test = None
        if "y_test" in data:
            y_test = np.asarray(data["y_test"])
            if y_test.ndim == 2:
                y_test = y_test[:, :, None]
            if y_test.ndim != 3:
                y_test = None

        return candidate, y_pred.astype(np.float64), None if y_test is None else y_test.astype(np.float64)

    return None, None, None


def infer_single_channel_floor(
    displacement_full: np.ndarray,
    test_indices_zero_based: np.ndarray,
    y_test_test_order: np.ndarray | None,
) -> int:
    if y_test_test_order is None:
        return 0
    if y_test_test_order.ndim != 3 or y_test_test_order.shape[2] != 1:
        return 0
    if y_test_test_order.shape[0] != len(test_indices_zero_based):
        return 0

    disp_test = displacement_full[test_indices_zero_based, CROP_START:, :6]
    n_time = min(disp_test.shape[1], y_test_test_order.shape[1])
    if n_time < 1:
        return 0

    y_test_single = y_test_test_order[:, :n_time, 0]
    mse_by_floor = [
        float(np.mean((y_test_single - disp_test[:, :n_time, floor_idx]) ** 2))
        for floor_idx in range(6)
    ]
    return int(np.argmin(mse_by_floor))


def pick_representative_rows(rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], List[str]]:
    warnings: List[str] = []
    prediction_ready = [r for r in rows if bool(r["has_prediction"])]
    if not prediction_ready:
        raise RuntimeError("No records with both ground truth and FNO predictions are available.")

    available_ranks = sorted({int(r["category_rank"]) for r in prediction_ready})
    if len(available_ranks) >= 3:
        target_ranks = [
            int(available_ranks[0]),
            int(available_ranks[len(available_ranks) // 2]),
            int(available_ranks[-1]),
        ]
    elif len(available_ranks) == 2:
        target_ranks = [int(available_ranks[0]), int(available_ranks[1]), int(available_ranks[1])]
        warnings.append(
            "Only two ShakeMap categories are present in prediction-ready records; one category is reused."
        )
    else:
        target_ranks = [int(available_ranks[0]), int(available_ranks[0]), int(available_ranks[0])]
        warnings.append(
            "Only one ShakeMap category is present in prediction-ready records; all three selections come from it."
        )

    selected: List[Dict[str, object]] = []
    used_pred_locals: set[int] = set()
    for rank in target_ranks:
        in_cat = sorted(
            [r for r in prediction_ready if int(r["category_rank"]) == rank],
            key=lambda x: float(x["pga_g"]),
        )
        if not in_cat:
            continue

        candidate_order = [len(in_cat) // 2] + list(range(len(in_cat)))
        chosen = None
        for cand_i in candidate_order:
            row = in_cat[int(cand_i)]
            pred_local_idx = int(row["pred_local_idx"])
            if pred_local_idx not in used_pred_locals:
                chosen = row
                break
        if chosen is None:
            chosen = in_cat[len(in_cat) // 2]

        used_pred_locals.add(int(chosen["pred_local_idx"]))
        selected.append(chosen)

    selected = sorted(selected[:3], key=lambda x: float(x["pga_g"]))
    if len(selected) != 3:
        raise RuntimeError(f"Expected 3 representative records, found {len(selected)}")

    return selected, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Create small Streamlit demo dataset from large local MAT files.")
    parser.add_argument("--dataset", default="dataset_EQ.mat", help="Path to full dataset MAT file.")
    parser.add_argument("--index", default="train_test_index.mat", help="Path to train/test index MAT file.")
    parser.add_argument("--output", default="data/app_demo_data.npz", help="Output NPZ path.")
    args = parser.parse_args()

    root = Path.cwd()
    dataset_path = root / args.dataset
    index_path = root / args.index
    output_path = root / args.output

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    dataset = scipy.io.loadmat(str(dataset_path))
    indices = scipy.io.loadmat(str(index_path))

    ground_full = np.asarray(dataset["ground_motion"], dtype=np.float64)
    displacement_full = np.asarray(dataset["displacement"], dtype=np.float64)
    time_full = np.asarray(dataset["time"], dtype=np.float64)
    test_indices_zero_based = np.asarray(indices["test"]).astype(int).ravel() - 1

    prediction_file, y_pred_test_order, y_test_test_order = load_prediction_data(root)
    if y_pred_test_order is None:
        raise RuntimeError("No valid prediction file with y_pred was found.")
    if y_pred_test_order.shape[0] != len(test_indices_zero_based):
        raise RuntimeError(
            "Prediction sample count does not match test indices: "
            f"predictions={y_pred_test_order.shape[0]} tests={len(test_indices_zero_based)}"
        )

    pga_all = np.max(np.abs(ground_full[:, CROP_START:]), axis=1)
    pred_lookup = {int(g_idx): i for i, g_idx in enumerate(test_indices_zero_based)}

    rows: List[Dict[str, object]] = []
    for global_idx in range(ground_full.shape[0]):
        pga_val = float(pga_all[global_idx])
        category_name, category_rank, category_description = assign_shakemap_category(pga_val)
        pred_local_idx = pred_lookup.get(global_idx, -1)
        rows.append(
            {
                "global_idx_zero": int(global_idx),
                "global_record_id": int(global_idx) + 1,
                "pga_g": pga_val,
                "category": category_name,
                "category_rank": int(category_rank),
                "category_description": category_description,
                "has_prediction": pred_local_idx >= 0,
                "pred_local_idx": int(pred_local_idx),
            }
        )

    selected_rows, selection_warnings = pick_representative_rows(rows)

    selected_global_idx = np.array([int(r["global_idx_zero"]) for r in selected_rows], dtype=np.int64)
    selected_pred_local = np.array([int(r["pred_local_idx"]) for r in selected_rows], dtype=np.int64)

    time_sel = time_full[selected_global_idx, CROP_START:]
    ground_sel = ground_full[selected_global_idx, CROP_START:]
    actual_sel = displacement_full[selected_global_idx, CROP_START:, :6]
    pred_sel = y_pred_test_order[selected_pred_local]

    common_lengths = []
    for i in range(3):
        common_lengths.append(
            min(time_sel.shape[1], ground_sel.shape[1], actual_sel.shape[1], pred_sel.shape[1])
        )
    n_time = int(min(common_lengths))

    time_sel = time_sel[:, :n_time]
    ground_sel = ground_sel[:, :n_time]
    actual_sel = actual_sel[:, :n_time, :]
    pred_sel = pred_sel[:, :n_time, :]

    single_channel_floor_idx = infer_single_channel_floor(
        displacement_full=displacement_full,
        test_indices_zero_based=test_indices_zero_based,
        y_test_test_order=y_test_test_order,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time=time_sel,
        ground_motion=ground_sel,
        actual_displacement=actual_sel,
        y_pred=pred_sel,
        pga_g=np.array([float(r["pga_g"]) for r in selected_rows], dtype=np.float64),
        category=np.array([str(r["category"]) for r in selected_rows], dtype="U32"),
        category_description=np.array([str(r["category_description"]) for r in selected_rows], dtype="U128"),
        record_id=np.array([int(r["global_record_id"]) for r in selected_rows], dtype=np.int64),
        prediction_file=np.array([prediction_file or ""], dtype="U128"),
        selection_warnings=np.array(selection_warnings, dtype="U256"),
        pga_note=np.array(
            [
                "PGA/category assignment is computed from all records in dataset_EQ.mat; "
                "the three demo records are selected from prediction-ready records."
            ],
            dtype="U256",
        ),
        single_channel_floor_idx=np.array([single_channel_floor_idx], dtype=np.int64),
        crop_start=np.array([CROP_START], dtype=np.int64),
    )

    print(f"Wrote demo data: {output_path}")
    for i, row in enumerate(selected_rows):
        print(
            f"{chr(ord('A') + i)}: record_id={row['global_record_id']} "
            f"PGA={row['pga_g']:.4f}g category={row['category']}"
        )


if __name__ == "__main__":
    main()
