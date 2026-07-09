#!/usr/bin/env python3
"""Create Fig. 5 from a clean packaged data file.

Default use:
    python make_figure5.py

This reads "FIG5 data.zip" and writes FIG5.svg, FIG5.pdf, and FIG5.png.
If the data package is missing, or if --rebuild-data is passed, the script
rebuilds the package from the saved PyTorch result files in the same folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


SIMULATION_FILE = "simulation_results_50trials_N16_B15_A10.pt"
TIMING_FILE = "timing_stats_N16_B15_A10.pt"
INVERSE_TIMING_FILE = "timing_benchmark_interleaved.pt"
DEFAULT_PACKAGE = "FIG5 data.zip"
DEFAULT_PREFIX = "FIG5"
INVERSE_DATASET_SIZE = 16000

BLUE = "#0072B2"
RED = "#B2182B"
NEUTRAL = "#333333"
LIGHT_NEUTRAL = "#777777"
VIRIDIS_PURPLE = "#440154"
VIRIDIS_TEAL = "#21918C"
VIRIDIS_GREEN = "#35B779"
JOINT_COLORS = [BLUE, RED]
JOINT_LABELS = ["Joint 1", "Joint 2"]


def to_numpy(value):
    """Convert tensor-like or array-like values to a NumPy array."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_summary(array: np.ndarray) -> dict:
    finite = np.asarray(array)[np.isfinite(array)]
    return {
        "shape": list(np.asarray(array).shape),
        "dtype": str(np.asarray(array).dtype),
        "finite_count": int(finite.size),
        "min": float(finite.min()) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
    }


def validate_data(data: dict[str, np.ndarray]) -> list[str]:
    required = {
        "trial_time_s": 1,
        "true_position_rad": 3,
        "neural_position_rad": 3,
        "true_velocity_rad_s": 3,
        "neural_velocity_rad_s": 3,
        "true_acceleration_rad_s2": 3,
        "neural_acceleration_rad_s2": 3,
        "position_rmse_rad": 2,
        "velocity_rmse_rad_s": 2,
        "acceleration_rmse_rad_s2": 2,
        "trajectory_time_ode_s": 1,
        "trajectory_time_neural_solver_s": 1,
        "forward_latency_ms": 1,
        "inverse_latency_16k_ms": 1,
    }

    report = []
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing required arrays: {missing}")

    for key, ndim in required.items():
        array = np.asarray(data[key])
        if array.ndim != ndim:
            raise ValueError(f"{key} has ndim={array.ndim}; expected {ndim}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{key} contains non-finite values")
        report.append(f"{key}: shape={array.shape}, dtype={array.dtype}")

    n_trials, n_time, n_joints = data["true_position_rad"].shape
    if n_joints != 2:
        raise ValueError(f"Expected 2 joints; found {n_joints}")
    if data["trial_time_s"].shape != (n_time,):
        raise ValueError("trial_time_s length does not match trajectory arrays")

    trajectory_arrays = [
        "neural_position_rad",
        "true_velocity_rad_s",
        "neural_velocity_rad_s",
        "true_acceleration_rad_s2",
        "neural_acceleration_rad_s2",
    ]
    for key in trajectory_arrays:
        if data[key].shape != (n_trials, n_time, n_joints):
            raise ValueError(f"{key} shape {data[key].shape} does not match trajectories")

    rmse_arrays = [
        "position_rmse_rad",
        "velocity_rmse_rad_s",
        "acceleration_rmse_rad_s2",
    ]
    for key in rmse_arrays:
        if data[key].shape != (n_trials, n_joints):
            raise ValueError(f"{key} shape {data[key].shape} does not match trial/joint count")

    timing_arrays = ["trajectory_time_ode_s", "trajectory_time_neural_solver_s"]
    for key in timing_arrays:
        if data[key].shape != (n_trials,):
            raise ValueError(f"{key} shape {data[key].shape} does not match trial count")
        if np.any(data[key] <= 0):
            raise ValueError(f"{key} contains non-positive timings")

    for key in ["forward_latency_ms", "inverse_latency_16k_ms"]:
        if np.any(data[key] <= 0):
            raise ValueError(f"{key} contains non-positive timings")

    if not np.all(np.diff(data["trial_time_s"]) > 0):
        raise ValueError("trial_time_s must be strictly increasing")

    return report


def compute_demo_index(data: dict[str, np.ndarray]) -> int:
    normalized_rmse = np.concatenate(
        [
            data["position_rmse_rad"] / np.median(data["position_rmse_rad"], axis=0),
            data["velocity_rmse_rad_s"] / np.median(data["velocity_rmse_rad_s"], axis=0),
            data["acceleration_rmse_rad_s2"]
            / np.median(data["acceleration_rmse_rad_s2"], axis=0),
        ],
        axis=1,
    )
    trajectory_score = normalized_rmse.mean(axis=1)
    return int(np.argmin(np.abs(trajectory_score - np.median(trajectory_score))))


def load_source_data(source_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required only when rebuilding FIG5 data.zip from .pt files. "
            "Install torch or provide an existing FIG5 data.zip package."
        ) from exc

    source_paths = {
        "simulation": source_dir / SIMULATION_FILE,
        "trajectory_timing": source_dir / TIMING_FILE,
        "inverse_timing": source_dir / INVERSE_TIMING_FILE,
    }
    for label, path in source_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {label} source file: {path}")

    simulation = torch.load(source_paths["simulation"], map_location="cpu")
    timing = torch.load(source_paths["trajectory_timing"], map_location="cpu")
    inverse_timing = torch.load(source_paths["inverse_timing"], map_location="cpu")
    benchmark_data = inverse_timing["benchmark_data"]
    benchmark_key = (
        INVERSE_DATASET_SIZE
        if INVERSE_DATASET_SIZE in benchmark_data
        else str(INVERSE_DATASET_SIZE)
    )

    data = {
        "trial_time_s": to_numpy(simulation["trial_time"]).astype(np.float64),
        "true_position_rad": to_numpy(simulation["all_true_q"]).astype(np.float32),
        "neural_position_rad": to_numpy(simulation["all_pred_q"]).astype(np.float32),
        "true_velocity_rad_s": to_numpy(simulation["all_true_dq"]).astype(np.float32),
        "neural_velocity_rad_s": to_numpy(simulation["all_pred_dq"]).astype(np.float32),
        "true_acceleration_rad_s2": to_numpy(simulation["all_true_ddq"]).astype(np.float32),
        "neural_acceleration_rad_s2": to_numpy(simulation["all_pred_ddq"]).astype(np.float32),
        "position_rmse_rad": to_numpy(simulation["rmse_pos"]).astype(np.float32),
        "velocity_rmse_rad_s": to_numpy(simulation["rmse_vel"]).astype(np.float32),
        "acceleration_rmse_rad_s2": to_numpy(simulation["rmse_acc"]).astype(np.float32),
        "trajectory_time_ode_s": to_numpy(timing["t_gen_true_per_traj"]).astype(np.float64),
        "trajectory_time_neural_solver_s": to_numpy(timing["t_gen_sim_per_traj"]).astype(np.float64),
        "forward_latency_ms": (
            to_numpy(timing["t_neural_calls_all"]).astype(np.float64) * 1000.0
        ),
        "inverse_latency_16k_ms": to_numpy(benchmark_data[benchmark_key]).astype(np.float64),
    }

    validate_data(data)
    demo_index = compute_demo_index(data)
    dt = float(np.median(np.diff(data["trial_time_s"])))
    duration = float(data["trial_time_s"][-1] - data["trial_time_s"][0])

    metadata = {
        "title": "Fig. 5 data package",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            label: {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for label, path in source_paths.items()
        },
        "array_descriptions": {
            "trial_time_s": "Time vector for each trajectory, in seconds.",
            "true_position_rad": "Numerical ODE angular position, shape trials x time x joints.",
            "neural_position_rad": "Optimization-based neural solver angular position, shape trials x time x joints.",
            "true_velocity_rad_s": "Numerical ODE angular velocity, shape trials x time x joints.",
            "neural_velocity_rad_s": "Optimization-based neural solver angular velocity, shape trials x time x joints.",
            "true_acceleration_rad_s2": "Numerical ODE angular acceleration, shape trials x time x joints.",
            "neural_acceleration_rad_s2": "Optimization-based neural solver angular acceleration, shape trials x time x joints.",
            "position_rmse_rad": "Trajectory-wise position RMSE, shape trials x joints.",
            "velocity_rmse_rad_s": "Trajectory-wise velocity RMSE, shape trials x joints.",
            "acceleration_rmse_rad_s2": "Trajectory-wise acceleration RMSE, shape trials x joints.",
            "trajectory_time_ode_s": "Wall-clock time to generate one trajectory with the numerical ODE solver.",
            "trajectory_time_neural_solver_s": "Wall-clock time to generate one trajectory with the neural solver.",
            "forward_latency_ms": "Single-query latency of the optimized neural forward solver, in milliseconds.",
            "inverse_latency_16k_ms": "Single-query latency of the inverse model trained on 16k samples, in milliseconds.",
        },
        "joint_labels": JOINT_LABELS,
        "demo_trajectory_index": demo_index,
        "trajectory_duration_s": duration,
        "time_step_s": dt,
        "inverse_dataset_size": INVERSE_DATASET_SIZE,
        "array_summaries": {key: array_summary(value) for key, value in data.items()},
    }
    return data, metadata


def write_text_to_zip(zip_handle: zipfile.ZipFile, name: str, text: str) -> None:
    zip_handle.writestr(name, text.encode("utf-8"))


def dataframe_csv_bytes(header: list[str], rows: list[list]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_data_package(data: dict[str, np.ndarray], metadata: dict, package_path: Path) -> None:
    validate_data(data)
    package_path.parent.mkdir(parents=True, exist_ok=True)

    npz_buffer = io.BytesIO()
    np.savez_compressed(
        npz_buffer,
        **data,
        metadata_json=np.array(json.dumps(metadata, indent=2)),
    )

    demo_index = int(metadata["demo_trajectory_index"])
    time = data["trial_time_s"]
    demo_rows = []
    for i, time_s in enumerate(time):
        demo_rows.append(
            [
                int(i),
                float(time_s),
                float(data["true_position_rad"][demo_index, i, 0]),
                float(data["true_position_rad"][demo_index, i, 1]),
                float(data["neural_position_rad"][demo_index, i, 0]),
                float(data["neural_position_rad"][demo_index, i, 1]),
                float(data["true_velocity_rad_s"][demo_index, i, 0]),
                float(data["true_velocity_rad_s"][demo_index, i, 1]),
                float(data["neural_velocity_rad_s"][demo_index, i, 0]),
                float(data["neural_velocity_rad_s"][demo_index, i, 1]),
                float(data["true_acceleration_rad_s2"][demo_index, i, 0]),
                float(data["true_acceleration_rad_s2"][demo_index, i, 1]),
                float(data["neural_acceleration_rad_s2"][demo_index, i, 0]),
                float(data["neural_acceleration_rad_s2"][demo_index, i, 1]),
            ]
        )

    rmse_rows = []
    n_trials = data["position_rmse_rad"].shape[0]
    for trial_index in range(n_trials):
        for joint_index, joint_label in enumerate(JOINT_LABELS):
            rmse_rows.append(
                [
                    trial_index,
                    joint_index + 1,
                    joint_label,
                    float(data["position_rmse_rad"][trial_index, joint_index]),
                    float(data["velocity_rmse_rad_s"][trial_index, joint_index]),
                    float(data["acceleration_rmse_rad_s2"][trial_index, joint_index]),
                ]
            )

    trajectory_timing_rows = [
        [
            trial_index,
            float(data["trajectory_time_ode_s"][trial_index]),
            float(data["trajectory_time_neural_solver_s"][trial_index]),
        ]
        for trial_index in range(n_trials)
    ]

    latency_rows = []
    for label, key in [
        ("forward_neural_solver", "forward_latency_ms"),
        ("inverse_model_16k", "inverse_latency_16k_ms"),
    ]:
        values = data[key]
        latency_rows.append(
            [
                label,
                int(values.size),
                float(np.min(values)),
                float(np.percentile(values, 25)),
                float(np.median(values)),
                float(np.percentile(values, 75)),
                float(np.percentile(values, 99)),
                float(np.max(values)),
            ]
        )

    readme = """FIG5 data package

This zip contains the complete numerical data needed to recreate Fig. 5.

Files:
- figure5_data.npz: compressed NumPy arrays used directly by make_figure5.py.
- metadata.json: source-file hashes, array descriptions, units, shapes, and summaries.
- rmse_values.csv: trajectory-wise RMSE values used in panel b.
- trajectory_timing.csv: per-trajectory ODE and neural-solver execution times used in panel c.
- latency_summary.csv: summary statistics for the latency samples used in panel d.
- demo_trajectory.csv: the representative trajectory plotted in panel a.

The full 50-trajectory time-series arrays are stored in figure5_data.npz to keep
the data package compact and lossless.
"""

    with zipfile.ZipFile(package_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("figure5_data.npz", npz_buffer.getvalue())
        write_text_to_zip(zf, "metadata.json", json.dumps(metadata, indent=2))
        write_text_to_zip(zf, "README.txt", readme)
        zf.writestr(
            "rmse_values.csv",
            dataframe_csv_bytes(
                [
                    "trial_index",
                    "joint_index",
                    "joint_label",
                    "position_rmse_rad",
                    "velocity_rmse_rad_s",
                    "acceleration_rmse_rad_s2",
                ],
                rmse_rows,
            ),
        )
        zf.writestr(
            "trajectory_timing.csv",
            dataframe_csv_bytes(
                ["trial_index", "ode_solver_s", "neural_solver_s"],
                trajectory_timing_rows,
            ),
        )
        zf.writestr(
            "latency_summary.csv",
            dataframe_csv_bytes(
                ["measurement", "n", "min_ms", "q1_ms", "median_ms", "q3_ms", "p99_ms", "max_ms"],
                latency_rows,
            ),
        )
        zf.writestr(
            "demo_trajectory.csv",
            dataframe_csv_bytes(
                [
                    "time_index",
                    "time_s",
                    "numerical_position_joint1_rad",
                    "numerical_position_joint2_rad",
                    "neural_position_joint1_rad",
                    "neural_position_joint2_rad",
                    "numerical_velocity_joint1_rad_s",
                    "numerical_velocity_joint2_rad_s",
                    "neural_velocity_joint1_rad_s",
                    "neural_velocity_joint2_rad_s",
                    "numerical_acceleration_joint1_rad_s2",
                    "numerical_acceleration_joint2_rad_s2",
                    "neural_acceleration_joint1_rad_s2",
                    "neural_acceleration_joint2_rad_s2",
                ],
                demo_rows,
            ),
        )


def read_data_package(package_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with zipfile.ZipFile(package_path, mode="r") as zf:
        with zf.open("figure5_data.npz") as npz_file:
            npz_bytes = io.BytesIO(npz_file.read())
    with np.load(npz_bytes, allow_pickle=False) as npz:
        metadata = json.loads(str(npz["metadata_json"]))
        data = {key: npz[key] for key in npz.files if key != "metadata_json"}
    validate_data(data)
    return data, metadata


def set_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
        }
    )


def style_axis(ax, spine_color: str | None = None, right: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    if right:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(True)
    else:
        ax.spines["right"].set_visible(False)
    if spine_color is not None:
        side = "right" if right else "left"
        ax.spines[side].set_color(spine_color)
        ax.tick_params(axis="y", colors=spine_color)
    ax.tick_params(axis="both", which="major", direction="out", length=2.5, width=0.5, pad=2)
    ax.tick_params(axis="both", which="minor", direction="out", length=1.5, width=0.4)
    ax.grid(False)


def add_panel_label(ax, label: str, x: float = -0.22, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def nice_step(value: float) -> float:
    if value <= 0 or not np.isfinite(value):
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10**exponent)
    for candidate in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        if fraction <= candidate:
            return float(candidate * (10**exponent))
    return float(10.0 * (10**exponent))


def nice_upper_limit(values, n_intervals: int = 3) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    max_value = float(values.max())
    step = nice_step(max_value / n_intervals)
    return step * n_intervals


def nice_padded_limits(values, n_intervals: int = 3, pad_fraction: float = 0.12) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    pad = span * pad_fraction if span > 0 else max(abs(upper), 1.0) * pad_fraction
    step = nice_step((span + 2 * pad) / n_intervals)
    nice_lower = np.floor((lower - pad) / step) * step
    nice_upper = np.ceil((upper + pad) / step) * step
    return float(nice_lower), float(nice_upper)


def nice_symmetric_limits(values, n_intervals: int = 4) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    max_abs = float(np.max(np.abs(values)))
    step = nice_step((2 * max_abs) / n_intervals)
    limit = step * n_intervals / 2
    return float(-limit), float(limit)


def tick_decimals(ticks) -> int:
    ticks = np.asarray(ticks, dtype=float)
    if ticks.size < 2:
        return 0
    step = float(np.min(np.abs(np.diff(ticks))))
    if step <= 0 or not np.isfinite(step):
        return 0
    for decimals in range(5):
        rounded = np.round(ticks, decimals)
        if np.max(np.abs(rounded - ticks)) <= max(step * 1e-6, 1e-12):
            return decimals
    return 4


def set_endpoint_yticks(ax, n_ticks: int = 4) -> None:
    lower, upper = ax.get_ylim()
    ticks = np.linspace(lower, upper, n_ticks)
    decimals = tick_decimals(ticks)
    labels = []
    for tick in ticks:
        if abs(tick) < 10 ** (-(decimals + 1)):
            tick = 0.0
        labels.append(f"{tick:.{decimals}f}")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


def set_endpoint_xticks(ax, n_ticks: int = 4, log: bool = False) -> None:
    lower, upper = ax.get_xlim()
    if log:
        ticks = np.geomspace(lower, upper, n_ticks)
    else:
        ticks = np.linspace(lower, upper, n_ticks)
    decimals = tick_decimals(ticks)
    labels = []
    for tick in ticks:
        if abs(tick) < 10 ** (-(decimals + 1)):
            tick = 0.0
        labels.append(f"{tick:.{decimals}f}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def compact_tick_label(value: float) -> str:
    if value >= 1:
        return f"{value:g}"
    if value >= 0.1:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def plot_editable_points(
    ax,
    x_values,
    y_values,
    color: str,
    markersize: float,
    alpha: float,
    zorder: int = 3,
) -> None:
    for x_value, y_value in zip(np.asarray(x_values), np.asarray(y_values)):
        ax.plot(
            [float(x_value)],
            [float(y_value)],
            marker="o",
            markersize=markersize,
            markerfacecolor=color,
            markeredgecolor="none",
            markeredgewidth=0,
            linestyle="None",
            alpha=alpha,
            zorder=zorder,
            rasterized=False,
        )


def padded_limits(values, pad_fraction: float = 0.12, lower_floor: float | None = None) -> tuple[float, float]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    lower = float(values.min())
    upper = float(values.max())
    pad = (upper - lower) * pad_fraction if upper > lower else max(abs(upper), 1.0) * pad_fraction
    lower -= pad
    upper += pad
    if lower_floor is not None:
        lower = max(lower_floor, lower)
    return lower, upper


def boxplot_display_limits(values, pad_fraction: float = 0.40, lower_floor: float | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_whisker = values[values >= q1 - 1.5 * iqr].min()
    upper_whisker = values[values <= q3 + 1.5 * iqr].max()
    span = upper_whisker - lower_whisker
    pad = span * pad_fraction
    if pad == 0:
        pad = max(abs(float(np.median(values))) * 0.015, 1e-9)
    step = nice_step((span + 2 * pad) / 3)
    lower = float(np.floor((lower_whisker - pad) / step) * step)
    upper = float(np.ceil((upper_whisker + pad) / step) * step)
    if lower_floor is not None:
        lower = max(lower_floor, lower)
    return lower, upper


def draw_violin(
    ax,
    values,
    position: float,
    color: str,
    width: float,
    rng: np.random.Generator,
    jitter: float,
    markersize: float,
    alpha: float,
    fill_alpha: float = 0.12,
    edge_alpha: float = 0.86,
    summary_color: str | None = None,
    summary_linewidth: float = 0.85,
) -> float:
    values = np.asarray(values, dtype=float)
    violin = ax.violinplot(
        [values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor(mpl.colors.to_rgba(color, fill_alpha))
    body.set_edgecolor(mpl.colors.to_rgba(color, edge_alpha))
    body.set_linewidth(0.55)
    body.set_alpha(None)
    body.set_zorder(1)
    body.set_rasterized(False)

    x = position + rng.normal(0, jitter, size=values.size)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    plot_editable_points(ax, x, values, color, markersize=markersize, alpha=alpha, zorder=2)
    summary_color = color if summary_color is None else summary_color
    ax.plot(
        [position, position],
        [q1, q3],
        color=summary_color,
        linewidth=summary_linewidth,
        solid_capstyle="round",
        zorder=4,
    )
    ax.plot(
        [position - width * 0.25, position + width * 0.25],
        [median, median],
        color=NEUTRAL,
        linewidth=0.65,
        solid_capstyle="round",
        zorder=5,
    )
    return float(median)


def draw_boxplot(ax, values, position: float, color: str, width: float) -> float:
    values = np.asarray(values, dtype=float)
    artists = ax.boxplot(
        [values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        manage_ticks=False,
        boxprops={"facecolor": mpl.colors.to_rgba(color, 0.14), "edgecolor": color, "linewidth": 0.75},
        medianprops={"color": NEUTRAL, "linewidth": 0.85},
        whiskerprops={"color": color, "linewidth": 0.65},
        capprops={"color": color, "linewidth": 0.65},
    )
    for group in artists.values():
        for artist in group:
            artist.set_rasterized(False)
    return float(np.median(values))


def interval_stats(values) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    lower = values[values >= q1 - 1.5 * iqr].min()
    upper = values[values <= q3 + 1.5 * iqr].max()
    return {
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "lower": float(lower),
        "upper": float(upper),
    }


def plot_horizontal_summary(
    ax,
    series: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    xlabel: str,
    panel_label: str,
    xlim: tuple[float, float],
    xticks: list[float] | None = None,
    show_points: bool = False,
    rng_seed: int = 17,
) -> None:
    y_positions = np.arange(len(series))[::-1].astype(float)
    rng = np.random.default_rng(rng_seed)

    for y_pos, values, label, color in zip(y_positions, series, labels, colors):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        stats = interval_stats(values)

        if show_points:
            jitter = rng.normal(0, 0.045, size=values.size)
            for x_value, y_jitter in zip(values, jitter):
                ax.plot(
                    [float(x_value)],
                    [float(y_pos + y_jitter)],
                    marker="o",
                    markersize=2.0,
                    markerfacecolor=color,
                    markeredgecolor="none",
                    linestyle="None",
                    alpha=0.20,
                    zorder=2,
                    rasterized=False,
                )

        ax.plot(
            [stats["lower"], stats["upper"]],
            [y_pos, y_pos],
            color=mpl.colors.to_rgba(color, 0.62),
            linewidth=0.8,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [stats["q1"], stats["q3"]],
            [y_pos, y_pos],
            color=mpl.colors.to_rgba(color, 0.92),
            linewidth=4.2,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            [stats["median"]],
            [y_pos],
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.95,
            linestyle="None",
            zorder=5,
            rasterized=False,
        )

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    if xticks is None:
        set_endpoint_xticks(ax, n_ticks=4, log=True)
    else:
        ax.set_xticks(xticks)
        ax.set_xticklabels([compact_tick_label(tick) for tick in xticks])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    for tick_label, color in zip(ax.get_yticklabels(), colors):
        tick_label.set_color(color)
    ax.set_ylim(-0.55, len(series) - 0.45)
    ax.set_xlabel(xlabel)
    style_axis(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    add_panel_label(ax, panel_label, x=-0.16, y=1.12)


def plot_vertical_interval_summary(
    ax,
    series: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    ylabel: str,
    panel_label: str,
    ylim: tuple[float, float],
    yticks: list[float],
    log_y: bool = False,
    show_points: bool = False,
    rng_seed: int = 17,
    realtime_line: float | None = None,
    realtime_label: str | None = None,
) -> None:
    positions = np.arange(1, len(series) + 1, dtype=float)
    rng = np.random.default_rng(rng_seed)

    for position, values, color in zip(positions, series, colors):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        stats = interval_stats(values)

        if show_points:
            jitter = rng.normal(0, 0.030, size=values.size)
            for x_offset, y_value in zip(jitter, values):
                ax.plot(
                    [float(position + x_offset)],
                    [float(y_value)],
                    marker="o",
                    markersize=1.9,
                    markerfacecolor=color,
                    markeredgecolor="none",
                    linestyle="None",
                    alpha=0.20,
                    zorder=2,
                    rasterized=False,
                )

        ax.plot(
            [position, position],
            [stats["lower"], stats["upper"]],
            color=mpl.colors.to_rgba(color, 0.62),
            linewidth=0.8,
            solid_capstyle="round",
            zorder=3,
        )
        ax.plot(
            [position, position],
            [stats["q1"], stats["q3"]],
            color=mpl.colors.to_rgba(color, 0.92),
            linewidth=4.0,
            solid_capstyle="round",
            zorder=4,
        )
        ax.plot(
            [position],
            [stats["median"]],
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.9,
            linestyle="None",
            zorder=5,
            rasterized=False,
        )

    if log_y:
        ax.set_yscale("log")
    ax.set_xlim(0.55, len(series) + 0.45)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([compact_tick_label(tick) for tick in yticks])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    for label, color in zip(ax.get_xticklabels(), colors):
        label.set_color(color)
    ax.set_ylabel(ylabel)
    style_axis(ax)
    if realtime_line is not None:
        ax.axhline(
            realtime_line,
            color=LIGHT_NEUTRAL,
            linewidth=0.55,
            linestyle=(0, (2.0, 1.6)),
            zorder=1,
        )
        if realtime_label is not None:
            ax.text(
                len(series) + 0.34,
                realtime_line,
                realtime_label,
                ha="right",
                va="bottom",
                fontsize=5.5,
                color=LIGHT_NEUTRAL,
            )
    add_panel_label(ax, panel_label, x=-0.22, y=1.12)


def plot_trajectory_panel(ax, time, true_values, neural_values, ylabel: str, panel_label: str | None = None) -> None:
    trace_slice = slice(None)
    tt = time[trace_slice]
    for joint_idx, color in enumerate(JOINT_COLORS):
        ax.plot(
            tt,
            true_values[trace_slice, joint_idx],
            color=color,
            linewidth=0.58,
            linestyle=(0, (2.2, 1.6)),
            alpha=0.58,
            solid_capstyle="round",
            rasterized=False,
        )
        ax.plot(
            tt,
            neural_values[trace_slice, joint_idx],
            color=color,
            linewidth=0.78,
            linestyle="-",
            alpha=0.96,
            solid_capstyle="round",
            rasterized=False,
        )
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(time.min()), float(time.max()))
    lower, upper = nice_symmetric_limits([true_values, neural_values], n_intervals=4)
    ax.set_ylim(lower, upper)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    set_endpoint_yticks(ax, n_ticks=5)
    style_axis(ax)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.20, y=1.20)


def plot_rmse_panel(ax, rmse_values, ylabel: str, panel_label: str | None = None, show_x: bool = False) -> None:
    positions = np.asarray([1.0, 1.55])
    rng = np.random.default_rng(23)
    for joint_idx, color in enumerate(JOINT_COLORS):
        draw_violin(
            ax,
            rmse_values[:, joint_idx],
            position=positions[joint_idx],
            color=color,
            width=0.22,
            rng=rng,
            jitter=0.014,
            markersize=1.55,
            alpha=0.16,
            fill_alpha=0.07,
            edge_alpha=0.72,
            summary_color=NEUTRAL,
            summary_linewidth=0.75,
        )

    ax.set_ylabel(ylabel)
    ax.set_xlim(0.72, 1.83)
    ax.set_ylim(0, nice_upper_limit(rmse_values, n_intervals=3))
    set_endpoint_yticks(ax, n_ticks=4)
    ax.set_xticks(positions)
    if show_x:
        ax.set_xticklabels(["J1", "J2"])
    else:
        ax.set_xticklabels([])
        ax.tick_params(labelbottom=False)
    style_axis(ax)
    if panel_label:
        add_panel_label(ax, panel_label, x=-0.42, y=1.20)


def mann_whitney_pvalue(x, y) -> float | None:
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return None
    _, p_value = scipy_stats.mannwhitneyu(x, y, alternative="greater")
    return float(p_value)


def add_bracket(ax, x0: float, x1: float, y: float, h: float, label: str) -> None:
    ax.plot(
        [x0, x0, x1, x1],
        [y, y + h, y + h, y],
        transform=ax.transAxes,
        color=NEUTRAL,
        linewidth=0.55,
        clip_on=False,
    )
    ax.text((x0 + x1) / 2, y + h + 0.012, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=7)


def plot_figure5(data: dict[str, np.ndarray], metadata: dict, output_prefix: Path) -> list[Path]:
    set_figure_style()
    demo_index = int(metadata.get("demo_trajectory_index", compute_demo_index(data)))
    duration_s = float(metadata.get("trajectory_duration_s", 5.0))

    fig = plt.figure(figsize=(7.20, 2.58), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        4,
        width_ratios=[3.05, 1.00, 1.18, 1.05],
        left=0.065,
        right=0.985,
        bottom=0.22,
        top=0.91,
        wspace=0.44,
    )

    gs_a = outer[0].subgridspec(3, 1, hspace=0.10)
    gs_b = outer[1].subgridspec(3, 1, hspace=0.13)
    ax_a = [fig.add_subplot(gs_a[row, 0]) for row in range(3)]
    ax_b = [fig.add_subplot(gs_b[row, 0]) for row in range(3)]
    ax_c = fig.add_subplot(outer[2])
    ax_d = fig.add_subplot(outer[3])

    time = data["trial_time_s"]
    plot_trajectory_panel(
        ax_a[0],
        time,
        data["true_position_rad"][demo_index],
        data["neural_position_rad"][demo_index],
        "Angle (rad)",
        panel_label="a.",
    )
    plot_trajectory_panel(
        ax_a[1],
        time,
        data["true_velocity_rad_s"][demo_index],
        data["neural_velocity_rad_s"][demo_index],
        "Vel. (rad/s)",
    )
    plot_trajectory_panel(
        ax_a[2],
        time,
        data["true_acceleration_rad_s2"][demo_index],
        data["neural_acceleration_rad_s2"][demo_index],
        "Acc. (rad/s$^2$)",
    )
    for ax in ax_a[:2]:
        ax.tick_params(labelbottom=False)
    ax_a[2].set_xlabel("Time (s)")

    legend_handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.1, label="Joint 1"),
        Line2D([0], [0], color=RED, linewidth=1.1, label="Joint 2"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.75, linestyle=(0, (2.2, 1.6)), label="Numerical"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.85, linestyle="-", label="Neural"),
    ]
    ax_a[2].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, -0.46),
        frameon=False,
        ncol=4,
        handlelength=1.7,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    plot_rmse_panel(ax_b[0], data["position_rmse_rad"], "RMSE (rad)", panel_label="b.")
    plot_rmse_panel(ax_b[1], data["velocity_rmse_rad_s"], "RMSE (rad/s)")
    plot_rmse_panel(ax_b[2], data["acceleration_rmse_rad_s2"], "RMSE (rad/s$^2$)", show_x=True)

    ode_times = data["trajectory_time_ode_s"]
    neural_times = data["trajectory_time_neural_solver_s"]
    ax_c_right = ax_c.twinx()
    rng_c = np.random.default_rng(31)
    ode_color = VIRIDIS_PURPLE
    neural_solver_color = VIRIDIS_TEAL
    forward_color = VIRIDIS_GREEN
    inverse_color = VIRIDIS_PURPLE

    draw_violin(ax_c, ode_times, 1.0, ode_color, 0.34, rng_c, 0.020, 2.40, 0.24)
    draw_violin(ax_c_right, neural_times, 2.0, neural_solver_color, 0.34, rng_c, 0.020, 2.40, 0.24)
    ax_c.set_xlim(0.55, 2.45)
    ax_c_right.set_xlim(0.55, 2.45)
    ax_c.set_xticks([1.0, 2.0])
    ax_c.set_xticklabels(["ODE\nSolver", "Neural\nSolver"])
    for label, color in zip(ax_c.get_xticklabels(), [ode_color, neural_solver_color]):
        label.set_color(color)
    ax_c.set_ylabel(f"Execution time per {duration_s:g} s trajectory (s)")
    ax_c.set_ylim(*nice_padded_limits(ode_times, n_intervals=3, pad_fraction=0.12))
    ax_c_right.set_ylim(*nice_padded_limits(neural_times, n_intervals=3, pad_fraction=0.12))
    set_endpoint_yticks(ax_c, n_ticks=4)
    set_endpoint_yticks(ax_c_right, n_ticks=4)
    style_axis(ax_c, spine_color=ode_color)
    style_axis(ax_c_right, spine_color=neural_solver_color, right=True)
    ax_c_right.tick_params(axis="x", bottom=False, labelbottom=False)
    add_panel_label(ax_c, "c.", x=-0.26, y=1.15)
    p_value = mann_whitney_pvalue(neural_times, ode_times)
    add_bracket(ax_c, 0.30, 0.76, 0.93, 0.025, "*" if p_value is None or p_value < 0.05 else "n.s.")

    forward_latency = data["forward_latency_ms"]
    inverse_latency = data["inverse_latency_16k_ms"]
    ax_d_right = ax_d.twinx()
    draw_boxplot(ax_d, forward_latency, 1.0, forward_color, 0.34)
    draw_boxplot(ax_d_right, inverse_latency, 2.0, inverse_color, 0.34)
    ax_d.set_xlim(0.55, 2.45)
    ax_d_right.set_xlim(0.55, 2.45)
    ax_d.set_xticks([1.0, 2.0])
    ax_d.set_xticklabels(["Frw.\nModel", "Inv.\nModel"])
    for label, color in zip(ax_d.get_xticklabels(), [forward_color, inverse_color]):
        label.set_color(color)
    ax_d.set_ylabel("Inference time (ms)")
    ax_d.set_ylim(*boxplot_display_limits(forward_latency, pad_fraction=0.40))
    ax_d_right.set_ylim(*boxplot_display_limits(inverse_latency, pad_fraction=0.40))
    set_endpoint_yticks(ax_d, n_ticks=4)
    set_endpoint_yticks(ax_d_right, n_ticks=4)
    style_axis(ax_d, spine_color=forward_color)
    style_axis(ax_d_right, spine_color=inverse_color, right=True)
    ax_d_right.tick_params(axis="x", bottom=False, labelbottom=False)
    add_panel_label(ax_d, "d.", x=-0.30, y=1.15)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return outputs


def plot_figure5_clean(data: dict[str, np.ndarray], metadata: dict, output_prefix: Path) -> list[Path]:
    set_figure_style()
    demo_index = int(metadata.get("demo_trajectory_index", compute_demo_index(data)))

    fig = plt.figure(figsize=(7.20, 2.64), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[3.12, 1.08, 2.38],
        left=0.065,
        right=0.985,
        bottom=0.22,
        top=0.91,
        wspace=0.42,
    )

    gs_a = outer[0].subgridspec(3, 1, hspace=0.10)
    gs_b = outer[1].subgridspec(3, 1, hspace=0.13)
    gs_cd = outer[2].subgridspec(2, 1, hspace=0.70)

    ax_a = [fig.add_subplot(gs_a[row, 0]) for row in range(3)]
    ax_b = [fig.add_subplot(gs_b[row, 0]) for row in range(3)]
    ax_c = fig.add_subplot(gs_cd[0, 0])
    ax_d = fig.add_subplot(gs_cd[1, 0])

    time = data["trial_time_s"]
    plot_trajectory_panel(
        ax_a[0],
        time,
        data["true_position_rad"][demo_index],
        data["neural_position_rad"][demo_index],
        "Angle (rad)",
        panel_label="a.",
    )
    plot_trajectory_panel(
        ax_a[1],
        time,
        data["true_velocity_rad_s"][demo_index],
        data["neural_velocity_rad_s"][demo_index],
        "Vel. (rad/s)",
    )
    plot_trajectory_panel(
        ax_a[2],
        time,
        data["true_acceleration_rad_s2"][demo_index],
        data["neural_acceleration_rad_s2"][demo_index],
        "Acc. (rad/s$^2$)",
    )
    for ax in ax_a[:2]:
        ax.tick_params(labelbottom=False)
    ax_a[2].set_xlabel("Time (s)")

    legend_handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.1, label="Joint 1"),
        Line2D([0], [0], color=RED, linewidth=1.1, label="Joint 2"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.75, linestyle=(0, (2.2, 1.6)), label="Numerical"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.85, linestyle="-", label="Neural"),
    ]
    ax_a[2].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, -0.46),
        frameon=False,
        ncol=4,
        handlelength=1.7,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    plot_rmse_panel(ax_b[0], data["position_rmse_rad"], "RMSE (rad)", panel_label="b.")
    plot_rmse_panel(ax_b[1], data["velocity_rmse_rad_s"], "RMSE (rad/s)")
    plot_rmse_panel(ax_b[2], data["acceleration_rmse_rad_s2"], "RMSE (rad/s$^2$)", show_x=True)

    plot_horizontal_summary(
        ax_c,
        series=[data["trajectory_time_ode_s"], data["trajectory_time_neural_solver_s"]],
        labels=["ODE solver", "Neural solver"],
        colors=[VIRIDIS_PURPLE, VIRIDIS_TEAL],
        xlabel="Execution time per 5 s trajectory (s)",
        panel_label="c.",
        xlim=(0.40, 6.0),
        xticks=[0.40, 1.0, 2.5, 6.0],
        show_points=True,
        rng_seed=31,
    )

    plot_horizontal_summary(
        ax_d,
        series=[data["forward_latency_ms"], data["inverse_latency_16k_ms"]],
        labels=["Forward model", "Inverse model"],
        colors=[VIRIDIS_GREEN, VIRIDIS_PURPLE],
        xlabel="Inference time (ms)",
        panel_label="d.",
        xlim=(0.05, 3.0),
        xticks=[0.05, 0.20, 0.80, 3.0],
        show_points=False,
        rng_seed=37,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return outputs


def plot_figure5_vertical_summary(
    data: dict[str, np.ndarray],
    metadata: dict,
    output_prefix: Path,
    linear_timing: bool = False,
) -> list[Path]:
    set_figure_style()
    demo_index = int(metadata.get("demo_trajectory_index", compute_demo_index(data)))

    fig = plt.figure(figsize=(7.20, 2.56), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        4,
        width_ratios=[3.02, 0.98, 0.93, 0.93],
        left=0.065,
        right=0.985,
        bottom=0.22,
        top=0.91,
        wspace=0.50,
    )

    gs_a = outer[0].subgridspec(3, 1, hspace=0.10)
    gs_b = outer[1].subgridspec(3, 1, hspace=0.13)
    ax_a = [fig.add_subplot(gs_a[row, 0]) for row in range(3)]
    ax_b = [fig.add_subplot(gs_b[row, 0]) for row in range(3)]
    ax_c = fig.add_subplot(outer[2])
    ax_d = fig.add_subplot(outer[3])

    time = data["trial_time_s"]
    plot_trajectory_panel(
        ax_a[0],
        time,
        data["true_position_rad"][demo_index],
        data["neural_position_rad"][demo_index],
        "Angle (rad)",
        panel_label="a.",
    )
    plot_trajectory_panel(
        ax_a[1],
        time,
        data["true_velocity_rad_s"][demo_index],
        data["neural_velocity_rad_s"][demo_index],
        "Vel. (rad/s)",
    )
    plot_trajectory_panel(
        ax_a[2],
        time,
        data["true_acceleration_rad_s2"][demo_index],
        data["neural_acceleration_rad_s2"][demo_index],
        "Acc. (rad/s$^2$)",
    )
    for ax in ax_a[:2]:
        ax.tick_params(labelbottom=False)
    ax_a[2].set_xlabel("Time (s)")

    legend_handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.1, label="Joint 1"),
        Line2D([0], [0], color=RED, linewidth=1.1, label="Joint 2"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.75, linestyle=(0, (2.2, 1.6)), label="Numerical"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.85, linestyle="-", label="Neural"),
    ]
    ax_a[2].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, -0.46),
        frameon=False,
        ncol=4,
        handlelength=1.7,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    plot_rmse_panel(ax_b[0], data["position_rmse_rad"], "RMSE (rad)", panel_label="b.")
    plot_rmse_panel(ax_b[1], data["velocity_rmse_rad_s"], "RMSE (rad/s)")
    plot_rmse_panel(ax_b[2], data["acceleration_rmse_rad_s2"], "RMSE (rad/s$^2$)", show_x=True)

    if linear_timing:
        timing_ylim = (0.0, 5.0)
        timing_ticks = [0.0, 1.25, 2.5, 3.75, 5.0]
        timing_log = False
    else:
        timing_ylim = (0.40, 6.0)
        timing_ticks = [0.40, 1.0, 2.5, 6.0]
        timing_log = True

    plot_vertical_interval_summary(
        ax_c,
        series=[data["trajectory_time_ode_s"], data["trajectory_time_neural_solver_s"]],
        labels=["ODE\nSolver", "Neural\nSolver"],
        colors=[VIRIDIS_PURPLE, VIRIDIS_TEAL],
        ylabel="Time per 5 s trajectory (s)",
        panel_label="c.",
        ylim=timing_ylim,
        yticks=timing_ticks,
        log_y=timing_log,
        show_points=True,
        rng_seed=31,
        realtime_line=5.0,
        realtime_label="real time",
    )

    plot_vertical_interval_summary(
        ax_d,
        series=[data["forward_latency_ms"], data["inverse_latency_16k_ms"]],
        labels=["Frw.\nModel", "Inv.\nModel"],
        colors=[VIRIDIS_GREEN, VIRIDIS_PURPLE],
        ylabel="Inference time (ms)",
        panel_label="d.",
        ylim=(0.05, 3.0),
        yticks=[0.05, 0.20, 0.80, 3.0],
        log_y=True,
        show_points=False,
        rng_seed=37,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return outputs


def plot_figure5_distribution_summary(
    data: dict[str, np.ndarray],
    metadata: dict,
    output_prefix: Path,
) -> list[Path]:
    set_figure_style()
    demo_index = int(metadata.get("demo_trajectory_index", compute_demo_index(data)))

    fig = plt.figure(figsize=(7.20, 2.42), constrained_layout=False)
    outer = fig.add_gridspec(
        1,
        4,
        width_ratios=[3.05, 1.08, 0.90, 0.90],
        left=0.058,
        right=0.992,
        bottom=0.20,
        top=0.92,
        wspace=0.34,
    )

    gs_a = outer[0].subgridspec(3, 1, hspace=0.10)
    gs_b = outer[1].subgridspec(3, 1, hspace=0.13)
    ax_a = [fig.add_subplot(gs_a[row, 0]) for row in range(3)]
    ax_b = [fig.add_subplot(gs_b[row, 0]) for row in range(3)]
    ax_c = fig.add_subplot(outer[2])
    ax_d = fig.add_subplot(outer[3])

    time = data["trial_time_s"]
    plot_trajectory_panel(
        ax_a[0],
        time,
        data["true_position_rad"][demo_index],
        data["neural_position_rad"][demo_index],
        "Angle (rad)",
        panel_label="a.",
    )
    plot_trajectory_panel(
        ax_a[1],
        time,
        data["true_velocity_rad_s"][demo_index],
        data["neural_velocity_rad_s"][demo_index],
        "Vel. (rad/s)",
    )
    plot_trajectory_panel(
        ax_a[2],
        time,
        data["true_acceleration_rad_s2"][demo_index],
        data["neural_acceleration_rad_s2"][demo_index],
        "Acc. (rad/s$^2$)",
    )
    for ax in ax_a[:2]:
        ax.tick_params(labelbottom=False)
    ax_a[2].set_xlabel("Time (s)")

    legend_handles = [
        Line2D([0], [0], color=BLUE, linewidth=1.1, label="Joint 1"),
        Line2D([0], [0], color=RED, linewidth=1.1, label="Joint 2"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.75, linestyle=(0, (2.2, 1.6)), label="Numerical"),
        Line2D([0], [0], color=NEUTRAL, linewidth=0.85, linestyle="-", label="Neural"),
    ]
    ax_a[2].legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, -0.42),
        frameon=False,
        ncol=4,
        handlelength=1.7,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    plot_rmse_panel(ax_b[0], data["position_rmse_rad"], "RMSE (rad)", panel_label="b.")
    plot_rmse_panel(ax_b[1], data["velocity_rmse_rad_s"], "RMSE (rad/s)")
    plot_rmse_panel(ax_b[2], data["acceleration_rmse_rad_s2"], "RMSE (rad/s$^2$)", show_x=True)

    ode_color = VIRIDIS_PURPLE
    neural_solver_color = VIRIDIS_TEAL
    forward_color = VIRIDIS_GREEN
    inverse_color = VIRIDIS_PURPLE

    rng_c = np.random.default_rng(31)
    draw_violin(ax_c, data["trajectory_time_ode_s"], 1.0, ode_color, 0.34, rng_c, 0.020, 2.2, 0.24)
    draw_violin(
        ax_c,
        data["trajectory_time_neural_solver_s"],
        2.0,
        neural_solver_color,
        0.34,
        rng_c,
        0.020,
        2.2,
        0.24,
    )
    ax_c.axhline(5.0, color=LIGHT_NEUTRAL, linewidth=0.55, linestyle=(0, (2.0, 1.6)), zorder=1)
    ax_c.text(2.34, 5.0, "real time", ha="right", va="bottom", fontsize=5.5, color=LIGHT_NEUTRAL)
    ax_c.set_xlim(0.55, 2.45)
    ax_c.set_ylim(0.0, 5.0)
    ax_c.set_yticks([0.0, 2.5, 5.0])
    ax_c.set_yticklabels(["0", "2.5", "5"])
    ax_c.set_xticks([1.0, 2.0])
    ax_c.set_xticklabels(["ODE\nSolver", "Neural\nSolver"])
    for label, color in zip(ax_c.get_xticklabels(), [ode_color, neural_solver_color]):
        label.set_color(color)
    ax_c.set_ylabel("Time (s)")
    style_axis(ax_c)
    add_panel_label(ax_c, "c.", x=-0.24, y=1.12)

    ax_d.set_yscale("log")
    draw_boxplot(ax_d, data["forward_latency_ms"], 1.0, forward_color, 0.34)
    draw_boxplot(ax_d, data["inverse_latency_16k_ms"], 2.0, inverse_color, 0.34)
    ax_d.axhline(2.0, color=LIGHT_NEUTRAL, linewidth=0.55, linestyle=(0, (2.0, 1.6)), zorder=1)
    ax_d.text(2.34, 2.0, "2 ms", ha="right", va="bottom", fontsize=5.5, color=LIGHT_NEUTRAL)
    ax_d.set_xlim(0.55, 2.45)
    ax_d.set_ylim(0.05, 3.2)
    latency_ticks = np.array([0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20])
    ax_d.set_yticks(latency_ticks)
    ax_d.set_yticklabels([compact_tick_label(tick) for tick in latency_ticks])
    ax_d.set_xticks([1.0, 2.0])
    ax_d.set_xticklabels(["Frw.\nModel", "Inv.\nModel"])
    for label, color in zip(ax_d.get_xticklabels(), [forward_color, inverse_color]):
        label.set_color(color)
    ax_d.set_ylabel("Latency (ms)")
    style_axis(ax_d)
    add_panel_label(ax_d, "d.", x=-0.26, y=1.12)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        output_prefix.with_suffix(".svg"),
        output_prefix.with_suffix(".pdf"),
        output_prefix.with_suffix(".png"),
    ]
    fig.savefig(outputs[0], format="svg", bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.03, facecolor="white")
    fig.savefig(outputs[2], dpi=600, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    return outputs


def inspect_svg(svg_path: Path) -> dict:
    text = svg_path.read_text(encoding="utf-8")
    return {
        "embedded_image_tags": text.count("<image"),
        "text_tags": text.count("<text"),
        "use_tags": text.count("<use"),
    }


def print_report(data: dict[str, np.ndarray], metadata: dict, package_path: Path, outputs: list[Path]) -> None:
    print("\nFIG5 data validation")
    print("--------------------")
    for line in validate_data(data):
        print(line)

    print("\nKey plotted summaries")
    print("---------------------")
    print(f"Representative trajectory index: {metadata.get('demo_trajectory_index')}")
    print(f"Trajectory duration: {metadata.get('trajectory_duration_s'):.3f} s")
    for label, key in [
        ("Position RMSE J1/J2, rad", "position_rmse_rad"),
        ("Velocity RMSE J1/J2, rad/s", "velocity_rmse_rad_s"),
        ("Acceleration RMSE J1/J2, rad/s^2", "acceleration_rmse_rad_s2"),
    ]:
        medians = np.median(data[key], axis=0)
        print(f"{label}: medians = {medians[0]:.6g}, {medians[1]:.6g}")
    print(
        "Trajectory time medians, s: "
        f"ODE = {np.median(data['trajectory_time_ode_s']):.6g}, "
        f"Neural solver = {np.median(data['trajectory_time_neural_solver_s']):.6g}"
    )
    print(
        "Latency medians, ms: "
        f"Forward = {np.median(data['forward_latency_ms']):.6g}, "
        f"Inverse 16k = {np.median(data['inverse_latency_16k_ms']):.6g}"
    )

    print("\nFiles written")
    print("-------------")
    print(f"Data package: {package_path.resolve()} ({package_path.stat().st_size:,} bytes)")
    with zipfile.ZipFile(package_path, "r") as zf:
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"  {name}: {info.file_size:,} bytes")
    for output in outputs:
        print(f"Figure output: {output.resolve()} ({output.stat().st_size:,} bytes)")
    svg_info = inspect_svg(outputs[0])
    print(
        "SVG check: "
        f"{svg_info['embedded_image_tags']} embedded image tags, "
        f"{svg_info['text_tags']} text tags, "
        f"{svg_info['use_tags']} marker/use tags"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create full manuscript Fig. 5 from packaged data.")
    parser.add_argument("--source-dir", default=".", help="Folder containing the saved .pt source files.")
    parser.add_argument("--data-package", default=DEFAULT_PACKAGE, help="Path to the FIG5 data zip file.")
    parser.add_argument("--output-prefix", default=None, help="Output prefix for SVG/PDF/PNG.")
    parser.add_argument(
        "--style",
        choices=["standard", "clean", "vertical-log", "vertical-linear", "distribution"],
        default="standard",
        help="Figure layout style.",
    )
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild the data package from .pt files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    package_path = Path(args.data_package)
    output_prefix = Path(
        args.output_prefix
        if args.output_prefix is not None
        else (
            "FIG5_clean"
            if args.style == "clean"
            else "FIG5_vertical_log"
            if args.style == "vertical-log"
            else "FIG5_vertical_linear"
            if args.style == "vertical-linear"
            else "FIG5_distribution"
            if args.style == "distribution"
            else DEFAULT_PREFIX
        )
    )

    if args.rebuild_data or not package_path.exists():
        print(f"Building data package from saved .pt files in {source_dir}")
        data, metadata = load_source_data(source_dir)
        write_data_package(data, metadata, package_path)
    else:
        print(f"Reading existing data package: {package_path}")
        data, metadata = read_data_package(package_path)

    if args.style == "clean":
        outputs = plot_figure5_clean(data, metadata, output_prefix)
    elif args.style == "vertical-log":
        outputs = plot_figure5_vertical_summary(data, metadata, output_prefix, linear_timing=False)
    elif args.style == "vertical-linear":
        outputs = plot_figure5_vertical_summary(data, metadata, output_prefix, linear_timing=True)
    elif args.style == "distribution":
        outputs = plot_figure5_distribution_summary(data, metadata, output_prefix)
    else:
        outputs = plot_figure5(data, metadata, output_prefix)
    print_report(data, metadata, package_path, outputs)


if __name__ == "__main__":
    main()
