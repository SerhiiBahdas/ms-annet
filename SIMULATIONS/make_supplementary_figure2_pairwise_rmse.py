#!/usr/bin/env python3
"""
Generate Supplementary Figure 2: an alternative, matrix-based view of the
pairwise RMSE comparisons shown as chord diagrams in the main figure.

The script reads the saved results_model_*.pt files, recomputes the same
two-sided Mann-Whitney U tests used in the notebook, applies Benjamini-Hochberg
FDR correction separately for each joint, and saves both the figure and the
underlying pairwise statistics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "FIGURES"
DATASET_SIZES = [
    1000,
    2000,
    4000,
    8000,
    16000,
    32000,
    64000,
    128000,
    256000,
]
ALPHA = 0.05


def format_sample_count(value: int) -> str:
    value = int(value)
    if value >= 1000:
        return f"{value // 1000}k"
    return str(value)


def load_rmse_table(base_dir: Path, dataset_sizes: list[int]) -> pd.DataFrame:
    records = []

    for size in dataset_sizes:
        result_path = base_dir / f"results_model_{size}.pt"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing saved result file: {result_path}")

        result_dict = torch.load(result_path, map_location="cpu")
        result_key = list(result_dict.keys())[0]
        result = result_dict[result_key]

        for rmse in result["rmse_j1"]:
            records.append(
                {"dataset_size": size, "joint": "J1", "rmse_nm": float(rmse)}
            )
        for rmse in result["rmse_j2"]:
            records.append(
                {"dataset_size": size, "joint": "J2", "rmse_nm": float(rmse)}
            )

    return pd.DataFrame.from_records(records)


def build_pairwise_statistics(
    df_rmse: pd.DataFrame,
    joint: str,
    dataset_sizes: list[int],
    alpha: float = ALPHA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_p_values = []
    pair_rows = []

    for i, size_a in enumerate(dataset_sizes):
        for j, size_b in enumerate(dataset_sizes):
            if j <= i:
                continue

            data_a = df_rmse.loc[
                (df_rmse["dataset_size"] == size_a) & (df_rmse["joint"] == joint),
                "rmse_nm",
            ].to_numpy(dtype=float)
            data_b = df_rmse.loc[
                (df_rmse["dataset_size"] == size_b) & (df_rmse["joint"] == joint),
                "rmse_nm",
            ].to_numpy(dtype=float)

            statistic, raw_p = stats.mannwhitneyu(
                data_a,
                data_b,
                alternative="two-sided",
            )
            raw_p_values.append(raw_p)
            pair_rows.append(
                {
                    "joint": joint,
                    "dataset_size_a": size_a,
                    "dataset_size_b": size_b,
                    "n_a": int(data_a.size),
                    "n_b": int(data_b.size),
                    "median_rmse_a_nm": float(np.median(data_a)),
                    "median_rmse_b_nm": float(np.median(data_b)),
                    "mann_whitney_u": float(statistic),
                    "raw_p_value": float(raw_p),
                }
            )

    reject, fdr_q_values, _, _ = multipletests(
        raw_p_values,
        alpha=alpha,
        method="fdr_bh",
    )

    for row, reject_null, fdr_q in zip(pair_rows, reject, fdr_q_values):
        row["fdr_q_value"] = float(fdr_q)
        row["significant_fdr_0_05"] = bool(reject_null)

    pairwise_df = pd.DataFrame.from_records(pair_rows)

    q_matrix = pd.DataFrame(
        np.nan,
        index=dataset_sizes,
        columns=dataset_sizes,
        dtype=float,
    )
    for row in pair_rows:
        size_a = row["dataset_size_a"]
        size_b = row["dataset_size_b"]
        q_value = row["fdr_q_value"]
        q_matrix.loc[size_a, size_b] = q_value
        q_matrix.loc[size_b, size_a] = q_value

    return pairwise_df, q_matrix


def draw_pairwise_matrix(
    ax: plt.Axes,
    q_matrix: pd.DataFrame,
    title: str,
    cmap: mpl.colors.Colormap,
    norm: mpl.colors.Normalize,
    alpha: float = ALPHA,
) -> None:
    dataset_sizes = list(q_matrix.index)
    n_sizes = len(dataset_sizes)

    ax.set_xlim(0, n_sizes)
    ax.set_ylim(n_sizes, 0)
    ax.set_aspect("equal")

    for i, size_a in enumerate(dataset_sizes):
        for j, size_b in enumerate(dataset_sizes):
            q_value = q_matrix.loc[size_a, size_b]

            if i == j or not np.isfinite(q_value):
                rect = Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor="#F2F2F2",
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
                continue

            score = -np.log10(max(float(q_value), np.finfo(float).tiny))
            significant = bool(q_value < alpha)
            rect = Rectangle(
                (j, i),
                1,
                1,
                facecolor=cmap(norm(score)),
                edgecolor="white",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            if not significant:
                outline = Rectangle(
                    (j + 0.045, i + 0.045),
                    0.91,
                    0.91,
                    facecolor="none",
                    edgecolor="#222222",
                    linewidth=0.75,
                )
                ax.add_patch(outline)
                ax.text(
                    j + 0.5,
                    i + 0.43,
                    "ns",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="#111111",
                )
                ax.text(
                    j + 0.5,
                    i + 0.62,
                    f"q={q_value:.2g}",
                    ha="center",
                    va="center",
                    fontsize=4.8,
                    color="#111111",
                )

    tick_positions = np.arange(n_sizes) + 0.5
    tick_labels = [format_sample_count(size) for size in dataset_sizes]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Training samples")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0, pad=2)


def make_figure(
    q_matrices: dict[str, pd.DataFrame],
    output_dir: Path,
    alpha: float = ALPHA,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_q_values = np.concatenate(
        [
            matrix.to_numpy(dtype=float)[np.isfinite(matrix.to_numpy(dtype=float))]
            for matrix in q_matrices.values()
        ]
    )
    max_score = float(np.ceil(np.nanmax(-np.log10(all_q_values))))
    max_score = max(max_score, -np.log10(alpha))
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max_score)
    joint_cmaps = {
        "J1": mpl.colors.LinearSegmentedColormap.from_list(
            "joint1_q_blue",
            ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#3182BD", "#08519C"],
        ),
        "J2": mpl.colors.LinearSegmentedColormap.from_list(
            "joint2_q_red",
            ["#FFF5F0", "#FEE0D2", "#FC9272", "#DE2D26", "#A50F15"],
        ),
    }

    with mpl.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.35),
            constrained_layout=True,
        )

        draw_pairwise_matrix(
            axes[0],
            q_matrices["J1"],
            "Joint 1",
            joint_cmaps["J1"],
            norm,
            alpha=alpha,
        )
        draw_pairwise_matrix(
            axes[1],
            q_matrices["J2"],
            "Joint 2",
            joint_cmaps["J2"],
            norm,
            alpha=alpha,
        )

        axes[0].text(
            -0.15,
            1.03,
            "a.",
            transform=axes[0].transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )
        axes[1].text(
            -0.15,
            1.03,
            "b.",
            transform=axes[1].transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        for ax, cmap in zip(axes, [joint_cmaps["J1"], joint_cmaps["J2"]]):
            scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_mappable.set_array([])
            colorbar = fig.colorbar(
                scalar_mappable,
                ax=ax,
                fraction=0.046,
                pad=0.022,
            )
            colorbar.set_label("FDR-adjusted q value\n(smaller = stronger evidence)")
            colorbar.set_ticks([0, -np.log10(alpha), 5, 10, 15])
            colorbar.set_ticklabels(["1", "0.05", "1e-5", "1e-10", "1e-15"])
            colorbar.ax.tick_params(length=2.0, width=0.5, pad=2)
            colorbar.outline.set_linewidth(0.5)

        legend_handle = Line2D(
            [0],
            [0],
            marker="s",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="#222222",
            linestyle="none",
            label=f"Not significant (FDR q >= {alpha})",
        )
        fig.legend(
            handles=[legend_handle],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            frameon=False,
            fontsize=6,
            handletextpad=0.4,
        )

        figure_base = output_dir / "Fig_S2_pairwise_RMSE_matrix"
        svg_path = figure_base.with_suffix(".svg")
        pdf_path = figure_base.with_suffix(".pdf")
        png_path = figure_base.with_suffix(".png")
        fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
        fig.savefig(
            png_path,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor="white",
        )
        plt.close(fig)

    return {"svg": svg_path, "pdf": pdf_path, "png": png_path}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_rmse = load_rmse_table(BASE_DIR, DATASET_SIZES)
    df_rmse.to_csv(OUTPUT_DIR / "Fig_S2_RMSE_values.csv", index=False)

    pairwise_tables = []
    q_matrices = {}
    for joint in ("J1", "J2"):
        pairwise_df, q_matrix = build_pairwise_statistics(
            df_rmse,
            joint,
            DATASET_SIZES,
            alpha=ALPHA,
        )
        pairwise_tables.append(pairwise_df)
        q_matrices[joint] = q_matrix
        q_matrix.to_csv(OUTPUT_DIR / f"Fig_S2_FDR_q_matrix_{joint}.csv")

    pairwise_statistics = pd.concat(pairwise_tables, ignore_index=True)
    pairwise_statistics.to_csv(
        OUTPUT_DIR / "Fig_S2_pairwise_RMSE_statistics.csv",
        index=False,
    )

    figure_paths = make_figure(q_matrices, OUTPUT_DIR, alpha=ALPHA)

    print("Supplementary Figure 2 generated from saved results_model_*.pt files.")
    for file_type, path in figure_paths.items():
        print(f"{file_type.upper()}: {path}")
    print(f"RMSE values: {OUTPUT_DIR / 'Fig_S2_RMSE_values.csv'}")
    print(f"Pairwise statistics: {OUTPUT_DIR / 'Fig_S2_pairwise_RMSE_statistics.csv'}")
    for joint in ("J1", "J2"):
        table = pairwise_statistics[pairwise_statistics["joint"] == joint]
        nonsignificant = table[~table["significant_fdr_0_05"]]
        print(
            f"{joint}: {int(table['significant_fdr_0_05'].sum())}/"
            f"{len(table)} significant comparisons after FDR correction."
        )
        if nonsignificant.empty:
            print(f"{joint}: no non-significant comparisons.")
        else:
            formatted_pairs = [
                (
                    f"{format_sample_count(row.dataset_size_a)} vs "
                    f"{format_sample_count(row.dataset_size_b)} "
                    f"(q={row.fdr_q_value:.4g})"
                )
                for row in nonsignificant.itertuples(index=False)
            ]
            print(f"{joint}: non-significant comparisons: {', '.join(formatted_pairs)}")


if __name__ == "__main__":
    main()
