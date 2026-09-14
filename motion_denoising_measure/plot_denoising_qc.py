"""
plot_denoising_qc.py
=====================
Generates all denoising-evaluation figures from the data files written by
denoising_evaluation.py (manifest.csv, qc_fc_*.npy, distance_matrix.npy in
a pipeline's output_dir). Kept separate from evaluation so a pipeline's
data is computed once and figures can be regenerated/restyled without
rerunning the (expensive) evaluation pass.

Usage:
    python plot_denoising_qc.py --result_dir <pipeline output_dir>
"""
import argparse
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize, to_rgb
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from nilearn import plotting
from scipy import stats

DEFAULT_CONNECTOME_ATLAS_PATH = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "templates/rois/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
)

# ROI order in qc_fc_r.npy/distance_matrix.npy follows this atlas's label order
# (1..400) -- this CSV lists the same 400 ROIs, in label order, with names
# encoding hemisphere + Yeo-7 network (e.g. "7Networks_LH_Vis_1").
DEFAULT_NETWORK_LABELS_CSV = (
    "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/"
    "templates/rois/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv"
)

# Standard Yeo-7 network colors (as used in the FreeSurfer/Schaefer LUTs).
YEO7_NETWORK_COLORS = {
    "Vis": "#781286",
    "SomMot": "#4682B4",
    "DorsAttn": "#00760E",
    "SalVentAttn": "#C43AFA",
    "Limbic": "#DCF8A4",
    "Cont": "#E69422",
    "Default": "#CB3F4A",
}


def load_network_labels(network_labels_csv: str) -> np.ndarray:
    """400 "LH_Vis"-style hemisphere+network labels, in the same ROI-label
    order as qc_fc_r.npy's rows/columns."""
    names = pd.read_csv(network_labels_csv)["ROI Name"]
    parts = names.str.split("_")
    return (parts.str[1] + "_" + parts.str[2]).to_numpy()


def load_results(result_dir: str) -> dict:
    manifest = pd.read_csv(os.path.join(result_dir, "manifest.csv"))
    r_mat = np.load(os.path.join(result_dir, "qc_fc_r.npy"))
    fdr_mat = np.load(os.path.join(result_dir, "qc_fc_fdr_significant.npy"))
    dist_mat = np.load(os.path.join(result_dir, "distance_matrix.npy"))
    iu = np.triu_indices(r_mat.shape[0], k=1)
    return {
        "r_mat": r_mat,
        "fdr_mat": fdr_mat,
        "dist_mat": dist_mat,
        "qcfc_r": r_mat[iu],
        "q_values": manifest["Q"].to_numpy(),
        "fd": manifest["mean_fd"].to_numpy(),
        "dvars_values": manifest["dvars"].to_numpy(),
        "tsnr_values": manifest["tsnr"].to_numpy(),
        "gs_std_values": manifest["gs_std"].to_numpy(),
        "fd_dvars_values": manifest["fd_dvars_r"].to_numpy(),
    }


def plot_qcfc_distance_dependence(
    distance_matrix,
    qcfc_matrix,
    title="Raw data",
    ax=None
):
    """
    distance_matrix : (R, R) Euclidean ROI-distance matrix in mm
    qcfc_matrix     : (R, R) edgewise QC-FC correlation matrix
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    # Unique undirected edges; exclude diagonal
    upper = np.triu_indices_from(distance_matrix, k=1)

    distance = distance_matrix[upper]
    qcfc = qcfc_matrix[upper]

    # Remove missing/infinite observations
    valid = np.isfinite(distance) & np.isfinite(qcfc)
    distance = distance[valid]
    qcfc = qcfc[valid]

    # QC-FC distance-dependence
    qcfc_dd_r, qcfc_dd_p = stats.pearsonr(distance, qcfc)

    # Linear regression
    regression = stats.linregress(distance, qcfc)
    slope = regression.slope
    intercept = regression.intercept

    # 2D kernel-density contours
    sns.kdeplot(
        x=distance,
        y=qcfc,
        fill=True,
        levels=10,
        thresh=0.03,
        bw_adjust=1.0,
        gridsize=150,
        cmap="Blues",
        ax=ax
    )

    # Zero-reference line
    ax.axhline(
        0,
        color="black",
        linewidth=2.5,
        zorder=4
    )

    # Linear regression line
    x_line = np.linspace(distance.min(), distance.max(), 200)
    y_line = intercept + slope * x_line

    ax.plot(
        x_line,
        y_line,
        color="red",
        linewidth=3,
        zorder=5
    )

    # Panel-specific limits, similar to Ciric et al.
    y_min = qcfc.min()
    y_max = qcfc.max()
    margin = 0.03 * (y_max - y_min)

    ax.set_ylim(y_min - margin, y_max + margin)

    # Display y-axis extrema like the paper
    ax.text(
        0.97, 0.98,
        f"{y_max:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11
    )

    ax.text(
        0.97, 0.02,
        f"{y_min:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11
    )

    ax.set_title(title)
    ax.set_xlabel("ROI-pair distance (mm)")
    ax.set_ylabel("QC–FC correlation (r)")

    # Optional regression-statistics annotation
    stats_text = (
        f"QC–FC–DD r = {qcfc_dd_r:.3f}\n"
        f"Slope = {slope:.4f} per mm\n"
        f"Intercept = {intercept:.3f}"
    )

    ax.text(
        0.03, 0.03,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85,
            edgecolor="none"
        )
    )

    sns.despine(ax=ax)

    return {
        "qcfc_dd_r": qcfc_dd_r,
        "qcfc_dd_p": qcfc_dd_p,
        "slope": slope,
        "intercept": intercept,
        "n_edges": len(qcfc)
    }


def plot_single_qcfc_distribution(
    qcfc_values,
    method_name="Raw",
    uses_gsr=False,
    plot_absolute=False,
    xlim=None,
    figsize=(6.5, 4.5)
):
    """
    Plot one Ciric-style QC-FC distribution using separate,
    perfectly aligned axes for the label and distribution.
    """

    values = np.asarray(qcfc_values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 3:
        raise ValueError("At least three valid QC-FC values are required.")

    median_signed = np.median(values)
    median_absolute = np.median(np.abs(values))

    if plot_absolute:
        plot_values = np.abs(values)

        if xlim is None:
            xlim = (0, 0.45)

        line_position = median_absolute
        line_colour = "#075985"
        x_label = r"Absolute QC–FC correlation, $|r|$"

    else:
        plot_values = values

        if xlim is None:
            xlim = (-0.30, 0.45)

        line_position = 0
        line_colour = "black"
        x_label = "QC–FC correlation (r)"

    # Colours
    fill_colour = "#43B7E9"
    density_edge_colour = "#405763"

    if uses_gsr:
        frame_colour = "#08A6C5"
    else:
        frame_colour = "#696969"

    # --------------------------------------------------------
    # Create separate, aligned label and density axes
    # --------------------------------------------------------

    fig = plt.figure(figsize=figsize)

    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[0.23, 1],
        wspace=0
    )

    label_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    # --------------------------------------------------------
    # Left label panel
    # --------------------------------------------------------

    label_ax.set_facecolor(frame_colour)
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(0, 1)

    label_ax.text(
        0.5,
        0.5,
        method_name,
        rotation=90,
        ha="center",
        va="center",
        color="white",
        fontsize=15,
        fontweight="bold"
    )

    label_ax.set_xticks([])
    label_ax.set_yticks([])

    for spine in label_ax.spines.values():
        spine.set_color(frame_colour)
        spine.set_linewidth(3)

    # --------------------------------------------------------
    # KDE density calculation
    # --------------------------------------------------------
    x_grid = np.linspace(xlim[0], xlim[1], 500)

    kde = stats.gaussian_kde(
        plot_values,
        bw_method="scott"
    )

    density = kde(x_grid)

    if density.max() > 0:
        density /= density.max()

    # --------------------------------------------------------
    # Density plot
    # --------------------------------------------------------
    ax.fill_between(
        x_grid,
        0,
        density,
        color=fill_colour,
        alpha=1,
        zorder=2
    )

    ax.plot(
        x_grid,
        density,
        color=density_edge_colour,
        linewidth=1,
        zorder=3
    )

    # Black zero line for signed data or median line for absolute data
    ax.axvline(
        line_position,
        color=line_colour,
        linewidth=4,
        zorder=4
    )

    # Median absolute QC-FC annotation
    ax.text(
        0.96,
        0.92,
        rf"Median $|QC\mathrm{{-}}FC|$ = {median_absolute:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_xlim(xlim)
    ax.set_ylim(0, 1.08)

    ax.set_xlabel(
        x_label,
        fontsize=12,
        labelpad=10
    )

    # Density magnitude is normalized, so y-axis values are unnecessary
    ax.set_yticks([])
    ax.set_ylabel("")

    ax.tick_params(
        axis="x",
        labelsize=10,
        pad=5
    )

    ax.set_facecolor("white")

    # Frame around the density plot
    for spine in ax.spines.values():
        spine.set_color(frame_colour)
        spine.set_linewidth(3)

    # The common border between both axes
    label_ax.spines["right"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    # Stable spacing around the complete figure
    fig.subplots_adjust(
        left=0.07,
        right=0.97,
        bottom=0.20,
        top=0.95
    )

    return fig, {
        "median_signed_qcfc": median_signed,
        "median_absolute_qcfc": median_absolute,
        "n_edges": len(values)
    }


def plot_q_vs_fd(mean_fd, q_values):

    result = stats.linregress(mean_fd, q_values)
    pearson_r, pearson_p = stats.pearsonr(mean_fd, q_values)
    spearman_r, spearman_p = stats.spearmanr(mean_fd, q_values)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    sns.regplot(
        x=mean_fd,
        y=q_values,
        ci=95,
        scatter_kws={
            "s": 42,
            "alpha": 0.72,
            "color": "#168AAD",
            "edgecolor": "white",
            "linewidths": 0.5
        },
        line_kws={
            "color": "#D55E00",
            "linewidth": 2.2
        },
        ax=ax
    )

    ax.text(
        0.97,
        0.96,
        (
            f"Pearson r = {pearson_r:.3f}, p = {pearson_p:.3g}\n"
            f"Spearman ρ = {spearman_r:.3f}, p = {spearman_p:.3g}\n"
            f"Slope = {result.slope:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "#CCCCCC",
            "alpha": 0.9
        }
    )

    ax.set_title(
        "Association between head motion and network modularity",
        loc="left",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel("Mean framewise displacement (mm)")
    ax.set_ylabel("Modularity Q")

    ax.grid(axis="both", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig


def plot_subject_fd_dvars_correlations(
    correlations,
    method_name="Raw",
    title="Within-run FD–DVARS correlations",
    uses_gsr=False,
    n_bootstrap=5000,
    figsize=(8.5, 4.8)
):
    """
    Plot one subject-level FD-DVARS correlation distribution
    using a Ciric-style label strip and aligned plot panel.
    """

    r = np.asarray(correlations, dtype=float)
    r = r[np.isfinite(r)]

    if len(r) < 3:
        raise ValueError(
            "At least three valid correlations are required."
        )

    # Prevent infinite Fisher transformations
    r = np.clip(r, -0.999999, 0.999999)

    # --------------------------------------------------------
    # Group statistics
    # --------------------------------------------------------

    fisher_z = np.arctanh(r)
    group_r = np.tanh(np.mean(fisher_z))
    median_r = np.median(r)

    # Subject-level bootstrap CI
    rng = np.random.default_rng(42)

    bootstrap_indices = rng.integers(
        low=0,
        high=len(r),
        size=(n_bootstrap, len(r))
    )

    bootstrap_mean_z = np.mean(
        fisher_z[bootstrap_indices],
        axis=1
    )

    bootstrap_mean_r = np.tanh(bootstrap_mean_z)

    ci_low, ci_high = np.percentile(
        bootstrap_mean_r,
        [2.5, 97.5]
    )

    # --------------------------------------------------------
    # Plotting limits
    # --------------------------------------------------------

    padding = 0.08

    x_min = max(
        -1,
        min(-0.10, r.min() - padding)
    )

    x_max = min(
        1,
        max(0.10, r.max() + padding)
    )

    x_grid = np.linspace(x_min, x_max, 500)

    kde = stats.gaussian_kde(
        r,
        bw_method="scott"
    )

    density = kde(x_grid)

    # Normalize KDE height for presentation
    if density.max() > 0:
        density /= density.max()

    # --------------------------------------------------------
    # Colours
    # --------------------------------------------------------

    fill_colour = "#43B7E9"
    density_edge_colour = "#26647A"
    estimate_colour = "#D55E00"

    if uses_gsr:
        frame_colour = "#08A6C5"
    else:
        frame_colour = "#696969"

    # --------------------------------------------------------
    # Create aligned label and density panels
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=figsize,
        facecolor="white"
    )

    grid = fig.add_gridspec(
        nrows=1,
        ncols=2,
        width_ratios=[0.20, 1],
        wspace=0
    )

    label_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    # --------------------------------------------------------
    # Coloured method-label rectangle
    # --------------------------------------------------------

    label_ax.set_facecolor(frame_colour)
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(0, 1)

    label_ax.text(
        0.5,
        0.5,
        method_name,
        rotation=90,
        ha="center",
        va="center",
        color="white",
        fontsize=16,
        fontweight="bold"
    )

    label_ax.set_xticks([])
    label_ax.set_yticks([])

    for spine in label_ax.spines.values():
        spine.set_color(frame_colour)
        spine.set_linewidth(3)

    # --------------------------------------------------------
    # Plotting rectangle
    # --------------------------------------------------------

    ax.set_facecolor("white")

    # Density distribution
    ax.fill_between(
        x_grid,
        0,
        density,
        color=fill_colour,
        alpha=0.85,
        zorder=2
    )

    ax.plot(
        x_grid,
        density,
        color=density_edge_colour,
        linewidth=1.3,
        zorder=3
    )

    # One rug mark for each subject
    ax.vlines(
        r,
        ymin=-0.075,
        ymax=-0.015,
        color=density_edge_colour,
        linewidth=0.8,
        alpha=0.55,
        zorder=4
    )

    # Zero-correlation reference
    ax.axvline(
        0,
        color="#222222",
        linestyle="--",
        linewidth=1.8,
        zorder=4
    )

    # Fisher-z group estimate
    ax.axvline(
        group_r,
        color=estimate_colour,
        linewidth=2.8,
        zorder=5
    )

    # Bootstrap 95% confidence interval
    ci_y = -0.115

    ax.plot(
        [ci_low, ci_high],
        [ci_y, ci_y],
        color=estimate_colour,
        linewidth=3.5,
        solid_capstyle="round",
        zorder=5
    )

    ax.scatter(
        group_r,
        ci_y,
        s=55,
        color=estimate_colour,
        edgecolor="white",
        linewidth=0.7,
        zorder=6
    )

    # --------------------------------------------------------
    # Title and statistical annotation
    # --------------------------------------------------------

    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=18
    )

    ax.text(
        0.03,
        0.94,
        (
            f"n = {len(r)}\n"
            f"Median r = {median_r:.3f}\n"
            f"Group r = {group_r:.3f}\n"
            f"95% CI [{ci_low:.3f}, {ci_high:.3f}]"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.4,
        zorder=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.95
        }
    )

    # Directly label important lines
    ax.text(
        group_r,
        1.015,
        "Group estimate",
        ha="center",
        va="bottom",
        fontsize=9,
        color=estimate_colour
    )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.15, 1.08)

    ax.set_xlabel(
        "Within-run correlation between FD and DVARS",
        fontsize=12,
        labelpad=10
    )

    ax.set_yticks([])
    ax.set_ylabel("")

    ax.tick_params(
        axis="x",
        labelsize=10,
        pad=5
    )

    ax.grid(
        axis="x",
        color="#D9D9D9",
        linewidth=0.7,
        alpha=0.50,
        zorder=0
    )

    # Frame around transparent plotting panel
    for spine in ax.spines.values():
        spine.set_color(frame_colour)
        spine.set_linewidth(3)

    # Ensure the shared boundary is aligned
    label_ax.spines["right"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)

    fig.subplots_adjust(
        left=0.06,
        right=0.97,
        bottom=0.20,
        top=0.88
    )

    results = {
        "n_subjects": len(r),
        "median_r": median_r,
        "fisher_mean_r": group_r,
        "ci_low": ci_low,
        "ci_high": ci_high
    }

    return fig, results


def plot_qcfc_matrix(fig_dir, r_mat, network_labels_csv=DEFAULT_NETWORK_LABELS_CSV) -> None:
    hemi_network = load_network_labels(network_labels_csv)  # e.g. "LH_Vis", len 400
    networks = np.array([label.split("_", 1)[1] for label in hemi_network])
    n_rois = len(hemi_network)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(r_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("QC-FC matrix")
    ax.set_xticks([])
    ax.set_yticks([])

    # Divide lines at every hemisphere+network block boundary (14 blocks:
    # 7 networks x 2 hemispheres, contiguous in this atlas's label order).
    boundaries = np.where(hemi_network[1:] != hemi_network[:-1])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=0.6, alpha=0.5)
        ax.axvline(b, color="black", linewidth=0.6, alpha=0.5)

    # make_axes_locatable derives each appended axis from ax's own (post
    # aspect-adjustment) box, so the network strips/colorbar always align to
    # the matrix's actual rendered extent exactly, regardless of figure size.
    divider = make_axes_locatable(ax)

    network_rgb = np.array([to_rgb(YEO7_NETWORK_COLORS[n]) for n in networks])
    left_ax = divider.append_axes("left", size="4%", pad=0.05)
    left_ax.imshow(network_rgb.reshape(n_rois, 1, 3), aspect="auto")
    left_ax.set_xticks([])
    left_ax.set_yticks([])
    left_ax.set_ylabel("ROI")

    top_ax = divider.append_axes("top", size="4%", pad=0.05)
    top_ax.imshow(network_rgb.reshape(1, n_rois, 3), aspect="auto")
    top_ax.set_xticks([])
    top_ax.set_yticks([])
    top_ax.set_xlabel("ROI")
    top_ax.xaxis.set_label_position("top")

    cax = divider.append_axes("right", size="5%", pad=0.6)
    fig.colorbar(im, cax=cax, label="QC-FC (r)")

    legend_handles = [Patch(facecolor=colour, label=name) for name, colour in YEO7_NETWORK_COLORS.items()]
    fig.legend(
        handles=legend_handles, title="Yeo-7 network", loc="lower center",
        ncol=7, bbox_to_anchor=(0.46, -0.04), frameon=False, fontsize=8, title_fontsize=9,
    )

    fig.savefig(os.path.join(fig_dir, "qc_fc_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qcfc_distribution(fig_dir, qcfc_r, method_name) -> None:
    # Distribution of QC-FC across all unique upper-triangle edges (raw,
    # signed r values -- no Fisher-z/recentering). Zero-line marks no
    # distance/motion dependence; the median |QC-FC| is annotated as text.
    # xlim is sized to the actual data range (default (-0.30, 0.45) is an
    # adult-study convention that clips this dataset's wider spread and
    # crowds the distribution against the method-label panel).
    finite_qcfc_r = qcfc_r[np.isfinite(qcfc_r)]
    margin = 0.05 * (finite_qcfc_r.max() - finite_qcfc_r.min())
    xlim = (finite_qcfc_r.min() - margin, finite_qcfc_r.max() + margin)
    fig, _ = plot_single_qcfc_distribution(
        qcfc_r, method_name=method_name, plot_absolute=False, xlim=xlim,
    )
    fig.savefig(os.path.join(fig_dir, "qc_fc_distribution.png"), dpi=200)
    plt.close(fig)


def plot_fd_dvars_correlation_figure(fig_dir, fd_dvars_values, method_name) -> None:
    fig, _ = plot_subject_fd_dvars_correlations(fd_dvars_values, method_name=method_name)
    fig.savefig(os.path.join(fig_dir, "fd_dvars_correlation.png"), dpi=200)
    plt.close(fig)


def plot_qcfc_dd_figure(fig_dir, dist_mat, r_mat) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_qcfc_distance_dependence(dist_mat, r_mat, title="QC-FC distance dependence", ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "qc_fc_distance_dependence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_modularity_q_violin(fig_dir, q_values) -> None:
    fig, ax = plt.subplots(figsize=(5, 6))
    sns.violinplot(y=q_values, ax=ax, width=0.7, inner="quartile", color="#54A24B", alpha=0.6)
    sns.stripplot(y=q_values, ax=ax, color="black", size=3, jitter=0.15, alpha=0.4)
    ax.set_xlabel(f"Subjects (n={len(q_values)})")
    ax.set_ylabel("Modularity Q")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Modularity Q")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "modularity_q_violin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_modularity_q_vs_fd(fig_dir, fd, q_values) -> None:
    fig = plot_q_vs_fd(fd, q_values)
    fig.savefig(os.path.join(fig_dir, "modularity_q_vs_fd.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dvars_tsnr_violin(fig_dir, dvars_values, tsnr_values) -> None:
    sim_df = pd.DataFrame({"Mean_DVARS": dvars_values, "Mean_tSNR": tsnr_values})
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    metrics = [
        ("Mean_DVARS", "Mean DVARS", "#4C78A8"),
        ("Mean_tSNR", "Mean tSNR", "#F58518"),
    ]
    for ax, (column, ylabel, colour) in zip(axes, metrics):
        sns.violinplot(
            data=sim_df, y=column, ax=ax, width=0.7,
            inner="quartile", color=colour, alpha=0.6,
        )
        sns.stripplot(
            data=sim_df, y=column, ax=ax,
            color="black", size=3, jitter=0.15, alpha=0.4,
        )
        ax.set_xlabel(f"Subjects (n={len(sim_df)})")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    plt.suptitle(f"DVARS and tSNR (nipype convention, n={len(sim_df)} subjects)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dvars_tsnr_violin.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dvars_tsnr_gs_boxplot(fig_dir, dvars_values, tsnr_values, gs_std_values) -> None:
    df = pd.DataFrame({
        "DVARS": dvars_values, "tSNR": tsnr_values, "Global signal std": gs_std_values,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    metrics = [
        ("DVARS", "#4C78A8"),
        ("tSNR", "#F58518"),
        ("Global signal std", "#54A24B"),
    ]
    for ax, (column, colour) in zip(axes, metrics):
        sns.boxplot(data=df, y=column, ax=ax, width=0.5, color=colour)
        sns.stripplot(data=df, y=column, ax=ax, color="black", size=3, jitter=0.15, alpha=0.4)
        ax.set_xlabel(f"Subjects (n={len(df)})")
        ax.set_ylabel(column)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"DVARS, tSNR, global signal std (n={len(df)} subjects)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dvars_tsnr_gs_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_connectome_figure(fig_dir, r_mat, fdr_mat, connectome_atlas_path) -> None:
    if not fdr_mat.any():
        print("no FDR-significant edges -- skipping connectome plot")
        return

    # Nilearn-style connectome: default schematic glass brain (no real
    # anatomy), node coords from the standard adult FSLMNI152 Schaefer
    # atlas -- a different template than the infant-space atlas used for
    # the distance matrix/QC-FC-DD. ROI identity (label order) still
    # matches; only the display coordinates come from the adult template,
    # for a familiar/legible schematic rather than an anatomically-exact
    # infant background.
    coords, _ = plotting.find_parcellation_cut_coords(
        connectome_atlas_path, return_label_names=True
    )
    # Every edge drawn already passed FDR correction -- draw all of them
    # (no further top-percentile filtering), colored by strength |r|:
    # green (weakest) -> yellow (moderate) -> red (strongest).
    strength = np.where(fdr_mat, np.abs(r_mat), 0)
    n_significant = int(fdr_mat.sum() / 2)
    vmax = float(np.abs(r_mat[fdr_mat]).max())
    cmap = "RdYlGn_r"

    # Aspect ratio matches nilearn's own default ortho proportions
    # (~2.09:1 width:height) scaled up -- anything taller lets each
    # GlassBrainAxes letterbox its equal-aspect brain drawing, leaving
    # large blank margins above/below.
    #
    # The 3 views (sagittal/coronal/axial) are drawn into individually
    # positioned axes -- rather than nilearn's single "ortho" call, which
    # leaves no room for a colorbar -- so the axial panel sits flush
    # against the coronal one, with the colorbar in its own column at
    # the far right.
    fig = plt.figure(figsize=(16, 7.7))
    x_rect = (0.00, 0.0, 0.30, 1.0)
    y_rect = (0.30, 0.0, 0.30, 1.0)
    z_rect = (0.60, 0.0, 0.32, 1.0)  # flush against y_rect; colorbar goes after it
    common_kwargs = dict(
        adjacency_matrix=strength,
        node_coords=coords,
        node_color="#333333",
        node_size=14,
        edge_cmap=cmap,
        edge_vmin=0.0,
        edge_vmax=vmax,
        edge_kwargs={"linewidth": 2.2, "alpha": 1.0},
        annotate=False,
        colorbar=False,
        figure=fig,
    )
    plotting.plot_connectome(
        display_mode="x", axes=x_rect,
        title=f"QC-FC significant edges (n={n_significant:,})", **common_kwargs,
    )
    plotting.plot_connectome(display_mode="y", axes=y_rect, **common_kwargs)
    plotting.plot_connectome(display_mode="z", axes=z_rect, **common_kwargs)

    # Measure the axial panel's actual rendered content (brain outline +
    # edges) so the colorbar -- placed in its own column just to the
    # right -- can match its true visual height exactly, rather than the
    # axes' full (letterboxed) bounding box.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img = plt.imread(buf)
    img_h, img_w = img.shape[:2]
    x0_px = int(z_rect[0] * img_w)
    x1_px = int((z_rect[0] + z_rect[2]) * img_w)
    nonwhite_rows = np.where(np.any(img[:, x0_px:x1_px, :3] < 0.98, axis=(1, 2)))[0]
    cbar_y0 = 1 - nonwhite_rows.max() / img_h
    cbar_y1 = 1 - nonwhite_rows.min() / img_h

    cbar_x0 = z_rect[0] + z_rect[2] + 0.015
    cbar_ax = fig.add_axes([cbar_x0, cbar_y0, 0.02, cbar_y1 - cbar_y0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("|QC-FC r|", fontsize=12)

    fig.savefig(os.path.join(fig_dir, "connectome.png"), dpi=150)
    plt.close(fig)


def make_plots(
    output_dir, method_name, connectome_atlas_path,
    r_mat, fdr_mat, dist_mat, qcfc_r, q_values, fd, dvars_values, tsnr_values, gs_std_values,
    fd_dvars_values, network_labels_csv=DEFAULT_NETWORK_LABELS_CSV,
) -> None:
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_qcfc_matrix(fig_dir, r_mat, network_labels_csv)
    plot_qcfc_distribution(fig_dir, qcfc_r, method_name)
    plot_fd_dvars_correlation_figure(fig_dir, fd_dvars_values, method_name)
    plot_qcfc_dd_figure(fig_dir, dist_mat, r_mat)
    plot_modularity_q_violin(fig_dir, q_values)
    plot_modularity_q_vs_fd(fig_dir, fd, q_values)
    plot_dvars_tsnr_violin(fig_dir, dvars_values, tsnr_values)
    plot_dvars_tsnr_gs_boxplot(fig_dir, dvars_values, tsnr_values, gs_std_values)
    plot_connectome_figure(fig_dir, r_mat, fdr_mat, connectome_atlas_path)

    print(f"saved plots -> {fig_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate denoising-evaluation figures from saved results")
    parser.add_argument(
        "--result_dir", required=True,
        help="Pipeline output_dir written by denoising_evaluation.py (contains manifest.csv, qc_fc_*.npy, ...)",
    )
    parser.add_argument(
        "--method_name", default="Raw",
        help="Label shown on plots, e.g. 'Raw' or 'ST v4 corrected'",
    )
    parser.add_argument("--connectome_atlas_path", default=DEFAULT_CONNECTOME_ATLAS_PATH)
    parser.add_argument("--network_labels_csv", default=DEFAULT_NETWORK_LABELS_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_results(args.result_dir)
    make_plots(
        args.result_dir, args.method_name, args.connectome_atlas_path,
        network_labels_csv=args.network_labels_csv, **data,
    )
