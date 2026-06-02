#!/usr/bin/env python3
"""
qc_chunk_analysis.py
====================
Chunk-level QC analysis for CycleGAN fMRI dataset.

Analyses per chunk, with SEPARATE outputs for each domain
(A_corrupted / B_motion_free), plus side-by-side comparison plots:
    1. Shape / affine / header consistency
    2. NaN / Inf / intensity range check
    3. FD distribution  (recomputed from .par, radius=50mm, chunk indices)
    4. DVARS distribution (computed from NIfTI chunks)
    5. tSNR distribution  (computed from NIfTI chunks)
    6. Global signal stability

Output layout
-------------
    <output_dir>/
        plots/
            val_overview_dashboard.png
            val_fd_distribution.png
            val_dvars_distribution.png
            val_tsnr_distribution.png
            val_global_signal_stability.png
            val_intensity_range.png
            val_shape_consistency.png
        analysis_output/
            corrupted/
                val_chunk_level_metrics.csv
                val_distribution_summary.csv
                val_qc_flags.csv
                val_shape_report.csv
            motion_free/
                val_chunk_level_metrics.csv
                val_distribution_summary.csv
                val_qc_flags.csv
                val_shape_report.csv

Usage
-----
    python qc_chunk_analysis.py \\
        --metadata_dir  cyclegan_dataset/metadata \\
        --dataset_dir   cyclegan_dataset \\
        --split         val \\
        --output_dir    qc_results \\
        --n_jobs        8

    # Both val and test
    python qc_chunk_analysis.py ... --split val test
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONSTANTS
# =============================================================================
FD_RADIUS        = 50.0
ALL_SPLITS       = ["train", "val", "test"]
DOMAIN_CORRUPTED = "A_corrupted"
DOMAIN_MOTFREE   = "B_motion_free"

# human-readable labels used in filenames and plot titles
DOMAIN_LABEL = {
    "corrupted"  : "High-Motion (Corrupted)",
    "motion_free": "Low-Motion (Motion-free)",
}
DOMAIN_SHORT = {
    "corrupted"  : "corrupted",
    "motion_free": "motion_free",
}
COLORS = {
    "corrupted"  : "#E05C5C",
    "motion_free": "#5C8AE0",
}

METRICS_FOR_SUMMARY = [
    "par_fd_mean", "par_fd_max", "par_fd_std",
    "csv_mean_fd", "csv_max_fd",
    "dvars_mean", "dvars_max",
    "tsnr_mean", "tsnr_median",
    "gs_mean", "gs_std",
    "intensity_min", "intensity_max", "intensity_mean",
]

# =============================================================================
# LOGGING
# =============================================================================
def setup_logging() -> logging.Logger:
    log = logging.getLogger("qc_analysis")
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                datefmt="%H:%M:%S")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    return log

log = setup_logging()


# =============================================================================
# ARGUMENT PARSING
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk-level QC analysis for CycleGAN fMRI dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata_dir", type=Path,
                        default=Path("cyclegan_dataset/metadata"))
    parser.add_argument("--dataset_dir",  type=Path,
                        default=Path("cyclegan_dataset"))
    parser.add_argument("--split", nargs="+", choices=ALL_SPLITS,
                        default=["val"], dest="splits")
    parser.add_argument("--output_dir", type=Path, default=Path("qc_results"))
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--fd_threshold", type=float, default=0.5,
                        help="FD threshold (mm) for flagging.")
    parser.add_argument("--tsnr_min", type=float, default=10.0,
                        help="Min acceptable tSNR.")
    return parser.parse_args()


# =============================================================================
# FD FROM .par   (rotations-first: rx ry rz tx ty tz, radians)
# =============================================================================
def compute_fd_from_par(par_path: str,
                        chunk_start: int,
                        chunk_end: int,
                        radius: float = FD_RADIUS) -> np.ndarray:
    try:
        params = np.loadtxt(par_path)
    except Exception:
        return np.array([np.nan])

    if params.ndim == 1:
        params = params[np.newaxis, :]

    chunk = params[chunk_start : chunk_end + 1]
    if len(chunk) < 2:
        return np.array([np.nan])

    chunk_mm = chunk.copy()
    chunk_mm[:, :3] *= radius          # radians → mm arc-length

    fd = np.abs(np.diff(chunk_mm, axis=0)).sum(axis=1)
    return fd


# =============================================================================
# NIfTI METRICS
# =============================================================================
def compute_nifti_metrics(chunk_path: str) -> dict:
    result = dict(
        shape=None, voxel_size=None, has_nan=None, has_inf=None,
        intensity_min=np.nan, intensity_max=np.nan, intensity_mean=np.nan,
        dvars_mean=np.nan, dvars_max=np.nan,
        tsnr_mean=np.nan, tsnr_median=np.nan,
        gs_mean=np.nan, gs_std=np.nan,
        n_vols=None, load_error=None,
    )
    try:
        img  = nib.load(chunk_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception as e:
        result["load_error"] = str(e)
        return result

    if data.ndim != 4:
        result["load_error"] = f"Not 4D: {data.shape}"
        return result

    vox = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    result.update(
        shape      = str(data.shape),
        voxel_size = str(np.round(vox, 3).tolist()),
        has_nan    = bool(np.any(np.isnan(data))),
        has_inf    = bool(np.any(np.isinf(data))),
        n_vols     = data.shape[3],
    )

    clean = data.copy()
    clean[~np.isfinite(clean)] = 0.0
    result.update(
        intensity_min  = float(clean.min()),
        intensity_max  = float(clean.max()),
        intensity_mean = float(clean.mean()),
    )

    if data.shape[3] > 1:
        diff  = np.diff(clean, axis=3)
        dvars = np.sqrt((diff ** 2).mean(axis=(0, 1, 2)))
        result["dvars_mean"] = float(dvars.mean())
        result["dvars_max"]  = float(dvars.max())

        ts_mean = clean.mean(axis=3)
        ts_std  = clean.std(axis=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tsnr = np.where(ts_std > 0, ts_mean / ts_std, 0.0)
        brain_mask = ts_mean > ts_mean.mean() * 0.1
        if brain_mask.any():
            result["tsnr_mean"]   = float(tsnr[brain_mask].mean())
            result["tsnr_median"] = float(np.median(tsnr[brain_mask]))

    spatial_mean = clean.mean(axis=(0, 1, 2))
    result["gs_mean"] = float(spatial_mean.mean())
    result["gs_std"]  = float(spatial_mean.std())

    return result


# =============================================================================
# PER-CHUNK WORKER
# =============================================================================
def process_chunk(row: pd.Series, domain: str) -> dict:
    record = dict(
        domain       = domain,
        subject_id   = row["subject_id"],
        session_id   = row["session_id"],
        run_id       = row["run_id"],
        task         = row.get("task", ""),
        chunk_start  = int(row["chunk_start"]),
        chunk_end    = int(row["chunk_end"]),
        chunk_path   = row.get("chunk_path", ""),
        csv_mean_fd  = row.get("chunk_mean_fd", np.nan),
        csv_max_fd   = row.get("chunk_max_fd",  np.nan),
    )

    par = row.get("motion_parameter_file", "")
    if par and Path(str(par)).exists():
        fd_vals = compute_fd_from_par(str(par),
                                      int(row["chunk_start"]),
                                      int(row["chunk_end"]))
        record["par_fd_mean"] = float(np.nanmean(fd_vals))
        record["par_fd_max"]  = float(np.nanmax(fd_vals))
        record["par_fd_std"]  = float(np.nanstd(fd_vals))
    else:
        record["par_fd_mean"] = np.nan
        record["par_fd_max"]  = np.nan
        record["par_fd_std"]  = np.nan

    chunk_path = row.get("chunk_path", "")
    nifti_m = (compute_nifti_metrics(str(chunk_path))
               if chunk_path and Path(str(chunk_path)).exists()
               else {"load_error": "chunk_path missing or not on disk"})

    record.update({
        "shape"         : nifti_m.get("shape"),
        "voxel_size"    : nifti_m.get("voxel_size"),
        "has_nan"       : nifti_m.get("has_nan"),
        "has_inf"       : nifti_m.get("has_inf"),
        "intensity_min" : nifti_m.get("intensity_min",  np.nan),
        "intensity_max" : nifti_m.get("intensity_max",  np.nan),
        "intensity_mean": nifti_m.get("intensity_mean", np.nan),
        "dvars_mean"    : nifti_m.get("dvars_mean",     np.nan),
        "dvars_max"     : nifti_m.get("dvars_max",      np.nan),
        "tsnr_mean"     : nifti_m.get("tsnr_mean",      np.nan),
        "tsnr_median"   : nifti_m.get("tsnr_median",    np.nan),
        "gs_mean"       : nifti_m.get("gs_mean",        np.nan),
        "gs_std"        : nifti_m.get("gs_std",         np.nan),
        "n_vols"        : nifti_m.get("n_vols"),
        "load_error"    : nifti_m.get("load_error"),
    })
    return record


# =============================================================================
# DOMAIN-LEVEL SUMMARY HELPERS
# =============================================================================
def domain_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS_FOR_SUMMARY:
        if metric not in df.columns:
            continue
        vals = df[metric].dropna()
        if vals.empty:
            continue
        rows.append(dict(
            metric   = metric,
            n        = len(vals),
            mean     = vals.mean(),
            std      = vals.std(),
            median   = vals.median(),
            q25      = vals.quantile(0.25),
            q75      = vals.quantile(0.75),
            min      = vals.min(),
            max      = vals.max(),
        ))
    return pd.DataFrame(rows)


def domain_flag_qc(df: pd.DataFrame,
                   fd_thr: float,
                   tsnr_min: float) -> pd.DataFrame:
    flags = []
    for _, row in df.iterrows():
        problems = []
        if row.get("has_nan"):    problems.append("has_nan")
        if row.get("has_inf"):    problems.append("has_inf")
        if row.get("load_error"): problems.append(f"load_error:{row['load_error']}")
        if pd.notna(row.get("par_fd_mean")) and row["par_fd_mean"] > fd_thr:
            problems.append(f"high_fd_mean:{row['par_fd_mean']:.3f}")
        if pd.notna(row.get("tsnr_mean")) and row["tsnr_mean"] < tsnr_min:
            problems.append(f"low_tsnr:{row['tsnr_mean']:.2f}")
        if problems:
            flags.append({**row.to_dict(), "qc_flags": "|".join(problems)})
    return pd.DataFrame(flags)


def domain_shape_report(df: pd.DataFrame) -> pd.DataFrame:
    shapes = df["shape"].dropna().value_counts()
    voxels = df["voxel_size"].dropna().value_counts()
    return pd.DataFrame([{
        "n_chunks"          : len(df),
        "unique_shapes"     : len(shapes),
        "most_common_shape" : shapes.index[0] if len(shapes) else None,
        "shape_inconsistent": len(shapes) > 1,
        "unique_voxsizes"   : len(voxels),
        "most_common_vox"   : voxels.index[0] if len(voxels) else None,
        "shape_counts"      : shapes.to_dict(),
    }])


def save_domain_csvs(df: pd.DataFrame,
                     domain: str,
                     split: str,
                     csv_dir: Path,
                     fd_thr: float,
                     tsnr_min: float) -> None:
    """Save all per-domain CSVs into analysis_output/<domain>/"""
    out = csv_dir / DOMAIN_SHORT[domain]
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / f"{split}_chunk_level_metrics.csv", index=False)

    domain_distribution_summary(df).to_csv(
        out / f"{split}_distribution_summary.csv", index=False)

    flags = domain_flag_qc(df, fd_thr, tsnr_min)
    flags.to_csv(out / f"{split}_qc_flags.csv", index=False)

    domain_shape_report(df).to_csv(
        out / f"{split}_shape_report.csv", index=False)

    log.info(f"    [{DOMAIN_SHORT[domain]}] metrics={len(df)}  "
             f"flagged={len(flags)}")


# =============================================================================
# PLOTTING
# =============================================================================
def _violin(ax, data_c, data_m, ylabel, title):
    clean_c = data_c[np.isfinite(data_c)]
    clean_m = data_m[np.isfinite(data_m)]

    # Build list of (position, data, color) only for non-empty arrays
    valid = [
        (pos, d, col)
        for pos, d, col in [
            (1, clean_c, COLORS["corrupted"]),
            (2, clean_m, COLORS["motion_free"]),
        ]
        if len(d) > 1          # violinplot needs at least 2 points
    ]

    if not valid:
        ax.set_title(f"{title}\n(no data)", fontsize=9, color="grey")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["High-Motion\n(Corrupted)",
                             "Low-Motion\n(Motion-free)"], fontsize=8)
        log.warning(f"  _violin: no finite data for '{title}' — skipping plot")
        return

    positions = [v[0] for v in valid]
    datasets  = [v[1] for v in valid]
    colors    = [v[2] for v in valid]

    parts = ax.violinplot(
        datasets, positions=positions,
        showmedians=True, showextrema=True,
    )
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)
    for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
        parts[part].set_color("black")
        parts[part].set_linewidth(1.2)

    # Grey "no data" label for any missing domain
    for pos, d, label in [
        (1, clean_c, "High-Motion\n(Corrupted)"),
        (2, clean_m, "Low-Motion\n(Motion-free)"),
    ]:
        if len(d) <= 1:
            ax.text(pos, 0, "no data", ha="center", va="center",
                    fontsize=8, color="grey", style="italic")
            log.warning(f"  _violin: '{title}' — "
                        f"'{label.replace(chr(10),' ')}' has {len(d)} points")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["High-Motion\n(Corrupted)", "Low-Motion\n(Motion-free)"],
                       fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # stats annotation
    for x, d, col in [(1, clean_c, COLORS["corrupted"]),
                      (2, clean_m, COLORS["motion_free"])]:
        if len(d) > 1:
            ylim = ax.get_ylim()
            ax.text(x, ylim[0] + (ylim[1] - ylim[0]) * 0.01,
                    f"n={len(d)}\nμ={d.mean():.3f}\nσ={d.std():.3f}",
                    ha="center", va="bottom", fontsize=6.5, color=col)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved plot → {path.name}")


def _legend(fig):
    fig.legend(handles=[
        Patch(facecolor=COLORS["corrupted"],   label="High-Motion (Corrupted)"),
        Patch(facecolor=COLORS["motion_free"], label="Low-Motion (Motion-free)"),
    ], loc="upper right", fontsize=8)


def plot_fd(df_c, df_m, plots_dir, split):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"[{split.upper()}]  FD Distribution — "
                 f"High-Motion vs Low-Motion", fontsize=12, fontweight="bold")
    for ax, col, label in zip(
        axes,
        ["par_fd_mean", "par_fd_max", "par_fd_std"],
        ["Mean FD (mm)", "Max FD (mm)", "FD Std (mm)"],
    ):
        _violin(ax, df_c[col].dropna().values, df_m[col].dropna().values,
                label, label)
    _legend(fig)
    plt.tight_layout()
    _save(fig, plots_dir / f"{split}_fd_distribution.png")


def plot_dvars(df_c, df_m, plots_dir, split):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"[{split.upper()}]  DVARS Distribution",
                 fontsize=12, fontweight="bold")
    for ax, col, label in zip(
        axes,
        ["dvars_mean", "dvars_max"],
        ["Mean DVARS", "Max DVARS"],
    ):
        _violin(ax, df_c[col].dropna().values, df_m[col].dropna().values,
                label, label)
    _legend(fig)
    plt.tight_layout()
    _save(fig, plots_dir / f"{split}_dvars_distribution.png")


def plot_tsnr(df_c, df_m, plots_dir, split):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"[{split.upper()}]  tSNR Distribution",
                 fontsize=12, fontweight="bold")
    for ax, col, label in zip(
        axes,
        ["tsnr_mean", "tsnr_median"],
        ["Mean tSNR", "Median tSNR"],
    ):
        _violin(ax, df_c[col].dropna().values, df_m[col].dropna().values,
                label, label)
    _legend(fig)
    plt.tight_layout()
    _save(fig, plots_dir / f"{split}_tsnr_distribution.png")


def plot_global_signal(df_c, df_m, plots_dir, split):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"[{split.upper()}]  Global Signal Stability",
                 fontsize=12, fontweight="bold")
    for ax, col, label in zip(
        axes,
        ["gs_mean", "gs_std"],
        ["Global Signal Mean", "Global Signal Std (stability)"],
    ):
        _violin(ax, df_c[col].dropna().values, df_m[col].dropna().values,
                label, label)
    _legend(fig)
    plt.tight_layout()
    _save(fig, plots_dir / f"{split}_global_signal_stability.png")


def plot_intensity(df_c, df_m, plots_dir, split):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"[{split.upper()}]  Intensity Range",
                 fontsize=12, fontweight="bold")
    for ax, col, label in zip(
        axes,
        ["intensity_min", "intensity_max", "intensity_mean"],
        ["Intensity Min", "Intensity Max", "Intensity Mean"],
    ):
        _violin(ax, df_c[col].dropna().values, df_m[col].dropna().values,
                label, label)
    _legend(fig)
    plt.tight_layout()
    _save(fig, plots_dir / f"{split}_intensity_range.png")


def plot_shape_consistency(df_c, df_m, plots_dir, split):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"[{split.upper()}]  Shape Consistency",
                 fontsize=12, fontweight="bold")
    for ax, df, domain, color in zip(
        axes,
        [df_c, df_m],
        ["corrupted", "motion_free"],
        [COLORS["corrupted"], COLORS["motion_free"]],
    ):
        counts = df["shape"].value_counts().head(10)
        ax.barh(counts.index.astype(str), counts.values,
                color=color, alpha=0.8)
        ax.set_xlabel("Count")
        ax.set_title(f"{DOMAIN_LABEL[domain]}  (n={len(df)})", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, plots_dir / f"{split}_shape_consistency.png")


def plot_overview_dashboard(df_c, df_m, plots_dir, split):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"[{split.upper()}]  QC Overview Dashboard\n"
        f"High-Motion (Corrupted) n={len(df_c)}  |  "
        f"Low-Motion (Motion-free) n={len(df_m)}",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

    metrics = [
        ("par_fd_mean",    "Mean FD (mm)"),
        ("dvars_mean",     "Mean DVARS"),
        ("tsnr_mean",      "Mean tSNR"),
        ("gs_std",         "GS Std (stability)"),
        ("intensity_mean", "Intensity Mean"),
    ]
    for i, (col, label) in enumerate(metrics):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _violin(ax,
                df_c[col].dropna().values,
                df_m[col].dropna().values,
                label, label)

    # NaN / Inf bar chart
    ax_flag = fig.add_subplot(gs[1, 2])
    categories = ["NaN\nHigh-Motion", "NaN\nLow-Motion",
                  "Inf\nHigh-Motion", "Inf\nLow-Motion"]
    vals = [
        int(df_c["has_nan"].sum()) if "has_nan" in df_c else 0,
        int(df_m["has_nan"].sum()) if "has_nan" in df_m else 0,
        int(df_c["has_inf"].sum()) if "has_inf" in df_c else 0,
        int(df_m["has_inf"].sum()) if "has_inf" in df_m else 0,
    ]
    bar_colors = [COLORS["corrupted"], COLORS["motion_free"]] * 2
    ax_flag.bar(categories, vals, color=bar_colors, alpha=0.8)
    ax_flag.set_title("NaN / Inf Chunks", fontsize=9)
    ax_flag.set_ylabel("Count")
    ax_flag.grid(axis="y", alpha=0.3)

    _legend(fig)
    _save(fig, plots_dir / f"{split}_overview_dashboard.png")



# =============================================================================
# PATCH MISSING chunk_path COLUMN
# =============================================================================
CHUNK_FILENAME_TEMPLATE = (
    "sub-{subject_id}_ses-{session_id}_task-{task}"
    "_run-{run_id}_chunk-{chunk_start}-{chunk_end}.nii.gz"
)

def patch_chunk_paths(df: pd.DataFrame,
                      dataset_dir: Path,
                      split: str,
                      domain_dir_name: str) -> pd.DataFrame:
    """
    If chunk_path column is missing or entirely empty, reconstruct it from
    the other metadata columns using the same filename template as the
    build script.  Logs how many paths were found vs missing on disk.
    """
    if "chunk_path" in df.columns and df["chunk_path"].notna().any():
        return df   # already populated, nothing to do

    log.warning(
        f"  chunk_path column missing or empty for {split}/{domain_dir_name} "
        f"— reconstructing from metadata columns..."
    )

    domain_dir = dataset_dir / split / domain_dir_name

    def _build_path(row):
        fname = CHUNK_FILENAME_TEMPLATE.format(
            subject_id  = row["subject_id"],
            session_id  = row["session_id"],
            task        = row["task"],
            run_id      = str(row["run_id"]).zfill(3),
            chunk_start = int(row["chunk_start"]),
            chunk_end   = int(row["chunk_end"]),
        )
        return str(domain_dir / fname)

    df = df.copy()
    df["chunk_path"] = df.apply(_build_path, axis=1)

    n_found   = df["chunk_path"].apply(lambda p: Path(p).exists()).sum()
    n_missing = len(df) - n_found
    log.info(
        f"  Reconstructed {len(df)} chunk paths — "
        f"found on disk: {n_found}  missing: {n_missing}"
    )
    if n_missing > 0:
        log.warning(
            f"  {n_missing} chunk files not found on disk. "
            f"NIfTI metrics will be NaN for those chunks."
        )
    return df


# =============================================================================
# LOAD METADATA
# =============================================================================
def load_split_metadata(metadata_dir: Path,
                        split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split == "train":
        c_path  = metadata_dir / "train_corrupted_all.csv"
        mf_path = metadata_dir / "train_motion_free.csv"
    else:
        c_path  = metadata_dir / f"{split}_corrupted_balanced.csv"
        mf_path = metadata_dir / f"{split}_motion_free.csv"

    for p in [c_path, mf_path]:
        if not p.exists():
            log.error(f"Metadata CSV not found: {p}")
            sys.exit(1)

    df_c  = pd.read_csv(c_path)
    df_mf = pd.read_csv(mf_path)
    log.info(f"  [{split}] corrupted={len(df_c)}  motion_free={len(df_mf)}")
    return df_c, df_mf


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================
def print_domain_summary(df: pd.DataFrame, domain: str, split: str):
    log.info(f"\n  ── {DOMAIN_LABEL[domain]} [{split}]  "
             f"(n={len(df)}) ────────────────────")
    for col, label in [
        ("par_fd_mean", "Mean FD      "),
        ("dvars_mean",  "Mean DVARS   "),
        ("tsnr_mean",   "Mean tSNR    "),
        ("gs_std",      "GS Std       "),
    ]:
        if col not in df.columns:
            continue
        v = df[col].dropna()
        if len(v):
            log.info(
                f"    {label}: "
                f"mean={v.mean():.4f}  std={v.std():.4f}  "
                f"median={v.median():.4f}  "
                f"IQR=[{v.quantile(0.25):.4f}, {v.quantile(0.75):.4f}]"
            )
    nan_c = int(df["has_nan"].sum()) if "has_nan" in df else 0
    inf_c = int(df["has_inf"].sum()) if "has_inf" in df else 0
    log.info(f"    NaN chunks : {nan_c}  |  Inf chunks : {inf_c}")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args   = parse_args()
    splits = [s for s in ALL_SPLITS if s in args.splits]

    log.info("=" * 65)
    log.info("  QC CHUNK ANALYSIS")
    log.info("=" * 65)
    log.info(f"  Metadata dir  : {args.metadata_dir}")
    log.info(f"  Dataset dir   : {args.dataset_dir}")
    log.info(f"  Splits        : {splits}")
    log.info(f"  Output dir    : {args.output_dir}")
    log.info(f"  Workers       : {args.n_jobs}")
    log.info(f"  FD threshold  : {args.fd_threshold} mm")
    log.info(f"  tSNR min      : {args.tsnr_min}")
    log.info("=" * 65)

    plots_dir = args.output_dir / "plots"
    csv_dir   = args.output_dir / "analysis_output"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True,   exist_ok=True)

    for split in splits:
        log.info(f"\n{'─' * 55}")
        log.info(f"  {split.upper()}")
        log.info(f"{'─' * 55}")

        df_c_meta, df_mf_meta = load_split_metadata(args.metadata_dir, split)

        # ── Patch missing chunk_path column (e.g. train CSVs saved before
        #    extraction so chunk_path was never written back) ────────────────
        df_c_meta  = patch_chunk_paths(df_c_meta,  args.dataset_dir,
                                       split, DOMAIN_CORRUPTED)
        df_mf_meta = patch_chunk_paths(df_mf_meta, args.dataset_dir,
                                       split, DOMAIN_MOTFREE)

        # ── Compute metrics in parallel ───────────────────────────────────
        tasks = (
            [(row, "corrupted")   for _, row in df_c_meta.iterrows()] +
            [(row, "motion_free") for _, row in df_mf_meta.iterrows()]
        )
        log.info(f"  Computing metrics for {len(tasks)} chunks "
                 f"({args.n_jobs} workers)...")

        records = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
            delayed(process_chunk)(row, domain) for row, domain in tasks
        )

        df_all = pd.DataFrame(records)
        df_all["split"] = split

        # ── Split by domain ───────────────────────────────────────────────
        df_corrupted   = df_all[df_all["domain"] == "corrupted"].copy()
        df_motion_free = df_all[df_all["domain"] == "motion_free"].copy()

        # ── Diagnostic: log any columns that are entirely NaN ─────────────
        log.info("  Checking for all-NaN metric columns...")
        for label, df_check in [("corrupted",   df_corrupted),
                                 ("motion_free", df_motion_free)]:
            all_nan = [c for c in METRICS_FOR_SUMMARY
                       if c in df_check.columns
                       and df_check[c].isna().all()]
            if all_nan:
                log.warning(
                    f"  [{label}] ALL-NaN columns (will show no data in plots): "
                    f"{all_nan}"
                )
                log.warning(
                    f"  [{label}] This usually means chunk_path is missing or "
                    f"NIfTI files were not found on disk. "
                    f"Check chunk_path column in the metadata CSV."
                )
            else:
                log.info(f"  [{label}] All metric columns have valid data OK")

        # ── Save separate CSVs per domain ─────────────────────────────────
        log.info("  Saving CSVs...")
        save_domain_csvs(df_corrupted,   "corrupted",
                         split, csv_dir, args.fd_threshold, args.tsnr_min)
        save_domain_csvs(df_motion_free, "motion_free",
                         split, csv_dir, args.fd_threshold, args.tsnr_min)

        # ── Plots (side-by-side comparison) ───────────────────────────────
        log.info("  Generating plots...")
        plot_fd(df_corrupted, df_motion_free, plots_dir, split)
        plot_dvars(df_corrupted, df_motion_free, plots_dir, split)
        plot_tsnr(df_corrupted, df_motion_free, plots_dir, split)
        plot_global_signal(df_corrupted, df_motion_free, plots_dir, split)
        plot_intensity(df_corrupted, df_motion_free, plots_dir, split)
        plot_shape_consistency(df_corrupted, df_motion_free, plots_dir, split)
        plot_overview_dashboard(df_corrupted, df_motion_free, plots_dir, split)

        # ── Console summary ───────────────────────────────────────────────
        print_domain_summary(df_corrupted,   "corrupted",   split)
        print_domain_summary(df_motion_free, "motion_free", split)

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  COMPLETE")
    log.info("=" * 65)
    log.info(f"  Plots             → {plots_dir.resolve()}")
    log.info(f"  CSVs (corrupted)  → {(csv_dir / 'corrupted').resolve()}")
    log.info(f"  CSVs (mot-free)   → {(csv_dir / 'motion_free').resolve()}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()