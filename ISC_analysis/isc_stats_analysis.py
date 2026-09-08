#!/usr/bin/env python3
"""
isc_stats_analysis.py
======================
Statistical analysis and publication-quality visualization of subject-level
7x7 network ISC/ISFC matrices, comparing raw vs. motion-corrected fMRI data
from the same subjects (paired design).

Input
-----
Two .npy files, each of shape (n_subjects, n_networks, n_networks):
    raw_iscs.npy       -- leave-one-subject-out network ISC/ISFC, raw data
    corrected_iscs.npy -- same, motion-corrected data

Each subject's matrix is symmetric in principle: the diagonal holds
within-network ISC, off-diagonal cells hold between-network ISFC. Same
subject ordering in both files.

IMPORTANT (see also `--help` and the module docstring in each analysis
function): the per-subject matrices produced by a leave-one-subject-out (LOO)
pipeline are built as corr(subject_i, mean(everyone else)), which is NOT
guaranteed to be symmetric in i/j for a single subject even though the
GROUP-mean matrix is. This script symmetrizes each subject's matrix
((M + M.T) / 2) after checking and reporting the pre-symmetrization
asymmetry, exactly mirroring the `symmetrize()` step already used for the
group-level summary elsewhere in this pipeline.

Statistical methods (see also print_interpretation() and the docstrings
below for a full explanation of each test):
  A. One-sample bootstrap test (mean Fisher-z > 0), subject-level resampling,
     same resampled indices shared across all 28 cells per iteration.
  B. Paired sign-flip permutation test (mean delta Fisher-z > 0), same random
     sign shared across all 28 cells of one subject per iteration.
  C. Global summary tests on each subject's mean-over-28-cells Fisher-z.
  BH-FDR correction (implemented directly, no external dependency) applied
  across the 28 unique upper-triangle cells (diagonal included) for each
  family of tests.

No BrainIAK dependency is required. If brainiak.isc.bootstrap_isc is
installed it is NOT substituted automatically (its API/statistic differs
slightly), but the method implemented here follows the same shifted-null
bootstrap logic BrainIAK uses (Chen et al., 2016, NeuroImage) -- see
`bootstrap_one_sample()`.

Example
-------
    python isc_stats_analysis.py \\
        --raw figures_raw_data_network_level/order_A_network_isfc_persubject.npy \\
        --corrected figures_motion_correction_spatiotemporal_data/order_A_motion_correction_spatiotemporal_network_isfc_persubject.npy \\
        --output_dir isc_stats_output/raw_vs_spatiotemporal \\
        --label_raw raw --label_corrected spatiotemporal
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NETWORK_NAMES = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]
RNG_SEED = 42
N_BOOT = 10_000
N_PERM = 10_000
ALPHA = 0.05
R_CLIP = 0.999999


# ===========================================================================
# 1. Input handling
# ===========================================================================

def load_and_validate(path: str, name: str = "data", n_networks_expected: int = None) -> np.ndarray:
    """
    Load a (n_subjects, n_networks, n_networks) ISC/ISFC array and validate it.

    Checks: 3D shape with a square last two dims, matching n_networks_expected
    if given, no NaN/Inf, reports per-subject asymmetry (LOO ISFC matrices are
    NOT symmetric by construction -- see module docstring), then returns a
    symmetrized, safely-clipped copy ready for Fisher transformation.
    """
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
        raise ValueError(f"{name}: expected (n_subjects, n_net, n_net), got {arr.shape}")
    n_subj, n_net, _ = arr.shape
    if n_networks_expected is not None and n_net != n_networks_expected:
        raise ValueError(f"{name}: expected n_networks={n_networks_expected}, got {n_net}")

    if np.isnan(arr).any():
        raise ValueError(f"{name}: contains NaN values")
    if np.isinf(arr).any():
        raise ValueError(f"{name}: contains infinite values")

    asym = np.abs(arr - arr.transpose(0, 2, 1))
    max_asym = asym.max()
    mean_asym = asym.mean()
    print(f"[validate] {name}: shape={arr.shape}  max|M-M.T|={max_asym:.4f}  mean|M-M.T|={mean_asym:.4f}")
    if max_asym > 0.05:
        warnings.warn(
            f"{name}: per-subject matrices are not closely symmetric (max diff "
            f"{max_asym:.3f}). This is EXPECTED for LOO ISFC (corr(subject, "
            f"group-mean) is not symmetric in i/j for one subject) -- "
            f"symmetrizing via (M + M.T)/2 before analysis, matching the "
            f"group-level symmetrize() step used elsewhere in this pipeline."
        )

    arr_sym = (arr + arr.transpose(0, 2, 1)) / 2.0
    arr_clipped = np.clip(arr_sym, -R_CLIP, R_CLIP)
    return arr_clipped


def fisher_z(r: np.ndarray) -> np.ndarray:
    """Fisher r-to-z, with a safe clip to avoid +/-inf at |r|=1."""
    return np.arctanh(np.clip(r, -R_CLIP, R_CLIP))


def inv_fisher_z(z: np.ndarray) -> np.ndarray:
    """Inverse Fisher transform, z back to r."""
    return np.tanh(z)


def upper_tri_indices(n: int, include_diag: bool = True):
    """Row/col index arrays for the unique upper-triangle cells of an n x n matrix."""
    k = 0 if include_diag else 1
    return np.triu_indices(n, k=k)


def extract_upper_tri(arr: np.ndarray, iu) -> np.ndarray:
    """(n_subj, n, n) -> (n_subj, n_pairs) using the given upper-tri index tuple."""
    return arr[:, iu[0], iu[1]]


def pair_labels(iu, names):
    return [f"{names[i]}" if i == j else f"{names[i]}-{names[j]}" for i, j in zip(*iu)]


# ===========================================================================
# BH-FDR (no external dependency)
# ===========================================================================

def bh_fdr(pvals: np.ndarray, alpha: float = ALPHA):
    """
    Benjamini-Hochberg FDR correction, implemented directly (statsmodels not
    required). Returns (adjusted_p, reject_mask).
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out, out <= alpha


# ===========================================================================
# 3. Analysis A -- one-sample bootstrap test (mean Fisher-z > 0)
# ===========================================================================

def bootstrap_one_sample(z: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED,
                          stat_fn=np.mean):
    """
    Subject-level nonparametric bootstrap test of H1: mean(z) > 0.

    z: (n_subj, n_cells) Fisher-z values (one column per network pair).

    Method (documented explicitly per the task spec):
      1. Draw n_boot resamples of subject indices (with replacement); the
         SAME resampled index set is applied to every column (cell) in a
         given iteration, preserving within-subject cross-cell dependency.
      2. `boot_dist[b, c] = stat_fn(z[resample_b, c])` -> the bootstrap
         sampling distribution of the group statistic, per cell.
      3. Percentile 95% CI = 2.5th/97.5th percentiles of boot_dist (used for
         the reported confidence interval on the point estimate).
      4. One-sided p-value uses a *null-shifted* bootstrap distribution:
         shifted = boot_dist - mean(boot_dist, axis=0). This re-centers the
         bootstrap distribution at zero using its own bootstrap mean (which
         approximates the observed statistic), giving an empirical proxy for
         the sampling distribution of the statistic UNDER H0: population
         value = 0, assuming the shape/spread of the sampling distribution
         is similar under H0 and H1 (the standard bootstrap-null
         approximation; this is the same logic BrainIAK's bootstrap_isc
         uses, Chen et al. 2016, NeuroImage).
         p = (#{shifted >= observed} + 1) / (n_boot + 1)   [one-sided, H1: >0]

    Returns dict of (n_cells,) arrays: obs, ci_low, ci_high, p_one_sided,
    plus the raw (n_boot, n_cells) boot_dist for downstream plotting.
    """
    rng = np.random.default_rng(seed)
    n_subj, n_cells = z.shape
    idx = rng.integers(0, n_subj, size=(n_boot, n_subj))
    boot_dist = stat_fn(z[idx], axis=1)                    # (n_boot, n_cells)

    obs = stat_fn(z, axis=0)                                # (n_cells,)
    ci_low = np.percentile(boot_dist, 2.5, axis=0)
    ci_high = np.percentile(boot_dist, 97.5, axis=0)

    shifted = boot_dist - boot_dist.mean(axis=0, keepdims=True)
    p_one = (np.sum(shifted >= obs, axis=0) + 1) / (n_boot + 1)

    return dict(obs=obs, ci_low=ci_low, ci_high=ci_high,
                p_one_sided=p_one, boot_dist=boot_dist)


# ===========================================================================
# 4. Analysis B -- paired sign-flip permutation test (mean delta_z > 0)
# ===========================================================================

def paired_sign_flip(delta_z: np.ndarray, n_perm: int = N_PERM, seed: int = RNG_SEED,
                      stat_fn=np.mean):
    """
    Paired sign-flip permutation test of H1: mean(delta_z) > 0.

    delta_z: (n_subj, n_cells) = corrected_z - raw_z.

    Each permutation multiplies EVERY cell of one subject's entire delta
    vector by the same random +/-1 (shared sign across the 7x7 matrix for
    that subject in that iteration) -- this is the correct null for a
    paired/within-subject design: under H0, which condition is "raw" and
    which is "corrected" is arbitrary for each subject, but consistently so
    across all cells of that subject.

    p = (#{null >= observed} + 1) / (n_perm + 1)   [one-sided, H1: mean>0]
    """
    rng = np.random.default_rng(seed)
    n_subj, n_cells = delta_z.shape
    obs = stat_fn(delta_z, axis=0)                          # (n_cells,)

    signs = rng.choice([-1.0, 1.0], size=(n_perm, n_subj))
    flipped = signs[:, :, None] * delta_z[None, :, :]       # (n_perm, n_subj, n_cells)
    null = stat_fn(flipped, axis=1)                          # (n_perm, n_cells)

    p_one = (np.sum(null >= obs, axis=0) + 1) / (n_perm + 1)
    return dict(obs=obs, null=null, p_one_sided=p_one)


def cohens_dz(delta: np.ndarray) -> np.ndarray:
    """Standardized paired effect size dz = mean(delta) / std(delta, ddof=1), per column."""
    return delta.mean(axis=0) / delta.std(axis=0, ddof=1)


def bootstrap_dz_ci(delta_z: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED):
    """
    Percentile bootstrap CI for the standardized paired effect size dz itself
    (mean/std is nonlinear, so this is NOT the same as rescaling the CI on the
    mean -- it needs its own resampling). Same shared resampled subject
    indices across all cells per iteration, as elsewhere in this script.
    """
    rng = np.random.default_rng(seed)
    n_subj, n_cells = delta_z.shape
    idx = rng.integers(0, n_subj, size=(n_boot, n_subj))
    resampled = delta_z[idx]                                   # (n_boot, n_subj, n_cells)
    boot_dz = resampled.mean(axis=1) / resampled.std(axis=1, ddof=1)   # (n_boot, n_cells)
    return dict(ci_low=np.percentile(boot_dz, 2.5, axis=0),
                ci_high=np.percentile(boot_dz, 97.5, axis=0))


# ===========================================================================
# 6. Sensitivity analyses
# ===========================================================================

def sensitivity_analyses(raw_r, corrected_r, raw_z, corrected_z, delta_z, iu, alpha=ALPHA):
    """
    Runs the lighter-weight sensitivity checks requested:
      - median vs mean as the group summary statistic (paired test)
      - two-sided vs one-sided paired p-values
      - Wilcoxon signed-rank as a secondary (non-primary) paired test
      - separate FDR within diagonal-only vs off-diagonal-only cells
      - paired test with vs without Fisher transformation
    Returns a dict of small result tables/arrays (printed by the caller,
    not turned into their own figures, per spec item 6 being "optional/
    secondary").
    """
    out = {}

    # --- median vs mean ---
    med = paired_sign_flip(delta_z, stat_fn=np.median)
    out["median_p_one_sided"] = med["p_one_sided"]
    out["median_obs"] = med["obs"]

    # --- two-sided p from the same sign-flip null (mean statistic) ---
    mean_res = paired_sign_flip(delta_z, stat_fn=np.mean)
    obs, null = mean_res["obs"], mean_res["null"]
    p_two = (np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1) / (null.shape[0] + 1)
    out["p_two_sided"] = p_two

    # --- Wilcoxon signed-rank (secondary, non-primary) ---
    wilcoxon_p = np.array([
        stats.wilcoxon(delta_z[:, c], alternative="greater").pvalue
        if np.any(delta_z[:, c] != 0) else 1.0
        for c in range(delta_z.shape[1])
    ])
    out["wilcoxon_p_one_sided"] = wilcoxon_p

    # --- separate FDR: diagonal vs off-diagonal ---
    is_diag = iu[0] == iu[1]
    p_primary = mean_res["p_one_sided"]
    fdr_diag, rej_diag = bh_fdr(p_primary[is_diag], alpha)
    fdr_offdiag, rej_offdiag = bh_fdr(p_primary[~is_diag], alpha)
    p_fdr_split = np.empty_like(p_primary)
    rej_split = np.empty_like(p_primary, dtype=bool)
    p_fdr_split[is_diag], rej_split[is_diag] = fdr_diag, rej_diag
    p_fdr_split[~is_diag], rej_split[~is_diag] = fdr_offdiag, rej_offdiag
    out["p_fdr_split_diag_offdiag"] = p_fdr_split
    out["reject_split_diag_offdiag"] = rej_split

    # --- with vs without Fisher transform (paired test directly on r) ---
    delta_r = corrected_r - raw_r
    r_res = paired_sign_flip(delta_r, stat_fn=np.mean)
    out["p_one_sided_no_fisher"] = r_res["p_one_sided"]

    return out


# ===========================================================================
# Global summary (Analysis C) helper
# ===========================================================================

def global_summary(z_by_cell: np.ndarray) -> np.ndarray:
    """(n_subj, n_cells) -> (n_subj,) mean over the 28 upper-tri cells per subject."""
    return z_by_cell.mean(axis=1)


# ===========================================================================
# 7 / 11. Optional advanced functions (documented, not run by default)
# ===========================================================================

def recompute_loo_isfc_from_timeseries(ts_stack: np.ndarray, subject_idx: np.ndarray) -> np.ndarray:
    """
    OPTIONAL / ADVANCED. Recompute the 7x7 LOO ISFC matrix directly from
    network time series for a given (possibly bootstrap-resampled, with
    replacement) set of subject indices, instead of resampling pre-computed
    per-subject ISC matrices.

    This is the methodologically stronger approach: it avoids treating the
    fixed per-subject LOO estimates as independent observations (see the
    "Important methodological note" in the module docstring) by rebuilding
    each subject's *reference* (the other subjects' mean) fresh inside every
    bootstrap iteration, from the resampled multiset actually drawn.

    Parameters
    ----------
    ts_stack     : (n_subj_total, T, n_net) network-mean time series, ALL
                   subjects available to resample from.
    subject_idx  : (k,) integer array -- one bootstrap resample's subject
                   indices, WITH replacement (k need not equal n_subj_total).

    Returns
    -------
    (k, n_net, n_net) array: one LOO ISFC matrix per (resampled) subject slot.

    Notes / simplification
    -----------------------
    With replacement, several resampled "slots" may point at the same
    physical subject. Each slot's LOO reference here is the mean of every
    OTHER SLOT in `subject_idx` (not "every other physical subject"), which
    is the direct generalization of leave-one-out to a resampled multiset.
    This is a simplified, single-tier version of this project's actual
    production LOO matching (which also handles missing session/run
    combinations via `select_representative_entry` in loo_isc.py) -- if the
    resampled data is complete (no missing acquisitions) the two agree.
    """
    k = len(subject_idx)
    ts = ts_stack[subject_idx]                      # (k, T, n_net)
    n_net = ts.shape[-1]
    out = np.empty((k, n_net, n_net))
    for i in range(k):
        others = np.delete(ts, i, axis=0).mean(axis=0)   # (T, n_net) LOO reference
        target = ts[i]                                    # (T, n_net)
        a = target - target.mean(axis=0, keepdims=True)
        b = others - others.mean(axis=0, keepdims=True)
        num = a.T @ b
        den = np.sqrt((a ** 2).sum(axis=0))[:, None] * np.sqrt((b ** 2).sum(axis=0))[None, :]
        out[i] = num / den
    return out


def temporal_null_isc(ts_stack: np.ndarray, n_null: int = 5000, seed: int = RNG_SEED,
                       method: str = "circular_shift"):
    """
    OPTIONAL. Temporal-null (circular time-shift / phase-randomization) test
    of whether observed ISC/ISFC is specifically tied to shared stimulus
    timing, rather than generic temporal autocorrelation structure.

    For each of n_null iterations: every subject's time series is
    INDEPENDENTLY circularly shifted (or phase-randomized) by a random
    amount, which preserves each subject's own autocorrelation / power
    spectrum but destroys cross-subject alignment to the common stimulus
    timing. ISC/ISFC is recomputed each iteration (leave-one-out, using the
    already-shifted series as both target and reference) to build a null
    distribution; the observed (unshifted) ISC/ISFC is compared against it.
    Two-sided or one-sided (H1: observed > null) comparison + BH-FDR across
    the 28 cells is the caller's responsibility (mirrors the primary tests).

    Parameters
    ----------
    ts_stack : (n_subj, T, n_net)
    method   : "circular_shift" (default) or "phase_randomize"

    Returns
    -------
    (n_null, n_net, n_net) null distribution of GROUP-MEAN LOO ISFC matrices.
    """
    rng = np.random.default_rng(seed)
    n_subj, T, n_net = ts_stack.shape
    null_group = np.empty((n_null, n_net, n_net))

    for it in range(n_null):
        if method == "circular_shift":
            shifts = rng.integers(0, T, size=n_subj)
            shifted = np.stack([np.roll(ts_stack[s], shifts[s], axis=0) for s in range(n_subj)])
        elif method == "phase_randomize":
            shifted = np.empty_like(ts_stack)
            for s in range(n_subj):
                fft = np.fft.rfft(ts_stack[s], axis=0)
                random_phase = rng.uniform(0, 2 * np.pi, size=fft.shape)
                random_phase[0] = 0  # keep DC term real
                shifted[s] = np.fft.irfft(np.abs(fft) * np.exp(1j * random_phase), n=T, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")

        per_subj = recompute_loo_isfc_from_timeseries(shifted, np.arange(n_subj))
        null_group[it] = np.tanh(np.arctanh(np.clip(per_subj, -R_CLIP, R_CLIP)).mean(axis=0))

    return null_group


# ===========================================================================
# 8. Results table
# ===========================================================================

def build_results_table(iu, names, raw_r, corrected_r, raw_res, corrected_res,
                         delta_res, dz, prop_improved, delta_ci):
    labels_i = [names[i] for i in iu[0]]
    labels_j = [names[j] for j in iu[1]]

    raw_fdr_p, raw_reject = bh_fdr(raw_res["p_one_sided"])
    cor_fdr_p, cor_reject = bh_fdr(corrected_res["p_one_sided"])
    delta_fdr_p, delta_reject = bh_fdr(delta_res["p_one_sided"])

    df = pd.DataFrame({
        "network_1": labels_i,
        "network_2": labels_j,
        "raw_mean_r": inv_fisher_z(raw_res["obs"]),
        "raw_ci_low_r": inv_fisher_z(raw_res["ci_low"]),
        "raw_ci_high_r": inv_fisher_z(raw_res["ci_high"]),
        "raw_p_uncorrected": raw_res["p_one_sided"],
        "raw_p_fdr": raw_fdr_p,
        "raw_significant_fdr": raw_reject,
        "corrected_mean_r": inv_fisher_z(corrected_res["obs"]),
        "corrected_ci_low_r": inv_fisher_z(corrected_res["ci_low"]),
        "corrected_ci_high_r": inv_fisher_z(corrected_res["ci_high"]),
        "corrected_p_uncorrected": corrected_res["p_one_sided"],
        "corrected_p_fdr": cor_fdr_p,
        "corrected_significant_fdr": cor_reject,
        "delta_mean_z": delta_res["obs"],
        "delta_ci_low_z": delta_ci["ci_low"],
        "delta_ci_high_z": delta_ci["ci_high"],
        "paired_effect_size_dz": dz,
        "proportion_improved": prop_improved,
        "delta_p_uncorrected": delta_res["p_one_sided"],
        "delta_p_fdr": delta_fdr_p,
        "delta_significant_fdr": delta_reject,
    })
    return df


def sig_stars(p_fdr):
    if p_fdr < 0.001:
        return "***"
    if p_fdr < 0.01:
        return "**"
    if p_fdr < 0.05:
        return "*"
    return ""


# ===========================================================================
# 9. Figures
# ===========================================================================

def _full_matrix_from_upper(values, iu, n, dtype=float, fill_value=0):
    """Reconstruct a full symmetric n x n matrix from unique upper-tri values.

    dtype/fill_value let this build either a numeric matrix (mean r values)
    or an object matrix of significance-star strings.
    """
    M = np.full((n, n), fill_value, dtype=dtype)
    values = np.asarray(values, dtype=dtype)
    M[iu[0], iu[1]] = values
    M[iu[1], iu[0]] = values
    return M


def _save_fig(fig, out_dir, stem):
    png = os.path.join(out_dir, f"{stem}.png")
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {png}")
    print(f"[figure] {pdf}")


def plot_heatmap(mean_r_full, star_full, names, title, out_dir, stem, vmin=None, vmax=None):
    n = len(names)
    vmax = vmax if vmax is not None else np.abs(mean_r_full).max()
    vmin = vmin if vmin is not None else -vmax
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mean_r_full, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            val = mean_r_full[i, j]
            star = star_full[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            label = f"{val:.2f}{star}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="ISC / ISFC (r)")
    fig.text(0.5, -0.02, "* pFDR<0.05   ** pFDR<0.01   *** pFDR<0.001", ha="center", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir, stem)


def figure_1_2_group_heatmaps(iu, names, raw_res, cor_res, raw_fdr_p, cor_fdr_p, out_dir):
    n = len(names)
    raw_mean_full = _full_matrix_from_upper(inv_fisher_z(raw_res["obs"]), iu, n)
    cor_mean_full = _full_matrix_from_upper(inv_fisher_z(cor_res["obs"]), iu, n)
    raw_star = _full_matrix_from_upper([sig_stars(p) for p in raw_fdr_p], iu, n, dtype=object, fill_value="")
    cor_star = _full_matrix_from_upper([sig_stars(p) for p in cor_fdr_p], iu, n, dtype=object, fill_value="")

    vmax = max(np.abs(raw_mean_full).max(), np.abs(cor_mean_full).max())
    plot_heatmap(raw_mean_full, raw_star, names, "Raw group ISC/ISFC (FDR-marked)",
                 out_dir, "fig1_raw_heatmap", vmin=-vmax, vmax=vmax)
    plot_heatmap(cor_mean_full, cor_star, names, "Corrected group ISC/ISFC (FDR-marked)",
                 out_dir, "fig2_corrected_heatmap", vmin=-vmax, vmax=vmax)
    return raw_mean_full, cor_mean_full, vmax


def figure_3_diff_heatmap(iu, names, raw_mean_full, cor_mean_full, delta_fdr_p, out_dir):
    n = len(names)
    diff_full = cor_mean_full - raw_mean_full
    star_full = _full_matrix_from_upper([sig_stars(p) for p in delta_fdr_p], iu, n, dtype=object, fill_value="")
    vmax = np.abs(diff_full).max()
    plot_heatmap(diff_full, star_full, names, "Corrected minus Raw (r), FDR-marked",
                 out_dir, "fig3_diff_heatmap", vmin=-vmax, vmax=vmax)
    return diff_full


def figure_4_three_panel(raw_mean_full, cor_mean_full, diff_full, names, vmax, out_dir):
    n = len(names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    panels = [(raw_mean_full, "Raw", -vmax, vmax, "RdBu_r"),
              (cor_mean_full, "Corrected", -vmax, vmax, "RdBu_r"),
              (diff_full, "Corrected - Raw", -np.abs(diff_full).max(), np.abs(diff_full).max(), "RdBu_r")]
    for ax, (mat, title, vmin_, vmax_, cmap) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin_, vmax=vmax_)
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Raw vs. Corrected vs. Difference (network ISC/ISFC)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig4_three_panel")


def figure_5_diagonal_paired(iu, names, raw_r, corrected_r, delta_res, delta_fdr_p, out_dir):
    is_diag = iu[0] == iu[1]
    diag_cols = np.where(is_diag)[0]
    diag_names = [names[iu[0][c]] for c in diag_cols]

    fig, axes = plt.subplots(1, len(diag_cols), figsize=(3.0 * len(diag_cols), 4.2), sharey=True)
    for ax, c, net_name in zip(axes, diag_cols, diag_names):
        rr, cr = raw_r[:, c], corrected_r[:, c]
        for s in range(len(rr)):
            ax.plot([0, 1], [rr[s], cr[s]], color="gray", alpha=0.35, lw=0.8, zorder=1)
        ax.scatter(np.zeros_like(rr), rr, color="#898781", s=18, zorder=2)
        ax.scatter(np.ones_like(cr), cr, color="#2a78d6", s=18, zorder=2)
        mean_r, mean_c = rr.mean(), cr.mean()
        ci = np.percentile([rr[np.random.default_rng(0).integers(0, len(rr), len(rr))].mean()
                             for _ in range(200)], [2.5, 97.5])  # quick visual CI (raw r, illustrative)
        ax.plot([0, 1], [mean_r, mean_c], color="black", lw=2.2, zorder=3)
        star = sig_stars(delta_fdr_p[c])
        ax.set_title(f"{net_name}\np(FDR)={delta_fdr_p[c]:.3g}{star}", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Raw", "Corrected"])
        ax.set_xlim(-0.3, 1.3)
    axes[0].set_ylabel("ISC (r)")
    fig.suptitle("Paired within-network ISC: raw vs. corrected", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig5_diagonal_paired")


def figure_6_forest(iu, names, dz, dz_ci, out_dir):
    labels = pair_labels(iu, names)
    is_diag = iu[0] == iu[1]
    order = np.argsort(dz)
    fig, ax = plt.subplots(figsize=(7, 8))
    y = np.arange(len(labels))
    colors = np.where(is_diag[order], "#2a78d6", "#898781")
    lo = np.clip(dz[order] - dz_ci["ci_low"][order], 0, None)
    hi = np.clip(dz_ci["ci_high"][order] - dz[order], 0, None)
    ax.errorbar(dz[order], y, xerr=[lo, hi],
                fmt="none", ecolor="#898781", elinewidth=1.2, capsize=2, zorder=2)
    ax.scatter(dz[order], y, c=colors, s=22, zorder=3)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(np.array(labels)[order], fontsize=7)
    ax.set_xlabel("Paired effect size (dz)")
    ax.set_title("Effect sizes, all 28 network-pair cells\n(blue = diagonal ISC, gray = off-diagonal ISFC)", fontsize=10)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig6_forest_effect_sizes")


def figure_7_pvalue_validation(delta_res, delta_fdr_p, iu, names, out_dir):
    labels = pair_labels(iu, names)
    p_unc = delta_res["p_one_sided"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(p_unc, delta_fdr_p, s=24, color="#2a78d6")
    for i, lbl in enumerate(labels):
        if delta_fdr_p[i] < 0.05:
            ax.annotate(lbl, (p_unc[i], delta_fdr_p[i]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.05, color="red", ls="--", lw=1, label="0.05")
    ax.axvline(0.05, color="red", ls="--", lw=1)
    ax.set_xlabel("Uncorrected p (one-sided)")
    ax.set_ylabel("FDR-adjusted p")
    ax.set_title("Uncorrected vs. FDR-adjusted p (paired test)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig7_pvalue_validation")


def figure_8_bootstrap_distributions(iu, names, raw_res, cor_res, out_dir,
                                      selected=("Cont", "Default", "DorsAttn", "SomMot", "Vis")):
    is_diag = iu[0] == iu[1]
    diag_cols = np.where(is_diag)[0]
    name_to_col = {names[iu[0][c]]: c for c in diag_cols}
    cols = [name_to_col[s] for s in selected if s in name_to_col]

    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.6), sharey=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, c, net_name in zip(axes, cols, selected):
        ax.hist(inv_fisher_z(raw_res["boot_dist"][:, c]), bins=40, alpha=0.5, color="#898781", label="Raw")
        ax.hist(inv_fisher_z(cor_res["boot_dist"][:, c]), bins=40, alpha=0.5, color="#2a78d6", label="Corrected")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.axvline(inv_fisher_z(raw_res["obs"][c]), color="#52514e", lw=1.4)
        ax.axvline(inv_fisher_z(cor_res["obs"][c]), color="#184f95", lw=1.4)
        ax.set_title(net_name, fontsize=9)
    axes[0].set_ylabel("Bootstrap count")
    axes[0].legend(fontsize=8)
    fig.suptitle("Bootstrap distributions of mean ISC (r), selected networks", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig8_bootstrap_distributions")


def figure_9_global_summary(subj_global_raw_z, subj_global_cor_z, global_delta_res,
                             global_ci, out_dir):
    raw_r = inv_fisher_z(subj_global_raw_z)
    cor_r = inv_fisher_z(subj_global_cor_z)
    fig, ax = plt.subplots(figsize=(4.5, 5))
    for s in range(len(raw_r)):
        ax.plot([0, 1], [raw_r[s], cor_r[s]], color="gray", alpha=0.35, lw=0.8)
    ax.scatter(np.zeros_like(raw_r), raw_r, color="#898781", s=20, zorder=3)
    ax.scatter(np.ones_like(cor_r), cor_r, color="#2a78d6", s=20, zorder=3)
    ax.plot([0, 1], [raw_r.mean(), cor_r.mean()], color="black", lw=2.5, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Raw", "Corrected"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("Subject global mean ISC (r, avg. over 28 cells)")
    mean_delta_r = inv_fisher_z(global_delta_res["obs"][0])
    dz = cohens_dz((subj_global_cor_z - subj_global_raw_z)[:, None])[0]
    ax.set_title(
        f"Global subject-level ISC, raw vs. corrected\n"
        f"mean delta(r)={mean_delta_r:.3f}  "
        f"95% CI z=[{global_ci['ci_low'][0]:.3f}, {global_ci['ci_high'][0]:.3f}]  "
        f"p={global_delta_res['p_one_sided'][0]:.3g}  dz={dz:.2f}",
        fontsize=8.5,
    )
    fig.tight_layout()
    _save_fig(fig, out_dir, "fig9_global_summary")


# ===========================================================================
# 10. Interpretation printout
# ===========================================================================

def print_interpretation(df, iu, names):
    n_raw_sig = int(df["raw_significant_fdr"].sum())
    n_cor_sig = int(df["corrected_significant_fdr"].sum())
    n_delta_sig = int(df["delta_significant_fdr"].sum())
    is_diag = iu[0] == iu[1]

    print("\n" + "=" * 70)
    print("INTERPRETATION SUMMARY")
    print("=" * 70)
    print(f"Raw:       {n_raw_sig}/28 cells significantly > 0 after FDR (q<0.05)")
    print(f"Corrected: {n_cor_sig}/28 cells significantly > 0 after FDR (q<0.05)")
    print(f"Corrected > Raw: {n_delta_sig}/28 cells survive FDR (q<0.05)")

    sig_diag = df.loc[is_diag & df["delta_significant_fdr"], "network_1"].tolist()
    print(f"Diagonal networks with significant improvement: {sig_diag if sig_diag else 'none'}")

    largest = df.reindex(df["paired_effect_size_dz"].abs().sort_values(ascending=False).index).head(5)
    print("\nLargest |effect size| cells (dz):")
    for _, row in largest.iterrows():
        print(f"  {row['network_1']}-{row['network_2']}: dz={row['paired_effect_size_dz']:.2f}  "
              f"p_fdr={row['delta_p_fdr']:.3g}")

    decreased = df[(df["delta_mean_z"] < 0) & df["delta_significant_fdr"]]
    if len(decreased):
        print(f"\nWARNING: {len(decreased)} cell(s) significantly DECREASE after correction:")
        for _, row in decreased.iterrows():
            print(f"  {row['network_1']}-{row['network_2']}")
    else:
        print("\nNo cells show a significant FDR-corrected decrease after correction.")

    print(
        "\nCaveat: a significant increase in ISC/ISFC is consistent with improved recovery\n"
        "of shared, stimulus-locked activity, but does NOT by itself prove motion correction\n"
        "is working correctly -- residual-motion checks (e.g. DVARS/tSNR/FD relationships)\n"
        "and a temporal-null test (see temporal_null_isc()) that destroys stimulus alignment\n"
        "while preserving each subject's own autocorrelation are needed to rule out\n"
        "correction-induced but stimulus-independent similarity (e.g. a shared, subject-\n"
        "invariant correction template) as an alternative explanation."
    )
    print(
        "\nLeave-one-out dependence: each subject's ISC/ISFC estimate is compared against a\n"
        "reference built from every OTHER subject, so subjects' estimates are not fully\n"
        "independent (their references overlap). The paired subject-level tests here are\n"
        "still a valid, standard practical group-level test (matching common ISC practice),\n"
        "but exact independence assumptions of classical parametric tests do not hold --\n"
        "this is why nonparametric bootstrap/permutation methods are used throughout."
    )
    print("=" * 70)


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="Path to raw_iscs.npy (n_subj, n_net, n_net)")
    ap.add_argument("--corrected", required=True, help="Path to corrected_iscs.npy (n_subj, n_net, n_net)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--label_raw", default="raw")
    ap.add_argument("--label_corrected", default="corrected")
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    names = NETWORK_NAMES

    print(f"\n--- 1. Loading and validating input ---")
    raw = load_and_validate(args.raw, name=args.label_raw, n_networks_expected=len(names))
    corrected = load_and_validate(args.corrected, name=args.label_corrected, n_networks_expected=len(names))
    if raw.shape != corrected.shape:
        raise ValueError(f"Shape mismatch: raw={raw.shape} corrected={corrected.shape}")
    n_subj, n_net, _ = raw.shape
    print(f"n_subjects={n_subj}  n_networks={n_net}")

    iu = upper_tri_indices(n_net, include_diag=True)
    n_cells = len(iu[0])
    print(f"Testing {n_cells} unique upper-triangle cells (diagonal included).")

    raw_r = extract_upper_tri(raw, iu)
    cor_r = extract_upper_tri(corrected, iu)
    raw_z = fisher_z(raw_r)
    cor_z = fisher_z(cor_r)
    delta_z = cor_z - raw_z

    print("\n--- 2. Fisher transformation applied for all inferential tests ---")

    print(f"\n--- 3. Analysis A: one-sample bootstrap (n_boot={args.n_boot}) ---")
    raw_res = bootstrap_one_sample(raw_z, n_boot=args.n_boot, seed=args.seed)
    cor_res = bootstrap_one_sample(cor_z, n_boot=args.n_boot, seed=args.seed + 1)
    raw_fdr_p, _ = bh_fdr(raw_res["p_one_sided"], args.alpha)
    cor_fdr_p, _ = bh_fdr(cor_res["p_one_sided"], args.alpha)

    print(f"\n--- 4. Analysis B: paired sign-flip permutation (n_perm={args.n_perm}) ---")
    delta_res = paired_sign_flip(delta_z, n_perm=args.n_perm, seed=args.seed + 2)
    delta_ci = bootstrap_one_sample(delta_z, n_boot=args.n_boot, seed=args.seed + 3)
    dz = cohens_dz(delta_z)
    dz_ci = bootstrap_dz_ci(delta_z, n_boot=args.n_boot, seed=args.seed + 8)
    prop_improved = (delta_z > 0).mean(axis=0)
    delta_ci_r = dict(ci_low=inv_fisher_z(delta_ci["ci_low"]), ci_high=inv_fisher_z(delta_ci["ci_high"]))

    print(f"\n--- 5. Analysis C: global summary ---")
    subj_global_raw_z = global_summary(raw_z)
    subj_global_cor_z = global_summary(cor_z)
    global_delta_z = (subj_global_cor_z - subj_global_raw_z)[:, None]
    global_raw_res = bootstrap_one_sample(subj_global_raw_z[:, None], n_boot=args.n_boot, seed=args.seed + 4)
    global_cor_res = bootstrap_one_sample(subj_global_cor_z[:, None], n_boot=args.n_boot, seed=args.seed + 5)
    global_delta_res = paired_sign_flip(global_delta_z, n_perm=args.n_perm, seed=args.seed + 6)
    global_ci = bootstrap_one_sample(global_delta_z, n_boot=args.n_boot, seed=args.seed + 7)
    print(f"  Global raw ISC:       mean r={inv_fisher_z(global_raw_res['obs'][0]):.4f}  p={global_raw_res['p_one_sided'][0]:.3g}")
    print(f"  Global corrected ISC: mean r={inv_fisher_z(global_cor_res['obs'][0]):.4f}  p={global_cor_res['p_one_sided'][0]:.3g}")
    print(f"  Global corrected > raw: delta r={inv_fisher_z(global_delta_res['obs'][0]):.4f}  "
          f"p={global_delta_res['p_one_sided'][0]:.3g}  "
          f"dz={cohens_dz(global_delta_z)[0]:.2f}")

    print(f"\n--- 6. Sensitivity analyses ---")
    sens = sensitivity_analyses(raw_r, cor_r, raw_z, cor_z, delta_z, iu, args.alpha)
    print(f"  Median vs mean (paired, primary=mean): "
          f"{(sens['median_p_one_sided'] < args.alpha).sum()}/{n_cells} cells p<{args.alpha} using median")
    print(f"  Two-sided vs one-sided: "
          f"{(sens['p_two_sided'] < args.alpha).sum()}/{n_cells} cells p<{args.alpha} two-sided "
          f"(vs {(delta_res['p_one_sided'] < args.alpha).sum()}/{n_cells} one-sided)")
    print(f"  Wilcoxon signed-rank (secondary): "
          f"{(sens['wilcoxon_p_one_sided'] < args.alpha).sum()}/{n_cells} cells p<{args.alpha}")
    print(f"  Split FDR (diag vs off-diag) significant: {sens['reject_split_diag_offdiag'].sum()}/{n_cells}")
    print(f"  Without Fisher transform (paired test on raw r): "
          f"{(sens['p_one_sided_no_fisher'] < args.alpha).sum()}/{n_cells} cells p<{args.alpha}")

    print(f"\n--- 7 / 11. Optional advanced functions ---")
    print("  recompute_loo_isfc_from_timeseries() and temporal_null_isc() are implemented "
          "and importable from this module, but are NOT run by default (require raw network "
          "time series of matching length per subject; see their docstrings).")

    print(f"\n--- 8. Results table ---")
    df = build_results_table(iu, names, raw_r, cor_r, raw_res, cor_res, delta_res, dz,
                              prop_improved, delta_ci_r)
    csv_path = os.path.join(args.output_dir, "isc_network_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[table] {csv_path}")

    print(f"\n--- 9. Figures ---")
    raw_mean_full, cor_mean_full, vmax = figure_1_2_group_heatmaps(iu, names, raw_res, cor_res, raw_fdr_p, cor_fdr_p, args.output_dir)
    delta_fdr_p, _ = bh_fdr(delta_res["p_one_sided"], args.alpha)
    diff_full = figure_3_diff_heatmap(iu, names, raw_mean_full, cor_mean_full, delta_fdr_p, args.output_dir)
    figure_4_three_panel(raw_mean_full, cor_mean_full, diff_full, names, vmax, args.output_dir)
    figure_5_diagonal_paired(iu, names, raw_r, cor_r, delta_res, delta_fdr_p, args.output_dir)
    figure_6_forest(iu, names, dz, dz_ci, args.output_dir)
    figure_7_pvalue_validation(delta_res, delta_fdr_p, iu, names, args.output_dir)
    figure_8_bootstrap_distributions(iu, names, raw_res, cor_res, args.output_dir)
    figure_9_global_summary(subj_global_raw_z, subj_global_cor_z, global_delta_res, global_ci, args.output_dir)

    print_interpretation(df, iu, names)


if __name__ == "__main__":
    main()
