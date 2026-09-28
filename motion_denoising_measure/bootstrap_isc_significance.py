"""
bootstrap_isc_significance.py
================================
Proper subject-level bootstrap significance for the order-A 7x7 network ISFC
comparison (raw unprocessed / raw preprocessed / denoised / delta), as opposed
to the parametric one-sample-t-test approach in plot_isc_comparison.py.

Why a separate bootstrap, not just resampling the already-computed per-subject
LOO-ISC values: those values already share subjects in each other's leave-one-
out reference means, so a naive bootstrap of them isn't a clean resample.
Instead, each bootstrap iteration resamples SUBJECTS with replacement and
recomputes the leave-one-out reference AND the ISC/ISFC matrix from scratch
within that resample (only a single, randomly-picked entry per resampled
subject-slot stands in for that subject's own repeat acquisitions -- adding,
not avoiding, subject-level resampling variability).

The denoised-vs-raw-preprocessed delta is bootstrapped too, not skipped: since
both use the exact same 46 subjects, drawing the SAME per-iteration resample
(same subject indices, same within-subject entry pick) for both conditions
preserves the paired covariance structure -- essentially free once the
standalone machinery exists, and the only way to propagate resampling
uncertainty into the delta correctly.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "ISC_analysis"))
import loo_isc as l  # noqa: E402

N_BOOT = 10000
SEED = 12345


def entries_array(all_series, subjects):
    """Per subject, network_tc entries stacked as (n_entries, n_networks, T),
    sorted by (session, run, segment_num) so datasets built from the same
    order-A rows (raw preprocessed vs. denoised) line up entry-for-entry --
    time_courses.py's ProcessPoolExecutor writes manifests in completion
    order, not input order, so this can't be assumed without sorting."""
    out = []
    for s in subjects:
        entries = sorted(all_series[s], key=lambda e: (e["session"], e["run"], e["segment_num"]))
        out.append(np.stack([e["network"] for e in entries]))
    return out


def bootstrap_group_isfc(selected):
    """selected: (n_slots, n_networks, T) -- one chosen entry per resampled slot.
    Vectorized leave-one-out ISFC + Fisher-z group average, matching
    loo_isc.py's compute_subject_isfc/group_isfc_summary exactly, just built
    from resampled slots instead of the fixed set of real subjects."""
    n = selected.shape[0]
    centered = selected - selected.mean(axis=2, keepdims=True)
    total = centered.sum(axis=0)
    loo_mean = (total[None, :, :] - centered) / (n - 1)  # zero-mean already (sum of zero-mean rows)

    num = np.einsum("sit,sjt->sij", centered, loo_mean)
    denom = np.sqrt((centered ** 2).sum(axis=2))[:, :, None] * np.sqrt((loo_mean ** 2).sum(axis=2))[:, None, :]
    isfc = num / denom

    group = l.inverse_fisher_z(l.fisher_z(isfc).mean(axis=0))
    return (group + group.T) / 2


def draw_resample(n_subjects, entry_counts, rng):
    subj_idx = rng.integers(0, n_subjects, size=n_subjects)
    entry_idx = np.array([rng.integers(0, entry_counts[i]) for i in subj_idx])
    return subj_idx, entry_idx


def bootstrap_standalone(entries, n_boot, rng):
    n = len(entries)
    n_networks, T = entries[0].shape[1:]
    entry_counts = [e.shape[0] for e in entries]
    boot = np.empty((n_boot, n_networks, n_networks))
    for b in range(n_boot):
        subj_idx, entry_idx = draw_resample(n, entry_counts, rng)
        selected = np.stack([entries[si][ei] for si, ei in zip(subj_idx, entry_idx)])
        boot[b] = bootstrap_group_isfc(selected)
    return boot


def bootstrap_paired(entries_a, entries_b, n_boot, rng):
    """Same resample (subject + within-subject entry draw) applied to both
    datasets each iteration -- entries_a[s] and entries_b[s] must already be
    entry-for-entry aligned (see entries_array's sort)."""
    n = len(entries_a)
    n_networks, T = entries_a[0].shape[1:]
    entry_counts = [e.shape[0] for e in entries_a]
    boot_a = np.empty((n_boot, n_networks, n_networks))
    boot_b = np.empty((n_boot, n_networks, n_networks))
    for b in range(n_boot):
        subj_idx, entry_idx = draw_resample(n, entry_counts, rng)
        sel_a = np.stack([entries_a[si][ei] for si, ei in zip(subj_idx, entry_idx)])
        sel_b = np.stack([entries_b[si][ei] for si, ei in zip(subj_idx, entry_idx)])
        boot_a[b] = bootstrap_group_isfc(sel_a)
        boot_b[b] = bootstrap_group_isfc(sel_b)
    return boot_a, boot_b


def upper_triangle_indices(n):
    return [(i, j) for i in range(n) for j in range(i, n)]


def bootstrap_pvalues(boot, names):
    """Two-sided percentile p-value per unique cell: 2x the smaller tail
    fraction crossing zero, capped at 1."""
    from statsmodels.stats.multitest import multipletests

    n = len(names)
    pairs = upper_triangle_indices(n)
    p_vals = []
    for i, j in pairs:
        vals = boot[:, i, j]
        frac_le = (vals <= 0).mean()
        frac_ge = (vals >= 0).mean()
        p = min(1.0, 2 * min(frac_le, frac_ge))
        p_vals.append(p)
    significant, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")

    sig_matrix = np.zeros((n, n), dtype=bool)
    rows = []
    for (i, j), p, q, sig in zip(pairs, p_vals, q_vals, significant):
        sig_matrix[i, j] = sig_matrix[j, i] = sig
        rows.append({"network_i": names[i], "network_j": names[j], "p": p, "q": q, "significant": sig})
    return sig_matrix, rows


def main(order):
    import csv

    out_dir = os.path.join(REPO_ROOT, "motion_denoising_pipeline_level_evaluation", "ISC", "comparison_data", f"order_{order}")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"order={order} n_boot={N_BOOT}", flush=True)

    _, rows_raw_unproc = l.load_timecourses_manifest(order, "raw")
    series_raw_unproc = l.load_all_series(rows_raw_unproc)
    subjects = sorted(series_raw_unproc.keys(), key=l.natural_key)
    names = l.load_network_names(os.path.join(l.DEST_ROOT, order, "time_courses"))
    entries_raw_unproc = entries_array(series_raw_unproc, subjects)

    print("bootstrapping raw_unprocessed...", flush=True)
    boot_raw_unproc = bootstrap_standalone(entries_raw_unproc, N_BOOT, rng)
    sig, rows = bootstrap_pvalues(boot_raw_unproc, names)
    with open(os.path.join(out_dir, f"order_{order}_raw_unprocessed_isfc_bootstrap_significance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    np.save(os.path.join(out_dir, f"order_{order}_raw_unprocessed_isfc_bootstrap_dist.npy"), boot_raw_unproc)
    print(f"  {sum(r['significant'] for r in rows)}/{len(rows)} unique cells FDR-significant", flush=True)

    _, rows_raw_prep = l.load_timecourses_manifest(order, "train_data_raw_preprocessed")
    _, rows_denoised = l.load_timecourses_manifest(order, "denoised_st_v4")
    series_raw_prep = l.load_all_series(rows_raw_prep)
    series_denoised = l.load_all_series(rows_denoised)
    assert sorted(series_raw_prep.keys()) == sorted(series_denoised.keys()) == sorted(subjects)
    entries_raw_prep = entries_array(series_raw_prep, subjects)
    entries_denoised = entries_array(series_denoised, subjects)

    print("bootstrapping raw_preprocessed + denoised_st_v4 (paired)...", flush=True)
    boot_raw_prep, boot_denoised = bootstrap_paired(entries_raw_prep, entries_denoised, N_BOOT, rng)
    boot_delta = boot_denoised - boot_raw_prep

    for tag, boot in [("raw_preprocessed", boot_raw_prep), ("denoised_st_v4", boot_denoised), ("delta", boot_delta)]:
        sig, rows = bootstrap_pvalues(boot, names)
        with open(os.path.join(out_dir, f"order_{order}_{tag}_isfc_bootstrap_significance.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        np.save(os.path.join(out_dir, f"order_{order}_{tag}_isfc_bootstrap_dist.npy"), boot)
        print(f"  {tag}: {sum(r['significant'] for r in rows)}/{len(rows)} unique cells FDR-significant", flush=True)

    print(f"saved bootstrap significance + distributions -> {out_dir}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    args = ap.parse_args()
    main(args.order)
