import argparse
import csv
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

DEST_ROOT = "/lustre/disk/home/shared/cusacklab/foundcog/bids/derivatives/isc_segmenting/isc_comparison_data_cyclegans"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_ROOT = os.path.join(SCRIPT_DIR, "figures")
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")


def source_dir_name(source):
    return "raw_data_network_level" if source == "raw" else source


def load_timecourses_manifest(order, source):
    root = DEST_ROOT if source == "raw" else os.path.join(DEST_ROOT, source)
    tc_root = os.path.join(root, order, "time_courses")
    manifest_path = os.path.join(tc_root, f"order_{order}_timecourses_manifest.csv")
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return tc_root, rows


def load_roi_labels(tc_root):
    with open(os.path.join(tc_root, "roi_labels.csv"), newline="") as f:
        return [r["roi_name"] for r in csv.DictReader(f)]


def load_network_names(tc_root):
    with open(os.path.join(tc_root, "network_names.json")) as f:
        return json.load(f)


def load_all_series(rows):
    all_series = defaultdict(list)
    for r in rows:
        entry = {
            "session": int(r["session"]),
            "run": int(r["run"]),
            "segment_num": int(r["segment_num"]),
            "roi": np.load(r["roi_tc_path"]),
            "network": np.load(r["network_tc_path"]),
        }
        all_series[r["subject"]].append(entry)
    return dict(all_series)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def inverse_fisher_z(z):
    return np.tanh(z)


def select_representative_entry(entries, session, run, segment_num):
    """Deterministically choose exactly one entry to represent a subject for a
    given target (session, run, segment_num) acquisition. Never averages raw
    BOLD across a subject's own multiple acquisitions -- each subject
    contributes exactly one time series to the group mean. Priority:
      1. same session + same run
      2. same session, closest run
      3. same run, closest session
      4. closest occurrence overall (any session/run)
    Ties within a tier prefer the same segment_num, then the lowest
    (session, run, segment_num) for full determinism.
    """

    def other_segment(e):
        return e["segment_num"] != segment_num

    exact = [e for e in entries if e["session"] == session and e["run"] == run]
    if exact:
        return min(exact, key=lambda e: (other_segment(e), e["segment_num"]))

    same_session = [e for e in entries if e["session"] == session]
    if same_session:
        return min(
            same_session,
            key=lambda e: (abs(e["run"] - run), other_segment(e), e["run"], e["segment_num"]),
        )

    same_run = [e for e in entries if e["run"] == run]
    if same_run:
        return min(
            same_run,
            key=lambda e: (abs(e["session"] - session), other_segment(e), e["session"], e["segment_num"]),
        )

    return min(
        entries,
        key=lambda e: (
            abs(e["session"] - session),
            abs(e["run"] - run),
            other_segment(e),
            e["session"],
            e["run"],
            e["segment_num"],
        ),
    )


def compute_loo_mean(all_series, target_subject, session, run, segment_num, key):
    reps = [
        select_representative_entry(entries, session, run, segment_num)[key]
        for subject, entries in all_series.items()
        if subject != target_subject
    ]
    return np.mean(reps, axis=0)


def rowwise_pearson(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
    return num / den


def compute_subject_isc(all_series, subject, key):
    entries = all_series[subject]
    r_values = []
    for e in entries:
        loo_mean = compute_loo_mean(all_series, subject, e["session"], e["run"], e["segment_num"], key)
        r_values.append(rowwise_pearson(e[key], loo_mean))

    z_mean = fisher_z(np.stack(r_values)).mean(axis=0)
    return inverse_fisher_z(z_mean)


def compute_all_isc(all_series, key):
    subjects = sorted(all_series.keys(), key=natural_key)
    isc = np.stack([compute_subject_isc(all_series, s, key) for s in subjects])
    return subjects, isc


def group_summary(isc):
    return inverse_fisher_z(fisher_z(isc).mean(axis=0))


def isc_significance(isc, alpha=0.05):
    """Two-sided one-sample t-test on each column's across-subject Fisher-z ISC values
    (against 0), BH-FDR corrected across columns -- same statistical convention used
    throughout this project's QC-FC/FC significance analyses (denoising_evaluation.py)."""
    z = fisher_z(isc)
    t, p = stats.ttest_1samp(z, popmean=0.0, axis=0)
    significant, q, _, _ = multipletests(p, alpha=alpha, method="fdr_bh")
    return t, p, q, significant


def full_pearson_matrix(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = a @ b.T
    denom = np.sqrt((a ** 2).sum(axis=1))[:, None] * np.sqrt((b ** 2).sum(axis=1))[None, :]
    return num / denom


def compute_subject_isfc(all_series, subject, key):
    entries = all_series[subject]
    r_matrices = []
    for e in entries:
        loo_mean = compute_loo_mean(all_series, subject, e["session"], e["run"], e["segment_num"], key)
        r_matrices.append(full_pearson_matrix(e[key], loo_mean))

    z_mean = fisher_z(np.stack(r_matrices)).mean(axis=0)
    return inverse_fisher_z(z_mean)


def compute_all_isfc(all_series, key):
    subjects = sorted(all_series.keys(), key=natural_key)
    isfc = np.stack([compute_subject_isfc(all_series, s, key) for s in subjects])
    return subjects, isfc


def symmetrize(matrix):
    return (matrix + matrix.T) / 2


def group_isfc_summary(isfc_stack):
    return inverse_fisher_z(fisher_z(isfc_stack).mean(axis=0))


def save_isfc_group_csv(path, names, matrix):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for name, row in zip(names, matrix):
            writer.writerow([name] + list(row))


def plot_isfc_heatmap(matrix, names, title, out_path):
    vmax = np.abs(matrix).max()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = matrix[i, j]
            color = "white" if abs(value) > vmax * 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title(title.replace(" — ", "\n"), fontsize=9)
    fig.colorbar(im, ax=ax, label="ISC (r)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_matrix(path, subjects, column_names, isc):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject"] + column_names)
        for subject, row in zip(subjects, isc):
            writer.writerow([subject] + list(row))


def save_group_summary(path, column_names, summary):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerow(list(summary))


def save_significance(path, column_names, summary, t, p, q, significant):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "group_isc", "t", "p", "q", "significant"])
        for name, isc_val, t_val, p_val, q_val, sig in zip(column_names, summary, t, p, q, significant):
            writer.writerow([name, isc_val, t_val, p_val, q_val, sig])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="A")
    ap.add_argument("--source", default="raw",
                     help="'raw', or a DEST_ROOT subdirectory name produced by motion_correct.py's "
                          "--out_dir_name (e.g. 'motion_corrected', 'motion_correction_with_residual').")
    args = ap.parse_args()
    # raw keeps its original filenames for backward compatibility; motion-corrected
    # outputs get an explicit tag so the two are never confused or overwritten.
    tag_suffix = "" if args.source == "raw" else f"_{args.source}"

    tc_root, rows = load_timecourses_manifest(args.order, args.source)
    roi_labels = load_roi_labels(tc_root)
    network_names = load_network_names(tc_root)
    all_series = load_all_series(rows)
    print(f"Subjects: {len(all_series)}", flush=True)

    root = DEST_ROOT if args.source == "raw" else os.path.join(DEST_ROOT, args.source)
    out_dir = os.path.join(root, args.order, "isc_results")
    os.makedirs(out_dir, exist_ok=True)

    subjects, roi_isc = compute_all_isc(all_series, "roi")
    save_matrix(os.path.join(out_dir, f"order_{args.order}{tag_suffix}_roi_isc.csv"), subjects, roi_labels, roi_isc)
    roi_group = group_summary(roi_isc)
    save_group_summary(os.path.join(out_dir, f"order_{args.order}{tag_suffix}_roi_isc_group.csv"), roi_labels, roi_group)
    roi_t, roi_p, roi_q, roi_sig = isc_significance(roi_isc)
    save_significance(
        os.path.join(out_dir, f"order_{args.order}{tag_suffix}_roi_isc_significance.csv"),
        roi_labels, roi_group, roi_t, roi_p, roi_q, roi_sig,
    )
    print(f"ROI ISC: {int(roi_sig.sum())}/{len(roi_sig)} FDR-significant (N={len(subjects)} subjects)", flush=True)

    subjects, network_isc = compute_all_isc(all_series, "network")
    save_matrix(os.path.join(out_dir, f"order_{args.order}{tag_suffix}_network_isc.csv"), subjects, network_names, network_isc)
    network_group = group_summary(network_isc)
    save_group_summary(os.path.join(out_dir, f"order_{args.order}{tag_suffix}_network_isc_group.csv"), network_names, network_group)
    net_t, net_p, net_q, net_sig = isc_significance(network_isc)
    save_significance(
        os.path.join(out_dir, f"order_{args.order}{tag_suffix}_network_isc_significance.csv"),
        network_names, network_group, net_t, net_p, net_q, net_sig,
    )
    print(f"Network ISC: {int(net_sig.sum())}/{len(net_sig)} FDR-significant (N={len(subjects)} subjects)", flush=True)

    print(f"Wrote ISC results to {out_dir}", flush=True)

    dir_name = source_dir_name(args.source)
    figures_dir = os.path.join(FIGURES_ROOT, dir_name)
    results_dir = os.path.join(RESULTS_ROOT, dir_name)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    isfc_subjects, network_isfc = compute_all_isfc(all_series, "network")
    group_network_isfc = symmetrize(group_isfc_summary(network_isfc))

    np.save(os.path.join(results_dir, f"order_{args.order}{tag_suffix}_network_isfc_persubject.npy"), network_isfc)
    save_isfc_group_csv(
        os.path.join(results_dir, f"order_{args.order}{tag_suffix}_network_isfc_group.csv"), network_names, group_network_isfc
    )

    plot_isfc_heatmap(
        group_network_isfc, network_names,
        f"Order {args.order} ({args.source}) — 7x7 network ISC (group, N={len(isfc_subjects)})",
        os.path.join(figures_dir, f"order_{args.order}{tag_suffix}_network_isfc_group_heatmap.png"),
    )
    print(f"Wrote 7x7 network ISFC results -> {results_dir}, figure -> {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
