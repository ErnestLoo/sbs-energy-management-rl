#!/usr/bin/env python3
"""
Figure generator — DDQN vs PPO vs Baseline.

Reads CSVs produced by the training scripts and the test harness, then emits the
7 planned figures. Plotting is fully decoupled from data collection: re-run this
any time without re-running training/testing.

Inputs (auto-detected latest):
    outputs/ddqn_5sbs_*/data/training_metrics.csv
    outputs/ppo_curr_5sbs_*/data/training_metrics.csv
    outputs/ddqn_5sbs_*/data/baseline_*_per_episode.npy   (per-episode baseline)
    outputs/test_*/testing_metrics.csv                    (test harness)

Outputs -> figures/compare_<timestamp>/
    Training (line, x=Episode):
      1 fig_train_reward.png       (Average Rewards; DDQN, PPO)
      2 fig_train_energy.png       (Total Energy; DDQN, PPO, Baseline)
      3 fig_train_sinr.png         (SINR + threshold lines)
      4 fig_train_efficiency.png   (bits/J)
    Testing (bar + 95% CI, 5 series: DDQN-G1, PPO-G1, DDQN-G2, PPO-G2, Baseline):
      5 fig_test_energy.png
      6 fig_test_sinr.png          (+ threshold lines)
      7 fig_test_efficiency.png
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import research_config as cfg

plt.rcParams["figure.dpi"]     = 300
plt.rcParams["savefig.dpi"]    = 300
plt.rcParams["axes.grid"]      = True
plt.rcParams["grid.alpha"]     = 0.3
plt.rcParams["axes.axisbelow"] = True

ALGO_STYLE = {
    "ddqn":     {"color": "tab:blue",   "label": "DDQN"},
    "ppo":      {"color": "tab:green",  "label": "PPO"},
    "baseline": {"color": "tab:orange", "label": "Baseline"},
}

# test-phase series order and styling (5 bars)
TEST_SERIES = [
    ("ddqn", "G1", "DDQN-G1", "tab:blue"),
    ("ppo",  "G1", "PPO-G1",  "tab:green"),
    ("ddqn", "G2", "DDQN-G2", "steelblue"),
    ("ppo",  "G2", "PPO-G2",  "limegreen"),
    ("baseline", "-", "Baseline", "tab:orange"),
]

OUT_DIR = os.path.join(cfg.FIGURES_DIR,
                       "compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# LOADING
# =============================================================================
def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(cfg.OUTPUTS_DIR, pattern)))
    return hits[-1] if hits else None


def load_training_csv(folder):
    """Return dict of np arrays for reward/energy/sinr/efficiency, or None."""
    if not folder:
        return None
    path = os.path.join(folder, "data", "training_metrics.csv")
    if not os.path.exists(path):
        print(f"  [WARN] no training_metrics.csv in {folder}")
        return None
    cols = {c: [] for c in cfg.TRAINING_CSV_COLUMNS}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for c in cfg.TRAINING_CSV_COLUMNS:
                cols[c].append(float(row[c]))
    return {c: np.array(v) for c, v in cols.items()}


def load_npy(folder, name):
    if not folder:
        return None
    p = os.path.join(folder, "data", name)
    return np.load(p) if os.path.exists(p) else None


def load_testing_csv(folder):
    """Return {(algo, group): {'energy':[], 'sinr':[], 'efficiency':[]}}."""
    if not folder:
        return {}
    path = os.path.join(folder, "testing_metrics.csv")
    if not os.path.exists(path):
        print(f"  [WARN] no testing_metrics.csv in {folder}")
        return {}
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["algorithm"], row["sleep_group"])
            d = data.setdefault(key, {"energy": [], "sinr": [], "efficiency": []})
            d["energy"].append(float(row["energy"]))
            d["sinr"].append(float(row["sinr"]))
            d["efficiency"].append(float(row["efficiency"]))
    return data


# =============================================================================
# HELPERS
# =============================================================================
def smooth(a, w=10):
    if len(a) >= w:
        return np.convolve(a, np.ones(w) / w, mode="valid"), w - 1
    return a, 0


def ci95(a):
    a = np.asarray(a)
    return 1.96 * np.std(a) / np.sqrt(len(a)) if len(a) else 0.0


def add_sinr_thresholds(ax):
    for val, color, lbl in cfg.SINR_THRESHOLDS:
        ax.axhline(val, color=color, linestyle=":", lw=1.3, label=lbl)


# =============================================================================
# TRAINING LINE PLOTS
# =============================================================================
def plot_training_line(train, metric, ylabel, title, fname,
                       include_baseline=False, baseline_arr=None,
                       sinr_thresholds=False):
    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    for algo in ("ddqn", "ppo"):
        d = train.get(algo)
        if d is None or metric not in d:
            continue
        arr = d[metric]
        ep = np.arange(1, len(arr) + 1)
        sty = ALGO_STYLE[algo]
        ax.plot(ep, arr, color=sty["color"], alpha=0.2)
        sm, off = smooth(arr)
        ax.plot(ep[off:], sm, color=sty["color"], lw=2, label=sty["label"])

    if include_baseline and baseline_arr is not None and len(baseline_arr):
        ep = np.arange(1, len(baseline_arr) + 1)
        ax.plot(ep, baseline_arr, color=ALGO_STYLE["baseline"]["color"],
                lw=1.8, ls="--", label="Baseline (Always Active)")

    if sinr_thresholds:
        add_sinr_thresholds(ax)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"  saved {fname}")


# =============================================================================
# TEST BAR PLOTS (mean + 95% CI, 5 series)
# =============================================================================
def plot_test_bar(test, metric, ylabel, title, fname,
                  val_fmt="{:.0f}", sinr_thresholds=False):
    labels, means, cis, colors = [], [], [], []
    for algo, group, label, color in TEST_SERIES:
        d = test.get((algo, group))
        if not d or not d[metric]:
            continue
        labels.append(label)
        means.append(float(np.mean(d[metric])))
        cis.append(ci95(d[metric]))
        colors.append(color)

    if not means:
        print(f"  [SKIP] {fname}: no test data")
        return

    plt.figure(figsize=(max(7, len(means) * 1.6), 6))
    ax = plt.gca()
    bars = ax.bar(labels, means, yerr=cis, color=colors,
                  edgecolor="black", lw=1.1, capsize=5)

    if sinr_thresholds:
        add_sinr_thresholds(ax)
        ax.legend(fontsize=8)

    y_top = (max(m + c for m, c in zip(means, cis))) * 1.18
    ax.set_ylim(0, y_top)
    for bar, m, c in zip(bars, means, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + c + y_top * 0.01,
                val_fmt.format(m), ha="center", va="bottom",
                fontweight="bold", fontsize=9)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"  saved {fname}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    ddqn_folder = _latest("ddqn_5sbs_*")
    ppo_folder  = _latest("ppo_curr_5sbs_*") or _latest("ppo_5sbs_*")
    test_folder = _latest("test_*")

    print("Loading data...")
    print(f"  DDQN train : {ddqn_folder}")
    print(f"  PPO  train : {ppo_folder}")
    print(f"  Test       : {test_folder}")

    train = {
        "ddqn": load_training_csv(ddqn_folder),
        "ppo":  load_training_csv(ppo_folder),
    }
    # per-episode baseline (saved by training scripts) — prefer DDQN's
    bl_energy = load_npy(ddqn_folder, "baseline_energy_per_episode.npy")
    bl_sinr   = load_npy(ddqn_folder, "baseline_sinr_per_episode.npy")
    bl_eff    = load_npy(ddqn_folder, "energy_efficiency_baseline.npy")

    test = load_testing_csv(test_folder)

    print(f"\nGenerating figures -> {OUT_DIR}/")

    # ---- Training (line, x = Episode) ----
    plot_training_line(train, "reward", "Average Rewards",
                       "Training — Learning Progress", "fig_train_reward.png")
    plot_training_line(train, "energy", "Total Energy (J)",
                       "Training — Energy Consumption", "fig_train_energy.png",
                       include_baseline=True, baseline_arr=bl_energy)
    plot_training_line(train, "sinr", "SINR (dB)",
                       "Training — Signal Quality", "fig_train_sinr.png",
                       include_baseline=True, baseline_arr=bl_sinr,
                       sinr_thresholds=True)
    plot_training_line(train, "efficiency", "Energy Efficiency (bits/J)",
                       "Training — Energy Efficiency", "fig_train_efficiency.png",
                       include_baseline=True, baseline_arr=bl_eff)

    # ---- Testing (bar + 95% CI, 5 series) ----
    plot_test_bar(test, "energy", "Total Energy (J)",
                  "Test — Energy Consumption (mean ± 95% CI)",
                  "fig_test_energy.png", val_fmt="{:.0f} J")
    plot_test_bar(test, "sinr", "SINR (dB)",
                  "Test — Signal Quality (mean ± 95% CI)",
                  "fig_test_sinr.png", val_fmt="{:.2f}", sinr_thresholds=True)
    plot_test_bar(test, "efficiency", "Energy Efficiency (bits/J)",
                  "Test — Energy Efficiency (mean ± 95% CI)",
                  "fig_test_efficiency.png", val_fmt="{:.0f}")

    print(f"\nDone. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
