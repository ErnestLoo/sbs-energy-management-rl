#!/usr/bin/env python3
"""
Algorithm Comparison — publication-ready figures.
Supports: DQN, DDQN, Q-Learning, PPO, MARL-CTDE vs Baseline.

Run after training all algorithms:
    python compare_algorithms.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# STYLING
# =============================================================================
plt.rcParams['figure.dpi']       = 300
plt.rcParams['savefig.dpi']      = 300
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['font.sans-serif']  = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.grid']        = True
plt.rcParams['grid.alpha']       = 0.3
plt.rcParams['axes.axisbelow']   = True

# Per-algorithm visual identity
ALGO_STYLE = {
    'dqn':     {'color': 'steelblue',   'label': 'DQN',        'ls': '-'},
    'ddqn':    {'color': 'blue',        'label': 'DDQN',       'ls': '-'},
    'qlearn':  {'color': 'green',       'label': 'Q-Learning', 'ls': '-'},
    'ppo':     {'color': 'darkorchid',  'label': 'PPO',        'ls': '-'},
    'marl':    {'color': 'crimson',     'label': 'MARL-CTDE',  'ls': '-'},
    'baseline':{'color': 'orange',      'label': 'Baseline',   'ls': '--'},
}

BASELINE_ENERGY     = 1034.0   # J  — overridden if real data found
BASELINE_SINR       = 10.85    # dB
BASELINE_EFFICIENCY = 79226.0  # bits/J

OUT_DIR = os.path.join("figures", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def _latest(*patterns):
    """Pick the most recent folder across one or more glob patterns that has data."""
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    for folder in reversed(sorted(candidates)):
        for fname in ["avg_reward_per_episode.npy", "episode_rewards.npy"]:
            if os.path.exists(os.path.join(folder, "data", fname)):
                return folder
    return None


def find_folders():
    return {
        'ddqn':   _latest("outputs/ddqn_5sbs_*"),
        'qlearn': _latest("outputs/qlearn_curr_5sbs_*", "outputs/qlearn_5sbs_*"),
        'ppo':    _latest("outputs/ppo_curr_5sbs_*",    "outputs/ppo_5sbs_*"),
    }


def _npy(folder, *names):
    """Try loading any of the given npy filenames from folder/data/."""
    data_dir = os.path.join(folder, "data")
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return np.load(path)
    return None


def load_algo_data(folder, algo):
    """Return dict with keys: rewards, energy, sinr, efficiency (all arrays or None)."""
    if folder is None:
        return None

    if algo == 'ddqn':
        return {
            'rewards':    _npy(folder, 'avg_reward_per_episode.npy'),
            'energy':     _npy(folder, 'rl_energy_per_episode.npy'),
            'sinr':       _npy(folder, 'sinr_per_episode.npy'),
            'efficiency': _npy(folder, 'energy_efficiency_per_episode.npy'),
            'bl_energy':  _npy(folder, 'baseline_energy_per_episode.npy'),
            'bl_sinr':    _npy(folder, 'baseline_sinr_per_episode.npy'),
            'bl_eff':     _npy(folder, 'energy_efficiency_baseline.npy'),
        }
    else:
        # dqn shares the same naming as ddqn (avg_reward / rl_energy)
        # qlearn / ppo / marl use episode_* naming
        return {
            'rewards':    _npy(folder, 'avg_reward_per_episode.npy', 'episode_rewards.npy'),
            'energy':     _npy(folder, 'rl_energy_per_episode.npy',  'episode_energies.npy'),
            'sinr':       _npy(folder, 'sinr_per_episode.npy',       'episode_sinrs.npy'),
            'efficiency': _npy(folder, 'energy_efficiency_per_episode.npy', 'episode_efficiencies.npy'),
            'bl_energy':  _npy(folder, 'baseline_energy_per_episode.npy'),
            'bl_sinr':    _npy(folder, 'baseline_sinr_per_episode.npy'),
            'bl_eff':     _npy(folder, 'energy_efficiency_baseline.npy'),
        }


def load_test_data(folder, algo):
    if folder is None:
        return {}
    test_dir = os.path.join(folder, "test_results")
    data = {}
    for metric in ('energies', 'sinrs', 'efficiencies'):
        p = os.path.join(test_dir, f"{algo}_test_{metric}.npy")
        if os.path.exists(p):
            data[metric] = np.load(p)
    return data


# =============================================================================
# PLOTTING UTILITIES
# =============================================================================

def _smooth(arr, window=10):
    if len(arr) >= window:
        return np.convolve(arr, np.ones(window)/window, mode='valid')
    return arr


def plot_training_curves(datasets, metric, ylabel, title, filename,
                          baseline_key=None, window=10):
    """Overlay learning curves for all available algorithms."""
    plt.figure(figsize=(11, 6))

    for algo, data in datasets.items():
        if data is None or data.get(metric) is None:
            continue
        arr = data[metric]
        ep  = np.arange(1, len(arr) + 1)
        sty = ALGO_STYLE[algo]
        plt.plot(ep, arr, color=sty['color'], alpha=0.25)
        sm = _smooth(arr, window)
        sm_ep = ep[window-1:] if len(arr) >= window else ep
        plt.plot(sm_ep, sm, color=sty['color'], linewidth=2,
                 linestyle=sty['ls'], label=sty['label'])

    # Draw a single baseline line if any dataset has it
    if baseline_key:
        for algo, data in datasets.items():
            if data and data.get(baseline_key) is not None:
                bl = data[baseline_key]
                ep = np.arange(1, len(bl) + 1)
                sty = ALGO_STYLE['baseline']
                plt.plot(ep, bl, color=sty['color'], linestyle=sty['ls'],
                         linewidth=1.5, label=sty['label'])
                break   # only one baseline line

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(ylabel,    fontsize=12)
    plt.title(title,      fontsize=14)
    plt.legend(fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()
    print(f"  Saved {filename}")


def plot_bar_comparison(datasets, metric, ylabel, title, filename,
                         baseline_val=None, val_fmt="{:.1f}", last_n=10):
    """Bar chart: last N-episode average for each algorithm vs baseline."""
    labels, values, colors = [], [], []

    # Baseline first
    if baseline_val is not None:
        labels.append('Baseline\n(Always On)')
        values.append(baseline_val)
        colors.append(ALGO_STYLE['baseline']['color'])

    order = ['qlearn', 'dqn', 'ddqn', 'ppo']
    for algo in order:
        data = datasets.get(algo)
        if data is None or data.get(metric) is None:
            continue
        arr = data[metric]
        labels.append(ALGO_STYLE[algo]['label'])
        values.append(float(np.mean(arr[-last_n:])))
        colors.append(ALGO_STYLE[algo]['color'])

    if not values:
        return

    plt.figure(figsize=(max(6, len(values)*1.6), 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.1, width=0.7)

    y_top = max(values) * 1.18
    plt.ylim(0, y_top)

    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + y_top*0.01,
                 val_fmt.format(val),
                 ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.ylabel(ylabel, fontsize=12)
    plt.title(title,   fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()
    print(f"  Saved {filename}")


def plot_test_summary(datasets, test_datasets, filename):
    """3-panel bar chart for test-phase results with 95% CI error bars."""
    available = [(algo, test_datasets[algo])
                 for algo in ('qlearn', 'dqn', 'ddqn', 'ppo')
                 if test_datasets.get(algo) and 'energies' in test_datasets[algo]]

    if not available:
        print("  No test data found — skipping test_phase_summary.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    def ci(arr):
        return 1.96 * np.std(arr) / np.sqrt(len(arr))

    # Collect baseline from any dataset that has it
    bl_energy = bl_sinr = bl_eff = None
    for algo, data in datasets.items():
        if data and data.get('bl_energy') is not None:
            bl_energy = float(np.mean(data['bl_energy']))
            bl_sinr   = float(np.mean(data['bl_sinr']))  if data.get('bl_sinr') is not None else BASELINE_SINR
            bl_eff    = float(np.mean(data['bl_eff']))   if data.get('bl_eff')  is not None else BASELINE_EFFICIENCY
            break
    if bl_energy is None:
        bl_energy, bl_sinr, bl_eff = BASELINE_ENERGY, BASELINE_SINR, BASELINE_EFFICIENCY

    for ax_idx, (metric_key, bl_val, title, ylabel, y_lim) in enumerate([
        ('energies',     bl_energy, 'Energy Consumption', 'Energy (J)',       None),
        ('sinrs',        bl_sinr,   'Signal Quality (SINR)', 'SINR (dB)',     None),
        ('efficiencies', bl_eff,    'Energy Efficiency',  'bits/J',           None),
    ]):
        ax = axes[ax_idx]
        labels = ['Baseline']
        means  = [bl_val]
        cis    = [0.0]
        colors = [ALGO_STYLE['baseline']['color']]

        for algo, td in available:
            arr = td[metric_key]
            labels.append(ALGO_STYLE[algo]['label'])
            means.append(float(np.mean(arr)))
            cis.append(float(ci(arr)))
            colors.append(ALGO_STYLE[algo]['color'])

        bars = ax.bar(labels, means, yerr=cis, color=colors,
                      edgecolor='black', linewidth=1.1, capsize=5)
        ax.set_title(title,  fontsize=14)
        ax.set_ylabel(ylabel, fontsize=12)

        if ax_idx == 1:  # SINR
            ax.axhline(y=12.5, color='forestgreen', linestyle='--', alpha=0.6, label='Good (12.5 dB)')
            ax.axhline(y=7.0,  color='red',         linestyle='--', alpha=0.6, label='Fair (7.0 dB)')
            ax.legend(fontsize=8)

        y_top = max(means) * 1.18
        ax.set_ylim(0, y_top)

        for bar, m, c in zip(bars, means, cis):
            if metric_key == 'efficiencies':
                lbl = f"{m/1000:.1f}k" + (f"±{c/1000:.1f}k" if c > 0 else "")
            elif metric_key == 'sinrs':
                lbl = f"{m:.2f}" + (f"±{c:.2f}" if c > 0 else "")
            else:
                lbl = f"{m:.0f}" + (f"±{c:.1f}" if c > 0 else "")
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + c + y_top*0.01,
                    lbl, ha='center', fontweight='bold', fontsize=8)

    plt.suptitle('Test Phase Performance Summary (95% CI)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"\n{'='*60}")
    print("Algorithm Comparison Figure Generator")
    print(f"{'='*60}")

    folders  = find_folders()
    datasets = {}
    for algo, folder in folders.items():
        if folder:
            print(f"  Loading {algo.upper():10s} from {folder}")
            datasets[algo] = load_algo_data(folder, algo)
        else:
            print(f"  [SKIP] No output folder found for {algo.upper()}")
            datasets[algo] = None

    test_datasets = {algo: load_test_data(folders[algo], algo) for algo in folders}

    print(f"\nGenerating figures → {OUT_DIR}/")

    # --- Learning curves ---
    plot_training_curves(datasets, 'rewards',    'Avg Reward',            'Learning Progress (Reward)',
                          'fig_training_reward.png', window=10)

    plot_training_curves(datasets, 'energy',     'Total Energy (J)',      'Energy Consumption per Episode',
                          'fig_training_energy.png', baseline_key='bl_energy', window=10)

    plot_training_curves(datasets, 'sinr',       'Global SINR (dB)',      'SINR per Episode',
                          'fig_training_sinr.png',   baseline_key='bl_sinr',   window=10)

    plot_training_curves(datasets, 'efficiency', 'Energy Efficiency (bits/J)',
                          'Energy Efficiency per Episode',
                          'fig_training_efficiency.png', baseline_key='bl_eff', window=10)

    # --- Bar charts (converged performance) ---
    # Resolve baseline from real data if possible
    bl_e = bl_s = bl_eff = None
    for d in datasets.values():
        if d and d.get('bl_energy') is not None:
            bl_e   = float(np.mean(d['bl_energy']))
            bl_s   = float(np.mean(d['bl_sinr']))  if d.get('bl_sinr')  is not None else BASELINE_SINR
            bl_eff = float(np.mean(d['bl_eff']))   if d.get('bl_eff')   is not None else BASELINE_EFFICIENCY
            break
    bl_e   = bl_e   or BASELINE_ENERGY
    bl_s   = bl_s   or BASELINE_SINR
    bl_eff = bl_eff or BASELINE_EFFICIENCY

    plot_bar_comparison(datasets, 'energy',     'Total Energy (J)',
                         'Converged Energy Consumption', 'fig_bar_energy.png',
                         baseline_val=bl_e,   val_fmt="{:.0f} J")

    plot_bar_comparison(datasets, 'sinr',       'Avg SINR (dB)',
                         'Converged SINR',            'fig_bar_sinr.png',
                         baseline_val=bl_s,   val_fmt="{:.2f} dB")

    plot_bar_comparison(datasets, 'efficiency', 'Energy Efficiency (bits/J)',
                         'Converged Energy Efficiency','fig_bar_efficiency.png',
                         baseline_val=bl_eff, val_fmt="{:.0f}")

    # --- Test phase summary ---
    plot_test_summary(datasets, test_datasets, 'fig_test_summary.png')

    print(f"\nDone. All figures in '{OUT_DIR}/'.")
    print("Algorithms found:", [a for a, d in datasets.items() if d is not None])


if __name__ == "__main__":
    main()
