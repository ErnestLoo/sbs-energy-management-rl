#!/usr/bin/env python3
"""
Curriculum Learning Ablation Study
Compares: DDQN with vs without curriculum, Q-Learning with vs without curriculum

Run after training all four variants:
    python fast_ddqn_multiagent.py       -> outputs/ddqn_5sbs_*
    python ddqn_no_curriculum.py         -> outputs/ddqn_nocurr_5sbs_*
    python qlearn_multiagent.py          -> outputs/qlearn_curr_5sbs_*
    python qlearn_no_curriculum.py       -> outputs/qlearn_nocurr_5sbs_*

    python compare_curriculum.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['figure.dpi']      = 300
plt.rcParams['savefig.dpi']     = 300
plt.rcParams['font.family']     = 'sans-serif'
plt.rcParams['axes.grid']       = True
plt.rcParams['grid.alpha']      = 0.3
plt.rcParams['axes.axisbelow']  = True

STYLE = {
    'ddqn_curr':    {'color': 'blue',       'label': 'DDQN + Curriculum',    'ls': '-'},
    'ddqn_nocurr':  {'color': 'steelblue',  'label': 'DDQN No Curriculum',   'ls': '--'},
    'qlearn_curr':  {'color': 'green',      'label': 'Q-Learning + Curriculum', 'ls': '-'},
    'qlearn_nocurr':{'color': 'limegreen',  'label': 'Q-Learning No Curriculum', 'ls': '--'},
    'baseline':     {'color': 'orange',     'label': 'Baseline',             'ls': '--'},
}

OUT_DIR = os.path.join("figures", "curriculum_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_ENERGY = 1034.0
BASELINE_SINR   = 10.85
BASELINE_EFF    = 79226.0


def _latest(pattern, required="avg_reward_per_episode.npy"):
    for folder in reversed(sorted(glob.glob(pattern))):
        for fname in [required, "episode_rewards.npy"]:
            if os.path.exists(os.path.join(folder, "data", fname)):
                return folder
    return None


def load(folder):
    if folder is None:
        return None
    d = os.path.join(folder, "data")
    def npy(*names):
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return np.load(p)
        return None
    return {
        'rewards':    npy('avg_reward_per_episode.npy', 'episode_rewards.npy'),
        'energy':     npy('rl_energy_per_episode.npy',  'episode_energies.npy'),
        'sinr':       npy('sinr_per_episode.npy',       'episode_sinrs.npy'),
        'efficiency': npy('energy_efficiency_per_episode.npy', 'episode_efficiencies.npy'),
    }


def smooth(arr, w=10):
    if len(arr) >= w:
        return np.convolve(arr, np.ones(w)/w, mode='valid'), w-1
    return arr, 0


def plot_metric(datasets, metric, ylabel, title, fname,
                baseline_val=None, threshold_lines=None):
    plt.figure(figsize=(11, 6))

    for key, data in datasets.items():
        if data is None or data.get(metric) is None:
            continue
        arr = data[metric]
        ep  = np.arange(1, len(arr)+1)
        sty = STYLE[key]
        plt.plot(ep, arr, color=sty['color'], alpha=0.2)
        sm, offset = smooth(arr)
        plt.plot(ep[offset:], sm, color=sty['color'], lw=2,
                 linestyle=sty['ls'], label=sty['label'])

    if baseline_val is not None:
        plt.axhline(baseline_val, color='orange', linestyle='--',
                    lw=1.5, label='Baseline (Always Active)')

    if threshold_lines:
        for val, color, lbl in threshold_lines:
            plt.axhline(val, color=color, linestyle=':', lw=1.2, label=lbl)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"  Saved {fname}")


def plot_bar(datasets, metric, ylabel, title, fname,
             baseline_val=None, fmt="{:.1f}", last_n=10):
    labels, values, colors = [], [], []

    if baseline_val is not None:
        labels.append('Baseline\n(Always On)')
        values.append(baseline_val)
        colors.append(STYLE['baseline']['color'])

    order = ['ddqn_curr', 'ddqn_nocurr', 'qlearn_curr', 'qlearn_nocurr']
    for key in order:
        data = datasets.get(key)
        if data is None or data.get(metric) is None:
            continue
        arr = data[metric]
        labels.append(STYLE[key]['label'])
        values.append(float(np.mean(arr[-last_n:])))
        colors.append(STYLE[key]['color'])

    if not values:
        return

    plt.figure(figsize=(max(6, len(values)*1.8), 6))
    bars = plt.bar(labels, values, color=colors, edgecolor='black', lw=1.1, width=0.65)
    y_top = max(values) * 1.18
    plt.ylim(0, y_top)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + y_top*0.01,
                 fmt.format(val), ha='center', va='bottom',
                 fontweight='bold', fontsize=9)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"  Saved {fname}")


def main():
    folders = {
        'ddqn_curr':    _latest("outputs/ddqn_5sbs_*"),
        'ddqn_nocurr':  _latest("outputs/ddqn_nocurr_5sbs_*"),
        'qlearn_curr':  _latest("outputs/qlearn_curr_5sbs_*"),
        'qlearn_nocurr':_latest("outputs/qlearn_nocurr_5sbs_*"),
    }

    datasets = {}
    for key, folder in folders.items():
        if folder:
            print(f"  Loading {key:20s} from {os.path.basename(folder)}")
            datasets[key] = load(folder)
        else:
            print(f"  [SKIP] No folder found for {key}")
            datasets[key] = None

    print(f"\nGenerating figures → {OUT_DIR}/")

    sinr_thresholds = [
        (12.5, 'forestgreen', 'Good (12.5 dB)'),
        (7.0,  'red',         'Fair (7.0 dB)'),
    ]

    plot_metric(datasets, 'rewards',    'Avg Reward',
                'Curriculum Ablation — Reward',
                'curriculum_reward.png')

    plot_metric(datasets, 'energy',     'Total Energy (J)',
                'Curriculum Ablation — Energy Consumption',
                'curriculum_energy.png', baseline_val=BASELINE_ENERGY)

    plot_metric(datasets, 'sinr',       'Avg SINR (dB)',
                'Curriculum Ablation — Signal Quality',
                'curriculum_sinr.png',
                baseline_val=BASELINE_SINR,
                threshold_lines=sinr_thresholds)

    plot_metric(datasets, 'efficiency', 'Energy Efficiency (bits/J)',
                'Curriculum Ablation — Energy Efficiency',
                'curriculum_efficiency.png', baseline_val=BASELINE_EFF)

    plot_bar(datasets, 'energy',     'Total Energy (J)',
             'Converged Energy — With vs Without Curriculum',
             'curriculum_bar_energy.png',
             baseline_val=BASELINE_ENERGY, fmt="{:.0f} J")

    plot_bar(datasets, 'sinr',       'Avg SINR (dB)',
             'Converged SINR — With vs Without Curriculum',
             'curriculum_bar_sinr.png',
             baseline_val=BASELINE_SINR, fmt="{:.2f} dB")

    plot_bar(datasets, 'efficiency', 'Energy Efficiency (bits/J)',
             'Converged Efficiency — With vs Without Curriculum',
             'curriculum_bar_efficiency.png',
             baseline_val=BASELINE_EFF, fmt="{:.0f}")

    print(f"\nDone. Figures in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()
