#!/usr/bin/env python3
"""
Algorithm Comparison Script for Capstone Report
================================================

This script compares DDQN and Q-Learning algorithms by:
1. Loading training data from outputs/
2. Loading test results from outputs/*/test_results/
3. Generating 17 publication-ready figures

Figures Generated:
- Fig 1-4: DDQN training (reward, energy, SINR, efficiency)
- Fig 5-8: Q-Learning training (reward, energy, SINR, efficiency)
- Fig 9-11: Algorithm comparison (energy, SINR, efficiency)
- Fig 12-17: Testing phase (box plots, histograms, summary)

Usage:
    python compare_algorithms.py

Output:
    - Training figures -> outputs/*/plots/
    - Comparison figures -> figures/
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
BASELINE_ENERGY = 1034.0
BASELINE_SINR = 10.77
BASELINE_EFFICIENCY = 79229.0

# Figure settings
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


# =============================================================================
# FIND OUTPUT FOLDERS
# =============================================================================
def find_output_folders():
    """Find DDQN and Q-Learning output folders"""
    ddqn_folders = sorted(glob.glob("outputs/ddqn_5sbs_*"))
    qlearn_folders = sorted(glob.glob("outputs/qlearn_5sbs_*"))
    
    ddqn_folder = ddqn_folders[-1] if ddqn_folders else None
    qlearn_folder = qlearn_folders[-1] if qlearn_folders else None
    
    return ddqn_folder, qlearn_folder


# =============================================================================
# LOAD DATA
# =============================================================================
def load_training_data(folder, algorithm):
    """Load training data from output folder"""
    data_dir = os.path.join(folder, "data")
    
    data = {'rewards': None, 'energy': None, 'sinr': None, 'efficiency': None}
    
    if algorithm == 'ddqn':
        if os.path.exists(os.path.join(data_dir, "avg_reward_per_episode.npy")):
            data['rewards'] = np.load(os.path.join(data_dir, "avg_reward_per_episode.npy"))
            data['energy'] = np.load(os.path.join(data_dir, "rl_energy_per_episode.npy"))
            data['sinr'] = np.load(os.path.join(data_dir, "sinr_per_episode.npy"))
            data['efficiency'] = np.load(os.path.join(data_dir, "energy_efficiency_per_episode.npy"))
    
    elif algorithm == 'qlearn':
        if os.path.exists(os.path.join(data_dir, "episode_rewards.npy")):
            data['rewards'] = np.load(os.path.join(data_dir, "episode_rewards.npy"))
            data['energy'] = np.load(os.path.join(data_dir, "episode_energies.npy"))
            data['sinr'] = np.load(os.path.join(data_dir, "episode_sinrs.npy"))
            data['efficiency'] = np.load(os.path.join(data_dir, "episode_efficiencies.npy"))
    
    return data


def load_test_data(folder, algorithm):
    """Load test results data"""
    test_dir = os.path.join(folder, "test_results")
    data = {}
    for metric in ['energies', 'sinrs', 'efficiencies', 'rewards']:
        filepath = os.path.join(test_dir, f"{algorithm}_test_{metric}.npy")
        if os.path.exists(filepath):
            data[metric] = np.load(filepath)
    return data


# =============================================================================
# TRAINING FIGURES (Fig 1-8)
# =============================================================================
def plot_training_reward(data, algorithm, output_path, fig_num):
    """Plot reward curve during training"""
    plt.figure(figsize=(8, 5))
    episodes = np.arange(1, len(data) + 1)
    
    color = '#3498db' if algorithm == 'ddqn' else '#2ecc71'
    label = 'DDQN' if algorithm == 'ddqn' else 'Q-Learning'
    
    plt.plot(episodes, data, color=color, alpha=0.3, linewidth=0.5)
    
    window = 10
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    ma_episodes = np.arange(window, len(data) + 1)
    plt.plot(ma_episodes, ma, color=color, linewidth=2, label=f'{label} (MA-{window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'{label} Training: Reward Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


def plot_training_energy(data, algorithm, output_path, fig_num):
    """Plot energy consumption during training"""
    plt.figure(figsize=(8, 5))
    episodes = np.arange(1, len(data) + 1)
    
    color = '#3498db' if algorithm == 'ddqn' else '#2ecc71'
    label = 'DDQN' if algorithm == 'ddqn' else 'Q-Learning'
    
    plt.plot(episodes, data, color=color, alpha=0.5, linewidth=1)
    plt.axhline(y=BASELINE_ENERGY, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    window = 10
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    ma_episodes = np.arange(window, len(data) + 1)
    plt.plot(ma_episodes, ma, color=color, linewidth=2, label=f'{label} (MA-{window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Energy Consumption (J)')
    plt.title(f'{label} Training: Energy Consumption')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


def plot_training_sinr(data, algorithm, output_path, fig_num):
    """Plot SINR during training"""
    plt.figure(figsize=(8, 5))
    episodes = np.arange(1, len(data) + 1)
    
    color = '#3498db' if algorithm == 'ddqn' else '#2ecc71'
    label = 'DDQN' if algorithm == 'ddqn' else 'Q-Learning'
    
    plt.plot(episodes, data, color=color, alpha=0.5, linewidth=1)
    plt.axhline(y=BASELINE_SINR, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    window = 10
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    ma_episodes = np.arange(window, len(data) + 1)
    plt.plot(ma_episodes, ma, color=color, linewidth=2, label=f'{label} (MA-{window})')
    
    plt.xlabel('Episode')
    plt.ylabel('SINR (dB)')
    plt.title(f'{label} Training: Signal Quality (SINR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


def plot_training_efficiency(data, algorithm, output_path, fig_num):
    """Plot energy efficiency during training"""
    plt.figure(figsize=(8, 5))
    episodes = np.arange(1, len(data) + 1)
    
    color = '#3498db' if algorithm == 'ddqn' else '#2ecc71'
    label = 'DDQN' if algorithm == 'ddqn' else 'Q-Learning'
    
    plt.plot(episodes, data, color=color, alpha=0.5, linewidth=1)
    plt.axhline(y=BASELINE_EFFICIENCY, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    window = 10
    ma = np.convolve(data, np.ones(window)/window, mode='valid')
    ma_episodes = np.arange(window, len(data) + 1)
    plt.plot(ma_episodes, ma, color=color, linewidth=2, label=f'{label} (MA-{window})')
    
    plt.xlabel('Episode')
    plt.ylabel('Energy Efficiency (bits/J)')
    plt.title(f'{label} Training: Energy Efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


# =============================================================================
# COMPARISON FIGURES (Fig 9-11)
# =============================================================================
def plot_comparison(ddqn_data, qlearn_data, ylabel, title, output_path, fig_num, baseline=None):
    """Plot comparison between algorithms"""
    plt.figure(figsize=(8, 5))
    
    window = 10
    ddqn_ma = np.convolve(ddqn_data, np.ones(window)/window, mode='valid')
    qlearn_ma = np.convolve(qlearn_data, np.ones(window)/window, mode='valid')
    ma_episodes = np.arange(window, len(ddqn_data) + 1)
    
    plt.plot(ma_episodes, ddqn_ma, '#3498db', linewidth=2, label='DDQN')
    plt.plot(ma_episodes, qlearn_ma, '#2ecc71', linewidth=2, label='Q-Learning')
    
    if baseline:
        plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


# =============================================================================
# TEST FIGURES (Fig 12-17)
# =============================================================================
def plot_test_boxplot(ddqn_data, qlearn_data, ylabel, title, output_path, fig_num, baseline=None):
    """Create box plot comparing test results"""
    plt.figure(figsize=(8, 5))
    
    bp = plt.boxplot([ddqn_data, qlearn_data], labels=['DDQN', 'Q-Learning'], patch_artist=True)
    
    colors = ['#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    means = [np.mean(ddqn_data), np.mean(qlearn_data)]
    plt.scatter([1, 2], means, color='black', marker='D', s=50, zorder=3, label='Mean')
    
    if baseline:
        plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='Baseline')
    
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


def plot_test_histogram(ddqn_data, qlearn_data, xlabel, title, output_path, fig_num):
    """Create histogram comparing test distributions"""
    plt.figure(figsize=(8, 5))
    
    plt.hist(ddqn_data, bins=20, alpha=0.6, color='#3498db', label='DDQN', edgecolor='black')
    plt.hist(qlearn_data, bins=20, alpha=0.6, color='#2ecc71', label='Q-Learning', edgecolor='black')
    
    plt.axvline(x=np.mean(ddqn_data), color='#2980b9', linestyle='--', linewidth=2)
    plt.axvline(x=np.mean(qlearn_data), color='#27ae60', linestyle='--', linewidth=2)
    
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


def plot_test_trends(ddqn_test, qlearn_test, output_path, fig_num):
    """Plot metrics across test iterations"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    iterations = np.arange(1, len(ddqn_test['energies']) + 1)
    
    axes[0].plot(iterations, ddqn_test['energies'], '#3498db', alpha=0.7, label='DDQN')
    axes[0].plot(iterations, qlearn_test['energies'], '#2ecc71', alpha=0.7, label='Q-Learning')
    axes[0].axhline(y=BASELINE_ENERGY, color='red', linestyle='--', label='Baseline')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Energy (J)')
    axes[0].set_title('Energy Consumption')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(iterations, ddqn_test['sinrs'], '#3498db', alpha=0.7, label='DDQN')
    axes[1].plot(iterations, qlearn_test['sinrs'], '#2ecc71', alpha=0.7, label='Q-Learning')
    axes[1].axhline(y=BASELINE_SINR, color='red', linestyle='--', label='Baseline')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('SINR (dB)')
    axes[1].set_title('Signal Quality')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(iterations, ddqn_test['efficiencies'], '#3498db', alpha=0.7, label='DDQN')
    axes[2].plot(iterations, qlearn_test['efficiencies'], '#2ecc71', alpha=0.7, label='Q-Learning')
    axes[2].axhline(y=BASELINE_EFFICIENCY, color='red', linestyle='--', label='Baseline')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Efficiency (bits/J)')
    axes[2].set_title('Energy Efficiency')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


def plot_test_summary(ddqn_test, qlearn_test, output_path, fig_num):
    """Create summary bar chart with error bars"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    
    x = np.arange(2)
    width = 0.6
    
    # Energy Reduction
    ddqn_red = (BASELINE_ENERGY - np.mean(ddqn_test['energies'])) / BASELINE_ENERGY * 100
    qlearn_red = (BASELINE_ENERGY - np.mean(qlearn_test['energies'])) / BASELINE_ENERGY * 100
    ddqn_ci = 1.96 * np.std(ddqn_test['energies']) / np.sqrt(len(ddqn_test['energies'])) / BASELINE_ENERGY * 100
    qlearn_ci = 1.96 * np.std(qlearn_test['energies']) / np.sqrt(len(qlearn_test['energies'])) / BASELINE_ENERGY * 100
    
    bars1 = axes[0].bar(x, [ddqn_red, qlearn_red], width, yerr=[ddqn_ci, qlearn_ci],
                        color=['#3498db', '#2ecc71'], capsize=5, alpha=0.8)
    axes[0].set_ylabel('Energy Reduction (%)')
    axes[0].set_title('Energy Savings')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['DDQN', 'Q-Learning'])
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, [ddqn_red, qlearn_red]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # SINR
    ddqn_sinr = np.mean(ddqn_test['sinrs'])
    qlearn_sinr = np.mean(qlearn_test['sinrs'])
    ddqn_sinr_ci = 1.96 * np.std(ddqn_test['sinrs']) / np.sqrt(len(ddqn_test['sinrs']))
    qlearn_sinr_ci = 1.96 * np.std(qlearn_test['sinrs']) / np.sqrt(len(qlearn_test['sinrs']))
    
    bars2 = axes[1].bar(x, [ddqn_sinr, qlearn_sinr], width, yerr=[ddqn_sinr_ci, qlearn_sinr_ci],
                        color=['#3498db', '#2ecc71'], capsize=5, alpha=0.8)
    axes[1].axhline(y=BASELINE_SINR, color='red', linestyle='--', label='Baseline')
    axes[1].set_ylabel('SINR (dB)')
    axes[1].set_title('Signal Quality')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['DDQN', 'Q-Learning'])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, [ddqn_sinr, qlearn_sinr]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.2f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Efficiency Improvement
    ddqn_imp = (np.mean(ddqn_test['efficiencies']) - BASELINE_EFFICIENCY) / BASELINE_EFFICIENCY * 100
    qlearn_imp = (np.mean(qlearn_test['efficiencies']) - BASELINE_EFFICIENCY) / BASELINE_EFFICIENCY * 100
    ddqn_imp_ci = 1.96 * np.std(ddqn_test['efficiencies']) / np.sqrt(len(ddqn_test['efficiencies'])) / BASELINE_EFFICIENCY * 100
    qlearn_imp_ci = 1.96 * np.std(qlearn_test['efficiencies']) / np.sqrt(len(qlearn_test['efficiencies'])) / BASELINE_EFFICIENCY * 100
    
    bars3 = axes[2].bar(x, [ddqn_imp, qlearn_imp], width, yerr=[ddqn_imp_ci, qlearn_imp_ci],
                        color=['#3498db', '#2ecc71'], capsize=5, alpha=0.8)
    axes[2].set_ylabel('Efficiency Improvement (%)')
    axes[2].set_title('Energy Efficiency')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['DDQN', 'Q-Learning'])
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, [ddqn_imp, qlearn_imp]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Testing Phase: Algorithm Performance Summary (100 Iterations)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Fig {fig_num}: {os.path.basename(output_path)}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  ALGORITHM COMPARISON - FIGURE GENERATION")
    print("=" * 70)
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find output folders
    ddqn_folder, qlearn_folder = find_output_folders()
    
    if not ddqn_folder or not qlearn_folder:
        print("\nERROR: Could not find output folders")
        return
    
    print(f"\n  DDQN folder: {ddqn_folder}")
    print(f"  Q-Learning folder: {qlearn_folder}")
    
    os.makedirs("figures", exist_ok=True)
    
    # Load training data
    print("\nLoading training data...")
    ddqn_train = load_training_data(ddqn_folder, 'ddqn')
    qlearn_train = load_training_data(qlearn_folder, 'qlearn')
    
    if ddqn_train['rewards'] is None or qlearn_train['rewards'] is None:
        print("ERROR: Could not load training data")
        return
    
    print(f"  DDQN: {len(ddqn_train['rewards'])} episodes")
    print(f"  Q-Learning: {len(qlearn_train['rewards'])} episodes")
    
    # DDQN training figures -> outputs/ddqn_*/plots/
    ddqn_plots = os.path.join(ddqn_folder, "plots")
    os.makedirs(ddqn_plots, exist_ok=True)
    
    print(f"\n--- DDQN Training Figures -> {ddqn_plots}/ ---")
    plot_training_reward(ddqn_train['rewards'], 'ddqn', os.path.join(ddqn_plots, "fig1_ddqn_reward.png"), 1)
    plot_training_energy(ddqn_train['energy'], 'ddqn', os.path.join(ddqn_plots, "fig3_ddqn_energy.png"), 3)
    plot_training_sinr(ddqn_train['sinr'], 'ddqn', os.path.join(ddqn_plots, "fig5_ddqn_sinr.png"), 5)
    plot_training_efficiency(ddqn_train['efficiency'], 'ddqn', os.path.join(ddqn_plots, "fig7_ddqn_efficiency.png"), 7)
    
    # Q-Learning training figures -> outputs/qlearn_*/plots/
    qlearn_plots = os.path.join(qlearn_folder, "plots")
    os.makedirs(qlearn_plots, exist_ok=True)
    
    print(f"\n--- Q-Learning Training Figures -> {qlearn_plots}/ ---")
    plot_training_reward(qlearn_train['rewards'], 'qlearn', os.path.join(qlearn_plots, "fig2_qlearn_reward.png"), 2)
    plot_training_energy(qlearn_train['energy'], 'qlearn', os.path.join(qlearn_plots, "fig4_qlearn_energy.png"), 4)
    plot_training_sinr(qlearn_train['sinr'], 'qlearn', os.path.join(qlearn_plots, "fig6_qlearn_sinr.png"), 6)
    plot_training_efficiency(qlearn_train['efficiency'], 'qlearn', os.path.join(qlearn_plots, "fig8_qlearn_efficiency.png"), 8)
    
    # Comparison figures -> figures/
    print(f"\n--- Comparison Figures -> figures/ ---")
    plot_comparison(ddqn_train['energy'], qlearn_train['energy'], 'Energy (J)',
                   'Algorithm Comparison: Energy Consumption', "figures/report_fig9_comparison_energy.png", 9, BASELINE_ENERGY)
    plot_comparison(ddqn_train['sinr'], qlearn_train['sinr'], 'SINR (dB)',
                   'Algorithm Comparison: Signal Quality', "figures/report_fig10_comparison_sinr.png", 10, BASELINE_SINR)
    plot_comparison(ddqn_train['efficiency'], qlearn_train['efficiency'], 'Efficiency (bits/J)',
                   'Algorithm Comparison: Energy Efficiency', "figures/report_fig11_comparison_efficiency.png", 11, BASELINE_EFFICIENCY)
    
    # Test figures
    print(f"\n--- Loading Test Data ---")
    ddqn_test = load_test_data(ddqn_folder, 'ddqn')
    qlearn_test = load_test_data(qlearn_folder, 'qlearn')
    
    if ddqn_test and qlearn_test and 'energies' in ddqn_test:
        print(f"  DDQN: {len(ddqn_test['energies'])} iterations")
        print(f"  Q-Learning: {len(qlearn_test['energies'])} iterations")
        
        print(f"\n--- Test Figures -> figures/ ---")
        plot_test_boxplot(ddqn_test['energies'], qlearn_test['energies'], 'Energy (J)',
                         'Testing: Energy Distribution', "figures/report_fig12_test_energy_boxplot.png", 12, BASELINE_ENERGY)
        plot_test_boxplot(ddqn_test['sinrs'], qlearn_test['sinrs'], 'SINR (dB)',
                         'Testing: SINR Distribution', "figures/report_fig13_test_sinr_boxplot.png", 13, BASELINE_SINR)
        plot_test_boxplot(ddqn_test['efficiencies'], qlearn_test['efficiencies'], 'Efficiency (bits/J)',
                         'Testing: Efficiency Distribution', "figures/report_fig14_test_efficiency_boxplot.png", 14, BASELINE_EFFICIENCY)
        plot_test_histogram(ddqn_test['energies'], qlearn_test['energies'], 'Energy (J)',
                           'Testing: Energy Histogram', "figures/report_fig15_test_energy_histogram.png", 15)
        plot_test_trends(ddqn_test, qlearn_test, "figures/report_fig16_test_iteration_trends.png", 16)
        plot_test_summary(ddqn_test, qlearn_test, "figures/report_fig17_test_summary_bars.png", 17)
    else:
        print("  Warning: Test data not found. Run ddqn_test.py and qlearn_test.py first.")
    
    print("\n" + "=" * 70)
    print("  FIGURE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
