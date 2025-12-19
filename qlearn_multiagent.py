#!/usr/bin/env python3
"""
Q-Learning Multi-Agent for Small Cell Energy Management
Adapted to work with the same environment as DDQN (5 SBS, 6 states, 4 actions)

This uses state discretization to enable tabular Q-Learning on continuous states.
"""

import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
from collections import defaultdict
import os
import pickle
from datetime import datetime

# === Configuration (Match DDQN) ===
N_AGENTS = 5
STATE_DIM_PER_AGENT = 6
ACTION_DIM = 4  # ACTIVE, SM1, SM2, SM3
NUM_UES = 50

# Training parameters
EPISODES = 150
MAX_STEPS = 1000
SIM_TIME = 10

# Q-Learning hyperparameters
LEARNING_RATE = 0.1
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# Output directory
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"qlearn_5sbs_6state_{RUN_TIMESTAMP}"
OUTPUT_DIR = f"outputs/{RUN_NAME}"


class StateDiscretizer:
    """
    Discretizes continuous state values into bins for Q-table lookup.
    """
    
    def __init__(self):
        # Define bin edges for each state factor
        self.ue_bins = [0, 3, 6, 9, 11]           # 4 bins: 0-2, 3-5, 6-8, 9+
        self.mode_bins = [0, 1, 2, 3, 4]          # 4 bins: already discrete
        self.power_bins = [0, 8, 13, 18, 25]      # 4 bins: SM3, SM2, SM1, ACTIVE
        self.sinr_bins = [-50, 5, 10, 15, 50]     # 4 bins: poor, fair, good, excellent
        self.trans_bins = [0, 1, 2]               # 2 bins: not transitioning, transitioning
        self.hour_bins = [0, 4, 6, 9, 17, 22, 25] # 6 bins: night, dawn, morning, work, evening, late
    
    def discretize(self, obs):
        """
        Convert continuous observation to discrete state tuple.
        
        Args:
            obs: numpy array of shape (30,) - 6 values per agent x 5 agents
            
        Returns:
            tuple: Discretized state for Q-table lookup
        """
        discrete_state = []
        
        for agent_idx in range(N_AGENTS):
            base = agent_idx * STATE_DIM_PER_AGENT
            
            # Extract continuous values
            ue_count = obs[base + 0]
            mode = obs[base + 1]
            power = obs[base + 2]
            sinr = obs[base + 3]
            transitioning = obs[base + 4]
            hour = obs[base + 5]
            
            # Discretize each value
            ue_bin = np.digitize(ue_count, self.ue_bins) - 1
            mode_bin = int(mode)  # Already discrete 0-3
            power_bin = np.digitize(power, self.power_bins) - 1
            sinr_bin = np.digitize(sinr, self.sinr_bins) - 1
            trans_bin = int(transitioning)  # Already 0 or 1
            hour_bin = np.digitize(hour, self.hour_bins) - 1
            
            # Clamp to valid ranges
            ue_bin = max(0, min(3, ue_bin))
            mode_bin = max(0, min(3, mode_bin))
            power_bin = max(0, min(3, power_bin))
            sinr_bin = max(0, min(3, sinr_bin))
            trans_bin = max(0, min(1, trans_bin))
            hour_bin = max(0, min(5, hour_bin))
            
            discrete_state.extend([ue_bin, mode_bin, power_bin, sinr_bin, trans_bin, hour_bin])
        
        return tuple(discrete_state)


class QLearningAgent:
    """
    Single Q-Learning agent for one SBS.
    """
    
    def __init__(self, agent_id, state_bins, action_dim, lr=0.1, gamma=0.95):
        self.agent_id = agent_id
        self.state_bins = state_bins  # Number of bins per state factor
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = EPSILON_START
        
        # Q-table as defaultdict (handles unseen states)
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        
        # Statistics
        self.update_count = 0
    
    def get_state_key(self, discrete_obs, agent_idx):
        base = agent_idx * 6  
        return tuple(discrete_obs[base:base + 6])
    
    def select_action(self, state_key, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            q_values = self.q_table[state_key]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)
    
    def update(self, state_key, action, reward, next_state_key, done):
        current_q = self.q_table[state_key][action]
        
        if done:
            target = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * next_max_q
        
        # Q-Learning update
        self.q_table[state_key][action] += self.lr * (target - current_q)
        self.update_count += 1
    
    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
    
    def get_q_table_size(self):
        return len(self.q_table)


class MultiAgentQLearning:
    
    def __init__(self):
        self.discretizer = StateDiscretizer()
        self.agents = [
            QLearningAgent(i, state_bins=6, action_dim=ACTION_DIM, lr=LEARNING_RATE, gamma=GAMMA)
            for i in range(N_AGENTS)
        ]
    
    def discretize_obs(self, obs):
        return self.discretizer.discretize(obs)
    
    def select_actions(self, discrete_obs, greedy=False):
        actions = []
        for i, agent in enumerate(self.agents):
            state_key = agent.get_state_key(discrete_obs, i)
            action = agent.select_action(state_key, greedy=greedy)
            actions.append(action)
        return actions
    
    def update_all(self, discrete_obs, actions, reward, discrete_next_obs, done):
        for i, agent in enumerate(self.agents):
            state_key = agent.get_state_key(discrete_obs, i)
            next_state_key = agent.get_state_key(discrete_next_obs, i)
            agent.update(state_key, actions[i], reward, next_state_key, done)
    
    def decay_epsilon_all(self):
        for agent in self.agents:
            agent.decay_epsilon()
    
    def get_epsilon(self):
        return self.agents[0].epsilon
    
    def get_total_q_table_size(self):
        return sum(agent.get_q_table_size() for agent in self.agents)
    
    def save(self, filepath):
        q_tables = {f"agent_{i}": dict(agent.q_table) for i, agent in enumerate(self.agents)}
        with open(filepath, 'wb') as f:
            pickle.dump(q_tables, f)
        print(f"Saved Q-tables to {filepath}")
    
    def load(self, filepath):
        with open(filepath, 'rb') as f:
            q_tables = pickle.load(f)
        for i, agent in enumerate(self.agents):
            agent.q_table = defaultdict(lambda: np.zeros(ACTION_DIM), q_tables[f"agent_{i}"])
        print(f"Loaded Q-tables from {filepath}")


def parse_info(info):
    if isinstance(info, (list, tuple)):
        info = info[0] if info else ""
    info_dict = {}
    if info:
        for item in info.split(";"):
            if "=" in item:
                key, value = item.split("=")
                info_dict[key] = float(value) if '.' in value else value
    return info_dict


def train():
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    
    print("=" * 60)
    print("Q-LEARNING MULTI-AGENT TRAINING")
    print("=" * 60)
    print(f"Agents: {N_AGENTS}")
    print(f"Episodes: {EPISODES}")
    print(f"Steps per Episode: {MAX_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Gamma: {GAMMA}")
    print(f"Epsilon: {EPSILON_START} → {EPSILON_END}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Initialize multi-agent Q-Learning
    maql = MultiAgentQLearning()
    
    # Metrics storage
    episode_rewards = []
    episode_energies = []
    episode_sinrs = []
    episode_efficiencies = []
    episode_epsilons = []
    episode_q_table_sizes = []
    
    # Energy efficiency calculation
    packets_per_ue = SIM_TIME / 0.05  # 50ms packet interval
    total_packets = packets_per_ue * NUM_UES
    total_bits = total_packets * 1024 * 8  # 1024 bytes per packet
    
    for episode in range(1, EPISODES + 1):
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{EPISODES}")
        print(f"{'='*60}")
        
        # Create environment
        env = ns3env.Ns3Env(port=5555, stepTime=0.01, startSim=True, simSeed=episode)
        obs = env.reset()
        
        if obs is None:
            print(f"[ERROR] Failed to get initial observation")
            env.close()
            continue
        
        # Discretize initial observation
        discrete_obs = maql.discretize_obs(obs)
        
        total_reward = 0.0
        total_energy = 0.0
        total_sinr = 0.0
        step_count = 0
        
        for step in range(MAX_STEPS):
            # Select actions
            actions = maql.select_actions(discrete_obs)
            
            # Execute actions
            next_obs, reward, done, info = env.step(np.array(actions, dtype=np.float32))
            
            if next_obs is None:
                break
            
            # Discretize next observation
            discrete_next_obs = maql.discretize_obs(next_obs)
            
            # Update all agents
            maql.update_all(discrete_obs, actions, reward, discrete_next_obs, done)
            
            # Parse info
            info_dict = parse_info(info)
            energy = float(info_dict.get('total_energy', 0))
            sinr = float(info_dict.get('global_sinr', 0))
            
            total_reward += reward
            total_energy = energy  # Cumulative from simulation
            total_sinr += sinr
            step_count += 1
            
            # Log every 100 steps
            if step % 100 == 0:
                print(f"  Step {step}: Reward={reward:.3f}, Actions={actions}, "
                      f"Energy={energy:.2f}J, SINR={sinr:.2f}dB")
            
            discrete_obs = discrete_next_obs
            
            if done:
                break
        
        # End of episode
        env.close()
        
        # Decay epsilon
        maql.decay_epsilon_all()
        
        # Calculate metrics
        avg_sinr = total_sinr / max(step_count, 1)
        efficiency = total_bits / max(total_energy, 1)
        q_table_size = maql.get_total_q_table_size()
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_sinrs.append(avg_sinr)
        episode_efficiencies.append(efficiency)
        episode_epsilons.append(maql.get_epsilon())
        episode_q_table_sizes.append(q_table_size)
        
        print(f"\n[Episode {episode} Summary]")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Total Energy: {total_energy:.2f} J")
        print(f"  Avg SINR: {avg_sinr:.2f} dB")
        print(f"  Efficiency: {efficiency:.0f} bits/J")
        print(f"  Epsilon: {maql.get_epsilon():.4f}")
        print(f"  Q-Table Size: {q_table_size} entries")
        
        # Save checkpoint every 25 episodes
        if episode % 25 == 0:
            maql.save(f"{OUTPUT_DIR}/models/qlearn_checkpoint_ep{episode}.pkl")
    
    # Save final model
    maql.save(f"{OUTPUT_DIR}/models/qlearn_final.pkl")
    
    # Save metrics
    np.save(f"{OUTPUT_DIR}/data/episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{OUTPUT_DIR}/data/episode_energies.npy", np.array(episode_energies))
    np.save(f"{OUTPUT_DIR}/data/episode_sinrs.npy", np.array(episode_sinrs))
    np.save(f"{OUTPUT_DIR}/data/episode_efficiencies.npy", np.array(episode_efficiencies))
    np.save(f"{OUTPUT_DIR}/data/episode_epsilons.npy", np.array(episode_epsilons))
    np.save(f"{OUTPUT_DIR}/data/episode_q_table_sizes.npy", np.array(episode_q_table_sizes))
    
    # Generate plots
    generate_plots(episode_rewards, episode_energies, episode_sinrs, 
                   episode_efficiencies, episode_epsilons, episode_q_table_sizes)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Reward: {episode_rewards[-1]:.2f}")
    print(f"Final Energy: {episode_energies[-1]:.2f} J")
    print(f"Final SINR: {episode_sinrs[-1]:.2f} dB")
    print(f"Final Efficiency: {episode_efficiencies[-1]:.0f} bits/J")
    print(f"Q-Table Size: {episode_q_table_sizes[-1]} entries")
    print(f"Output saved to: {OUTPUT_DIR}")
    
    return maql


def generate_plots(rewards, energies, sinrs, efficiencies, epsilons, q_sizes):
    """Generate training plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Q-Learning Multi-Agent Training (5 SBS, 6 States)', fontsize=14)
    
    episodes = range(1, len(rewards) + 1)
    
    # Reward
    axes[0, 0].plot(episodes, rewards, 'b-', alpha=0.7)
    axes[0, 0].plot(episodes, np.convolve(rewards, np.ones(10)/10, mode='valid'), 'b-', linewidth=2)
    axes[0, 0].set_title('Total Reward per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Energy
    axes[0, 1].plot(episodes, energies, 'r-', alpha=0.7)
    axes[0, 1].plot(episodes, np.convolve(energies, np.ones(10)/10, mode='valid'), 'r-', linewidth=2)
    axes[0, 1].set_title('Total Energy per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Energy (J)')
    axes[0, 1].grid(True)
    
    # SINR
    axes[0, 2].plot(episodes, sinrs, 'g-', alpha=0.7)
    axes[0, 2].plot(episodes, np.convolve(sinrs, np.ones(10)/10, mode='valid'), 'g-', linewidth=2)
    axes[0, 2].axhline(y=10, color='orange', linestyle='--', label='Good threshold')
    axes[0, 2].set_title('Average SINR per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('SINR (dB)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Efficiency
    axes[1, 0].plot(episodes, efficiencies, 'm-', alpha=0.7)
    axes[1, 0].plot(episodes, np.convolve(efficiencies, np.ones(10)/10, mode='valid'), 'm-', linewidth=2)
    axes[1, 0].set_title('Energy Efficiency per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Efficiency (bits/J)')
    axes[1, 0].grid(True)
    
    # Epsilon
    axes[1, 1].plot(episodes, epsilons, 'c-', linewidth=2)
    axes[1, 1].set_title('Exploration Rate (Epsilon)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True)
    
    # Q-Table Size
    axes[1, 2].plot(episodes, q_sizes, 'orange', linewidth=2)
    axes[1, 2].set_title('Q-Table Size (Total Entries)')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Number of States')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/qlearn_training.png", dpi=150)
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}/plots/")


if __name__ == "__main__":
    train()
