#!/usr/bin/env python3
"""
PPO Multi-Agent for Small Cell Energy Management
Proximal Policy Optimization with independent agents for 5 SBS

Architecture:
- Actor-Critic with separate networks
- GAE (Generalized Advantage Estimation)
- Clipped PPO objective
- Independent agents (one per SBS)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from ns3gym import ns3env
import os
from datetime import datetime
from collections import deque

# === Configuration ===
N_AGENTS = 5
STATE_DIM_PER_AGENT = 6
ACTION_DIM = 4  # ACTIVE, SM1, SM2, SM3
NUM_UES = 50

# Training parameters
EPISODES = 150
MAX_STEPS = 1000
SIM_TIME = 10

# PPO Hyperparameters
GAMMA = 0.99              # Discount factor
GAE_LAMBDA = 0.95         # GAE parameter
CLIP_RATIO = 0.2          # PPO clip ratio
ACTOR_LR = 3e-4           # Actor learning rate
CRITIC_LR = 1e-3          # Critic learning rate
EPOCHS_PER_UPDATE = 10    # K epochs for PPO update
BATCH_SIZE = 64           # Mini-batch size
BUFFER_SIZE = 2048        # Experience buffer size
ENTROPY_COEF = 0.01       # Entropy bonus coefficient

PACKET_INTERVAL   = 0.02778
PACKET_SIZE_BYTES = 1400

# Output directory — rooted at this file's location (capstone root, not additions/)
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME      = f"ppo_curr_5sbs_6state_{RUN_TIMESTAMP}"
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "outputs", RUN_NAME)


class Actor(keras.Model):
    """Policy network - outputs action probabilities"""
    
    def __init__(self, state_dim, action_dim, name='actor'):
        super(Actor, self).__init__(name=name)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(64, activation='relu')
        self.logits = layers.Dense(action_dim)  # No activation - raw logits
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.logits(x)  # Return logits for numerical stability


class Critic(keras.Model):
    """Value network - estimates state value"""
    
    def __init__(self, state_dim, name='critic'):
        super(Critic, self).__init__(name=name)
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc3 = layers.Dense(64, activation='relu')
        self.value = layers.Dense(1)
    
    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.value(x)


class PPOAgent:
    """Single PPO agent for one SBS"""
    
    def __init__(self, agent_id, state_dim, action_dim):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Networks
        self.actor = Actor(state_dim, action_dim, name=f'actor_{agent_id}')
        self.critic = Critic(state_dim, name=f'critic_{agent_id}')
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=ACTOR_LR)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=CRITIC_LR)
        
        # Experience buffer for this agent
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
        
        # Build models
        dummy_state = tf.zeros((1, state_dim))
        self.actor(dummy_state)
        self.critic(dummy_state)
    
    def get_state(self, full_obs, agent_idx):
        """Extract this agent's state from full observation"""
        base = agent_idx * STATE_DIM_PER_AGENT
        return full_obs[base:base + STATE_DIM_PER_AGENT]
    
    def select_action(self, state, training=True, episode_num=None):
        """Select action using policy network with optional curriculum masking"""
        state_t = tf.convert_to_tensor([state], dtype=tf.float32)

        # Curriculum learning: restrict available actions by episode
        if episode_num is not None:
            if episode_num <= 5:
                valid_actions = [0]
            elif episode_num <= 10:
                valid_actions = [0, 1]
            elif episode_num <= 15:
                valid_actions = [0, 1, 2]
            else:
                valid_actions = [0, 1, 2, 3]
        else:
            valid_actions = [0, 1, 2, 3]

        logits = self.actor(state_t).numpy()[0]
        mask = np.full(self.action_dim, -1e9)
        for a in valid_actions:
            mask[a] = 0.0
        masked_logits = tf.convert_to_tensor([logits + mask], dtype=tf.float32)

        action_probs = tf.nn.softmax(masked_logits)

        if training:
            action = int(tf.random.categorical(masked_logits, 1)[0, 0].numpy())
        else:
            action = int(tf.argmax(action_probs, axis=1)[0].numpy())

        value = self.critic(state_t)[0, 0].numpy()
        log_prob = float(tf.nn.log_softmax(masked_logits)[0, action].numpy())

        return action, value, log_prob
    
    def store_transition(self, state, action, reward, value, log_prob):
        """Store experience in buffer"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
    
    def compute_gae(self, next_value, done):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.buffer['rewards'])
        values = np.array(self.buffer['values'])
        
        # Add next value for bootstrapping
        values_extended = np.append(values, next_value)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value if not done else 0
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + GAMMA * next_value_t - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * gae
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns
    
    def update(self, next_value, done):
        """Update policy and value networks using PPO"""
        if len(self.buffer['states']) < BATCH_SIZE:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value, done)
        
        # Convert to tensors
        states = tf.convert_to_tensor(self.buffer['states'], dtype=tf.float32)
        actions = tf.convert_to_tensor(self.buffer['actions'], dtype=tf.int32)
        old_log_probs = tf.convert_to_tensor(self.buffer['log_probs'], dtype=tf.float32)
        advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Multiple epochs of optimization
        batch_size = min(BATCH_SIZE, len(states))
        indices = np.arange(len(states))
        
        actor_losses = []
        critic_losses = []
        entropies = []
        
        for epoch in range(EPOCHS_PER_UPDATE):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                batch_states = tf.gather(states, batch_idx)
                batch_actions = tf.gather(actions, batch_idx)
                batch_old_log_probs = tf.gather(old_log_probs, batch_idx)
                batch_advantages = tf.gather(advantages_tf, batch_idx)
                batch_returns = tf.gather(returns_tf, batch_idx)
                
                # Update actor
                with tf.GradientTape() as tape:
                    logits = self.actor(batch_states)
                    log_probs = tf.nn.log_softmax(logits)
                    
                    # Get log prob of taken actions
                    action_masks = tf.one_hot(batch_actions, self.action_dim)
                    new_log_probs = tf.reduce_sum(log_probs * action_masks, axis=1)
                    
                    # Probability ratio
                    ratio = tf.exp(new_log_probs - batch_old_log_probs)
                    
                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - CLIP_RATIO, 1 + CLIP_RATIO) * batch_advantages
                    
                    # Entropy bonus for exploration
                    probs = tf.nn.softmax(logits)
                    entropy = -tf.reduce_sum(probs * log_probs, axis=1)
                    entropy_bonus = ENTROPY_COEF * tf.reduce_mean(entropy)
                    
                    # Actor loss (negative because we want to maximize)
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2)) - entropy_bonus
                
                actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                
                # Update critic
                with tf.GradientTape() as tape:
                    values = self.critic(batch_states)
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - tf.squeeze(values)))
                
                critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                entropies.append(tf.reduce_mean(entropy).numpy())
        
        # Clear buffer
        self.clear_buffer()
        
        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses),
            'entropy': np.mean(entropies)
        }
    
    def clear_buffer(self):
        """Clear experience buffer"""
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }


class MultiAgentPPO:
    """Multi-agent PPO coordinator"""
    
    def __init__(self):
        self.agents = [
            PPOAgent(i, STATE_DIM_PER_AGENT, ACTION_DIM)
            for i in range(N_AGENTS)
        ]
    
    def select_actions(self, obs, training=True, episode_num=None):
        """Select actions for all agents"""
        actions = []
        values = []
        log_probs = []

        for i, agent in enumerate(self.agents):
            state = agent.get_state(obs, i)
            action, value, log_prob = agent.select_action(state, training, episode_num=episode_num)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)

        return actions, values, log_probs
    
    def store_transitions(self, obs, actions, reward, values, log_probs):
        """Store transitions for all agents"""
        for i, agent in enumerate(self.agents):
            state = agent.get_state(obs, i)
            agent.store_transition(state, actions[i], reward, values[i], log_probs[i])
    
    def update_all(self, next_obs, done):
        """Update all agents"""
        losses = {'actor': [], 'critic': [], 'entropy': []}
        
        for i, agent in enumerate(self.agents):
            next_state = agent.get_state(next_obs, i)
            next_value = agent.critic(tf.convert_to_tensor([next_state], dtype=tf.float32))[0, 0].numpy()
            
            loss_dict = agent.update(next_value, done)
            losses['actor'].append(loss_dict['actor_loss'])
            losses['critic'].append(loss_dict['critic_loss'])
            losses['entropy'].append(loss_dict['entropy'])
        
        return {
            'actor_loss': np.mean(losses['actor']),
            'critic_loss': np.mean(losses['critic']),
            'entropy': np.mean(losses['entropy'])
        }
    
    def save(self, filepath):
        """Save all agent models"""
        for i, agent in enumerate(self.agents):
            agent.actor.save_weights(f"{filepath}_actor_{i}.h5")
            agent.critic.save_weights(f"{filepath}_critic_{i}.h5")
        print(f"Saved PPO models to {filepath}")
    
    def load(self, filepath):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.actor.load_weights(f"{filepath}_actor_{i}.h5")
            agent.critic.load_weights(f"{filepath}_critic_{i}.h5")
        print(f"Loaded PPO models from {filepath}")


def parse_info(info):
    """Parse info string from environment"""
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
    """Main training loop"""
    os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)
    
    print("=" * 60)
    print("PPO MULTI-AGENT TRAINING")
    print("=" * 60)
    print(f"Agents: {N_AGENTS}")
    print(f"Episodes: {EPISODES}")
    print(f"Steps per Episode: {MAX_STEPS}")
    print(f"Clip Ratio: {CLIP_RATIO}")
    print(f"GAE Lambda: {GAE_LAMBDA}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Initialize multi-agent PPO
    mappo = MultiAgentPPO()
    
    # Metrics storage
    episode_rewards = []
    episode_energies = []
    episode_sinrs = []
    episode_efficiencies = []
    episode_actor_losses = []
    episode_critic_losses = []
    
    # Energy efficiency calculation
    packets_per_ue = SIM_TIME / 0.05
    total_packets = packets_per_ue * NUM_UES
    total_bits = total_packets * 1024 * 8
    
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
        
        total_reward = 0.0
        total_energy = 0.0
        total_sinr = 0.0
        step_count = 0
        
        for step in range(MAX_STEPS):
            # Select actions
            actions, values, log_probs = mappo.select_actions(obs, training=True, episode_num=episode)
            
            # Execute actions
            next_obs, reward, done, info = env.step(np.array(actions, dtype=np.float32))
            
            if next_obs is None:
                break
            
            # Store transitions
            mappo.store_transitions(obs, actions, reward, values, log_probs)
            
            # Parse info
            info_dict = parse_info(info)
            energy = float(info_dict.get('total_energy', 0))
            sinr = float(info_dict.get('global_sinr', 0))
            
            total_reward += reward
            total_energy = energy
            total_sinr += sinr
            step_count += 1
            
            # Log every 100 steps
            if step % 100 == 0:
                print(f"  Step {step}: Reward={reward:.3f}, Actions={actions}, "
                      f"Energy={energy:.2f}J, SINR={sinr:.2f}dB")
            
            obs = next_obs
            
            if done:
                break
        
        # End of episode - update networks
        losses = mappo.update_all(obs, done)
        env.close()
        
        # Calculate metrics
        avg_sinr = total_sinr / max(step_count, 1)
        efficiency = total_bits / max(total_energy, 1)
        
        # Store metrics
        episode_rewards.append(total_reward)
        episode_energies.append(total_energy)
        episode_sinrs.append(avg_sinr)
        episode_efficiencies.append(efficiency)
        episode_actor_losses.append(losses['actor_loss'])
        episode_critic_losses.append(losses['critic_loss'])
        
        print(f"\n[Episode {episode} Summary]")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Total Energy: {total_energy:.2f} J")
        print(f"  Avg SINR: {avg_sinr:.2f} dB")
        print(f"  Efficiency: {efficiency:.0f} bits/J")
        print(f"  Actor Loss: {losses['actor_loss']:.4f}")
        print(f"  Critic Loss: {losses['critic_loss']:.4f}")
        print(f"  Entropy: {losses['entropy']:.4f}")
        
        # Save checkpoint every 25 episodes
        if episode % 25 == 0:
            mappo.save(f"{OUTPUT_DIR}/models/ppo_checkpoint_ep{episode}")
    
    # Save final model
    mappo.save(f"{OUTPUT_DIR}/models/ppo_final")
    
    # Save metrics
    np.save(f"{OUTPUT_DIR}/data/episode_rewards.npy", np.array(episode_rewards))
    np.save(f"{OUTPUT_DIR}/data/episode_energies.npy", np.array(episode_energies))
    np.save(f"{OUTPUT_DIR}/data/episode_sinrs.npy", np.array(episode_sinrs))
    np.save(f"{OUTPUT_DIR}/data/episode_efficiencies.npy", np.array(episode_efficiencies))
    np.save(f"{OUTPUT_DIR}/data/episode_actor_losses.npy", np.array(episode_actor_losses))
    np.save(f"{OUTPUT_DIR}/data/episode_critic_losses.npy", np.array(episode_critic_losses))
    
    # Generate individual training plots
    generate_plots(episode_rewards, episode_energies, episode_sinrs,
                   episode_efficiencies, episode_actor_losses, episode_critic_losses)

    # =========================================================================
    # BASELINE SIMULATION — mirrors the pattern used by DDQN / Q-Learning
    # =========================================================================
    print("\n[INFO] Running baseline (always-active) simulation...")
    os.environ["NS3_BASELINE"] = "1"
    baseline_energies, baseline_sinrs, baseline_powers = [], [], []

    for ep in range(1, EPISODES + 1):
        b_env = ns3env.Ns3Env(port=5555, stepTime=0.01, startSim=True, simSeed=ep)
        b_env.reset()
        done, step_count = False, 0
        b_energy = b_power = sinr_sum = sinr_steps = 0.0
        while not done and step_count < MAX_STEPS:
            _, _, done, info = b_env.step(np.array([0]*N_AGENTS, dtype=np.float32))
            d = parse_info(info)
            b_energy  = float(d.get('total_energy', 0.0))
            b_power   = float(d.get('total_power',  0.0))
            sinr_sum += float(d.get('global_sinr',  0.0))
            sinr_steps += 1; step_count += 1
        b_env.close()
        baseline_energies.append(b_energy)
        baseline_sinrs.append(sinr_sum / sinr_steps if sinr_steps else 0.0)
        baseline_powers.append(b_power)

    os.environ["NS3_BASELINE"] = "0"

    packets_per_ue = SIM_TIME / PACKET_INTERVAL
    total_bits     = packets_per_ue * NUM_UES * PACKET_SIZE_BYTES * 8
    ee_baseline    = [total_bits / e if e > 0 else 0 for e in baseline_energies]

    np.save(f"{OUTPUT_DIR}/data/baseline_energy_per_episode.npy", np.array(baseline_energies))
    np.save(f"{OUTPUT_DIR}/data/baseline_sinr_per_episode.npy",   np.array(baseline_sinrs))
    np.save(f"{OUTPUT_DIR}/data/baseline_power_per_episode.npy",  np.array(baseline_powers))
    np.save(f"{OUTPUT_DIR}/data/energy_efficiency_baseline.npy",  np.array(ee_baseline))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Reward: {episode_rewards[-1]:.2f}")
    print(f"Output saved to: {OUTPUT_DIR}")

    return mappo


def generate_plots(rewards, energies, sinrs, efficiencies, actor_losses, critic_losses):
    """Generate training plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PPO Multi-Agent Training (5 SBS, 6 States)', fontsize=14)
    
    episodes = np.arange(1, len(rewards) + 1)
    window = 10
    
    def plot_metric(ax, data, title, ylabel, color, goal=None):
        ax.plot(episodes, data, color=color, alpha=0.3, label='Raw')
        
        if len(data) >= window:
            ma = np.convolve(data, np.ones(window)/window, mode='valid')
            ma_episodes = episodes[window-1:]
            ax.plot(ma_episodes, ma, color=color, linewidth=2, label=f'Avg ({window})')
        
        if goal:
            ax.axhline(y=goal, color='orange', linestyle='--', label='Goal')
        
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=8)
    
    plot_metric(axes[0, 0], rewards, 'Total Reward', 'Reward', 'b')
    plot_metric(axes[0, 1], energies, 'Total Energy', 'Energy (J)', 'r')
    plot_metric(axes[0, 2], sinrs, 'Avg SINR', 'SINR (dB)', 'g', goal=10)
    plot_metric(axes[1, 0], efficiencies, 'Efficiency', 'bits/J', 'm')
    plot_metric(axes[1, 1], actor_losses, 'Actor Loss', 'Loss', 'orange')
    plot_metric(axes[1, 2], critic_losses, 'Critic Loss', 'Loss', 'purple')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/ppo_training.png", dpi=150)
    plt.close()
    
    print(f"Plots saved to {OUTPUT_DIR}/plots/")


if __name__ == "__main__":
    train()
