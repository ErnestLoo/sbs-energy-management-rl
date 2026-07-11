#!/usr/bin/env python3
"""
Unified test harness — DDQN vs PPO vs Baseline, real ns-3 evaluation.

For each of the 50 fixed seeds (research_config.TEST_SEEDS) it evaluates, on the
IDENTICAL network per seed:

    DDQN  under G1 (graduated) and G2 (on/off)
    PPO   under G1 (graduated) and G2 (on/off)
    Baseline (always ACTIVE)              — one series, seed-matched

Energy and SINR are read from the ns-3 info string (real simulation, NOT the old
Python approximation). G2 is produced by action-masking the G1-trained agents at
test time (no retraining), so the curriculum stays identical across groups.

Greedy policy only (no exploration, no learning).

Output:  outputs/test_<timestamp>/testing_metrics.csv   (schema: research_config.TESTING_CSV_COLUMNS)

Usage:
    python run_tests.py                       # auto-detect latest model folders
    python run_tests.py --ddqn <folder> --ppo <folder>
    python run_tests.py --iterations 50
"""

import os
import sys
import csv
import glob
import argparse
import numpy as np
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from ns3gym import ns3env

import research_config as cfg
from ppo_multiagent import Actor   # reuse the exact PPO policy network


# =============================================================================
# POLICY WRAPPERS (greedy, action-masked)
# =============================================================================
def _mask_actions(state, allowed):
    """Intersect the group's allowed actions with the transition constraint."""
    is_transitioning = state[cfg.IDX_TRANSITIONING]
    if is_transitioning:
        return [0]
    return allowed


class DDQNPolicy:
    """Loads the 5 trained DDQN keras models; greedy masked action selection."""

    def __init__(self, models_dir):
        self.models = []
        for i in range(cfg.N_AGENTS):
            path = os.path.join(models_dir, f"trained_ddqn_agent_{i}_model.h5")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing DDQN model: {path}")
            self.models.append(keras.models.load_model(path, compile=False))

    def act(self, obs, allowed):
        actions = []
        for i, model in enumerate(self.models):
            s = obs[i * cfg.STATE_DIM_PER_AGENT:(i + 1) * cfg.STATE_DIM_PER_AGENT]
            valid = _mask_actions(s, allowed)
            q = model.predict(s[np.newaxis], verbose=0)[0]
            masked = np.full(cfg.ACTION_DIM, -np.inf)
            for a in valid:
                masked[a] = q[a]
            actions.append(int(np.argmax(masked)))
        return actions


class PPOPolicy:
    """Loads the 5 trained PPO actor networks; greedy masked action selection."""

    def __init__(self, models_dir):
        self.actors = []
        base = self._find_weight_base(models_dir)
        for i in range(cfg.N_AGENTS):
            actor = Actor(cfg.STATE_DIM_PER_AGENT, cfg.ACTION_DIM, name=f"actor_{i}")
            actor(tf.zeros((1, cfg.STATE_DIM_PER_AGENT)))   # build
            actor.load_weights(f"{base}_actor_{i}.weights.h5")
            self.actors.append(actor)

    @staticmethod
    def _find_weight_base(models_dir):
        # prefer ppo_final, else the latest ppo_checkpoint_ep*
        final = os.path.join(models_dir, "ppo_final")
        if os.path.exists(final + "_actor_0.weights.h5"):
            return final
        ckpts = sorted(glob.glob(os.path.join(models_dir, "ppo_checkpoint_ep*_actor_0.weights.h5")))
        if ckpts:
            return ckpts[-1].replace("_actor_0.weights.h5", "")
        raise FileNotFoundError(f"No PPO actor weights found in {models_dir}")

    def act(self, obs, allowed):
        actions = []
        for i, actor in enumerate(self.actors):
            s = obs[i * cfg.STATE_DIM_PER_AGENT:(i + 1) * cfg.STATE_DIM_PER_AGENT]
            valid = _mask_actions(s, allowed)
            logits = actor(tf.convert_to_tensor([s], dtype=tf.float32)).numpy()[0]
            masked = np.full(cfg.ACTION_DIM, -np.inf)
            for a in valid:
                masked[a] = logits[a]
            actions.append(int(np.argmax(masked)))
        return actions


# =============================================================================
# SINGLE-ITERATION ROLLOUT (real ns-3)
# =============================================================================
def run_iteration(policy, allowed, seed, baseline=False):
    """
    Run one episode on ns-3 with a fixed seed. Returns (energy, sinr, efficiency)
    read from the real simulation info string.
    """
    env = ns3env.Ns3Env(port=cfg.NS3_PORT, stepTime=cfg.STEP_TIME,
                         startSim=True, simSeed=seed)
    obs = env.reset()

    total_energy = 0.0
    sinr_sum, sinr_steps = 0.0, 0
    done, step = False, 0

    while not done and step < cfg.MAX_STEPS:
        if baseline:
            actions = [0] * cfg.N_AGENTS
        else:
            actions = policy.act(np.asarray(obs, dtype=np.float32), allowed)

        obs, _, done, info = env.step(np.array(actions, dtype=np.uint32))
        if obs is None:
            break

        d = cfg.parse_ns3_info(info)
        total_energy = float(d.get("total_energy", total_energy))
        sinr_sum += float(d.get("global_sinr", 0.0))
        sinr_steps += 1
        step += 1

    env.close()
    avg_sinr = sinr_sum / sinr_steps if sinr_steps else 0.0
    return total_energy, avg_sinr, cfg.energy_efficiency(total_energy)


# =============================================================================
# MAIN
# =============================================================================
def _latest(pattern):
    hits = sorted(glob.glob(os.path.join(cfg.OUTPUTS_DIR, pattern)))
    return hits[-1] if hits else None


def main():
    ap = argparse.ArgumentParser(description="Unified DDQN/PPO/Baseline test harness")
    ap.add_argument("--ddqn", help="DDQN output folder (contains models/)")
    ap.add_argument("--ppo",  help="PPO output folder (contains models/)")
    ap.add_argument("--iterations", type=int, default=cfg.N_TEST_ITERATIONS)
    args = ap.parse_args()

    ddqn_folder = args.ddqn or _latest("ddqn_5sbs_*")
    ppo_folder  = args.ppo  or _latest("ppo_curr_5sbs_*") or _latest("ppo_5sbs_*")

    if not ddqn_folder or not ppo_folder:
        sys.exit(f"ERROR: could not locate model folders "
                 f"(ddqn={ddqn_folder}, ppo={ppo_folder}). Pass --ddqn/--ppo.")

    seeds = cfg.TEST_SEEDS[:args.iterations]
    print(f"DDQN models : {ddqn_folder}")
    print(f"PPO  models : {ppo_folder}")
    print(f"Iterations  : {len(seeds)} seeds {seeds[0]}..{seeds[-1]}")

    print("Loading policies...")
    ddqn = DDQNPolicy(os.path.join(ddqn_folder, "models"))
    ppo  = PPOPolicy(os.path.join(ppo_folder, "models"))

    # what to run: (algorithm, sleep_group, policy_or_None)
    plan = [
        ("ddqn",     "G1", ddqn),
        ("ddqn",     "G2", ddqn),
        ("ppo",      "G1", ppo),
        ("ppo",      "G2", ppo),
        ("baseline", "-",  None),
    ]

    out_dir = os.path.join(cfg.OUTPUTS_DIR,
                           "test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "testing_metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cfg.TESTING_CSV_COLUMNS)

        for algo, group, policy in plan:
            allowed = cfg.SLEEP_GROUPS.get(group, [0])
            is_baseline = policy is None
            print(f"\n=== {algo.upper()} [{group}] ===")
            for it, seed in enumerate(seeds):
                energy, sinr, eff = run_iteration(
                    policy, allowed, seed, baseline=is_baseline)
                writer.writerow([it, algo, group, seed,
                                 f"{energy:.4f}", f"{sinr:.4f}", f"{eff:.2f}"])
                f.flush()
                if (it + 1) % 10 == 0:
                    print(f"  iter {it+1}/{len(seeds)}: "
                          f"E={energy:.1f}J SINR={sinr:.2f}dB EE={eff:.0f}")

    print(f"\nDone. Wrote {csv_path}")
    print("Now run:  python compare_algorithms.py")


if __name__ == "__main__":
    main()
