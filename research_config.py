#!/usr/bin/env python3
"""
Shared research configuration for the DDQN vs PPO energy-management study.

Single source of truth for:
  - environment dimensions (agents / state / actions)
  - power model (matches smallCellEnergyModel.h)
  - the canonical energy-efficiency formula (same constant for every algorithm,
    so cross-algorithm comparison is fair regardless of the exact bit count)
  - the fixed 50-seed test set (identical seeds for DDQN / PPO / baseline)
  - sleep-mode groups G1 (graduated) and G2 (on/off)
  - SINR quality thresholds for plotting

Import this everywhere instead of re-declaring constants per script.
"""

import os

# =============================================================================
# ENVIRONMENT
# =============================================================================
N_AGENTS            = 5
STATE_DIM_PER_AGENT = 6
ACTION_DIM          = 4          # 0=ACTIVE, 1=SM1, 2=SM2, 3=SM3
NUM_UES             = 50
SIM_TIME            = 10.0       # seconds per episode/iteration
MAX_STEPS           = 1000
STEP_TIME           = 0.01       # ns3-gym decision interval (s)
NS3_PORT            = 5555

# state vector indices (per agent)
IDX_UE_COUNT        = 0
IDX_CURRENT_MODE    = 1
IDX_POWER           = 2
IDX_SINR            = 3
IDX_TRANSITIONING   = 4
IDX_HOUR            = 5

# =============================================================================
# POWER MODEL  (Watts) — must match smallCellEnergyModel.h
# =============================================================================
POWER_ACTIVE = 20.7
POWER_SM1    = 15.0
POWER_SM2    = 10.0
POWER_SM3    = 3.36
POWER_MAP    = {0: POWER_ACTIVE, 1: POWER_SM1, 2: POWER_SM2, 3: POWER_SM3}

# =============================================================================
# ENERGY-EFFICIENCY FORMULA (canonical, shared)
# EE = TOTAL_BITS / energy_consumed   (bits / Joule)
# Uses the ns-3 traffic convention (36 pkt/s, 1400-byte packets).
# The constant is identical across algorithms, so it never biases the
# comparison — it only sets the absolute scale of the y-axis.
# =============================================================================
PACKET_INTERVAL   = 0.02778      # ~36 packets / second
PACKET_SIZE_BYTES = 1400
TOTAL_BITS        = (SIM_TIME / PACKET_INTERVAL) * NUM_UES * PACKET_SIZE_BYTES * 8


def energy_efficiency(energy_joules):
    """bits/J for a given episode/iteration energy. Safe against divide-by-zero."""
    return TOTAL_BITS / energy_joules if energy_joules and energy_joules > 0 else 0.0


# =============================================================================
# SLEEP-MODE GROUPS (testing)
#   G1 graduated : all four modes available
#   G2 on/off    : only ACTIVE (0) or deep sleep SM3 (3)
# =============================================================================
SLEEP_GROUPS = {
    "G1": [0, 1, 2, 3],   # graduated
    "G2": [0, 3],         # on/off
}
SLEEP_GROUP_LABELS = {"G1": "Graduated (SM1/2/3)", "G2": "On/Off"}

# =============================================================================
# CURRICULUM SCHEDULE (training) — identical for DDQN and PPO
# =============================================================================
def curriculum_actions(episode_num):
    """Valid actions for a given training episode under the curriculum."""
    if episode_num is None:
        return [0, 1, 2, 3]
    if episode_num <= 5:
        return [0]
    if episode_num <= 10:
        return [0, 1]
    if episode_num <= 15:
        return [0, 1, 2]
    return [0, 1, 2, 3]

# =============================================================================
# TESTING PROTOCOL
# =============================================================================
N_TEST_ITERATIONS = 150
TEST_SEEDS        = list(range(N_TEST_ITERATIONS))   # 0..149, shared by all

# =============================================================================
# SINR QUALITY THRESHOLDS (dB) — for dotted lines on SINR plots
#   Poor <7 | Fair 7-10 | Good 10-12.5 | Excellent >12.5
# =============================================================================
SINR_THRESHOLDS = [
    (7.0,  "red",    "Poor / Fair (7 dB)"),
    (10.0, "orange", "Fair / Good (10 dB)"),
    (12.5, "green",  "Good / Excellent (12.5 dB)"),
]

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(SCRIPT_DIR, "outputs")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# CSV schemas (column order) — used by training scripts, test harness, plotter
TRAINING_CSV_COLUMNS = ["episode", "reward", "energy", "sinr", "efficiency"]
TESTING_CSV_COLUMNS  = ["iteration", "algorithm", "sleep_group", "seed",
                        "energy", "sinr", "efficiency"]


def parse_ns3_info(info):
    """Parse the ns-3 info string 'k1=v1;k2=v2;...' into a float dict."""
    if isinstance(info, (list, tuple)):
        info = info[0] if info else ""
    out = {}
    if info:
        for item in str(info).split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out
