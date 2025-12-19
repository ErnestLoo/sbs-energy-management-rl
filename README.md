# Multi-Agent Reinforcement Learning for 5G Small Cell Energy Management

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![NS-3](https://img.shields.io/badge/NS--3-3.45-green.svg)](https://www.nsnam.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)

## Overview
This project implements multi-agent reinforcement learning algorithms (DDQN and Q-Learning) for energy-efficient management of Small Cell Base Stations (SBS) in 5G networks using NS-3 simulator with ns3-gym integration.

## Results
| Algorithm | Energy Reduction | Efficiency (bits/J) | SINR (dB) |
|-----------|------------------|---------------------|-----------|
| DDQN      | 33.8%            | 119,216 ± 466       | 8.47      |
| Q-Learning| 29.2%            | 111,502 ± 596       | 9.32      |
| Baseline  | -                | 79,226              | 10.85     |

## Prerequisites
- Ubuntu 24.04 LTS (or 22.04)
- NS-3 version 3.45
- Python 3.10+
- ns3-gym

## Installation

### Step 1: Install NS-3
```bash
# Install dependencies
sudo apt update
sudo apt install g++ python3 python3-pip cmake ninja-build git libzmq5-dev

# Clone and build NS-3
cd ~
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev
git checkout ns-3.45
./ns3 configure --enable-examples
./ns3 build
```

### Step 2: Install ns3-gym
```bash
cd ~/ns-3-dev/contrib
git clone https://github.com/tkn-tub/ns3-gym.git gym
pip install --user ./gym/model/ns3gym
```

### Step 3: Clone This Project
```bash
cd ~/ns-3-dev/scratch
git clone https://github.com//ErnestLoo/sbs-energy-management-rl.git capstone
cd capstone
pip install -r requirements.txt
```

### Step 4: Rebuild NS-3
```bash
cd ~/ns-3-dev
./ns3 build
```

## Usage

### Train DDQN Agent
```bash
cd ~/ns-3-dev/scratch/capstone
python3 fast_ddqn_multiagent.py
```

### Train Q-Learning Agent
```bash
cd ~/ns-3-dev/scratch/capstone
python3 qlearn_multiagent.py
```

### Test Trained DDQN Policy
```bash
cd ~/ns-3-dev/scratch/capstone
python3 ddqn_test.py
```

### Test Trained Q-Learning Policy
```bash
cd ~/ns-3-dev/scratch/capstone
python3 qlearn_test.py
```

### Compare Algorithms
```bash
cd ~/ns-3-dev/scratch/capstone
python3 compare_algorithms.py
```

## Project Structure
```
capstone/
├── scratch.cc                  # NS-3 simulation (C++)
├── smallCellEnergyModel.h      # Custom energy model
├── fast_ddqn_multiagent.py     # DDQN training script
├── qlearn_multiagent.py        # Q-Learning training script
├── ddqn_test.py                # DDQN evaluation
├── qlearn_test.py              # Q-Learning evaluation
├── compare_algorithms.py       # Algorithm comparison
├── figures/                    # Result figures
└── outputs/                    # Trained models & data
```

## Configuration

### Network Parameters
| Parameter | Value |
|-----------|-------|
| Number of SBS | 5 |
| Number of UEs | 50 |
| Simulation Time | 10 seconds |
| Training Episodes | 150 |
| Test Iterations | 100 |

### Sleep Modes
| Mode | Power (W) | Tx Power (dBm) | Wake-up Latency |
|------|-----------|----------------|-----------------|
| Active | 20.7 | 30 | 0 ms |
| SM1 | 15.0 | 20 | 10 ms |
| SM2 | 10.0 | 10 | 50 ms |
| SM3 | 3.36 | 0 | 100 ms |

### State Space (6 factors per agent)
| Index | Factor | Range |
|-------|--------|-------|
| 0 | UE Count | 0-10 |
| 1 | Current Mode | 0-3 |
| 2 | Power | 3.36-20.7 W |
| 3 | SINR | Variable dB |
| 4 | Transitioning | 0-1 |
| 5 | Hour of Day | 0-23 |

## Author
- **Name:** Loo Hui Yan
- **Student ID:** 22018469
- **Supervisor:** Charis Kwan Shwu Chen
- **Institution:** Sunway University

---

## **2. requirements.txt**
```
tensorflow>=2.10.0
numpy>=1.23.0
matplotlib>=3.6.0
gymnasium>=0.26.0
pyzmq>=24.0.0
```
