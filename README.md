<div align="center">

# 🧠 An Intelligent Energy Brain for Smart Buildings

### *Autonomous HVAC Control via Deep Reinforcement Learning & Physics-Based Digital Twin Simulation*

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Custom_Env-0081A7?style=for-the-badge&logo=openaigym&logoColor=white)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-DQN-FF6F00?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Digital_Twin-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-MLP-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Logging-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Status](https://img.shields.io/badge/Status-Production_Ready-00C853?style=for-the-badge)

<br/>

> **A self-learning HVAC controller that fuses Deep Q-Networks (DQN), online comfort prediction (MLPRegressor), and Newton's Law of Cooling into a real-time digital twin dashboard — optimizing thermal comfort while minimizing energy consumption across all four Indian seasons.**

<br/>

[📖 Technical Report](#-technical-deep-dive) · [🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#-system-architecture) · [📊 Dashboard](#-digital-twin-dashboard) · [🧪 Training](#-model-training)

</div>

---

## 📌 Project Overview

Traditional HVAC (Heating, Ventilation, and Air Conditioning) systems rely on rigid rule-based thermostat logic — operating on fixed schedules and static setpoints regardless of real-time environmental dynamics. This approach results in **significant energy waste** (HVAC accounts for ~40% of total building energy consumption globally) and **suboptimal thermal comfort**.

This project replaces legacy control logic with an **Intelligent Energy Brain** — a Deep Reinforcement Learning agent that:

- 🧠 **Learns** optimal cooling strategies through trial-and-error interaction with a physics-accurate simulation
- 🌡️ **Adapts** in real-time to outdoor temperature fluctuations, occupancy changes, and seasonal transitions
- 👤 **Personalizes** comfort setpoints using an online-trained neural network (MLPRegressor)
- ⚡ **Minimizes** energy consumption via a multi-objective reward function
- 🖥️ **Visualizes** all dynamics through an interactive Streamlit Digital Twin dashboard

---

## 👥 Team Members

| Name | Role & Contribution |
|:-----|:-------------------|
| **Ayan Kumar Batabyal** | Designed the HVAC simulation environment, physics-based thermal model (Newton's Law of Cooling), multi-objective reward function, online comfort prediction model (MLPRegressor), and implemented the complete ML/RL pipeline |
| **Jahar Kumar Paul** | Developed the Streamlit Digital Twin dashboard (UI), SVG-based room visualization, real-time interaction controls, and performance monitoring interface |
| **Sayan Goswami** | Assisted with system design, integration testing, evaluation metrics, and project coordination |

> *This project was developed collaboratively as a group project.*

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DIGITAL TWIN DASHBOARD                         │
│                        (Streamlit — app/app.py)                        │
│  ┌──────────┐  ┌──────────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Live KPI │  │  SVG Room    │  │  Season   │  │ Manual Override  │  │
│  │ Metrics  │  │  Occupancy   │  │  Control  │  │  AC ON/OFF/Auto  │  │
│  │  Panel   │  │  Renderer    │  │  Selector │  │  Temp ±1°C       │  │
│  └────┬─────┘  └──────┬───────┘  └─────┬─────┘  └────────┬─────────┘  │
│       │               │               │                   │            │
└───────┼───────────────┼───────────────┼───────────────────┼────────────┘
        │               │               │                   │
        ▼               ▼               ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING ENGINE                       │
│  ┌──────────────────────────┐    ┌─────────────────────────────────┐   │
│  │  DQN Agent (SB3)        │    │  Online Comfort Model           │   │
│  │  ─────────────────────  │    │  ──────────────────────────     │   │
│  │  Policy:  MlpPolicy     │    │  Architecture: MLP (32→16)     │   │
│  │  γ:       0.99           │    │  Activation:   ReLU            │   │
│  │  Buffer:  100K           │    │  Optimizer:    Adam (lr=0.01)  │   │
│  │  ε-decay: 20% fraction  │    │  Warm Start:   Online Updates  │   │
│  │  Actions: {0,1,2,3}     │    │  Features:     [T_out, Occ,    │   │
│  │                          │    │                 Hour, Context]  │   │
│  └────────────┬─────────────┘    └──────────────┬──────────────────┘   │
│               │     ┌───────────────────────┐   │                     │
│               └─────┤  Action Selection     ├───┘                     │
│                     │  + Setpoint Prediction │                        │
│                     └───────────┬────────────┘                        │
└─────────────────────────────────┼─────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  PHYSICS-BASED SIMULATION ENGINE                       │
│                    (env/smart_hvac_env.py)                             │
│                                                                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────────┐  │
│  │  Thermal Model   │  │ Seasonal Profile │  │  Reward Function    │  │
│  │  ──────────────  │  │ ───────────────  │  │  ────────────────   │  │
│  │  Newton's Law    │  │ Summer: 30-40°C  │  │  w_energy   = 4.0  │  │
│  │  of Cooling      │  │ Monsoon: 25-31°C │  │  w_deviation= 2.0  │  │
│  │  dT = ΣQ·dt / C  │  │ Autumn: 18-26°C  │  │  w_stable   = 1.0  │  │
│  │  R = 2.0 K/kW    │  │ Winter:  5-15°C  │  │  w_violation= 5.0  │  │
│  │  C = 2.0 kWh/K   │  │                  │  │                     │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────────┘  │
│                                                                        │
│  Observation Space (13-dim):  [T_in, T_out, Setpoint, T_out_forecast  │
│                                ×3, Occ_forecast×3, Setpoint, AC_on,   │
│                                AC_on_steps, AC_off_steps]             │
│  Action Space (Discrete 4):   [Maintain, Noop, Noop, Toggle AC]      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Technical Deep-Dive

### 1. Physics-Based Thermal Simulation (`SmartHVACEnv`)

The environment implements a **first-principles thermodynamic model** based on the **RC (Resistance-Capacitance) thermal network analogy**:

#### 🌡️ Heat Transfer Equation (Newton's Law of Cooling)

```
Q_transfer = (T_outdoor − T_indoor) / R
```

| Parameter | Symbol | Value | Description |
|:----------|:------:|:-----:|:------------|
| Thermal Resistance | `R` | 2.0 K/kW | Building envelope insulation quality |
| Thermal Capacitance | `C` | 2.0 kWh/K | Thermal mass of the room (furniture, walls, air) |
| AC Cooling Power | `Q_ac` | -0.6 kW | Standard cooling rate (Normal mode) |
| Turbo Cooling Power | `Q_turbo` | Dynamic | Overpowers heat load + 0.5°C/step |
| Occupant Heat Gain | `Q_occ` | 0.05 kW/person | Metabolic heat generation per occupant |

#### 📐 Temperature Update Rule

```
T_next = T_current + (Q_transfer + Q_occupants + Q_hvac) × Δt / C + ε
```

Where `ε ~ N(0, 0.05)` adds stochastic thermal noise for simulation realism.

---

### 2. Deep Q-Network (DQN) Agent

The RL agent uses **Stable-Baselines3's DQN** implementation with the following configuration:

| Hyperparameter | Value | Rationale |
|:---------------|:-----:|:----------|
| Policy | `MlpPolicy` | Fully-connected network for continuous state input |
| Learning Rate | `1e-4` | Stable convergence for complex reward landscape |
| Replay Buffer | `100,000` | Retains cross-seasonal experience for transfer learning |
| Batch Size | `64` | Balanced gradient estimation |
| Discount Factor (γ) | `0.99` | Emphasizes long-term comfort over short-term energy savings |
| Exploration Fraction | `0.2` | 20% of training dedicated to ε-greedy exploration |
| Training Steps | `300,000` | Sufficient to learn all four seasonal control strategies |

#### Action Space

| Action ID | Behavior |
|:---------:|:---------|
| `0` | Maintain current AC state |
| `1` | No operation |
| `2` | No operation |
| `3` | Toggle AC (ON ↔ OFF) |

---

### 3. Multi-Objective Reward Function

The reward function balances **four competing objectives**:

```
R = −w_energy × (E / E_max)          ← Penalize energy consumption
    − w_deviation × (δ / δ_max)       ← Penalize comfort deviation
    + w_stable × 𝟙(within band)       ← Reward temperature stability
    − w_violation × 𝟙(hot & AC off)   ← Penalize discomfort violations
    + interaction_penalty              ← Penalize manual override reliance
```

| Weight | Value | Objective |
|:-------|:-----:|:----------|
| `w_energy` | 4.0 | Energy efficiency (highest priority) |
| `w_violation` | 5.0 | Prevent overheating (critical safety) |
| `w_deviation` | 2.0 | Comfort deviation minimization |
| `w_stable` | 1.0 | Stability within comfort band (±2°C) |

---

### 4. Online Comfort Prediction Model

A **warm-start MLPRegressor** (Scikit-learn) learns human comfort preferences in real-time:

```
Comfort Setpoint = MLP([T_outdoor, Occupancy, Hour, Context])
```

| Property | Value |
|:---------|:------|
| Architecture | `32 → 16` hidden units |
| Activation | ReLU |
| Optimizer | Adam (lr = 0.01) |
| Pre-training | 500 synthetic samples |
| Online Updates | Every 10 interaction steps |
| Warm Start | Enabled (incremental learning) |

This model dynamically adjusts the target temperature based on outdoor conditions and occupancy — replacing static thermostat setpoints with **personalized, context-aware comfort targets**.

---

### 5. Seasonal Temperature Modeling

The environment simulates **four distinct Indian seasons** with sinusoidal diurnal temperature cycles:

```
T_outdoor(t) = T_base + A × sin(2π × t / Period) + ε
```

| Season | Base Temp (°C) | Amplitude (°C) | Range (°C) |
|:-------|:--------------:|:--------------:|:----------:|
| ☀️ Summer | 35.0 | 5.0 | 30 – 40 |
| 🌧️ Monsoon | 28.0 | 3.0 | 25 – 31 |
| 🍂 Autumn | 22.0 | 4.0 | 18 – 26 |
| ❄️ Winter | 10.0 | 5.0 | 5 – 15 |

---

## 📊 Digital Twin Dashboard

The Streamlit-based Digital Twin provides **real-time visualization and control**:

| Feature | Description |
|:--------|:------------|
| 🌡️ **Live KPI Panel** | Room temp, outdoor temp, ML target, AC state, energy consumption |
| 🏢 **SVG Room Renderer** | Visual occupancy grid with dynamic person/seat icons |
| 🎮 **Control Panel** | Auto-run toggle, step-by-step execution, speed control |
| 🍂 **Season Selector** | Switch between Summer / Monsoon / Autumn / Winter |
| 🌡️ **Temp Override** | Manual ±1°C setpoint adjustment (triggers manual mode) |
| ⚡ **AC Master Control** | Force ON / Force OFF / Return to AI control |
| 🚀 **Turbo Cooling** | Override physics to guarantee temperature reduction |
| 📈 **Performance Stats** | Simulated time, AC duty cycle, active cooling duration |
| 🔄 **Online Training** | DQN learns from live interactions via experience replay |

---

## 📂 Project Structure

```
An-Intelligent-Energy-Brain-for-Smart-Buildings/
│
├── 📂 app/
│   └── app.py                                  # Streamlit Digital Twin dashboard
│                                                # (SVG renderer, KPI panel, controls)
│
├── 📂 env/
│   └── smart_hvac_env.py                       # Custom Gymnasium environment
│                                                # (Physics engine, reward function,
│                                                #  comfort model, seasonal profiles)
│
├── 📂 training/
│   ├── train.py                                # Basic DQN training script (100K steps)
│   └── train_hvac_dqn.py                       # Advanced training with TensorBoard
│                                                # logging & progress bar (300K steps)
│
├── 📂 models/
│   └── dqn_smart_hvac_all_improvements.zip     # Pre-trained DQN policy weights
│
├── 📂 reports/
│   ├── smart_hvac_project_report.pdf           # Detailed project report
│   └── smart_hvac_project_ppt.pdf              # Project presentation slides
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.10 or higher
- **pip** package manager
- **Git** for cloning

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ayan074/An-Intelligent-Energy-Brain-for-Smart-Buildings-Smart-HVAC-Project.git
cd An-Intelligent-Energy-Brain-for-Smart-Buildings-Smart-HVAC-Project
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Digital Twin Dashboard

```bash
streamlit run app/app.py
```

The dashboard will launch at `http://localhost:8501` in your default browser.

### 5️⃣ (Optional) Retrain the DQN Agent

```bash
# Basic training (100K steps, ~5 min)
cd training
python train.py

# Advanced training with TensorBoard (300K steps, ~15 min)
python train_hvac_dqn.py
```

### 6️⃣ (Optional) Monitor Training with TensorBoard

```bash
tensorboard --logdir=./tb_logs
```

---

## ⚙️ How It Works — Step-by-Step Flow

```
┌─────────────┐     ┌────────────────┐     ┌──────────────────┐
│  1. OBSERVE  │────▶│  2. PREDICT    │────▶│  3. ACT          │
│  State s(t)  │     │  Q(s,a) → a*   │     │  Toggle AC       │
│  [13-dim]    │     │  via DQN       │     │  ON/OFF          │
└─────────────┘     └────────────────┘     └────────┬─────────┘
                                                     │
      ┌──────────────────────────────────────────────┘
      │
      ▼
┌─────────────────┐     ┌────────────────┐     ┌──────────────┐
│  4. SIMULATE    │────▶│  5. REWARD     │────▶│  6. LEARN     │
│  Physics Engine │     │  R(s,a,s')     │     │  Update DQN   │
│  T_next = f(.)  │     │  Multi-obj     │     │  via Replay   │
└─────────────────┘     └────────────────┘     └──────┬───────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │  7. VISUALIZE  │
                                              │  Digital Twin  │
                                              │  Dashboard     │
                                              └────────────────┘
```

1. **Observe** — The agent receives a 13-dimensional state vector: indoor temp, outdoor temp, setpoint, 3-step outdoor forecast, 3-step occupancy forecast, current setpoint, AC state, and consecutive ON/OFF step counters.

2. **Predict** — The DQN's neural network estimates Q-values for all 4 actions and selects the action with the highest expected cumulative reward (ε-greedy during training).

3. **Act** — The selected action is executed (toggle AC or maintain state), subject to human override priority and intelligent safety logic (e.g., blocking AC start when already below setpoint).

4. **Simulate** — The physics engine computes the next indoor temperature using Newton's Law of Cooling, accounting for occupant heat, HVAC cooling, and environmental heat transfer.

5. **Reward** — A multi-objective reward signal is computed, balancing energy efficiency, thermal comfort deviation, stability, and violation penalties.

6. **Learn** — The experience tuple `(s, a, r, s')` is stored in the replay buffer, and the DQN is updated via mini-batch gradient descent (online learning supported).

7. **Visualize** — All state variables are rendered in real-time on the Streamlit Digital Twin dashboard.

---

## 🧪 Technologies & Frameworks

| Category | Technology | Purpose |
|:---------|:-----------|:--------|
| **Language** | Python 3.10+ | Core implementation |
| **RL Framework** | Stable-Baselines3 | DQN agent training & inference |
| **Environment** | Gymnasium (Custom) | Physics-based HVAC simulation |
| **ML Library** | Scikit-learn | Online comfort prediction (MLPRegressor) |
| **Dashboard** | Streamlit | Real-time Digital Twin visualization |
| **Visualization** | Matplotlib, SVG | Charts, room occupancy rendering |
| **Logging** | TensorBoard | Training metrics visualization |
| **Numerical** | NumPy | Mathematical computations |
| **Progress** | tqdm | Training progress bars |

---

## 🏆 Key Results

| Metric | Value |
|:-------|:------|
| Training Algorithm | Deep Q-Network (DQN) |
| Observation Space | 13-dimensional continuous |
| Action Space | Discrete (4 actions) |
| Training Duration | 300,000 timesteps |
| Seasonal Coverage | 4 seasons (Summer, Monsoon, Autumn, Winter) |
| Comfort Band | ±2.0°C from adaptive setpoint |
| Online Learning | ✅ Supported (live experience replay) |
| Human Override | ✅ Supported (Force ON/OFF/Temperature) |
| Turbo Mode | ✅ Available (guaranteed cooling) |

---

## 📜 License

This project is developed for academic and research purposes.

---

<div align="center">

**Built with 🧠 Intelligence & ⚡ Efficiency**

*An Intelligent Energy Brain for Smart Buildings — Where AI meets Thermodynamics*

</div>
