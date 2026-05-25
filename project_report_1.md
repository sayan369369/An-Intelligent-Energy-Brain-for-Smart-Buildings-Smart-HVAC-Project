# 📋 Project Report 1: An Intelligent Energy Brain for Smart Buildings

## Understanding the Smart HVAC Project — A Complete Guide

**Author:** Ayan Kumar Batabyal  
**Date:** May 2026  
**Project Type:** Deep Reinforcement Learning + Digital Twin Simulation

---

## 1. What is This Project?

### The Problem
Traditional air conditioning (HVAC) systems in buildings are **dumb** — they follow simple rules like "if temperature > 25°C, turn ON AC." This wastes massive energy (HVAC uses ~40% of building electricity globally) and doesn't adapt to real-world changes like:
- How many people are in the room (more people = more body heat)
- What season it is (summer vs winter need completely different strategies)
- Individual comfort preferences (some people prefer cooler rooms)

### The Solution
This project builds an **AI-powered brain** that replaces the dumb thermostat. Instead of fixed rules, it uses:
1. **Deep Reinforcement Learning (DQN)** — The AI learns by trial-and-error, like a human learning to drive
2. **Physics Simulation** — A virtual building that behaves like a real one (heat transfer, insulation, etc.)
3. **Online Learning** — The AI keeps improving even while running, adapting to new situations
4. **Digital Twin Dashboard** — A live control panel to watch and interact with the AI in real-time

---

## 2. How Does the Project Work? (Simple Explanation)

Think of it like training a **virtual building manager**:

```
Step 1: Build a virtual building (Physics Simulation)
Step 2: Put an AI agent inside it (DQN Agent)
Step 3: Let the AI practice controlling the AC for 300,000 rounds
Step 4: The AI learns: "In summer with 200 people, keep AC ON"
                        "In winter with 10 people, keep AC OFF"
Step 5: Deploy the trained AI on a real-time dashboard
Step 6: The AI continues learning from live interactions
```

### The Loop (Every 3 Minutes of Simulated Time):

| Step | What Happens | Technical Term |
|:-----|:-------------|:---------------|
| 1 | AI reads the room state (temp, people, weather) | **Observation** (13-dim vector) |
| 2 | AI decides: should I turn AC ON or OFF? | **Action Selection** (ε-greedy policy) |
| 3 | The physics engine calculates new room temperature | **State Transition** (Newton's Law of Cooling) |
| 4 | AI gets a score: +points for comfort, -points for energy waste | **Reward Signal** |
| 5 | AI stores this experience and updates its brain | **Experience Replay + Gradient Descent** |
| 6 | Dashboard shows everything live | **Digital Twin Visualization** |

---

## 3. Project Components Explained

### Component A: The Physics Engine (`env/smart_hvac_env.py`)

This is the **heart of the project** — a virtual building that follows real physics.

**What it simulates:**
- **Heat flowing through walls** — Using Newton's Law of Cooling: `Q = (T_outside - T_inside) / R`
- **Body heat from people** — Each person generates 0.05 kW of heat
- **AC cooling effect** — The AC removes heat at -0.6 kW (or more in Turbo mode)
- **Seasonal weather** — Summer (30-40°C), Monsoon (25-31°C), Autumn (18-26°C), Winter (5-15°C)
- **Random noise** — Small random fluctuations to simulate real-world unpredictability

**Key Parameters:**
| Parameter | Value | What It Means |
|:----------|:-----:|:-------------|
| R (Thermal Resistance) | 2.0 | How well the building is insulated |
| C (Thermal Capacitance) | 2.0 | How much heat the room can absorb before temp changes |
| Timestep | 3 minutes | AI makes a decision every 3 simulated minutes |
| Max Occupancy | 250 | Maximum people the room can hold |
| Comfort Band | ±2°C | Acceptable range around the target temperature |

---

### Component B: The AI Agent (DQN — Deep Q-Network)

**What is DQN?**
DQN is a Reinforcement Learning algorithm that uses a neural network to learn which actions are best in each situation.

**How it learns:**
1. The AI tries random actions initially (exploration)
2. It records what happened: `(state, action, reward, next_state)`
3. It stores thousands of these experiences in a "replay buffer"
4. It randomly samples batches and trains its neural network
5. Over time, it learns which actions give the highest long-term reward

**DQN Configuration:**
| Setting | Value | Why |
|:--------|:-----:|:----|
| Neural Network | MlpPolicy (fully connected) | Good for continuous state inputs |
| Learning Rate | 0.0001 | Slow but stable learning |
| Replay Buffer | 100,000 experiences | Remembers winter strategies while in summer |
| Discount Factor (γ) | 0.99 | Cares about long-term comfort, not just this moment |
| Exploration | 20% of training | Tries random actions to discover better strategies |
| Training Steps | 300,000 | Enough to learn all four seasons |

**The 4 Actions the AI Can Take:**
| Action | Effect |
|:------:|:-------|
| 0 | Keep AC in current state (do nothing) |
| 1 | No operation |
| 2 | No operation |
| 3 | Toggle AC (if ON → OFF, if OFF → ON) |

---

### Component C: The Reward Function (How the AI Scores Itself)

The reward function is what teaches the AI **what "good" means**. It balances four goals:

```
Reward = - 4.0 × (energy used)           ← "Don't waste electricity"
         - 2.0 × (comfort deviation)      ← "Keep room at the right temp"
         + 1.0 × (within comfort band)    ← "Bonus for being comfortable"
         - 5.0 × (too hot AND AC is off)  ← "NEVER let people overheat"
```

**Why these weights?**
- Energy penalty (4.0) is high → AI learns to turn AC OFF when not needed
- Violation penalty (5.0) is highest → AI NEVER ignores overheating
- Stability bonus (1.0) is low → Nice-to-have, not critical

---

### Component D: The Comfort Prediction Model (MLPRegressor)

**Problem:** Different situations need different target temperatures. You don't want 24°C when it's winter and the room is empty.

**Solution:** A small neural network (MLP with 32→16 hidden layers) that predicts the ideal temperature based on:
- Outdoor temperature
- Number of people in the room
- Time of day
- Context flags

**How it works:**
1. Pre-trained on 500 synthetic data points (initial knowledge)
2. Updated online every 10 steps with real interaction data
3. Uses `warm_start=True` — doesn't forget old knowledge when learning new things

---

### Component E: The Digital Twin Dashboard (`app/app.py`)

A **Streamlit web application** that lets you watch and control the AI in real-time.

**Dashboard Features:**
| Feature | What It Shows |
|:--------|:-------------|
| Room Temp Metric | Current indoor temperature with deviation from ideal |
| Outside Temp | Current outdoor temperature based on season |
| ML Target | What temperature the comfort model thinks is ideal |
| System State | Whether AC is Running, Idle, Forced ON, or Forced OFF |
| Energy Used | Total energy consumed in kWh |
| SVG Room Map | Visual grid showing occupied seats with person icons |
| Performance Panel | Simulated time elapsed, AC duty cycle, cooling duration |
| Season Control | Switch between Summer/Monsoon/Autumn/Winter |
| Temp Override | Manual ±1°C adjustment (triggers manual mode) |
| AC Master | Force AC ON/OFF or return control to AI |
| Turbo Mode | Override physics for guaranteed cooling |
| Online Training | AI learns from live dashboard interactions |

---

## 4. File-by-File Breakdown

### `env/smart_hvac_env.py` (410 lines) — The Brain's World

| Section | Lines | Purpose |
|:--------|:-----:|:--------|
| `ComfortSetpointModel` class | 15-43 | Neural network that predicts ideal temperature |
| `SmartHVACEnv.__init__` | 49-125 | Sets up physics parameters, observation/action spaces |
| `_pretrain_comfort_model` | 128-139 | Generates 500 synthetic training samples |
| `_get_obs` | 142-158 | Builds the 13-dim observation vector with forecasts |
| `_step_thermal_model` | 161-185 | Original physics (Newton's Law + noise) |
| `reset` | 188-206 | Resets environment to initial state |
| `step` | 209-404 | **Core loop**: action → physics → reward → next state |
| `set_season` | 407-409 | Changes seasonal temperature profile |

### `app/app.py` (392 lines) — The Dashboard

| Section | Lines | Purpose |
|:--------|:-----:|:--------|
| Imports & Path Setup | 1-19 | Loads libraries and finds project root |
| CSS Styling | 31-62 | Custom metric card styling (blue accent, compact layout) |
| Model Loading | 67-84 | Loads pre-trained DQN with TensorBoard logger fix |
| Session State | 89-104 | Persistent state across Streamlit reruns |
| `run_simulation_step` | 118-175 | One simulation step: occupancy change → agent action → physics → online training → logging |
| Sidebar Controls | 180-266 | Auto-run, speed, season, temp override, AC master, turbo |
| SVG Room Renderer | 271-306 | Generates visual room layout with occupancy icons |
| Main Layout | 311-384 | KPI metrics, room visualization, performance stats |
| Auto-run Loop | 389-392 | Continuous simulation when auto-run is enabled |

### `training/train.py` (40 lines) — Quick Training

Simple training script: creates environment → trains DQN for 100K steps → saves model.

### `training/train_hvac_dqn.py` (70 lines) — Full Training

Advanced training: 300K steps + TensorBoard logging + tqdm progress bar + larger replay buffer.

---

## 5. How to Run the Project

### Step 1: Install Python 3.10+
Download from [python.org](https://python.org)

### Step 2: Clone & Setup
```bash
git clone https://github.com/Ayan074/An-Intelligent-Energy-Brain-for-Smart-Buildings-Smart-HVAC-Project.git
cd An-Intelligent-Energy-Brain-for-Smart-Buildings-Smart-HVAC-Project
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Step 3: Run Dashboard (Uses Pre-Trained Model)
```bash
streamlit run app/app.py
```
Opens at `http://localhost:8501` — you can immediately interact with the trained AI.

### Step 4: (Optional) Retrain from Scratch
```bash
cd training
python train_hvac_dqn.py       # Takes ~15 minutes for 300K steps
```

---

## 6. Key Technical Concepts Used

| Concept | Where Used | What It Means |
|:--------|:-----------|:-------------|
| **Deep Q-Network (DQN)** | Agent training | RL algorithm that uses neural networks to approximate Q-values |
| **Experience Replay** | Training + Online | Stores past experiences and randomly samples them for training |
| **ε-Greedy Exploration** | Action selection | Sometimes takes random actions to discover better strategies |
| **Newton's Law of Cooling** | Physics engine | Heat flows from hot to cold proportional to temperature difference |
| **RC Thermal Network** | Building model | Models building as a resistor-capacitor circuit for heat flow |
| **MLPRegressor** | Comfort model | Multi-Layer Perceptron that predicts ideal temperature |
| **Online/Continual Learning** | Dashboard | Model keeps learning from new data without full retraining |
| **Digital Twin** | Dashboard | Virtual replica of a physical system for monitoring and control |
| **Gymnasium (OpenAI Gym)** | Environment | Standard API for RL environments (observation, action, reward, done) |
| **Warm Start** | Comfort model | Continues training from previous weights instead of starting over |
| **Sinusoidal Modeling** | Seasonal temps | Uses sin waves to simulate daily temperature cycles |
| **Multi-Objective Optimization** | Reward function | Balancing competing goals (comfort vs energy vs safety) |

---

## 7. What Makes This Project Special?

| Feature | Why It Matters |
|:--------|:--------------|
| **Physics-Based Simulation** | Not a toy environment — uses real thermodynamic equations |
| **Online Learning** | AI improves while running, not just during offline training |
| **Human-in-the-Loop** | Users can override AI and the AI learns from those overrides |
| **Seasonal Adaptation** | Handles all four Indian seasons with different strategies |
| **Multi-Objective Reward** | Balances comfort, energy, stability, and safety simultaneously |
| **Digital Twin Dashboard** | Professional real-time monitoring and control interface |
| **Turbo Mode** | Safety feature that guarantees cooling in extreme conditions |
| **Occupancy-Aware** | Adjusts strategy based on how many people are present |

---

## 8. Summary

This project demonstrates how **AI can replace traditional rule-based HVAC control** with an intelligent, adaptive system. The DQN agent learns through 300,000 rounds of simulated experience in a physics-accurate virtual building, mastering energy-efficient cooling strategies across all seasons. The Streamlit Digital Twin dashboard makes the invisible AI decisions visible and controllable, enabling human oversight while the AI continuously improves through online learning.

**In one sentence:** *It's an AI brain that learns to control your building's AC — keeping you comfortable while saving energy, and you can watch it think in real-time.*

---

<div align="center">

*Project Report 1 — Prepared for Personal Reference & Understanding*

</div>
