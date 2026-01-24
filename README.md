# 🏢 Smart HVAC Controller using Deep Reinforcement Learning (Digital Twin)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-DQN-orange)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 Project Overview

This project implements a **Smart HVAC Control System** using **Deep Reinforcement Learning (DQN)** and a **physics-based simulation environment**, combined with an interactive **Streamlit Digital Twin dashboard**.

The system learns to intelligently control air-conditioning based on:

- Indoor temperature  
- Outdoor temperature (seasonal variation)  
- Occupancy  
- Human comfort preferences  
- Energy efficiency  

The goal is to **maximize comfort while minimizing energy consumption**.

---

## 👥 Team Members (Group Project)

| Name | Contribution |
|------|-------------|
| **Ayan Kumar Batabyal** | Designed the HVAC simulation environment, physics model, reward function, comfort prediction model, and implemented the full Machine Learning / Reinforcement Learning pipeline |
| **Jahar Kumar Paul** | Developed the Streamlit dashboard (UI), visualization system, and real-time interaction controls |
| **Sayan Goswami** | Assisted with system design, testing, evaluation, and project coordination |

> This project was developed collaboratively as a group project.

---

## 🧠 Key Features

- Deep Q-Learning (DQN) based HVAC control  
- Physics-based thermal environment simulation  
- Online comfort preference learning (MLP model)  
- Seasonal temperature modeling (Summer, Monsoon, Autumn, Winter)  
- Occupancy-aware decision making  
- Human manual override support  
- Energy consumption tracking  
- Real-time visualization using Streamlit (Digital Twin)  
- Online training support  

---
User / Environment
↓
Smart HVAC Environment (Physics + Comfort Model)
↓
Reinforcement Learning Agent (DQN)
↓
Action (AC ON / OFF / Control)
↓
Streamlit Digital Twin Dashboard (Visualization & Control)


---

## 🧪 Technologies Used

- Python  
- Gymnasium (custom environment)  
- Stable-Baselines3 (DQN)  
- NumPy  
- Scikit-learn (MLPRegressor for comfort model)  
- Streamlit (UI)  
- Matplotlib  
- TensorBoard  
- tqdm  

---

## 📂 Project Structure



smart-hvac-project/
│
├── app/
│   └── app.py                     # Streamlit dashboard
│
├── env/
│   └── smart_hvac_env.py          # Custom HVAC simulation environment
│
├── training/
│   ├── train.py
│   └── train_hvac_dqn.py          # DQN training scripts
│
├── models/
│   └── dqn_smart_hvac_all_improvements.zip
│
├── reports/
│   ├── smart_hvac_project_report.pdf
│   └── smart_hvac_project_ppt.pdf
│
├── assets/
│
├── requirements.txt
└── README.md


---

## ⚙️ How It Works

### 1. Environment (by Ayan Kumar Batabyal)
- Simulates heat transfer  
- Models human comfort preferences  
- Tracks energy usage  
- Generates rewards for the RL agent  
- Handles occupancy dynamics and seasonal effects  

### 2. Learning Agent
- Uses Deep Q-Network (DQN)  
- Learns optimal cooling strategies  
- Trained using experience replay  

### 3. Dashboard (by Jahar Kumar Paul)
- Visual occupancy map  
- Live temperature monitoring  
- Manual override controls  
- AC force ON/OFF  
- Performance statistics  
- Energy usage graphs  

---

## 🏆 Training Performance

- Algorithm: **Deep Q-Network (DQN)**
- Observation space: Multi-dimensional thermal + occupancy state
- Training steps: up to **300,000**
- Supports **online learning**



## 🏗️ System Architecture

