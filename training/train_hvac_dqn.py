import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from smart_hvac_env import SmartHVACEnv  # <--- CRITICAL: Imports your updated physics

# -----------------------------
# Config
# -----------------------------
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = tqdm(total=total_timesteps)

    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# We need more steps now because the agent must learn 4 different seasons
TOTAL_TIMESTEPS = 300_000  
# This path MUST match what app.py looks for
MODEL_SAVE_PATH = "dqn_smart_hvac_all_improvements" 
TENSORBOARD_LOGDIR = "./tb_logs"

def train_and_run():
    # 1. Setup Logging
    if not os.path.exists(TENSORBOARD_LOGDIR):
        os.makedirs(TENSORBOARD_LOGDIR)
    
    # 2. Initialize the UPDATED Environment
    # This automatically loads your new R=4.0, C=5.0, AC=-15.0kW, and Seasonal logic
    env = Monitor(SmartHVACEnv())
    
    # 3. Create the DQN Agent
    # We increase buffer_size to help it remember Winter actions while in Summer
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,       
        buffer_size=100_000,      # Larger memory for seasonal patterns
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,               # Focus on long-term comfort
        exploration_fraction=0.2, # Explore 20% of the time to find new seasonal strategies
        verbose=1,
        tensorboard_log=TENSORBOARD_LOGDIR
    )
    
    print(f"🚀 Starting training for {TOTAL_TIMESTEPS} steps...")
    print("   - Physics: Seasons enabled, Strong AC (-15kW)")
    print("   - Goal: Learn to manage Winter vs Summer automatically")
    
    # 4. Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=ProgressCallback(TOTAL_TIMESTEPS)
    )
    
    # 5. Save
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Saved new seasonal brain to {MODEL_SAVE_PATH}.zip")
    print("   - You can now run 'streamlit run gui/app.py'")

if __name__ == "__main__":
    train_and_run()