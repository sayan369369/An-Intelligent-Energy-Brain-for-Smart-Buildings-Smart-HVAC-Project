# train.py
import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

# Import the UPDATED environment class
# Make sure smart_hvac_env.py contains the NEW code I gave you previously
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.smart_hvac_env import SmartHVACEnv 
env = SmartHVACEnv()

# Config
TOTAL_TIMESTEPS = 100_000 # Train enough to learn the new logic
MODEL_PATH = "dqn_smart_hvac_all_improvements" # This will OVERWRITE your old model

def train():
    # 1. Initialize the NEW environment (Observation shape = 14)
    env = Monitor(SmartHVACEnv())
    
    # 2. Create a NEW model (Input layer will automatically size to 14)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        verbose=1,
    )
    
    print(f"🚀 Starting training on {env.observation_space.shape[0]}-dim observation space...")
    
    # 3. Train
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    # 4. Save the new 14-input model
    model.save(MODEL_PATH)
    print(f"✅ Saved updated model to {MODEL_PATH}.zip")

if __name__ == "__main__":
    train()