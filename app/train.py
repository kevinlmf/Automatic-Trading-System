import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from app.env import TradingEnv

# === 1. Load OHLCV Data ===
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_ohlcv.csv")
df = pd.read_csv(DATA_PATH)

# === 2. Create the Trading Environment ===
def create_env():
    return TradingEnv(df)

env = DummyVecEnv([create_env])  # Stable-Baselines3 requires vectorized envs

# === 3. Initialize the PPO Agent ===
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./tensorboard_logs"
)

# === 4. Train the Agent ===
model.learn(total_timesteps=10_000)

# === 5. Save the Trained Model ===
model.save("ppo_trading_agent")
print("✅ PPO model saved as ppo_trading_agent.zip")
