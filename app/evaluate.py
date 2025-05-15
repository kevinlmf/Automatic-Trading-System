import os
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from app.env import TradingEnv  # ✅ 长期结构标准写法

# === 1. Load OHLCV data ===
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/sample_ohlcv.csv")
df = pd.read_csv(DATA_PATH)

# === 2. Load trained PPO model ===
model_path = os.path.join(os.path.dirname(__file__), "../ppo_trading_agent")
model = PPO.load(model_path)

# === 3. Set up environment ===
env = TradingEnv(df)
obs = env.reset()

balance_history = []
price_history = []
action_history = []

done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    balance_history.append(info["balance"])
    action_history.append(int(action))
    price_history.append(df.iloc[env.current_step]["close"])

# === 4. Plot PnL curve ===
plt.figure(figsize=(12, 6))
plt.plot(balance_history, label="Balance")
plt.title("Backtest PnL Curve")
plt.xlabel("Step")
plt.ylabel("Account Balance")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# === 5. Print stats ===
final_return = balance_history[-1] - env.initial_balance
print("✅ Final Net Profit: ${:.2f}".format(final_return))
print("📊 Total steps:", len(balance_history))
print("📈 Sell actions taken:", sum(1 for a in action_history if a == 2))
