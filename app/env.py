import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class TradingEnv(gym.Env):
    """A custom RL trading environment using OHLCV data."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, window_size=10, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.df.shape[1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # ✅ Use Gym's seed handler
        if seed is not None:
            np.random.seed(seed)  # Optional: control numpy randomness

        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.current_step = self.window_size
        self.total_reward = 0

        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        return window.to_numpy().astype(np.float32)

    def step(self, action):
        done = False
        reward = 0
        current_price = self.df.iloc[self.current_step]["close"]

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price

        elif action == 2:  # Sell
            if self.position == 1:
                profit = current_price - self.entry_price
                reward = profit
                self.balance += profit
                self.position = 0
                self.entry_price = 0

        self.total_reward += reward
        self.current_step += 1

        if self.current_step >= len(self.df):
            done = True

        obs = self._get_observation()
        info = {"balance": self.balance, "position": self.position}
        return obs, reward, done, False, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
