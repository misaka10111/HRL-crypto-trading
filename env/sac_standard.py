import os
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class SacStandard(gym.Env):
    """
    Gymnasium environment, for SB3
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(SacStandard, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.num_crypto_assets = 1
        self.total_assets = self.num_crypto_assets + 1  # BTC + USDT
        self.commission_fee_percent = 0.001  # 0.1%
        
        # Observation Space
        num_features = len(self.df.columns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.num_crypto_assets, num_features), 
            dtype=np.float32
        )

        # Action Space
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.total_assets,), 
            dtype=np.float32
        )

        # Internal State Variables
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        current_features = self.df.iloc[self.current_step].values
        return current_features.reshape(self.num_crypto_assets, -1).astype(np.float32)

    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
        }
    
    def _get_current_prices(self):
        current_row = self.df.iloc[self.current_step]
        return np.array([current_row['Close']], dtype=np.float32)