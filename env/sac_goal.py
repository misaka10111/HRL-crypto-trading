import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class GoalConditionedCryptoEnv(gym.Env):
    """
    Goal-Conditioned low-level trading environment
    High-level policy provides Goal (target weights); the low-level agent outputs Execution (actual execution weights)    
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0, goal_change_freq=48, custom_mean=None, custom_std=None):
        super(GoalConditionedCryptoEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.num_crypto_assets = 1 
        self.total_assets = self.num_crypto_assets + 1 
        self.commission_fee_percent = 0.001
        self.goal_change_freq = goal_change_freq

        self.price_data = self.df['Close'].values
        self.feature_df = self.df.drop(columns=['Close'])
        self.feature_data = self.feature_df.values

        # for local normalization
        if custom_mean is not None and custom_std is not None:
            self.obs_mean = custom_mean
            self.obs_std = custom_std
        else:
            self.obs_mean = self.feature_df.mean().values.astype(np.float32)
            self.obs_std = self.feature_df.std().values.astype(np.float32)
            self.obs_std[self.obs_std == 0] = 1e-8  # prevent division by zero

        # expanded observation space (flattened)
        self.num_market_features = self.feature_df.shape[1]
        total_obs_dim = self.num_market_features + self.total_assets + self.total_assets
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )

        # Action Space
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.total_assets,), 
            dtype=np.float32
        )

        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        self.current_goal_weights = np.zeros(self.total_assets)

    def _get_observation(self):
        raw_features = self.feature_data[self.current_step].astype(np.float32)
        norm_features = (raw_features - self.obs_mean) / self.obs_std
        
        current_prices = self._get_current_prices()
        actual_weights = self._get_actual_weights(current_prices)
        
        obs = np.concatenate([
            norm_features, 
            actual_weights, 
            self.current_goal_weights
        ])
        return obs

    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
        }

    def _get_current_prices(self):
        return np.array([self.price_data[self.current_step]], dtype=np.float32)
    
    # generate random high-level instruction
    def _sample_random_goal(self):
        if np.random.rand() < 0.2:
            w = np.zeros(self.total_assets)
            w[0] = 1.0 # 20% full cash
            return w.astype(np.float32)
        elif np.random.rand() < 0.2:
            w = np.zeros(self.total_assets)
            w[1] = 1.0 # 20% full BTC
            return w.astype(np.float32)
            
        random_weights = np.random.dirichlet(np.ones(self.total_assets), size=1)[0]
        return random_weights.astype(np.float32)
    
    def _get_actual_weights(self, current_prices):
        if self.portfolio_value <= 0:
            w = np.zeros(self.total_assets)
            w[0] = 1.0
            return w.astype(np.float32)
            
        crypto_values = self.crypto_holdings * current_prices
        actual_weights = np.zeros(self.total_assets)
        actual_weights[0] = self.cash_balance / self.portfolio_value
        actual_weights[1:] = crypto_values / self.portfolio_value
        return actual_weights.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        
        self.current_goal_weights = self._sample_random_goal()
        
        return self._get_observation(), self._get_info()
    
    