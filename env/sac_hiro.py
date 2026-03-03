import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from sac_goal import GoalConditionedCryptoEnv


class HighLevelCryptoEnv(gym.Env):
    """
    HIRO-like high-level manager environment.
    Outputs target position (Goal) and calls the frozen low-level SAC model to execute for c steps.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, low_level_model_path: str, macro_step_freq=48, initial_balance=10000.0):
        super(HighLevelCryptoEnv, self).__init__()
        
        self.df = df
        self.macro_step_freq = macro_step_freq
        self.initial_balance = initial_balance
        self.num_crypto_assets = 1
        self.total_assets = self.num_crypto_assets + 1 
        
        self.low_level_env = GoalConditionedCryptoEnv(
            df=self.df, 
            initial_balance=self.initial_balance, 
            goal_change_freq=self.macro_step_freq
        )
        
        print(f"Loading pre-trained low-level worker from {low_level_model_path}...")
        self.low_level_model = SAC.load(low_level_model_path)
        
        # Action Space: outputs Goal (target position weights) to the low-level agent
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.total_assets,), 
            dtype=np.float32
        )
        
        # Observation Space (current market features and actual positions)
        self.num_market_features = len(self.df.columns)
        total_obs_dim = self.num_market_features + self.total_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )

        # High-level risk management records
        self.peak_portfolio_value = self.initial_balance
        self.previous_drawdown = 0.0
        self.risk_penalty_weight = 0.5
        
        self.current_low_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # reset low-level environment and get initial observation
        self.current_low_obs, info = self.low_level_env.reset(seed=seed)
        
        # reset high-level risk state
        self.peak_portfolio_value = self.initial_balance
        self.previous_drawdown = 0.0
        
        return self._get_high_level_obs(), info
    
    def _get_high_level_obs(self):
        # extract features and actual positions from low-level env for next high-level decision
        features = self.current_low_obs[:self.num_market_features]
        actual_weights = self.current_low_obs[self.num_market_features : self.num_market_features + self.total_assets]
        
        return np.concatenate([features, actual_weights])
