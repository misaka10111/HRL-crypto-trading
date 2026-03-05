import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
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

    def step(self, macro_action: np.ndarray):
        """
        High-level agent sets Goal, low-level agent executes for macro_step_freq steps
        """
        # normalize high-level action to use as low-level Goal
        goal_weights = np.clip(macro_action, 0.0, 1.0)
        weight_sum = np.sum(goal_weights)
        if weight_sum > 0:
            goal_weights = goal_weights / weight_sum
        else:
            goal_weights = np.zeros(self.total_assets)
            goal_weights[0] = 1.0
            
        # force inject Goal into low-level environment
        self.low_level_env.current_goal_weights = goal_weights
        
        # record starting portfolio value for macro return calculation
        portfolio_value_start = self.low_level_env.portfolio_value
        
        terminated = False
        truncated = False
        macro_steps_taken = 0
        
        # Low-level Worker execution loop
        for _ in range(self.macro_step_freq):
            # update Goal in low-level observation
            self.current_low_obs[-self.total_assets:] = goal_weights # type: ignore
            
            # query frozen low-level SAC model for execution action
            # deterministic=True ensures pure execution without random exploration
            low_level_action, _ = self.low_level_model.predict(self.current_low_obs, deterministic=True) # type: ignore
            
            # execute in low-level physical environment
            self.current_low_obs, low_reward, terminated, truncated, info = self.low_level_env.step(low_level_action)
            
            macro_steps_taken += 1
            if terminated or truncated:
                break
                
        # high-level manager reward
        portfolio_value_end = self.low_level_env.portfolio_value
        
        if portfolio_value_start > 0:
            # Macro base reward (total log return over c steps)
            macro_log_return = math.log(portfolio_value_end / portfolio_value_start)
            macro_base_reward = macro_log_return * 100.0
            
            # Macro risk penalty (incremental drawdown penalty)
            if portfolio_value_end > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value_end
                
            current_drawdown = (self.peak_portfolio_value - portfolio_value_end) / self.peak_portfolio_value
            drawdown_delta = current_drawdown - self.previous_drawdown
            
            if drawdown_delta > 0:
                macro_drawdown_penalty = self.risk_penalty_weight * drawdown_delta * 100.0
            else:
                macro_drawdown_penalty = 0.0
                
            self.previous_drawdown = current_drawdown
            
            # Final high-level reward: profitability and risk management over the macro step
            macro_reward = macro_base_reward - macro_drawdown_penalty
        else:
            macro_reward = -1.0
            
        return self._get_high_level_obs(), macro_reward, terminated, truncated, info


if __name__ == "__main__":
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    import os

    print("loading data...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'btcusd_5-min_data.csv')
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True).sort_index()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)

    # specify low-level model path
    # assumes the saved low-level model is named sac_low_level_executioner.zip
    LOW_LEVEL_MODEL_PATH = os.path.join(BASE_DIR, "sac_low_level_executioner.zip")
    
    if not os.path.exists(LOW_LEVEL_MODEL_PATH):
        raise FileNotFoundError(f"cannot find {LOW_LEVEL_MODEL_PATH}")

    # instantiate high-level environment
    base_env = HighLevelCryptoEnv(
        df=train_df, 
        low_level_model_path=LOW_LEVEL_MODEL_PATH, 
        macro_step_freq=48,  # makes a decision every 48 low-level steps
        initial_balance=10000.0
    )
    
    # wrap environment for monitoring and vectorization
    base_env = Monitor(base_env)
    vec_env = DummyVecEnv([lambda: base_env])
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # initialize high-level SAC Agent
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=1e-4, 
        batch_size=256,
        verbose=1, 
        tensorboard_log="./sac_hiro_manager_tensorboard/"
    )
    
    # train
    print("Training High-level Manager...")
    high_level_steps_per_epoch = len(train_df) // 48
    model.learn(
        total_timesteps=high_level_steps_per_epoch * 5,
        log_interval=1
    )
    
    # save
    print("training finished, saving...")
    model.save("sac_high_level_manager")
    env.save("vec_normalize_hiro.pkl")