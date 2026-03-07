import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from sac_goal import GoalConditionedCryptoEnv

class HighLevelCryptoEnv(gym.Env):
    """
    HIRO-like high-level manager environment.
    Outputs target position (Goal) and calls the frozen low-level SAC model to execute for c steps.
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, low_level_model_path: str, macro_step_freq=48, initial_balance=10000.0, custom_mean=None, custom_std=None):
        super(HighLevelCryptoEnv, self).__init__()
        
        self.df = df
        self.macro_step_freq = macro_step_freq
        self.initial_balance = initial_balance
        self.num_crypto_assets = 1
        self.total_assets = self.num_crypto_assets + 1 
        
        # 修复点 1: 将归一化参数传递给底层执行环境，防止分布偏移
        self.low_level_env = GoalConditionedCryptoEnv(
            df=self.df, 
            initial_balance=self.initial_balance, 
            goal_change_freq=self.macro_step_freq,
            custom_mean=custom_mean,
            custom_std=custom_std
        )
        
        # Load the pre-trained low-level executioner
        self.low_level_model = SAC.load(low_level_model_path, device="cpu")
        
        # Action Space: outputs Goal (target position weights) to the low-level agent
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.total_assets,), 
            dtype=np.float32
        )
        
        if 'Close' in self.df.columns:
            self.feature_data = self.df.drop(columns=['Close']).values
        else:
            self.feature_data = self.df.values       
            
        self.num_market_features = self.feature_data.shape[1]
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
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset low-level environment and get initial observation
        self.current_low_obs, info = self.low_level_env.reset(seed=seed)
        
        self.current_step = info.get("step", getattr(self.low_level_env, "current_step", 0))
        
        # Reset high-level risk state
        self.peak_portfolio_value = self.initial_balance
        self.previous_drawdown = 0.0
        
        return self._get_high_level_obs(), info
    
    def _get_high_level_obs(self):
        # Boundary protection to prevent IndexError caused by current_step overflow at the end of an episode
        safe_step = min(self.current_step, len(self.feature_data) - 1)
        features = self.feature_data[safe_step].flatten().astype(np.float32)
        
        # actual dimension = num_market_features
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
        
        # Low-level Worker execution loop
        for _ in range(self.macro_step_freq):
            # update Goal in low-level observation
            self.current_low_obs[-self.total_assets:] = goal_weights 
            
            # query frozen low-level SAC model for execution action
            # deterministic=True ensures pure execution without random exploration
            low_level_action, _ = self.low_level_model.predict(self.current_low_obs, deterministic=True) 
            
            # execute in low-level physical environment
            self.current_low_obs, low_reward, terminated, truncated, info = self.low_level_env.step(low_level_action)
            
            # Keep track of the actual step sequence for accurate stationary feature fetching
            self.current_step = info.get("step", getattr(self.low_level_env, "current_step", self.current_step + 1))
            
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
            
            macro_drawdown_penalty = self.risk_penalty_weight * drawdown_delta * 100.0 if drawdown_delta > 0 else 0.0
            self.previous_drawdown = current_drawdown
            
            # Final high-level reward: profitability minus risk penalty
            macro_reward = macro_base_reward - macro_drawdown_penalty
        else:
            macro_reward = -1.0
            
        return self._get_high_level_obs(), macro_reward, terminated, truncated, info


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback for logging Trading Metrics to TensorBoard.
    """
    def __init__(self, steps_per_year=2190, verbose=0): # 105120 / 48 macro steps = 2190
        super(TradingMetricsCallback, self).__init__(verbose)
        self.steps_per_year = steps_per_year
        self.episode_returns = []
        self.last_portfolio_value = 10000.0

    def _on_step(self) -> bool:
        step_reward = self.locals.get("rewards", [0.0])[0]
        done = self.locals.get("dones", [False])[0]
        infos = self.locals.get("infos", [{}])[0]

        self.episode_returns.append(step_reward)

        if "terminal_info" in infos:
            self.last_portfolio_value = infos["terminal_info"].get("portfolio_value", self.last_portfolio_value)
        elif "portfolio_value" in infos:
            self.last_portfolio_value = infos["portfolio_value"]

        if done:
            final_portfolio_value = self.last_portfolio_value
            returns_array = np.array(self.episode_returns)
            
            if len(returns_array) > 1 and np.std(returns_array) > 0:
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                annualized_sharpe = (mean_return / std_return) * np.sqrt(self.steps_per_year)
            else:
                annualized_sharpe = 0.0

            self.logger.record("trading/final_portfolio_value", final_portfolio_value)
            self.logger.record("trading/annualized_sharpe_ratio", annualized_sharpe)

            self.episode_returns = []

        return True


def make_env(df, low_level_model_path, seed, custom_mean, custom_std):
    """
    Utility function for multiprocess environment creation.
    """
    def _init():
        env = HighLevelCryptoEnv(
            df=df, 
            low_level_model_path=low_level_model_path, 
            macro_step_freq=48,
            initial_balance=10000.0,
            custom_mean=custom_mean,
            custom_std=custom_std
        )
        env = Monitor(env)
        env.action_space.seed(seed)
        return env
    return _init


if __name__ == "__main__":
    print("loading data...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'btcusd_5-min_features.csv')
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True).sort_index()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)

    LOW_LEVEL_MODEL_PATH = "./model/sac_goal.zip"
    if not os.path.exists(LOW_LEVEL_MODEL_PATH):
        raise FileNotFoundError(f"Cannot find {LOW_LEVEL_MODEL_PATH}")
        
    OBS_MEAN_PATH = "./model/obs_mean.npy"
    OBS_STD_PATH = "./model/obs_std.npy"
    if not os.path.exists(OBS_MEAN_PATH) or not os.path.exists(OBS_STD_PATH):
        raise FileNotFoundError("cannot find obs_mean or obs_std")
        
    global_obs_mean = np.load(OBS_MEAN_PATH)
    global_obs_std = np.load(OBS_STD_PATH)

    print(f"Data volume: {len(df)} steps")
    print(f"Training set volume: {len(train_df)} steps")

    num_cpu = 8 
    vec_env = SubprocVecEnv([make_env(train_df, LOW_LEVEL_MODEL_PATH, i, global_obs_mean, global_obs_std) for i in range(num_cpu)])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # 5-min intervals: 105120 / 48 = 2190
    trading_callback = TradingMetricsCallback(steps_per_year=2190)

    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4, 
        batch_size=1024,
        buffer_size=500000,
        train_freq=(8, "step"),  # Train after every 8 collected macro steps
        gradient_steps=4,
        device="cuda",  # GPU
        verbose=1, 
        tensorboard_log="./tensorboard/sac_hiro/"
    )
    
    # train
    print("Training High-level Manager...")
    high_level_steps_per_epoch = len(train_df) // 48
    model.learn(
        total_timesteps=high_level_steps_per_epoch * 5,
        callback=trading_callback,
        log_interval=4
    )
    
    print("Training finished, saving models...")
    model.save("./model/sac_hiro")
    env.save("./model/vec_normalize_sac_hiro.pkl")