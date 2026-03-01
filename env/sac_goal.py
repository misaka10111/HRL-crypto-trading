import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor



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
    
    def step(self, action: np.ndarray):
        # Simulate high-level intervention
        if self.current_step > 0 and self.current_step % self.goal_change_freq == 0:
            self.current_goal_weights = self._sample_random_goal()

        # Parse the low-level Agent's execution action
        execution_weights = np.clip(action, 0.0, 1.0)
        weight_sum = np.sum(execution_weights)
        if weight_sum > 0:
            execution_weights = execution_weights / weight_sum
        else:
            execution_weights = np.zeros(self.total_assets)
            execution_weights[0] = 1.0

        current_prices = self._get_current_prices()
        current_crypto_values = self.crypto_holdings * current_prices
        
        # Compute rebalancing amounts and execute trades
        target_values = execution_weights * self.portfolio_value
        target_crypto_values = target_values[1:] 
        crypto_value_diffs = target_crypto_values - current_crypto_values

        trade_cost_fiat = 0.0 # absolute friction cost

        # Sell
        for i in range(self.num_crypto_assets):
            if crypto_value_diffs[i] < 0:
                trade_amount_fiat = abs(crypto_value_diffs[i])
                crypto_to_sell = trade_amount_fiat / current_prices[i]
                crypto_to_sell = min(crypto_to_sell, self.crypto_holdings[i])
                
                gross_fiat = crypto_to_sell * current_prices[i]
                fee = gross_fiat * self.commission_fee_percent
                net_fiat = gross_fiat - fee
                
                self.crypto_holdings[i] -= crypto_to_sell
                self.cash_balance += net_fiat
                trade_cost_fiat += fee

        # Buy
        for i in range(self.num_crypto_assets):
            if crypto_value_diffs[i] > 0:
                trade_amount_fiat = crypto_value_diffs[i]
                trade_amount_fiat = min(trade_amount_fiat, self.cash_balance)
                
                if trade_amount_fiat > 0:
                    fee = trade_amount_fiat * self.commission_fee_percent
                    net_fiat = trade_amount_fiat - fee
                    crypto_bought = net_fiat / current_prices[i]
                    
                    self.cash_balance -= trade_amount_fiat
                    self.crypto_holdings[i] += crypto_bought
                    trade_cost_fiat += fee

        # Update
        self.current_step += 1
        self.portfolio_value = self.cash_balance + np.sum(self.crypto_holdings * current_prices)

        # Reward
        actual_weights_after_trade = self._get_actual_weights(current_prices)
        
        # Penalty 1: Tracking error
        tracking_error = np.sum(np.abs(actual_weights_after_trade - self.current_goal_weights))
        
        # Tolerance threshold
        if tracking_error < 0.01:
            tracking_error = 0.0
        
        # Penalty 2: Transaction friction
        cost_penalty = trade_cost_fiat / max(self.portfolio_value, 1.0)
        
        # balance "closely tracking the goal" against "minimizing transaction fees"
        step_reward = - (tracking_error * 2.0) - (cost_penalty * 100.0)

        # Termination 
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        if self.portfolio_value < self.initial_balance * 0.1:
            terminated = True

        info = self._get_info()
        info["tracking_error"] = tracking_error

        return self._get_observation(), step_reward, terminated, truncated, info
    
def make_env(df, seed, custom_mean, custom_std):
    def _init():
        env = GoalConditionedCryptoEnv(df, initial_balance=10000.0, custom_mean=custom_mean, custom_std=custom_std)
        env = Monitor(env)
        env.action_space.seed(seed)
        return env
    return _init


if __name__ == "__main__":
    print("loading data...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'btcusd_5-min_features.csv')
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True)
    df = df.sort_index()

    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"data volume: {len(df)} steps")
    print(f"training set volume: {len(train_df)} steps ; testing set volume: {len(test_df)} steps")

    train_feature_df = train_df.drop(columns=['Close'])
    global_obs_mean = train_feature_df.mean().values.astype(np.float32)
    global_obs_std = train_feature_df.std().values.astype(np.float32)
    global_obs_std[global_obs_std == 0] = 1e-8

    num_cpu = 8 
    vec_env = SubprocVecEnv([make_env(train_df, i, global_obs_mean, global_obs_std) for i in range(num_cpu)])
    # norm_obs only used to normalize rewards and prevent gradient explosion
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        batch_size=1024,
        buffer_size=500000,
        train_freq=(8, "step"),  # train after every 8 collected steps
        gradient_steps=4,
        device="cuda",  # GPU
        verbose=1, 
        tensorboard_log="./sac_goal_crypto_tensorboard/"
    )
    
    print("training...")
    model.learn(
        total_timesteps=len(train_df) * 3, 
        log_interval=4
    )

    # save
    print("training finished, saving...")
    model.save("./model/goal_sac")
    env.save("./model/vec_normalize_sac_goal.pkl")
    np.save("obs_mean.npy", np.array(global_obs_mean))
    np.save("obs_std.npy", np.array(global_obs_std))