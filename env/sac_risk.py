import os
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class SacRiskAware(gym.Env):
    """
    Gymnasium environment, for SB3
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0):
        super(SacRiskAware, self).__init__()
        
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

        # Record the historical maximum asset for calculating the current drawdown
        self.peak_portfolio_value = self.initial_balance 
        # Risk penalty factor, controlling the model’s conservativeness
        self.risk_penalty_weight = 0.5

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        self.peak_portfolio_value = self.initial_balance  # Reset the historical maximum asset
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: np.ndarray):
        # Parse actions and normalize them
        target_weights = np.clip(action, 0.0, 1.0)
        weight_sum = np.sum(target_weights)
        if weight_sum > 0:
            target_weights = target_weights / weight_sum
        else:
            target_weights = np.zeros(self.total_assets)
            target_weights[0] = 1.0

        current_prices = self._get_current_prices()
        self.previous_portfolio_value = self.portfolio_value
        current_crypto_values = self.crypto_holdings * current_prices
        
        # trading
        target_values = target_weights * self.portfolio_value
        target_crypto_values = target_values[1:] 
        crypto_value_diffs = target_crypto_values - current_crypto_values

        # sell
        for i in range(self.num_crypto_assets):
            if crypto_value_diffs[i] < 0:
                trade_amount_fiat = abs(crypto_value_diffs[i])
                crypto_to_sell = trade_amount_fiat / current_prices[i]
                crypto_to_sell = min(crypto_to_sell, self.crypto_holdings[i])
                
                gross_fiat = crypto_to_sell * current_prices[i]
                net_fiat = gross_fiat * (1 - self.commission_fee_percent)
                
                self.crypto_holdings[i] -= crypto_to_sell
                self.cash_balance += net_fiat

        # buy
        for i in range(self.num_crypto_assets):
            if crypto_value_diffs[i] > 0:
                trade_amount_fiat = crypto_value_diffs[i]
                trade_amount_fiat = min(trade_amount_fiat, self.cash_balance)
                
                if trade_amount_fiat > 0:
                    net_fiat = trade_amount_fiat * (1 - self.commission_fee_percent)
                    crypto_bought = net_fiat / current_prices[i]
                    
                    self.cash_balance -= trade_amount_fiat
                    self.crypto_holdings[i] += crypto_bought

        # update
        self.current_step += 1
        self.portfolio_value = self.cash_balance + np.sum(self.crypto_holdings * current_prices)

        # calculate reward
        if self.previous_portfolio_value > 0:
            log_return = math.log(self.portfolio_value / self.previous_portfolio_value)
            base_reward = log_return * 100.0  
            
            # Drawdown Penalty
            # update the historical maximum asset
            if self.portfolio_value > self.peak_portfolio_value:
                self.peak_portfolio_value = self.portfolio_value
                
            # drawdown percentage
            current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
            
            # tolerate small drawdowns, averse to large drawdowns
            drawdown_penalty = self.risk_penalty_weight * (current_drawdown ** 2) * 100.0
            
            # final reward
            step_reward = base_reward - drawdown_penalty
            
        else:
            step_reward = -1.0

        # termination
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        if self.portfolio_value < self.initial_balance * 0.1:
            terminated = True

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

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


class TradingMetricsCallback(BaseCallback):
    def __init__(self, steps_per_year=105120, verbose=0):
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
    

if __name__ == "__main__":
    print("loading data...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'btcusd_5-min_data.csv')
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True)
    df = df.sort_index()

    # training set: 80%
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"data volume: {len(df)} steps")
    print(f"training set volume: {len(train_df)} steps ; testing set volume: {len(test_df)} steps")

    # Instantiate env
    base_env = SacRiskAware(train_df, initial_balance=10000.0)
    base_env = Monitor(base_env)
    vec_env = DummyVecEnv([lambda: base_env])
    
    # Automatically normalize features and rewards
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # training
    trading_callback = TradingMetricsCallback(steps_per_year=105120)
    
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        batch_size=256,
        verbose=1, 
        tensorboard_log="./tensorboard/sac_risk_aware/"
    )
    
    print("training...")
    model.learn(
        total_timesteps=len(train_df) * 3, 
        callback=trading_callback,
        log_interval=1
    )
    
    # save
    print("training finished, saving...")
    model.save("./model/sac_risk")
    env.save("./model/vec_normalize_sac_risk.pkl")