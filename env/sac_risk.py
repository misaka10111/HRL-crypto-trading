import os
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

class SacRiskAware(gym.Env):
    """
    Risk-Aware Gymnasium environment for SB3
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000.0, max_steps=2000):
        super(SacRiskAware, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.max_steps = max_steps 
        self.num_crypto_assets = 1
        self.total_assets = self.num_crypto_assets + 1  # BTC + USDT
        self.commission_fee_percent = 0.001  # 0.1%
        
        self.price_data = self.df['Close'].values
        self.feature_data = self.df.drop(columns=['Close']).values
        
        # Observation Space
        num_features = self.feature_data.shape[1]
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
        self.start_step = 0
        self.current_step = 0
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        # Risk Variables
        self.peak_portfolio_value = self.initial_balance
        self.risk_penalty_weight = 0.5
        self.previous_drawdown = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        max_start_idx = len(self.df) - self.max_steps - 1
        self.start_step = np.random.randint(0, max_start_idx) if max_start_idx > 0 else 0
        self.current_step = self.start_step
        
        self.cash_balance = self.initial_balance
        self.crypto_holdings = np.zeros(self.num_crypto_assets)
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        self.peak_portfolio_value = self.initial_balance
        self.previous_drawdown = 0.0
        
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
            drawdown_delta = current_drawdown - self.previous_drawdown
            
            if drawdown_delta > 0:
                # Punishment is only given when the drawdowns worsens
                drawdown_penalty = self.risk_penalty_weight * drawdown_delta * 100.0
            else:
                drawdown_penalty = 0.0
                
            # for next step
            self.previous_drawdown = current_drawdown
            
            # final reward
            step_reward = base_reward - drawdown_penalty
                
        else:
            step_reward = -1.0

        # termination
        steps_taken = self.current_step - self.start_step
        terminated = steps_taken >= self.max_steps or self.current_step >= len(self.df) - 1
        truncated = False

        if self.portfolio_value < self.initial_balance * 0.1:
            terminated = True
            step_reward -= 100.0 # heavy penalty for bankruptcy

        return self._get_observation(), step_reward, terminated, truncated, self._get_info()

    def _get_observation(self):
        return self.feature_data[self.current_step].reshape(self.num_crypto_assets, -1).astype(np.float32)

    def _get_info(self):
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "max_drawdown": (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        }
    
    def _get_current_prices(self):
        return np.array([self.price_data[self.current_step]], dtype=np.float32)


class TradingMetricsCallback(BaseCallback):
    def __init__(self, steps_per_year=105120, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.steps_per_year = steps_per_year
        self.episode_returns = []
        self.last_portfolio_value = 10000.0
        self.drawdowns = []

    def _on_step(self) -> bool:
        step_reward = self.locals.get("rewards", [0.0])[0]
        done = self.locals.get("dones", [False])[0]
        infos = self.locals.get("infos", [{}])[0]

        self.episode_returns.append(step_reward)
        
        if "max_drawdown" in infos:
            self.drawdowns.append(infos["max_drawdown"])

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
                
            max_dd = max(self.drawdowns) if self.drawdowns else 0.0

            self.logger.record("trading/final_portfolio_value", final_portfolio_value)
            self.logger.record("trading/annualized_sharpe_ratio", annualized_sharpe)
            self.logger.record("trading/max_drawdown", max_dd)

            self.episode_returns = []
            self.drawdowns = []

        return True
    
def make_env(df, seed):
    def _init():
        env = SacRiskAware(df, initial_balance=10000.0, max_steps=2000)
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

    # training set: 80%
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"data volume: {len(df)} steps")
    print(f"training set volume: {len(train_df)} steps ; testing set volume: {len(test_df)} steps")

    num_cpu = 8 
    vec_env = SubprocVecEnv([make_env(train_df, i) for i in range(num_cpu)])
    
    # Automatically normalize features and rewards
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # training
    trading_callback = TradingMetricsCallback(steps_per_year=105120)
    
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=3e-4,
        batch_size=1024,
        buffer_size=500000,
        train_freq=(8, "step"),
        gradient_steps=4,
        device="cuda", 
        verbose=1, 
        tensorboard_log="./tensorboard/sac_risk_aware/"
    )
    
    print("training...")
    model.learn(
        total_timesteps=len(train_df) * 3, 
        callback=trading_callback,
        log_interval=4
    )
    
    # save
    print("training finished, saving...")
    model.save("./model/sac_risk")
    env.save("./model/vec_normalize_sac_risk.pkl")