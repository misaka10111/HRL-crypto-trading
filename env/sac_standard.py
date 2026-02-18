import os
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


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
            step_reward = log_return * 100.0  
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