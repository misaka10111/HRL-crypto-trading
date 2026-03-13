import os
import sys
import csv
import time
import pickle
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
from datetime import datetime, timedelta, timezone
from stable_baselines3 import SAC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class SimulatedTrading:
    def __init__(self, symbol='BTC/USDT', timeframe='5m', timeframe_minutes=5, initial_balance=10000.0, macro_step_freq=48):
        self.symbol = symbol
        self.timeframe = timeframe
        self.timeframe_minutes = timeframe_minutes
        self.macro_step_freq = macro_step_freq
        
        # Initialize Kraken exchange
        self.exchange = ccxt.kraken({
            'enableRateLimit': True,  # Rate limiting
        })
        
        # Paper trading account state
        self.portfolio_value = initial_balance
        self.cash_balance = initial_balance
        self.crypto_holdings = 0.0
        self.commission_fee_percent = 0.0016 # Kraken specific
        
        # State tracking
        self.current_step = 0
        self.current_goal_weights = np.array([1.0, 0.0], dtype=np.float32)
        
        self._load_models()

        # CSV logging
        self.log_file = os.path.join(BASE_DIR, "trading", "dry_run_log.csv")
        self._init_csv_log()

    def _init_csv_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp (UTC)", "Macro_Step", "Micro_Step", "BTC_Price", 
                    "Total_Portfolio_Value", "Cash_Balance", "BTC_Holdings",
                    "Target_Cash_Pct", "Target_BTC_Pct",
                    "Actual_Cash_Pct", "Actual_BTC_Pct", "Trade_Action"
                ])

    def _log_state_to_csv(self, timestamp_str, current_price, trade_action_msg):
        actual_weights = self.get_actual_weights(current_price)
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp_str,
                self.current_step // self.macro_step_freq,
                self.current_step,
                f"{current_price:.2f}",
                f"{self.portfolio_value:.2f}",
                f"{self.cash_balance:.2f}",
                f"{self.crypto_holdings:.6f}",
                f"{self.current_goal_weights[0]:.4f}",
                f"{self.current_goal_weights[1]:.4f}",
                f"{actual_weights[0]:.4f}",
                f"{actual_weights[1]:.4f}",
                trade_action_msg
            ])

    def _load_models(self):
        print("Loading HRL models and normalization parameters...")
        model_dir = os.path.join(BASE_DIR, "model")
        
        # Low-level model
        self.low_level_model = SAC.load(os.path.join(model_dir, "sac_goal"))
        self.ll_obs_mean = np.load(os.path.join(model_dir, "obs_mean.npy"))
        self.ll_obs_std = np.load(os.path.join(model_dir, "obs_std.npy"))
        self.num_market_features = len(self.ll_obs_mean)
        
        # High-level model
        self.high_level_model = SAC.load(os.path.join(model_dir, "sac_hiro_2"))
        vec_norm_path = os.path.join(model_dir, "vec_normalize_sac_hiro_2.pkl")
        with open(vec_norm_path, "rb") as f:
            vec_env = pickle.load(f)
            self.hl_obs_mean = vec_env.obs_rms.mean
            self.hl_obs_var = vec_env.obs_rms.var
            self.hl_obs_std = np.sqrt(self.hl_obs_var + 1e-8)
            
        print("Models loaded successfully.")

    def _wait_for_next_candle(self):
        now = datetime.now(timezone.utc)
        next_minute = ((now.minute // self.timeframe_minutes) + 1) * self.timeframe_minutes
        # Delay trigger by 5 seconds for buffer
        next_run = now.replace(minute=0, second=5, microsecond=0) + timedelta(minutes=next_minute)
        sleep_seconds = (next_run - now).total_seconds()
        
        print(f"Waiting for next candle close (Expected fetch time: {next_run.strftime('%H:%M:%S')} UTC) ...")
        time.sleep(max(0, sleep_seconds))

    # Feature engineering
    def _calculate_features(self, df):
        df = df.copy()
        df['Datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('Datetime', inplace=True)

        # Technical indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.log_return(length=1, append=True)
        
        df['Vol_Change'] = df['Volume'].pct_change()
        df['High_Low_Spread'] = (df['High'] - df['Low']) / df['Close']

        # Time cycle features
        time_of_day = df.index.hour + df.index.minute / 60.0
        df['Time_Sin'] = np.sin(2 * np.pi * time_of_day / 24.0)
        df['Time_Cos'] = np.cos(2 * np.pi * time_of_day / 24.0)
        df['Day_Sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
        df['Day_Cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)

        # Remove invalid columns and non-stationary features
        cols_to_drop = [col for col in df.columns if col.startswith(('BBL', 'BBM', 'BBU'))]
        cols_to_drop.extend(['timestamp', 'Open', 'High', 'Low', 'Volume', 'Close'])
        cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df = df.drop(columns=cols_to_drop)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df) == 0:
            raise ValueError("Insufficient K-line data, resulting in empty data after removing NaNs.")

        latest_features = df.iloc[-1].values.astype(np.float32)
        
        if len(latest_features) != self.num_market_features:
            print(f"Feature dimension mismatch... Model expects {self.num_market_features} dimensions, actual is {len(latest_features)} dimensions")
            
        return latest_features
    
    def get_realtime_features(self, max_retries=3, retry_delay=5):
        for attempt in range(max_retries):
            try:
                # Set limit to 200 to ensure MACD and BBands have enough preceding data for calculation
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=200)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                current_price = df['Close'].iloc[-1]

                raw_features = self._calculate_features(df)
                return raw_features, current_price
            
            except Exception as e:
                print(f"[Network Error] Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                else:
                    raise

        raw_features = self._calculate_features(df)
        return raw_features, current_price

    def get_actual_weights(self, current_price):
        crypto_value = self.crypto_holdings * current_price
        self.portfolio_value = self.cash_balance + crypto_value
        if self.portfolio_value <= 0:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([self.cash_balance / self.portfolio_value, crypto_value / self.portfolio_value], dtype=np.float32)

    def execute_trade_simulation(self, execution_weights, current_price):
        target_crypto_value = execution_weights[1] * self.portfolio_value
        current_crypto_value = self.crypto_holdings * current_price
        value_diff = target_crypto_value - current_crypto_value
        trade_msg = "Hold"
        
        # Minimum trade threshold to avoid frequent friction fees
        if abs(value_diff) >= 10.0:
            if value_diff < 0:
                trade_amount = abs(value_diff)
                crypto_to_sell = min(trade_amount / current_price, self.crypto_holdings)
                gross_fiat = crypto_to_sell * current_price
                fee = gross_fiat * self.commission_fee_percent
                
                self.crypto_holdings -= crypto_to_sell
                self.cash_balance += (gross_fiat - fee)
                trade_msg = f"Sold {crypto_to_sell:.6f} BTC"
                print(f"[Virtual Execution] {trade_msg} | Avg Price: {current_price:.2f} | Fee: ${fee:.2f}")
                
            elif value_diff > 0:
                trade_amount = min(value_diff, self.cash_balance)
                if trade_amount > 0:
                    fee = trade_amount * self.commission_fee_percent
                    net_fiat = trade_amount - fee
                    crypto_bought = net_fiat / current_price
                    
                    self.cash_balance -= trade_amount
                    self.crypto_holdings += crypto_bought
                    trade_msg = f"Bought {crypto_bought:.6f} BTC"
                    print(f"[Virtual Execution] {trade_msg} | Avg Price: {current_price:.2f} | Fee: ${fee:.2f}")

        self.portfolio_value = self.cash_balance + (self.crypto_holdings * current_price)
        return trade_msg

    def run_step(self):
        current_time_str = datetime.now(timezone.utc).strftime('%H:%M:%S')
        print(f"\n[{current_time_str} UTC] --- Macro: {self.current_step // self.macro_step_freq} | Micro: {self.current_step} ---")        
        
        try:
            raw_features, current_price = self.get_realtime_features()
        except Exception as e:
            print(f"[Network Error] Failed to fetch data: {e}")
            self.current_step += 1  # align to the macro cycle
            return 
            
        actual_weights = self.get_actual_weights(current_price)
        
        # High-level decision
        if self.current_step % self.macro_step_freq == 0:
            hl_raw_obs = np.concatenate([raw_features, actual_weights])
            hl_norm_obs = np.clip((hl_raw_obs - self.hl_obs_mean) / self.hl_obs_std, -10.0, 10.0)
            macro_action, _ = self.high_level_model.predict(hl_norm_obs, deterministic=True)
            goal_weights = np.clip(macro_action, 0.0, 1.0)
            weight_sum = np.sum(goal_weights)
            self.current_goal_weights = goal_weights / weight_sum if weight_sum > 0 else np.array([1.0, 0.0])
            print(f"[High-level Manager] Set new macro goal: Cash {self.current_goal_weights[0]:.1%} | BTC {self.current_goal_weights[1]:.1%}")

        # Low-level execution
        ll_norm_features = (raw_features - self.ll_obs_mean) / self.ll_obs_std
        ll_obs = np.concatenate([ll_norm_features, actual_weights, self.current_goal_weights])
        
        execution_action, _ = self.low_level_model.predict(ll_obs, deterministic=True)
        exec_weights = np.clip(execution_action, 0.0, 1.0)
        exec_sum = np.sum(exec_weights)
        exec_weights = exec_weights / exec_sum if exec_sum > 0 else np.array([1.0, 0.0])
        
        trade_msg = self.execute_trade_simulation(exec_weights, current_price)
        
        actual_weights_post = self.get_actual_weights(current_price)
        print(f"[Total Asset] ${self.portfolio_value:.2f} | Current Allocation: Cash {actual_weights_post[0]:.1%} | BTC {actual_weights_post[1]:.1%}")
        
        # Log to CSV
        self._log_state_to_csv(current_time_str, current_price, trade_msg)

        self.current_step += 1

    def start_loop(self):
        print("Starting Kraken Dry Run simulation engine...\n")
        try:
            while True:
                self.run_step()
                self._wait_for_next_candle()
        except KeyboardInterrupt:
            print(f"\n\nSimulation engine terminated. Final total capital: ${self.portfolio_value:.2f}")

if __name__ == "__main__":
    trader = SimulatedTrading(symbol='BTC/USDT')
    trader.start_loop()