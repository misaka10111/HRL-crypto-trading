import os
import sys
import pickle
import numpy as np
import ccxt
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