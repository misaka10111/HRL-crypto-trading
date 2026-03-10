import os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def calculate_metrics(equity_curve, steps_per_year):
    curve = np.array(equity_curve)
    # calculate single step returns
    returns = np.diff(curve) / curve[:-1]
    
    # total return
    total_return = (curve[-1] - curve[0]) / curve[0] * 100
    
    # annualized sharpe ratio
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(steps_per_year)
    else:
        sharpe = 0.0
        
    # maximum drawdown
    running_max = np.maximum.accumulate(curve)
    drawdowns = (running_max - curve) / running_max
    max_drawdown = np.max(drawdowns) * 100
    
    return total_return, sharpe, max_drawdown

def run_backtest(env_class, model_path, vec_norm_path, df, env_kwargs, is_hrl=False):
    """
    General backtesting execution function
    """
    print(f"\nbacktesting model: {os.path.basename(model_path)} ...")
    
    # initialize environment
    base_env = env_class(df=df, **env_kwargs)
    vec_env = DummyVecEnv([lambda: base_env])

    # load normalization wrapper and freeze parameters
    env = VecNormalize.load(vec_norm_path, vec_env)
    env.training = False
    env.norm_reward = False

    # load model
    model = SAC.load(model_path)

    # execute inference
    obs = env.reset()
    done = False
    
    dates = []
    portfolio_values = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info_list = env.step(action)
        
        info = info_list[0]
        step_idx = info.get('step', 0)
        current_value = info.get('portfolio_value', base_env.initial_balance)
        
        # prevent step overflow of df index
        safe_idx = min(step_idx, len(df) - 1)
        dates.append(df.index[safe_idx])
        portfolio_values.append(current_value)

    # calculate metrics
    # hrl is 4 hours per step, single layer is 5 mins per step
    steps_per_yr = 2190 if is_hrl else 105120 
    metrics = calculate_metrics(portfolio_values, steps_per_yr)
    
    return dates, portfolio_values, metrics
