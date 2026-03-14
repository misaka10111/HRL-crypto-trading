import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sac_standard import SacStandard
from sac_risk import SacRiskAware
from sac_hiro import HighLevelCryptoEnv


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
        action, _ = model.predict(obs, deterministic=True) # type: ignore
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


if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR = os.path.join(ROOT_DIR, 'model')
    
    # 20% test set
    print("loading test data...")
    data_path = "./env/btcusd_5-min_features.csv"
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True).sort_index()

    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # restore timestamp index for plotting
    test_df.index = df.index[split_idx:] 
    print(f"test set volume: {len(test_df)} steps ; timespan: {test_df.index[0]} to {test_df.index[-1]}")

    INITIAL_BALANCE = 10000.0
    results = {}

    # backtest 1: standard SAC
    try:
        d1, v1, m1 = run_backtest(
            env_class=SacStandard,
            model_path=os.path.join(BASE_DIR, "standard_sac_2.zip"),
            vec_norm_path=os.path.join(BASE_DIR, "vec_normalize_sac_2.pkl"),
            df=test_df,
            env_kwargs={'initial_balance': INITIAL_BALANCE, 'is_eval': True}
        )
        results['Standard SAC'] = {'dates': d1, 'values': v1, 'metrics': m1}
    except Exception as e:
        print(f"standard SAC backtest failed or skipped: {e}")

    # backtest 2: risk-aware SAC
    try:
        d2, v2, m2 = run_backtest(
            env_class=SacRiskAware, 
            model_path=os.path.join(BASE_DIR, "sac_risk_2.zip"),
            vec_norm_path=os.path.join(BASE_DIR, "vec_normalize_sac_risk_2.pkl"),
            df=test_df,
            env_kwargs={'initial_balance': INITIAL_BALANCE, 'is_eval': True}
        )
        results['Risk-Aware SAC'] = {'dates': d2, 'values': v2, 'metrics': m2}
    except Exception as e:
        print(f"risk-aware SAC backtest failed or skipped: {e}")

    # backtest 3: HRL
    try:
        d4, v4, m4 = run_backtest(
            env_class=HighLevelCryptoEnv,
            model_path=os.path.join(BASE_DIR, "sac_hiro.zip"),
            vec_norm_path=os.path.join(BASE_DIR, "vec_normalize_sac_hiro.pkl"),
            df=test_df,
            env_kwargs={
                'low_level_model_path': os.path.join(BASE_DIR, "sac_goal.zip"),
                'macro_step_freq': 48,
                'initial_balance': INITIAL_BALANCE,
                'is_eval': True
            },
            is_hrl=True
        )
        results['HRL'] = {'dates': d4, 'values': v4, 'metrics': m4}
    except Exception as e:
        print(f"HRL backtest failed: {e}")

    # calculate another baseline (buy & hold)
    print("\ncalculating buy & hold baseline...")
    btc_prices = test_df['Close'].values
    buy_amount = INITIAL_BALANCE / btc_prices[0]
    bh_values = buy_amount * btc_prices
    bh_metrics = calculate_metrics(bh_values, 105120)
    results['Buy & Hold BTC'] = {'dates': test_df.index, 'values': bh_values, 'metrics': bh_metrics}

    # print table
    print("\n" + "="*65)
    print(f"{'Strategy':<20} | {'Total Return':>12} | {'Sharpe Ratio':>12} | {'Max Drawdown':>12}")
    print("-" * 65)
    for name, data in results.items():
        ret, sharpe, mdd = data['metrics']
        print(f"{name:<20} | {ret:>11.2f}% | {sharpe:>12.2f} | {mdd:>11.2f}%")
    print("="*65)

    # plot equity curve comparison
    plt.figure(figsize=(14, 7))
    plt.title("Out-of-Sample Backtest Equity Curve Comparison", fontsize=16)
    
    colors = {
        'Standard SAC': 'red',
        'Risk-Aware SAC': 'orange',
        'HRL': 'green',
        'Buy & Hold BTC': 'blue'
    }

    for name, data in results.items():
        plt.plot(data['dates'], data['values'], label=name, color=colors.get(name, 'black'), alpha=0.8, linewidth=1.5)

    plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', alpha=0.5, label='Initial Balance')
    
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Portfolio Value (USD)", fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # save
    save_path = "./figures/backtest.png"
    plt.savefig(save_path, dpi=300)
    print(f"\ncomparison chart saved to: {save_path}")
    plt.show()