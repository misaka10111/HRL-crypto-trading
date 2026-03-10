import numpy as np


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