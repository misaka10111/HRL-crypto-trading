# Notes during Development

## Data Preprocess

- process: fetch data -> resample data -> build features
- fetch_data.py can be used to fetch data by API in the future. For now, we get history data from Kaggle.

1. We download [CSV files from Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) for "select bitcoin exchanges for the time period of Jan 2012 to Present (Measured by UTC day), with minute to minute updates of OHLC (Open, High, Low, Close) and Volume in BTC".
2. We use pandas[(doc)](pandas.pydata.org/pandas-docs/stable/) library to resample data from 1-minute to 5-minute.
   - About year range: Early data quality and market mechanisms were unstable. Bitcoin entered a more mature and liquid market phase after 2016, with improved data quality and more stable market microstructure. This period also covers multiple market regimes (bull and bear markets), which improves the robustness and generalisability of the model. So we select data starting from 2017.
   - About intervals: Although 1-minute data is available, it contains substantial microstructure noise and significantly increases computational complexity. Resampling to 5-minute intervals should provides a better trade-off between information density and model stability, which is more suitable for medium-frequency trading strategies.
   - Reinforcement learning requires continuous time steps without any gaps; otherwise, the state sequence of the environment will be disrupted, leading to abnormal training. Therefore, it is necessary to forward fill the missing K-lines with the price of the previous K-line to maintain the continuity of the time series.
3. Remove non-stationary features (absolute prices): Neural networks struggle to generalize when fed unbounded, shifting data. If we train an RL agent on raw Bitcoin prices ranging from 10,000 to 20,000, it will be completely lost when the price hits 70,000 because the input numbers are far outside its learned distribution. By converting raw prices into stationary features (like percentage returns, volatility bandwidths, or moving average convergence), we force the data to fluctuate within a stable, consistent range (e.g., mostly between -1 and 1). This allows the neural network to recognize market patterns regardless of the absolute price level.
4. Add cyclical time encoding: Raw time is linear. If we feed the model hours as numbers from 0 to 23, the neural network calculates the distance between 23:00 and 00:00 as a massive gap (23 - 0 = 23). By using cyclical encoding (sine and cosine transformations), we map the time onto a 2D circle.

## Environment

1. try Gymnasium with a simple example using CartPole (current code is copied from <https://gymnasium.farama.org/introduction/basic_usage/> for study purpose)
2. learn to train an Q-learning agent with [Gymnasium's sample code](https://gymnasium.farama.org/introduction/train_agent/) for Blackjack
3. [Custom Environment Skeleton](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/):
    - build a custom trading environment from scratch by inheriting `gymnasium.Env`.
    - use `spaces.Box` to define continuous state and action spaces.
    - standardizing the `reset()` method to return `(obs, info)` is strictly required by newer Gymnasium versions.
4. Core trading logic and reward design:
   - converting raw continuous actions from the agent into portfolio weights requires normalization (clipping `[0, 1]` and dividing by the sum) to ensure weights always sum to exactly 1
   - always execute "sell" actions before "buy" actions to free up cash balance, otherwise the environment might try to spend money it doesn't have
   - using log returns for step rewards is mathematically preferred over simple returns to maintain time-additivity, but need to care log of zero or negative values
5. Stable-Baselines3 callbacks and metrics logging:
   - custom callbacks (inheriting [`BaseCallback` from SB3](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html)) are the cleanest way to log domain-specific metrics (like Annualized Sharpe Ratio) to TensorBoard without polluting the core environment step logic
   - in vectorized environments, when an episode ends (`done=True`), the environment automatically resets. The `info` dict will contain the new state's info, not the final state's, so remember to extract the true final metrics from `infos["terminal_info"]`
6. Training pipeline and normalization (`VecNormalize`):
   - time-series data splitting must follow chronological order (first 80% train, last 20% test) to prevent look-ahead bias
   - wrap the environment in `VecNormalize` to keep observations and rewards centered, preventing gradient explosion or `NaN` errors during training; moving average statistics need to save alongside the model
7. Adding risk management to SAC:
   - pure return-seeking agents lack the concept of "capital preservation". In TensorBoard figures, they often take on massive uncompensated volatility (deeply negative Sharpe ratio) and speedrun bankruptcy by quickly crashing the portfolio to the 10% early termination threshold.
   - calculating the current drawdown from a continuously updated `peak_portfolio_value` and squaring it (`current_drawdown ** 2`) as a penalty.  It mathematically tolerates minor, normal market noise but heavily punishes severe drawdowns, forcing the agent to learn to cut losses.
   - state management: when reward calculations depend on historical variables (like the peak portfolio value), it is strictly required to manually reset them back to their defaults inside the `reset()` method. Forgetting this causes state leakage across episodes, severely confusing the agent with penalties carried over from a previous "lifetime".
8. Fix penalty problem: The previous absolute squared calculation caused two major issues. First, squaring a small fractional drawdown value (e.g., 0.001) resulted in a vanishingly small penalty (0.000001), effectively neutralizing the risk control signal for the neural network. Second, punishing the agent continuously for historical drawdowns, even when its current actions were successfully recovering the portfolio value, provided conflicting reward signals. By switching to a linear, delta-based penalty, the agent receives precise, proportionate, and immediate negative feedback solely for actions that directly cause the portfolio to shrink.
9. Why risk-aware model underperform the standard baseline:
   - the base state features (raw 5-min prices) lack strong predictive signals
   - when an environment is fundamentally unprofitable (high fees + noisy data), an aggressively penalized agent becomes terrified. Instead of learning a smart strategy, it realizes that any action leads to drawdown penalties.
   - risk management is just there to smooth the curve of an already profitable strategy
   - next step: use featured data to train the model
10. Goal-Conditioned SAC:
    - The environment simulates high-level instructions (target weights) via _sample_random_goal, while the SAC agent outputs actual execution weights.
    - Tracking error penalty: The absolute difference between actual and goal weights is penalized, with a tolerance threshold of 0.01.
    - Transaction friction: Trading fees relative to portfolio value are used as a penalty (scaled by 100) to discourage excessive rebalancing.

## Tensorboard Figures

- rollout
  - ep_len_mean: Average episode length per rollout, measures how many steps the agent takes per episode on average. An increase typically means the agent learned to survive longer; a decrease may indicate a policy shift or early termination.
  - ep_rew_mean: Average episode reward per rollout, the most important metric, reflecting overall agent performance. Ideally it rises steadily and converges; prolonged oscillation or decline suggests instability.
- time
  - fps: Training speed (frames per second), reflects computational efficiency — how many environment steps are processed per second. A sudden drop may indicate a computational bottleneck.
- trading
  - annualized_sharpe_ratio: Measures return per unit of risk. Higher is better; persistently negative values suggest the strategy takes excessive risk without sufficient return.
  - final_portfolio_value: Total asset value at the end of each episode, directly reflecting profit/loss. High variance suggests an unstable strategy; ideally it converges upward over time.
- train
  - actor_loss: Actor network loss, the optimization objective for the Actor network. Unlike supervised learning, Actor loss need not decrease monotonically; only persistent growth or violent oscillation is concerning.
  - critic_loss: Critic network loss, measures how accurately the Critic predicts Q-values. It's naturally high early on and should decrease and stabilize; persistently high values indicate inaccurate value estimation.
  - ent_coef: Entropy coefficient, SAC's automatic temperature parameter controlling policy randomness. It's typically larger early on to encourage exploration, and should gradually decrease as the policy matures.
  - ent_coef_loss: Entropy coefficient loss, the loss signal used to auto-tune the entropy coefficient, reflecting the gap between current and target policy entropy. Oscillation around zero is normal behavior.
  - learning_rate: Controls the step size of each parameter update. A fixed learning rate appears as a flat horizontal line; scheduled rates decay over time. Too large causes instability; too small causes slow convergence.
