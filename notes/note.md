# Notes during Development

## Data Preprocess

- process: fetch data -> resample data -> build features
- fetch_data.py can be used to fetch data by API in the future. For now, we get history data from Kaggle.

1. We download [CSV files from Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) for "select bitcoin exchanges for the time period of Jan 2012 to Present (Measured by UTC day), with minute to minute updates of OHLC (Open, High, Low, Close) and Volume in BTC".

2. We use pandas[(doc)](pandas.pydata.org/pandas-docs/stable/) library to resample data from 1-minute to 5-minute.
   - About year range: Early data quality and market mechanisms were unstable. Bitcoin entered a more mature and liquid market phase after 2016, with improved data quality and more stable market microstructure. This period also covers multiple market regimes (bull and bear markets), which improves the robustness and generalisability of the model. So we select data starting from 2017.
   - About intervals: Although 1-minute data is available, it contains substantial microstructure noise and significantly increases computational complexity. Resampling to 5-minute intervals should provides a better trade-off between information density and model stability, which is more suitable for medium-frequency trading strategies.
   - Reinforcement learning requires continuous time steps without any gaps; otherwise, the state sequence of the environment will be disrupted, leading to abnormal training. Therefore, it is necessary to forward fill the missing K-lines with the price of the previous K-line to maintain the continuity of the time series.

3.

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
