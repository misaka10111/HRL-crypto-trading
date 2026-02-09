## Data Preprocess
- fetch_data.py can be used to fetch data by API in the future. For now, we get history data from Kaggle.
1. We download [CSV files from Kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data) for "select bitcoin exchanges for the time period of Jan 2012 to Present (Measured by UTC day), with minute to minute updates of OHLC (Open, High, Low, Close) and Volume in BTC".
2. We use pandas[(doc)](pandas.pydata.org/pandas-docs/stable/) library to resample data from 1-minute to 5-minute.
- About year range: Early data quality and market mechanisms were unstable. Bitcoin entered a more mature and liquid market phase after 2016, with improved data quality and more stable market microstructure. This period also covers multiple market regimes (bull and bear markets), which improves the robustness and generalisability of the model. So we select data starting from 2017.
- About intervals: Although 1-minute data is available, it contains substantial microstructure noise and significantly increases computational complexity. Resampling to 5-minute intervals should provides a better trade-off between information density and model stability, which is more suitable for medium-frequency trading strategies.

## Environment
1. try Gymnasium with a simple example using CartPole (current code is copied from https://gymnasium.farama.org/introduction/basic_usage/ for study purpose)
2. learn to train an Q-learning agent with [Gymnasium's sample code](https://gymnasium.farama.org/introduction/train_agent/) for Blackjack