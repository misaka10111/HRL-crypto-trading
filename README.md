# Hierarchical Reinforcement Learning for Multi-Crypto Assets Mid-Frequency Trading

This project implements a hierarchical RL approach to manage a portfolio of crypto assets, decomposing the complex trading problem into:

- High-level policy: Strategic asset allocation and portfolio rebalancing
- Low-level policies: Execution of trades and market timing for individual assets


1. setup python virtual environment
    ```bash
    py -3.13 -m venv .venv  # create venv
    pip install -r requirements.txt # install dependencies
    ```
2. if you need to train model or run a backtest:
   1. download data from [kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
   2. run data_preprocess/resample_data.py to resample data
   3. run data_preprocess/build_features.py to get featured data
3. run trading/simu_trade.py to see simulated trading process