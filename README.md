# Hierarchical Reinforcement Learning for Mid-Frequency Cryptocurrency Trading

The project utilizes the HIRO (Data-Efficient Hierarchical RL) framework:

- High-level Policy (Manager): Operates at a macro frequency (e.g., every 4 hours) to set target portfolio weights (goals) based on market trends.
- Low-level Policy (Worker): Operates at a micro frequency (every 5 minutes) to reach the manager's target while minimizing tracking error and transaction costs.
- Soft Actor-Critic (SAC): Used as the base off-policy algorithm for both levels to ensure robust exploration via maximum entropy optimization.

See online simulation: <https://hrl-crypto-trading.streamlit.app/>

## Project Structure

- data_preprocess/: scripts for data ingestion and feature engineering
- env/: Gymnasium environments for different RL architectures
- trading/: simulation tools
- model/: storing trained model weights (.zip) and VecNormalize statistics (.pkl).

## Usage

1. setup python virtual environment

    ```bash
    py -3.13 -m venv .venv  # create venv
    pip install -r requirements-for-install.txt # install dependencies
    ```

2. if you want to train model or run a backtest:
   1. download data from [kaggle](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
   2. run data_preprocess/resample_data.py to resample data
   3. run data_preprocess/build_features.py to get featured data
3. if you want to host your own website to see simulation:
   1. use [streamlit](https://docs.streamlit.io) as frontend
   2. use [supabase](https://supabase.com/) as database
   3. add project website and private key to streamlit
   4. run trading/simu_trade.py to generate data of simulated trading
   5. access your streamlit website
