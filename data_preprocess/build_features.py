import pandas as pd
import pandas_ta as ta
import numpy as np

def main():
    input_file = "./data_preprocess/btcusd_5-min_data.csv"
    output_file = "./data_preprocess/btcusd_5-min_features.csv"
    data = pd.read_csv(input_file, index_col="Datetime", parse_dates=True)
    data.index = pd.to_datetime(data.index)

    # RSI (Momentum)
    data.ta.rsi(length=14, append=True)

    # MACD (Trend)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Bollinger Bands (Volatility)
    data.ta.bbands(length=20, std=2, append=True)
    # BBL, BBM, BBU are absolute price, only keep BBP BBB
    cols_to_drop = [col for col in data.columns if col.startswith(('BBL', 'BBM', 'BBU'))]

    # Log Return (as part of State)
    data.ta.log_return(length=1, append=True)
    data['Vol_Change'] = data['Volume'].pct_change()
    data['High_Low_Spread'] = (data['High'] - data['Low']) / data['Close']

    # Time
    time_of_day = data.index.hour + data.index.minute / 60.0
    data['Time_Sin'] = np.sin(2 * np.pi * time_of_day / 24.0)
    data['Time_Cos'] = np.cos(2 * np.pi * time_of_day / 24.0)
    data['Day_Sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7.0)
    data['Day_Cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7.0)

    # drop non-stationary features
    cols_to_drop.extend(['Open', 'High', 'Low', 'Volume'])
    data = data.drop(columns=cols_to_drop)

    # drop NaN
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    print(f"Generated Features: {data.columns.tolist()}")

    # Save
    data.to_csv(output_file)
    print(f"Saved shape: {data.shape} to {output_file}")

if __name__ == "__main__":
    main()