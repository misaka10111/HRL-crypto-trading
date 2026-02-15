import pandas


def main():
    input_file = "btcusd_1-min_data.csv"
    output_file = "btcusd_5-min_data.csv"
    start_date = "2017-01-01"
    frequency = "5min"
    
    # header of the csv: Timestamp,Open,High,Low,Close,Volume
    data = pandas.read_csv(input_file)
    data["Datetime"] = pandas.to_datetime(data["Timestamp"], unit="s")
    data = data.set_index("Datetime").sort_index()

    # Slice (start from 2017)
    data = data.loc[start_date:]

    # Resample
    resampled = data.resample(frequency).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    # Forward fill missing Candlestick
    resampled["Close"] = resampled["Close"].ffill()
    resampled["Open"] = resampled["Open"].fillna(resampled["Close"])
    resampled["High"] = resampled["High"].fillna(resampled["Close"])
    resampled["Low"] = resampled["Low"].fillna(resampled["Close"])
    resampled["Volume"] = resampled["Volume"].fillna(0)
    
    resampled = resampled.dropna().to_csv(output_file)
    print(f"Saved shape: {data.shape} to {output_file}")


if __name__ == "__main__":
    main()