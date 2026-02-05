import pandas


def main():
    # header of the csv: Timestamp,Open,High,Low,Close,Volume
    data = pandas.read_csv("btcusd_1-min_data.csv")
    data["Timestamp"] = pandas.to_datetime(data["timestamp"])
    data = data.set_index("Timestamp").sort_index()

    data_5min = data.resample("5T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()


if __name__ == "__main__":
    main()