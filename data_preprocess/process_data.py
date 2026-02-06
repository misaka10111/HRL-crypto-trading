import pandas


def main():
    # header of the csv: Timestamp,Open,High,Low,Close,Volume
    data = pandas.read_csv("btcusd_1-min_data.csv")
    data["Timestamp"] = pandas.to_datetime(data["timestamp"])
    data = data.set_index("Timestamp").sort_index()

    # Slice (start from 2017)
    data = data.loc["2017-01-01":]

    # Resample
    data.resample("5T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna().to_csv("btcusd_5-min_data.csv")


if __name__ == "__main__":
    main()