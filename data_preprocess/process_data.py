import pandas


def main():
    # header of the csv: Timestamp,Open,High,Low,Close,Volume
    data = pandas.read_csv("btcusd_1-min_data.csv")
    data["Timestamp"] = pandas.to_datetime(data["timestamp"])
    data = data.set_index("Timestamp").sort_index()


if __name__ == "__main__":
    main()