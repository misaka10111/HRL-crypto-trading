import pandas
import pandas_ta as ta

def main():
    input_file = "btcusd_5-min_data.csv"
    output_file = "btcusd_5-min_features.csv"
    data = pandas.read_csv(input_file, index_col="Datetime", parse_dates=True)
        
    # RSI (Momentum)
    data.ta.rsi(length=14, append=True)

    # MACD (Trend)
    data.ta.macd(fast=12, slow=26, signal=9, append=True)

    # Bollinger Bands (Volatility)
    data.ta.bbands(length=20, std=2, append=True)

    # Log Return (as part of State)
    data.ta.log_return(length=1, append=True)

    data = data.dropna()
    print(f"Generated Features: {data.columns.tolist()}")

    # Save
    data.to_csv(output_file)
    print(f"Saved shape: {data.shape} to {output_file}")

if __name__ == "__main__":
    main()