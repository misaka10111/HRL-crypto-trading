# Use CoinCap API 3.0 to get data (https://pro.coincap.io/api-docs)

import sys
import requests
import json


def main():
    try:
        # Bitcoin data
        response = requests.get("https://rest.coincap.io/v3/assets/bitcoin?apiKey=6f077700b46dfbfee732c95d8aea6f8318369fcb1d766d403cc84d47c05b4fb1")
        response_dict = response.json()
        print(response_dict)
    except requests.RequestException:
        print("Request Error")


if __name__ == "__main__":
    main()