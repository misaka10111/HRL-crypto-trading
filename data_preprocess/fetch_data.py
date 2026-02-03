# Use CoinCap API 3.0 to get data (https://pro.coincap.io/api-docs)

import sys
import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

DAYS_TO_FETCH = 0.01  # 1 interval for testing
DATA_INTERVAL = "m15"  # mid-frequency


def main():
    load_dotenv()  # load .env

    cryptos = ["bitcoin", "ethereum", "tether"]

    # Time
    current_time = datetime.now()
    start_time = get_ms(current_time - timedelta(days=DAYS_TO_FETCH))
    end_time = get_ms(current_time)
    # print(start_time)
    # print(end_time)

    # API
    api_key = os.getenv("COINCAP_API_KEY")
    if not api_key:
        sys.exit("API key does not exist")

    # Request
    request_headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    request_parameters = {
        "interval": DATA_INTERVAL,
        "start": start_time,
        "end": end_time,
    }

    try:
        for c in cryptos:
            response = requests.get(
                get_url(c),
                headers=request_headers,
                params=request_parameters,
            )
            response_json = response.json()
            print(json.dumps(response_json, indent=4))
    except requests.RequestException:
        sys.exit("Request Error")


# Used for API request
def get_ms(dt: datetime):
    return int(dt.timestamp() * 1000)


def get_url(crypto_id: str):
    return f"https://rest.coincap.io/v3/assets/{crypto_id}/history"


if __name__ == "__main__":
    main()
