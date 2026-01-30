# Use CoinCap API 3.0 to get data (https://pro.coincap.io/api-docs)

import sys
import os
import requests
import json


def main():
    crypto_ids = {
        'Bitcoin': 'bitcoin',
        'Ethereum': 'ethereum',
        'Tether': 'tether',
    }
    
    api_key = os.getenv("COINCAP_API_KEY")
    if not api_key:
        sys.exit("API key does not exist")
        
    all_data = {}

    try:
        for name, crypto_id in crypto_ids.items():
            url = f"https://rest.coincap.io/v3/assets/{crypto_id}?apiKey={api_key}"
            response = requests.get(url)
            response_dict = response.json()
            all_data[name] = response_dict
            print(json.dumps(response_dict, indent=4))
    except requests.RequestException:
        sys.exit("Request Error")


if __name__ == "__main__":
    main()