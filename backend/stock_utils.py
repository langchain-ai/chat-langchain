import requests
from typing import Optional, List
from backend.constants import STOCK_API_URL

# Replace with your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
STOCK_API_KEY="YOUR_STOCK_API_KEY" #for test only
# This list should contain all the stock symbols you want to recognize
VALID_STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
# Add more symbols as needed


def is_stock_symbol(text: str) -> bool:
    return text.upper() in VALID_STOCK_SYMBOLS


def fetch_stock_data_from_api_with_header(symbol: str) -> Optional[dict]:
    try:
        response = requests.get(
            f"{STOCK_API_URL}/quote",
            params={"symbol": symbol},
            headers={"X-API-KEY": STOCK_API_KEY},
        )
        response.raise_for_status()
        data = response.json()
        return {
            "symbol": data.get("symbol"),
            "name": data.get("companyName"),
            "price": data.get("latestPrice"),
            "change": data.get("changePercent"),
            "volume": data.get("volume"),
            "market_cap": data.get("marketCap")
        }
    except requests.RequestException:
        return None


def fetch_stock_data_from_api(symbol: str) -> Optional[dict]:
    try:
        response = requests.get(f"{STOCK_API_URL}/stock", params={"symbol": symbol})
        response.raise_for_status()
        data = response.json()
        return {
            "symbol": data.get("symbol"),
            "name": data.get("companyName"),
            "price": data.get("latestPrice"),
            "change": data.get("changePercent"),
            "volume": data.get("volume"),
            "market_cap": data.get("marketCap")
        }
    except requests.RequestException:
        return None



def extract_and_fetch_stock_data(query: str) -> List[dict]:
    words = query.upper().split()
    stock_data = []
    for word in words:
        if is_stock_symbol(word):
            data = fetch_stock_data_from_api(word)
            if data:
                stock_data.append(data)
    return stock_data

def format_stock_info(stock_data: List[dict]) -> str:
    if not stock_data:
        return ""
    
    stock_info = "Stock information:\n"
    for stock in stock_data:
        stock_info += (f"Symbol: {stock['symbol']}, "
                       f"Price: ${stock['price']:.2f}, "
                       f"Change: {stock['change']:.2f}%, "
                       f"Volume: {stock['volume']:,}\n")
    return stock_info
