import os
import time
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def get_candles(symbol="BTCUSD", resolution="15m", limit=30):
    end_time = int(time.time())
    start_time = end_time - (limit * 15 * 60)
    urls = [
        "https://cdn-ind.testnet.deltaex.org/v2/history/candles",
        "https://api.india.delta.exchange/v2/history/candles",
        "https://api.delta.exchange/v2/history/candles"
    ]
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "start": start_time,
        "end": end_time
    }
    for url in urls:
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if data.get("success") and data.get("result"):
                return data["result"][-15:]
        except Exception:
            continue
    return None

def analyze_with_gemini(candles):
    prompt = f"""
    Tum ek expert crypto aur price-action trader ho. Niche BTCUSD (15-minute timeframe) ki pichhli 15 candles ka data (Time, Open, High, Low, Close, Volume) diya gaya hai:
    {candles}

    Is data ka analysis karke sirf 5-6 chhoti lines mein Hinglish mein crisp update do:
    1. Current Price & Trend (Bullish / Bearish / Sideways)
    2. Key Support & Resistance levels
    3. Trade Setup (BUY / SELL / WAIT) — agar clear setup ho tabhi Entry, Stop Loss aur Target batao, warna 'WAIT (No clear setup)' likho.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    res = requests.post(url, json=payload, timeout=25)
    data = res.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"📊 *JAANI BHAI AI MARKET UPDATE (BTCUSD 15M)*\n\n{message}",
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload, timeout=15)

if __name__ == "__main__":
    candles = get_candles("BTCUSD", "15m", 30)
    if candles:
        analysis = analyze_with_gemini(candles)
        send_telegram(analysis)
        print("Alert successfully bhej diya gaya!")
    else:
        print("Delta Exchange se data nahi mil paya.")
