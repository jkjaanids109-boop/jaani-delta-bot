import os
import time
import requests

# Sabhi secrets se faltu space hatana
TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()

def get_candles(symbol="BTCUSD", resolution="15m", limit=30):
    end_time = int(time.time())
    start_time = end_time - (limit * 15 * 60)
    urls = [
        "https://api.india.delta.exchange/v2/history/candles",
        "https://cdn-ind.testnet.deltaex.org/v2/history/candles",
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
            res = requests.get(url, params=params, timeout=12)
            if res.status_code == 200:
                data = res.json()
                if data.get("success") and data.get("result"):
                    return data["result"][-15:]
        except Exception:
            continue
    return None

def calculate_fallback_analysis(candles, ai_error=""):
    latest = candles[-1]
    first = candles[0]
    current_price = float(latest["close"])
    open_price = float(first["open"])
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    resistance = max(highs)
    support = min(lows)

    if current_price > open_price * 1.002:
        trend = "Bullish (Tezi)"
    elif current_price < open_price * 0.998:
        trend = "Bearish (Mandi)"
    else:
        trend = "Sideways (Range-bound)"

    report = (
        f"1. Current Price: ${current_price:,.2f} | Trend: {trend}\n"
        f"2. Key Support: ${support:,.2f} | Key Resistance: ${resistance:,.2f}\n"
        f"3. Trade Setup: WAIT (Price level check karein — Support ke paas Buy aur Resistance ke paas Sell dekhein)"
    )
    if ai_error:
        report += f"\n\n[Note: AI Key check karein — {ai_error}]"
    return report

def analyze_with_gemini(candles):
    if not GEMINI_API_KEY:
        return calculate_fallback_analysis(candles, "GEMINI_API_KEY khali hai")

    prompt = f"""
    Tum ek expert crypto aur price-action trader ho. Niche BTCUSD (15-minute timeframe) ki pichhli 15 candles ka data diya gaya hai:
    {candles}

    Is data ka analysis karke sirf 5-6 chhoti lines mein Hinglish mein crisp update do (kisi bhi word mein * ya _ ka use mat karna):
    1. Current Price & Trend (Bullish / Bearish / Sideways)
    2. Key Support & Resistance levels
    3. Trade Setup (BUY / SELL / WAIT) — agar clear setup ho tabhi Entry, Stop Loss aur Target batao, warna 'WAIT (No clear setup)' likho.
    """
    models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
    last_error = ""

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            res = requests.post(url, json=payload, timeout=25)
            data = res.json()
            if "candidates" in data and data["candidates"]:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            last_error = data.get("error", {}).get("message", "Unknown API response")
        except Exception as e:
            last_error = str(e)

    # Agar AI fail ho jaye tab bhi bot crash nahi hoga, technical report bhejega
    return calculate_fallback_analysis(candles, last_error)

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_TOKEN ya TELEGRAM_CHAT_ID missing hai!")

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    full_text = f"📊 JAANI BHAI LIVE MARKET UPDATE (BTCUSD 15M)\n\n{message}"

    res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": full_text}, timeout=15)
    if res.status_code != 200:
        raise RuntimeError(f"Telegram Error: {res.text}")

if __name__ == "__main__":
    candles = get_candles("BTCUSD", "15m", 30)
    if candles:
        analysis = analyze_with_gemini(candles)
        send_telegram(analysis)
        print("Success! Telegram par message bhej diya gaya hai.")
    else:
        send_telegram("⚠️ Delta Exchange se abhi candle data nahi mil paya, agle 15 min mein fir check hoga.")
        print("Delta Exchange data unavailable.")
