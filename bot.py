import os
import time
import requests

TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()

def get_candles(symbol="BTCUSD", resolution="15m", limit=45):
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
                    return data["result"][-40:]
        except Exception:
            continue
    return None

def build_actionable_signal(candles):
    current_price = float(candles[-1]["close"])
    prev_close = float(candles[-5]["close"])

    # Pichhle 10 ghante (40 candles) ke asli Swing High aur Swing Low
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    resistance = max(highs)
    support = min(lows)
    range_size = max(resistance - support, current_price * 0.005)

    # Trend check
    if current_price > prev_close * 1.0015:
        trend = "BULLISH (Tezi) 🟢"
    elif current_price < prev_close * 0.9985:
        trend = "BEARISH (Mandi) 🔴"
    else:
        trend = "SIDEWAYS (Range) 🟡"

    # Exact Action Levels
    sl_buffer = range_size * 0.25
    tp1_move = range_size * 0.45
    tp2_move = range_size * 0.85

    buy_entry = support
    buy_sl = support - sl_buffer
    buy_tp1 = support + tp1_move
    buy_tp2 = support + tp2_move

    sell_entry = resistance
    sell_sl = resistance + sl_buffer
    sell_tp1 = resistance - tp1_move
    sell_tp2 = resistance - tp2_move

    # Check karein ki abhi price kahan khada hai
    dist_to_res = abs(resistance - current_price) / current_price
    dist_to_sup = abs(current_price - support) / current_price

    if dist_to_sup <= 0.003:
        live_call = f"🔥 LIVE ACTION: BUY NOW (Price Support ${support:,.2f} ke paas hai!)"
    elif dist_to_res <= 0.003:
        live_call = f"🔥 LIVE ACTION: SELL NOW (Price Resistance ${resistance:,.2f} ke paas hai!)"
    else:
        live_call = "⏳ LIVE ACTION: BEECH MEIN TRADE NA LEIN (Niche diye gaye Support ya Resistance par aane dein)"

    signal_text = (
        f"💰 Live Price: ${current_price:,.2f}\n"
        f"📈 Trend: {trend}\n"
        f"🧱 Strong Resistance: ${resistance:,.2f}\n"
        f"🛡️ Strong Support: ${support:,.2f}\n\n"
        f"{live_call}\n\n"
        f"🟢 BUY SETUP (Support par aane par):\n"
        f"• Entry: ${buy_entry:,.2f}\n"
        f"• Stop Loss: ${buy_sl:,.2f}\n"
        f"• Target 1: ${buy_tp1:,.2f} | Target 2: ${buy_tp2:,.2f}\n\n"
        f"🔴 SELL SETUP (Resistance par aane par):\n"
        f"• Entry: ${sell_entry:,.2f}\n"
        f"• Stop Loss: ${sell_sl:,.2f}\n"
        f"• Target 1: ${sell_tp1:,.2f} | Target 2: ${sell_tp2:,.2f}"
    )
    return signal_text

def get_ai_note(candles, base_signal):
    if not GEMINI_API_KEY.startswith("AIza"):
        return ""
    prompt = f"""
    Tum ek pro BTCUSD trader ho. Niche technical levels diye gaye hain:
    {base_signal}
    Pichhli 10 candles: {candles[-10:]}
    Sirf 2 chhoti lines mein Hinglish mein batao ki abhi market ka momentum kaisa hai aur breakout ke chances kahan zyada hain. (* ya _ mat lagana)
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    try:
        res = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=15)
        data = res.json()
        if "candidates" in data and data["candidates"]:
            ai_tip = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return f"\n\n🤖 AI Confirmation:\n{ai_tip}"
    except Exception:
        pass
    return ""

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    full_text = f"📊 JAANI BHAI PRO SIGNAL (BTCUSD 15M)\n{'-'*32}\n{message}"
    res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": full_text}, timeout=15)
    if res.status_code != 200:
        raise RuntimeError(f"Telegram Error: {res.text}")

if __name__ == "__main__":
    candles = get_candles("BTCUSD", "15m", 45)
    if candles:
        signal = build_actionable_signal(candles)
        ai_note = get_ai_note(candles, signal)
        send_telegram(signal + ai_note)
        print("Pro signal bhej diya gaya!")
    else:
        print("Delta Exchange se data nahi mila.")
