import os
import time
import requests

TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()

def get_live_market_data(symbol="BTCUSD", resolution="15m"):
    base_urls = [
        "https://api.india.delta.exchange",
        "https://api.delta.exchange"
    ]
    end_time = int(time.time())
    start_time = end_time - (60 * 15 * 60)  # Pichhli 60 candles

    for base in base_urls:
        try:
            # 1. Exact Live Ticker Price lena
            ticker_res = requests.get(f"{base}/v2/tickers/{symbol}", timeout=10)
            live_price = None
            if ticker_res.status_code == 200:
                t_data = ticker_res.json().get("result", {})
                live_price = float(t_data.get("close") or t_data.get("mark_price") or 0)

            # 2. Candles lena aur sahi time order (purani se nayi) mein lagana
            c_res = requests.get(
                f"{base}/v2/history/candles",
                params={"symbol": symbol, "resolution": resolution, "start": start_time, "end": end_time},
                timeout=12
            )
            if c_res.status_code == 200:
                c_data = c_res.json()
                if c_data.get("success") and c_data.get("result"):
                    candles = sorted(c_data["result"], key=lambda x: x["time"])
                    if not live_price:
                        live_price = float(candles[-1]["close"])
                    return live_price, candles[-40:]
        except Exception:
            continue
    return None, None

def calc_ema(prices, period):
    ema = prices[0]
    k = 2 / (period + 1)
    for p in prices[1:]:
        ema = (p * k) + (ema * (1 - k))
    return ema

def calc_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def analyze_market(live_price, candles):
    closes = [float(c["close"]) for c in candles]
    closes[-1] = live_price  # Latest close ko live price se sync karna

    # Technical Indicators
    ema9 = calc_ema(closes[-25:], 9)
    ema21 = calc_ema(closes[-35:], 21)
    rsi = calc_rsi(closes, 14)

    # Pichhli 16 candles (4 ghante) ke nazdiki aur asli Swing High / Low
    recent = candles[-16:]
    resistance = max(float(c["high"]) for c in recent)
    support = min(float(c["low"]) for c in recent)

    # ATR (Average Candle Size) for realistic SL & Target
    atr = sum(float(c["high"]) - float(c["low"]) for c in candles[-14:]) / 14
    atr = max(atr, live_price * 0.0015)

    # Trend & Momentum Check
    if live_price > ema9 > ema21:
        trend = "STRONG BULLISH (Tezi) 🟢"
    elif live_price < ema9 < ema21:
        trend = "STRONG BEARISH (Mandi) 🔴"
    else:
        trend = "SIDEWAYS / CHOPPY 🟡"

    # Realistic Action Levels
    sl_points = atr * 1.2
    tp1_points = atr * 1.8
    tp2_points = atr * 3.0

    buy_entry = support
    buy_sl = buy_entry - sl_points
    buy_tp1 = buy_entry + tp1_points
    buy_tp2 = buy_entry + tp2_points

    sell_entry = resistance
    sell_sl = sell_entry + sl_points
    sell_tp1 = sell_entry - tp1_points
    sell_tp2 = sell_entry - tp2_points

    # Live Action Trigger (agar price level ke 0.2% ke andar ho)
    if abs(live_price - support) / live_price <= 0.002 and rsi < 45:
        action = f"🔥 BUY NOW! Price Support (${support:,.2f}) par hai aur RSI ({rsi:.1f}) oversold zone mein hai."
    elif abs(resistance - live_price) / live_price <= 0.002 and rsi > 55:
        action = f"🔥 SELL NOW! Price Resistance (${resistance:,.2f}) par hai aur RSI ({rsi:.1f}) overbought zone mein hai."
    else:
        action = "⏳ WAIT: Abhi price beech mein hai — niche diye gaye Support ya Resistance par aane par hi entry lein."

    report = (
        f"💰 Live Price: ${live_price:,.2f}\n"
        f"📈 Trend: {trend}\n"
        f"📊 RSI (14): {rsi:.1f} | EMA 9/21: ${ema9:,.0f} / ${ema21:,.0f}\n"
        f"🧱 Resistance (4H High): ${resistance:,.2f}\n"
        f"🛡️ Support (4H Low): ${support:,.2f}\n\n"
        f"{action}\n\n"
        f"🟢 BUY PLAN (Support par):\n"
        f"• Entry: ${buy_entry:,.2f}\n"
        f"• Stop Loss: ${buy_sl:,.2f}\n"
        f"• Target 1: ${buy_tp1:,.2f} | Target 2: ${buy_tp2:,.2f}\n\n"
        f"🔴 SELL PLAN (Resistance par):\n"
        f"• Entry: ${sell_entry:,.2f}\n"
        f"• Stop Loss: ${sell_sl:,.2f}\n"
        f"• Target 1: ${sell_tp1:,.2f} | Target 2: ${sell_tp2:,.2f}"
    )
    return report

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    full_text = f"⚡ JAANI BHAI LIVE BTCUSD (15M)\n{'-'*32}\n{message}"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": full_text}, timeout=15)

if __name__ == "__main__":
    live_price, candles = get_live_market_data("BTCUSD", "15m")
    if live_price and candles:
        msg = analyze_market(live_price, candles)
        send_telegram(msg)
        print("Accurate signal bhej diya gaya!")
    else:
        print("Market data fetch nahi ho paya.")
