import os
import time
import requests

TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
IS_MANUAL_RUN = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

BASE_URLS = [
    "https://api.india.delta.exchange",
    "https://api.delta.exchange"
]

def fetch_json(endpoint, params=None, timeout=15):
    for base in BASE_URLS:
        try:
            res = requests.get(f"{base}{endpoint}", params=params, timeout=timeout)
            if res.status_code == 200:
                data = res.json()
                if data.get("success"):
                    return data.get("result")
        except Exception:
            continue
    return None

def get_weekly_candles(symbol="BTCUSD", resolution="15m"):
    end_time = int(time.time())
    # 7 din = 672 candles (15m) + buffer
    start_time = end_time - (685 * 15 * 60)
    raw = fetch_json(
        "/v2/history/candles",
        params={"symbol": symbol, "resolution": resolution, "start": start_time, "end": end_time}
    )
    if not raw:
        return None
    candles = sorted(raw, key=lambda x: x["time"])
    return candles[-672:]

def get_derivatives_and_options():
    tickers = fetch_json("/v2/tickers", timeout=18)
    info = {
        "live_price": None,
        "funding_rate": 0.0,
        "futures_oi": 0.0,
        "max_call_strike": None,
        "max_call_oi": 0.0,
        "max_put_strike": None,
        "max_put_oi": 0.0,
        "pcr": None
    }
    if not tickers:
        return info

    total_call_oi = 0.0
    total_put_oi = 0.0

    for t in tickers:
        sym = t.get("symbol", "")
        ctype = t.get("contract_type", "")
        oi = float(t.get("oi") or 0)

        if sym == "BTCUSD":
            info["live_price"] = float(t.get("close") or t.get("mark_price") or 0)
            info["funding_rate"] = float(t.get("funding_rate") or 0)
            info["futures_oi"] = oi

        elif sym.startswith("C-BTC-") or ctype == "call_options":
            if "BTC" in sym:
                total_call_oi += oi
                if oi > info["max_call_oi"]:
                    info["max_call_oi"] = oi
                    parts = sym.split("-")
                    if len(parts) >= 3:
                        info["max_call_strike"] = float(parts[2])

        elif sym.startswith("P-BTC-") or ctype == "put_options":
            if "BTC" in sym:
                total_put_oi += oi
                if oi > info["max_put_oi"]:
                    info["max_put_oi"] = oi
                    parts = sym.split("-")
                    if len(parts) >= 3:
                        info["max_put_strike"] = float(parts[2])

    if total_call_oi > 0:
        info["pcr"] = round(total_put_oi / total_call_oi, 2)

    return info

def calc_ema(prices, period):
    ema = prices[0]
    k = 2 / (period + 1)
    for p in prices[1:]:
        ema = (p * k) + (ema * (1 - k))
    return ema

def calc_rsi(prices, period=14):
    gains, losses = [], []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + (avg_gain / avg_loss)))

def analyze_institutional_market(candles, deriv):
    live_price = deriv["live_price"] or float(candles[-1]["close"])
    candles[-1]["close"] = live_price

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    vols = [float(c.get("volume") or 0) for c in candles]

    # 1. Weekly Levels & VWAP
    weekly_high = max(highs)
    weekly_low = min(lows)

    # Pichhle 24 ghante (96 candles) ke PDH / PDL
    prev_day = candles[-192:-96] if len(candles) >= 192 else candles[:-16]
    pdh = max(float(c["high"]) for c in prev_day)
    pdl = min(float(c["low"]) for c in prev_day)

    # Weekly VWAP
    pv_sum = sum(((float(c["high"]) + float(c["low"]) + float(c["close"])) / 3.0) * float(c.get("volume") or 1) for c in candles)
    v_sum = sum(float(c.get("volume") or 1) for c in candles)
    weekly_vwap = pv_sum / v_sum if v_sum > 0 else live_price

    # 2. Weekly Swing Structure (HH/HL vs LH/LL)
    swing_highs, swing_lows = [], []
    for i in range(12, len(candles) - 12):
        window_h = highs[i-12:i+13]
        window_l = lows[i-12:i+13]
        if highs[i] == max(window_h):
            swing_highs.append(highs[i])
        if lows[i] == min(window_l):
            swing_lows.append(lows[i])

    structure = "RANGE / CONSOLIDATION 🟡"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            structure = "BULLISH (HH + HL) 🟢"
        elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
            structure = "BEARISH (LH + LL) 🔴"
        elif swing_highs[-1] < swing_highs[-2]:
            structure = "LOWER HIGH (Bearish Pressure) 🟠"
        elif swing_lows[-1] > swing_lows[-2]:
            structure = "HIGHER LOW (Bullish Base) 🟢"

    # Active S&R (4H Swing + PDH/PDL)
    res_4h = max(highs[-16:])
    sup_4h = min(lows[-16:])
    key_resistance = pdh if abs(pdh - live_price) < abs(res_4h - live_price) * 1.5 and pdh > live_price else res_4h
    key_support = pdl if abs(live_price - pdl) < abs(live_price - sup_4h) * 1.5 and pdl < live_price else sup_4h

    # Equal Highs / Equal Lows (Liquidity Pools)
    liquidity_note = "Normal Liquidity"
    if len(swing_highs) >= 2 and abs(swing_highs[-1] - swing_highs[-2]) / live_price <= 0.0018:
        liquidity_note = f"⚠️ Equal Highs (${swing_highs[-1]:,.0f}) — Upward SL Hunt Possible"
    elif len(swing_lows) >= 2 and abs(swing_lows[-1] - swing_lows[-2]) / live_price <= 0.0018:
        liquidity_note = f"⚠️ Equal Lows (${swing_lows[-1]:,.0f}) — Downward SL Hunt Possible"

    # 3. Candlestick Pattern & Liquidity Sweep (Last closed candle)
    c_prev = candles[-3]
    c_last = candles[-2]
    o1, h1, l1, cl1 = float(c_last["open"]), float(c_last["high"]), float(c_last["low"]), float(c_last["close"])
    o0, cl0 = float(c_prev["open"]), float(c_prev["close"])
    body = abs(cl1 - o1)
    upper_wick = h1 - max(o1, cl1)
    lower_wick = min(o1, cl1) - l1

    candle_signal = "Neutral Candle"
    bull_candle, bear_candle = False, False

    if l1 < key_support and cl1 > key_support:
        candle_signal = "🔥 Bullish Liquidity Sweep (Support ke niche SL khakar upar close)"
        bull_candle = True
    elif h1 > key_resistance and cl1 < key_resistance:
        candle_signal = "🔥 Bearish Liquidity Sweep (Resistance ke upar SL khakar niche close)"
        bear_candle = True
    elif lower_wick > body * 1.7 and lower_wick > upper_wick:
        candle_signal = "🟢 Bullish Rejection Pin Bar (Niche se buying pressure)"
        bull_candle = True
    elif upper_wick > body * 1.7 and upper_wick > lower_wick:
        candle_signal = "🔴 Bearish Rejection Pin Bar (Upar se selling pressure)"
        bear_candle = True
    elif cl1 > o1 and cl0 < o0 and cl1 > o0 and o1 <= cl0:
        candle_signal = "🟢 Bullish Engulfing Candle"
        bull_candle = True
    elif cl1 < o1 and cl0 > o0 and cl1 < o0 and o1 >= cl0:
        candle_signal = "🔴 Bearish Engulfing Candle"
        bear_candle = True

    # 4. Volume & Momentum (EMA 21/50 + RSI 14)
    avg_vol = sum(vols[-22:-2]) / 20 if len(vols) >= 22 else (sum(vols) / max(len(vols), 1))
    vol_ratio = (vols[-2] / avg_vol) if avg_vol > 0 else 1.0
    ema21 = calc_ema(closes[-50:], 21)
    ema50 = calc_ema(closes[-100:], 50)
    rsi = calc_rsi(closes[-30:], 14)

    # 5. Confluence Scoring (Out of 10)
    bull_score, bear_score = 0, 0
    reasons_bull, reasons_bear = [], []

    if "BULLISH" in structure or "HIGHER LOW" in structure:
        bull_score += 2
        reasons_bull.append(f"Weekly Structure ({structure.split()[0]})")
    if "BEARISH" in structure or "LOWER HIGH" in structure:
        bear_score += 2
        reasons_bear.append(f"Weekly Structure ({structure.split()[0]})")

    if live_price > weekly_vwap:
        bull_score += 2
        reasons_bull.append(f"Price Weekly VWAP (${weekly_vwap:,.0f}) ke upar")
    else:
        bear_score += 2
        reasons_bear.append(f"Price Weekly VWAP (${weekly_vwap:,.0f}) ke niche")

    if bull_candle:
        bull_score += 2
        reasons_bull.append(candle_signal.replace("🔥 ", "").replace("🟢 ", ""))
    if bear_candle:
        bear_score += 2
        reasons_bear.append(candle_signal.replace("🔥 ", "").replace("🔴 ", ""))

    if vol_ratio >= 1.4:
        if cl1 > o1:
            bull_score += 2
            reasons_bull.append(f"High Volume Spike ({vol_ratio:.1f}x)")
        else:
            bear_score += 2
            reasons_bear.append(f"High Volume Spike ({vol_ratio:.1f}x)")

    pcr = deriv["pcr"]
    if pcr:
        if pcr >= 1.1:
            bull_score += 1
            reasons_bull.append(f"Strong Put Writing (PCR {pcr})")
        elif pcr <= 0.85:
            bear_score += 1
            reasons_bear.append(f"Heavy Call Writing (PCR {pcr})")

    if abs(live_price - key_support) / live_price <= 0.0035 and rsi < 45:
        bull_score += 1
        reasons_bull.append(f"Key Support (${key_support:,.0f}) + RSI {rsi:.0f}")
    if abs(key_resistance - live_price) / live_price <= 0.0035 and rsi > 55:
        bear_score += 1
        reasons_bear.append(f"Key Resistance (${key_resistance:,.0f}) + RSI {rsi:.0f}")

    # Trade Setup & Projection
    atr = sum(highs[i] - lows[i] for i in range(-14, 0)) / 14
    atr = max(atr, live_price * 0.0018)

    is_a_plus = False
    if bull_score >= 7 and bull_score > bear_score:
        is_a_plus = True
        direction = f"🟢 A+ BUY SIGNAL (Score: {bull_score}/10)"
        entry = live_price
        sl = min(key_support, live_price - atr * 1.3)
        risk = max(entry - sl, atr * 1.1)
        tp1 = entry + risk * 1.8
        tp2 = entry + risk * 3.0
        why_text = " + ".join(reasons_bull)
        plan = (
            f"🎯 ACTIONABLE BUY PLAN:\n"
            f"• Entry: ${entry:,.2f}\n"
            f"• Stop Loss: ${sl:,.2f}\n"
            f"• Target 1: ${tp1:,.2f} | Target 2: ${tp2:,.2f}\n"
            f"• Next Major Wall: ${key_resistance:,.2f}"
        )
    elif bear_score >= 7 and bear_score > bull_score:
        is_a_plus = True
        direction = f"🔴 A+ SELL SIGNAL (Score: {bear_score}/10)"
        entry = live_price
        sl = max(key_resistance, live_price + atr * 1.3)
        risk = max(sl - entry, atr * 1.1)
        tp1 = entry - risk * 1.8
        tp2 = entry - risk * 3.0
        why_text = " + ".join(reasons_bear)
        plan = (
            f"🎯 ACTIONABLE SELL PLAN:\n"
            f"• Entry: ${entry:,.2f}\n"
            f"• Stop Loss: ${sl:,.2f}\n"
            f"• Target 1: ${tp1:,.2f} | Target 2: ${tp2:,.2f}\n"
            f"• Next Major Floor: ${key_support:,.2f}"
        )
    else:
        bias = "Bullish Lean" if bull_score > bear_score else "Bearish Lean"
        direction = f"⏳ NO A+ TRADE YET ({bias} | Bull {bull_score}/10 vs Bear {bear_score}/10)"
        why_text = "Abhi sabhi confirmations (Volume + Structure + Level) ek saath nahi mile hain."
        plan = (
            f"📍 KEY ZONES TO WATCH:\n"
            f"• Buy Zone (Support/PDL): ${key_support:,.2f}\n"
            f"• Sell Zone (Resistance/PDH): ${key_resistance:,.2f}"
        )

    call_wall = f"${deriv['max_call_strike']:,.0f}" if deriv["max_call_strike"] else "N/A"
    put_wall = f"${deriv['max_put_strike']:,.0f}" if deriv["max_put_strike"] else "N/A"
    pcr_txt = f"{pcr}" if pcr else "N/A"
    fr_pct = deriv["funding_rate"] * 100

    msg = (
        f"🏛️ JAANI BHAI INSTITUTIONAL DESK (BTCUSD 15M)\n"
        f"{'='*34}\n"
        f"💰 Live Price: ${live_price:,.2f}\n"
        f"🧭 7D Structure: {structure}\n"
        f"⚖️ Weekly VWAP: ${weekly_vwap:,.2f} ({'Above 🟢' if live_price > weekly_vwap else 'Below 🔴'})\n"
        f"📊 Volume: {vol_ratio:.1f}x Avg | RSI(14): {rsi:.1f}\n\n"
        f"🧱 Weekly High: ${weekly_high:,.2f} | PDH: ${pdh:,.2f}\n"
        f"🛡️ Weekly Low: ${weekly_low:,.2f} | PDL: ${pdl:,.2f}\n"
        f"💧 Liquidity: {liquidity_note}\n\n"
        f"⛓️ OPTIONS & FUTURES CONTROL:\n"
        f"• Highest Call OI (Ceiling): {call_wall}\n"
        f"• Highest Put OI (Floor): {put_wall}\n"
        f"• Options PCR: {pcr_txt} | Funding: {fr_pct:+.4f}%\n\n"
        f"🕯️ Candlestick: {candle_signal}\n"
        f"{'-'*34}\n"
        f"{direction}\n"
        f"💡 Wajah: {why_text}\n\n"
        f"{plan}"
    )
    return is_a_plus, msg

def send_telegram(text):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)

if __name__ == "__main__":
    candles = get_weekly_candles("BTCUSD", "15m")
    if candles:
        deriv = get_derivatives_and_options()
        is_a_plus, report = analyze_institutional_market(candles, deriv)
        if is_a_plus or IS_MANUAL_RUN:
            send_telegram(report)
            print("Institutional report Telegram par bhej di gayi!")
        else:
            print("A+ setup abhi nahi bana — no-spam filter ke tahat message skip kiya gaya.")
    else:
        print("Weekly candle data fetch nahi ho paya.")
