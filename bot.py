import os
import io
import time
import requests
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TELEGRAM_TOKEN = (os.environ.get("TELEGRAM_TOKEN") or "").strip()
TELEGRAM_CHAT_ID = (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()
IS_MANUAL_RUN = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"

BASE_URLS = [
    "https://api.india.delta.exchange",
    "https://api.delta.exchange"
]

IST = timezone(timedelta(hours=5, minutes=30))


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
    start_time = end_time - (685 * 15 * 60)
    raw = fetch_json(
        "/v2/history/candles",
        params={"symbol": symbol, "resolution": resolution, "start": start_time, "end": end_time}
    )
    if not raw or len(raw) < 40:
        return None
    candles = sorted(raw, key=lambda x: x["time"])
    return candles[-672:]


def get_derivatives_and_options():
    tickers = fetch_json("/v2/tickers", timeout=18)
    info = {
        "live_price": None,
        "funding_rate": 0.0,
        "max_call_strike": None,
        "max_call_oi": 0.0,
        "max_put_strike": None,
        "max_put_oi": 0.0,
        "pcr": None
    }
    if not tickers:
        return info

    total_call_oi, total_put_oi = 0.0, 0.0

    for t in tickers:
        sym = t.get("symbol", "")
        ctype = t.get("contract_type", "")
        oi = float(t.get("oi") or 0)

        if sym == "BTCUSD":
            info["live_price"] = float(t.get("close") or t.get("mark_price") or 0)
            info["funding_rate"] = float(t.get("funding_rate") or 0)

        elif sym.startswith("C-BTC-") or ctype == "call_options":
            if "BTC" in sym:
                total_call_oi += oi
                if oi > info["max_call_oi"]:
                    info["max_call_oi"] = oi
                    parts = sym.split("-")
                    if len(parts) >= 3:
                        try:
                            info["max_call_strike"] = float(parts[2])
                        except ValueError:
                            pass

        elif sym.startswith("P-BTC-") or ctype == "put_options":
            if "BTC" in sym:
                total_put_oi += oi
                if oi > info["max_put_oi"]:
                    info["max_put_oi"] = oi
                    parts = sym.split("-")
                    if len(parts) >= 3:
                        try:
                            info["max_put_strike"] = float(parts[2])
                        except ValueError:
                            pass

    if total_call_oi > 0:
        info["pcr"] = round(total_put_oi / total_call_oi, 2)

    return info


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
    return 100 - (100 / (1 + (avg_gain / avg_loss)))


def generate_chart_image(candles, live_price, support, resistance, vwap, trade_plan=None):
    plot_candles = candles[-48:]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=130)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    for idx, c in enumerate(plot_candles):
        o = float(c["open"])
        h = float(c["high"])
        l = float(c["low"])
        cl = live_price if idx == len(plot_candles) - 1 else float(c["close"])

        color = "#22c55e" if cl >= o else "#ef4444"
        ax.plot([idx, idx], [l, h], color=color, linewidth=1.2, zorder=2)
        lower = min(o, cl)
        height = max(abs(cl - o), (h - l) * 0.03)
        rect = Rectangle((idx - 0.32, lower), 0.64, height, facecolor=color, edgecolor=color, zorder=3)
        ax.add_patch(rect)

    ax.axhline(resistance, color="#f87171", linestyle="--", linewidth=1.4, label=f"Resistance: ${resistance:,.0f}")
    ax.axhline(support, color="#4ade80", linestyle="--", linewidth=1.4, label=f"Support: ${support:,.0f}")
    ax.axhline(vwap, color="#38bdf8", linestyle="-.", linewidth=1.2, label=f"Weekly VWAP: ${vwap:,.0f}")
    ax.axhline(live_price, color="#facc15", linestyle=":", linewidth=1.3, label=f"Live Price: ${live_price:,.1f}")

    if trade_plan:
        ax.axhline(trade_plan["sl"], color="#dc2626", linestyle="-", linewidth=1.3, label=f"Stop Loss: ${trade_plan['sl']:,.0f}")
        ax.axhline(trade_plan["tp1"], color="#10b981", linestyle="-", linewidth=1.3, label=f"Target 1: ${trade_plan['tp1']:,.0f}")

    ax.set_title(f"JAANI BHAI LIVE BTCUSD (15M) | Price: ${live_price:,.2f}", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.grid(color="#1e293b", linestyle="-", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def send_telegram_photo_and_text(image_buf, caption_hindi):
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    res = requests.post(
        url_photo,
        data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption_hindi[:1020]},
        files={"photo": ("btcusd_chart.png", image_buf, "image/png")},
        timeout=20
    )
    if res.status_code != 200:
        url_msg = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url_msg, json={"chat_id": TELEGRAM_CHAT_ID, "text": caption_hindi}, timeout=15)


def run_market_analysis():
    candles = get_weekly_candles("BTCUSD", "15m")
    if not candles:
        return

    deriv = get_derivatives_and_options()
    live_price = deriv["live_price"] or float(candles[-1]["close"])
    candles[-1]["close"] = live_price
    now_ist = datetime.now(IST)

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    vols = [float(c.get("volume") or 0) for c in candles]

    prev_day = candles[-192:-96] if len(candles) >= 192 else candles[:-16]
    pdh = max(float(c["high"]) for c in prev_day)
    pdl = min(float(c["low"]) for c in prev_day)

    pv_sum = sum(((float(c["high"]) + float(c["low"]) + float(c["close"])) / 3.0) * float(c.get("volume") or 1) for c in candles)
    v_sum = sum(float(c.get("volume") or 1) for c in candles)
    weekly_vwap = pv_sum / v_sum if v_sum > 0 else live_price

    swing_highs, swing_lows = [], []
    for i in range(12, len(candles) - 12):
        if highs[i] == max(highs[i-12:i+13]):
            swing_highs.append(highs[i])
        if lows[i] == min(lows[i-12:i+13]):
            swing_lows.append(lows[i])

    structure_hi = "साइडवेज़ रेंज 🟡"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            structure_hi = "तेज़ी का ट्रेंड (HH + HL) 🟢"
        elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
            structure_hi = "मंदी का ट्रेंड (LH + LL) 🔴"
        elif swing_highs[-1] < swing_highs[-2]:
            structure_hi = "कमज़ोर / लोअर हाई दबाव 🟠"
        elif swing_lows[-1] > swing_lows[-2]:
            structure_hi = "मज़बूत आधार / हायर लो 🟢"

    res_4h = max(highs[-16:])
    sup_4h = min(lows[-16:])
    key_resistance = pdh if (pdh > live_price and abs(pdh - live_price) < abs(res_4h - live_price) * 1.4) else res_4h
    key_support = pdl if (pdl < live_price and abs(live_price - pdl) < abs(live_price - sup_4h) * 1.4) else sup_4h

    c_prev = candles[-3]
    c_last = candles[-2]
    mins_left = 15 - (now_ist.minute % 15)

    o1, h1, l1, cl1 = float(c_last["open"]), float(c_last["high"]), float(c_last["low"]), float(c_last["close"])
    o0, cl0 = float(c_prev["open"]), float(c_prev["close"])
    body = abs(cl1 - o1)
    upper_wick = h1 - max(o1, cl1)
    lower_wick = min(o1, cl1) - l1

    candle_hi = "सामान्य कैंडल"
    bull_candle, bear_candle = False, False

    if l1 < key_support and cl1 > key_support:
        candle_hi = "🔥 बुलिश लिक्विडिटी स्वीप (सपोर्ट के नीचे SL खाकर ऊपर बंद)"
        bull_candle = True
    elif h1 > key_resistance and cl1 < key_resistance:
        candle_hi = "🔥 बेयरिश लिक्विडिटी स्वीप (रेज़िस्टेंस के ऊपर SL खाकर नीचे बंद)"
        bear_candle = True
    elif lower_wick > body * 1.6 and lower_wick > upper_wick:
        candle_hi = "🟢 नीचे से मज़बूत खरीदारी रिजेक्शन (Pin Bar)"
        bull_candle = True
    elif upper_wick > body * 1.6 and upper_wick > lower_wick:
        candle_hi = "🔴 ऊपर से मज़बूत बिकवाली रिजेक्शन (Pin Bar)"
        bear_candle = True
    elif cl1 > o1 and cl0 < o0 and cl1 > o0 and o1 <= cl0:
        candle_hi = "🟢 बुलिश एंगल्फिंग कैंडल"
        bull_candle = True
    elif cl1 < o1 and cl0 > o0 and cl1 < o0 and o1 >= cl0:
        candle_hi = "🔴 बेयरिश एंगल्फिंग कैंडल"
        bear_candle = True

    avg_vol = sum(vols[-22:-2]) / 20 if len(vols) >= 22 else 1.0
    vol_ratio = (vols[-2] / avg_vol) if avg_vol > 0 else 1.0
    rsi = calc_rsi(closes[-30:], 14)

    bull_score, bear_score = 0, 0
    reasons_bull, reasons_bear = [], []

    if "तेज़ी" in structure_hi or "मज़बूत आधार" in structure_hi:
        bull_score += 2
        reasons_bull.append("7-दिन स्ट्रक्चर तेज़ी में")
    if "मंदी" in structure_hi or "लोअर हाई" in structure_hi:
        bear_score += 2
        reasons_bear.append("7-दिन स्ट्रक्चर मंदी में")

    if live_price > weekly_vwap:
        bull_score += 2
        reasons_bull.append("भाव Weekly VWAP के ऊपर")
    else:
        bear_score += 2
        reasons_bear.append("भाव Weekly VWAP के नीचे")

    if bull_candle:
        bull_score += 2
        reasons_bull.append(candle_hi.replace("🔥 ", "").replace("🟢 ", ""))
    if bear_candle:
        bear_score += 2
        reasons_bear.append(candle_hi.replace("🔥 ", "").replace("🔴 ", ""))

    if vol_ratio >= 1.35:
        if cl1 > o1:
            bull_score += 2
            reasons_bull.append(f"तेज़ वॉल्यूम ({vol_ratio:.1f}x)")
        else:
            bear_score += 2
            reasons_bear.append(f"तेज़ वॉल्यूम ({vol_ratio:.1f}x)")

    pcr = deriv["pcr"]
    if pcr:
        if pcr >= 1.05:
            bull_score += 1
            reasons_bull.append(f"ऑप्शंस Put सपोर्ट (PCR {pcr})")
        elif pcr <= 0.85:
            bear_score += 1
            reasons_bear.append(f"ऑप्शंस Call दबाव (PCR {pcr})")

    if abs(live_price - key_support) / live_price <= 0.0035 and rsi < 45:
        bull_score += 1
        reasons_bull.append(f"सपोर्ट ज़ोन + RSI {rsi:.0f}")
    if abs(key_resistance - live_price) / live_price <= 0.0035 and rsi > 55:
        bear_score += 1
        reasons_bear.append(f"रेज़िस्टेंस ज़ोन + RSI {rsi:.0f}")

    atr = sum(highs[i] - lows[i] for i in range(-14, 0)) / 14
    atr = max(atr, live_price * 0.0018)

    call_wall = f"${deriv['max_call_strike']:,.0f}" if deriv["max_call_strike"] else "N/A"
    put_wall = f"${deriv['max_put_strike']:,.0f}" if deriv["max_put_strike"] else "N/A"
    pcr_txt = f"{pcr}" if pcr else "N/A"
    fr_pct = deriv["funding_rate"] * 100

    near_support = 0 <= (live_price - key_support) <= atr * 0.75
    near_resistance = 0 <= (key_resistance - live_price) <= atr * 0.75
    early_warning = (mins_left <= 5) and (near_support or near_resistance)

    is_a_plus = False
    trade_plan = None

    if bull_score >= 7 and bull_score > bear_score:
        is_a_plus = True
        sl = min(key_support, live_price - atr * 1.25)
        risk = max(live_price - sl, atr * 1.0)
        tp1 = live_price + risk * 1.8
        tp2 = live_price + risk * 3.0
        trade_plan = {"sl": sl, "tp1": tp1}
        header_hi = f"🟢 पक्का BUY सिग्नल (स्कोर: {bull_score}/10)"
        wajah_hi = " + ".join(reasons_bull)
        action_box_hi = (
            f"🎯 ट्रेड प्लान (BUY):\n"
            f"• एंट्री: ${live_price:,.2f}\n"
            f"• स्टॉप लॉस: ${sl:,.2f}\n"
            f"• टारगेट 1: ${tp1:,.2f} | टारगेट 2: ${tp2:,.2f}"
        )

    elif bear_score >= 7 and bear_score > bull_score:
        is_a_plus = True
        sl = max(key_resistance, live_price + atr * 1.25)
        risk = max(sl - live_price, atr * 1.0)
        tp1 = live_price - risk * 1.8
        tp2 = live_price - risk * 3.0
        trade_plan = {"sl": sl, "tp1": tp1}
        header_hi = f"🔴 पक्का SELL सिग्नल (स्कोर: {bear_score}/10)"
        wajah_hi = " + ".join(reasons_bear)
        action_box_hi = (
            f"🎯 ट्रेड प्लान (SELL):\n"
            f"• एंट्री: ${live_price:,.2f}\n"
            f"• स्टॉप लॉस: ${sl:,.2f}\n"
            f"• टारगेट 1: ${tp1:,.2f} | टारगेट 2: ${tp2:,.2f}"
        )

    elif early_warning:
        target_zone = key_support if near_support else key_resistance
        side_hi = "BUY (सपोर्ट)" if near_support else "SELL (रेज़िस्टेंस)"
        header_hi = f"⚠️ 4–5 मिनट पहले तैयारी अलर्ट ({side_hi})"
        wajah_hi = f"भाव ${target_zone:,.2f} के बिल्कुल करीब है और 15M कैंडल बंद होने में {mins_left} मिनट बाकी हैं।"
        action_box_hi = (
            f"👀 तैयार रहें जानी भाई:\n"
            f"• एक्शन ज़ोन: ${target_zone:,.2f}\n"
            f"• चार्ट खोल लें — अगले 4–5 मिनट में एंट्री बन सकती है।"
        )

    else:
        header_hi = f"📊 लाइव मार्केट रिपोर्ट (Bull {bull_score}/10 | Bear {bear_score}/10)"
        wajah_hi = "अभी भाव बीच में है — पूरे कन्फर्मेशन का इंतज़ार है।"
        action_box_hi = (
            f"📍 नज़र रखने वाले लेवल्स:\n"
            f"• खरीदारी ज़ोन (Support): ${key_support:,.2f}\n"
            f"• बिकवाली ज़ोन (Resistance): ${key_resistance:,.2f}"
        )

    hindi_message = (
        f"🏛️ जानी भाई लाइव BTCUSD (15M)\n"
        f"{header_hi}\n"
        f"━━━━━━━━━━━━━━\n"
        f"💰 लाइव भाव: ${live_price:,.2f}\n"
        f"🧭 7-दिन स्ट्रक्चर: {structure_hi}\n"
        f"⚖️ Weekly VWAP: ${weekly_vwap:,.2f} ({'ऊपर 🟢' if live_price > weekly_vwap else 'नीचे 🔴'})\n"
        f"📊 वॉल्यूम: {vol_ratio:.1f}x | RSI: {rsi:.1f}\n"
        f"🧱 रेज़िस्टेंस: ${key_resistance:,.2f} (PDH: ${pdh:,.0f})\n"
        f"🛡️ सपोर्ट: ${key_support:,.2f} (PDL: ${pdl:,.0f})\n"
        f"⛓️ ऑप्शंस: Call वॉल {call_wall} | Put वॉल {put_wall} | PCR {pcr_txt} | Funding {fr_pct:+.3f}%\n"
        f"🕯️ कैंडल: {candle_hi}\n"
        f"━━━━━━━━━━━━━━\n"
        f"💡 वजह: {wajah_hi}\n\n"
        f"{action_box_hi}"
    )

    if IS_MANUAL_RUN or is_a_plus or early_warning:
        chart_buf = generate_chart_image(candles, live_price, key_support, key_resistance, weekly_vwap, trade_plan)
        send_telegram_photo_and_text(chart_buf, hindi_message)
        print("Chart aur Hindi report bhej di gayi!")
    else:
        print("A+ setup nahi bana — no-spam ke tahat skip kiya gaya.")


if __name__ == "__main__":
    run_market_analysis()
