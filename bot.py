import os
import json
import time
import datetime
import requests
import pytz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf

# ==============================================================================
# 1. कॉन्फ़िगरेशन (Token/Chat ID सीधे GitHub Secrets से आएँगे)
# ==============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

REST_BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"

PAPER_MODE = True                 # पहले 7-10 दिन True रखें
RISK_PER_TRADE_INR = 1000.0       # प्रति ट्रेड फिक्स्ड रिस्क (₹ में)
USD_TO_INR = 85.0
LOT_SIZE_BTC = 0.001
ROUND_TRIP_FEE_PCT = 0.00118      # 0.05% x 2 Taker + 18% GST

RSI_PERIOD = 14
EMA_1H_PERIOD = 20
ATR_PERIOD = 14
VOLUME_MULTIPLIER = 1.35
PINBAR_WICK_RATIO = 2.5
MIN_SL_PCT = 0.0035
MIN_NET_RR = 1.5

COOLDOWN_NORMAL_MIN = 90
COOLDOWN_DOUBLE_SL_MIN = 180
STATE_FILE = "bot_state.json"

IST = pytz.timezone("Asia/Kolkata")
RES_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


# ==============================================================================
# 2. स्टेट मेमोरी (bot_state.json — ताकि GitHub पर पुराना ट्रेड याद रहे)
# ==============================================================================
def load_state() -> dict:
    default = {
        "state": "SCANNING",
        "trade": None,
        "signal_msg_id": None,
        "cooldown_until": 0,
        "consecutive_sls": 0,
        "last_candle_ts": 0
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                default.update(data)
        except Exception:
            pass
    return default


def save_state(st: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


# ==============================================================================
# 3. टेलीग्राम अलर्ट
# ==============================================================================
def send_tg_message(text: str, reply_to: int = None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG LOG]", text)
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        r = requests.post(url, json=payload, timeout=15).json()
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print("TG Error:", e)
        return None


def send_tg_photo(photo_path: str, caption: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG PHOTO]", caption)
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as f:
            r = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
                files={"photo": f},
                timeout=25
            ).json()
        if os.path.exists(photo_path):
            os.remove(photo_path)
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print("TG Photo Error:", e)
        return None


# ==============================================================================
# 4. Delta India डेटा इंजन
# ==============================================================================
def fetch_candles(resolution: str, count: int = 300) -> pd.DataFrame:
    sec = RES_SECONDS[resolution]
    now = int(time.time())
    start = now - (count + 10) * sec
    url = f"{REST_BASE_URL}/v2/history/candles"
    params = {"resolution": resolution, "symbol": SYMBOL, "start": start, "end": now}

    for _ in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json().get("result", [])
            if not raw:
                continue
            df = pd.DataFrame(raw)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["time"] = pd.to_numeric(df["time"], errors="coerce")
            df = df.dropna(subset=["time", "open", "high", "low", "close"])
            df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

            # अधूरी (Unclosed) कैंडल हटाना
            if len(df) > 0 and (df.iloc[-1]["time"] + sec) > int(time.time()):
                df = df.iloc[:-1].reset_index(drop=True)

            df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            if len(df) >= 50:
                return df.tail(count).reset_index(drop=True)
        except Exception as e:
            print(f"Fetch error ({resolution}):", e)
            time.sleep(2)
    return pd.DataFrame()


def get_live_ticker():
    try:
        r = requests.get(f"{REST_BASE_URL}/v2/tickers/{SYMBOL}", timeout=10).json()
        res = r.get("result", {})
        ltp = float(res.get("close", 0))
        mark = float(res.get("mark_price", ltp))
        fr = float(res.get("funding_rate", 0) or 0)
        return ltp, mark, fr
    except Exception:
        return None, None, 0.0


# ==============================================================================
# 5. प्राइस एक्शन, इंडिकेटर्स और गेटकीपर
# ==============================================================================
def wilders_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calc_weekly_vwap(df_15m: pd.DataFrame) -> float:
    last_dt = df_15m["datetime"].iloc[-1]
    week_start = (last_dt - pd.Timedelta(days=last_dt.weekday())).normalize()
    wdf = df_15m[df_15m["datetime"] >= week_start]
    if wdf.empty or wdf["volume"].sum() == 0:
        return float(df_15m["close"].iloc[-1])
    tp = (wdf["high"] + wdf["low"] + wdf["close"]) / 3.0
    return float((tp * wdf["volume"]).sum() / wdf["volume"].sum())


def find_swings(df: pd.DataFrame, n: int = 3):
    highs, lows = [], []
    h_vals, l_vals = df["high"].values, df["low"].values
    for i in range(n, len(df) - n):
        if h_vals[i] == max(h_vals[i - n:i + n + 1]):
            highs.append((i, float(h_vals[i])))
        if l_vals[i] == min(l_vals[i - n:i + n + 1]):
            lows.append((i, float(l_vals[i])))
    return highs, lows


def get_trend(df: pd.DataFrame) -> str:
    highs, lows = find_swings(df.tail(120).reset_index(drop=True), n=3)
    if len(highs) < 2 or len(lows) < 2:
        return "RANGE"
    if highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]:
        return "UP"
    if highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]:
        return "DOWN"
    return "RANGE"


def check_15m_patterns(df_15m: pd.DataFrame, key_levels: list, atr_val: float):
    c, p = df_15m.iloc[-1], df_15m.iloc[-2]
    c_range = c["high"] - c["low"]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["open"], c["close"])
    lower_wick = min(c["open"], c["close"]) - c["low"]

    if c_range < 0.5 * atr_val or c_range == 0:
        return None, None

    close_pos = (c["close"] - c["low"]) / c_range
    eff_body = max(body, 0.05 * c_range)

    for lvl in key_levels:
        if c["low"] < lvl and c["close"] > lvl and close_pos >= 0.55:
            return "BUY", "Liquidity Sweep"
        if c["high"] > lvl and c["close"] < lvl and close_pos <= 0.45:
            return "SELL", "Liquidity Sweep"

    if lower_wick >= PINBAR_WICK_RATIO * eff_body and close_pos >= 0.65:
        return "BUY", "Pin Bar (Hammer)"
    if upper_wick >= PINBAR_WICK_RATIO * eff_body and close_pos <= 0.35:
        return "SELL", "Pin Bar (Shooting Star)"

    p_bull, c_bull = p["close"] > p["open"], c["close"] > c["open"]
    if (not p_bull) and c_bull and (c["close"] > p["open"]) and (c["open"] <= p["close"]):
        return "BUY", "Bullish Engulfing"
    if p_bull and (not c_bull) and (c["close"] < p["open"]) and (c["open"] >= p["close"]):
        return "SELL", "Bearish Engulfing"

    return None, None


def analyze_market(data: dict) -> dict:
    df_1d, df_4h, df_1h, df_15m = data["1d"], data["4h"], data["1h"], data["15m"]
    df_15m = df_15m.copy()
    df_15m["rsi"] = wilders_rsi(df_15m["close"], RSI_PERIOD)
    df_15m["atr"] = calc_atr(df_15m, ATR_PERIOD)
    df_15m["vol_sma"] = df_15m["volume"].shift(1).rolling(20).mean()

    ema_1h = df_1h["close"].ewm(span=EMA_1H_PERIOD, adjust=False).mean().iloc[-1]
    price = float(df_15m["close"].iloc[-1])
    rsi_15m = float(df_15m["rsi"].iloc[-1])
    atr_15m = float(df_15m["atr"].iloc[-1])
    atr_sma = float(df_15m["atr"].rolling(20).mean().iloc[-1])
    vol_ratio = float(df_15m["volume"].iloc[-1]) / float(df_15m["vol_sma"].iloc[-1] or 1.0)
    vwap = calc_weekly_vwap(df_15m)

    trend_1d, trend_4h = get_trend(df_1d), get_trend(df_4h)
    trend_1h = "UP" if price > ema_1h else "DOWN"
    pdh, pdl = float(df_1d["high"].iloc[-1]), float(df_1d["low"].iloc[-1])

    sw_h_4h, sw_l_4h = find_swings(df_4h.tail(150).reset_index(drop=True), n=3)
    demand_4h, supply_4h = [v for _, v in sw_l_4h[-4:]], [v for _, v in sw_h_4h[-4:]]

    sw_h_15m, sw_l_15m = find_swings(df_15m.tail(60).reset_index(drop=True), n=3)
    rec_h = sw_h_15m[-1][1] if sw_h_15m else float(df_15m["high"].tail(15).max())
    rec_l = sw_l_15m[-1][1] if sw_l_15m else float(df_15m["low"].tail(15).min())

    htf_levels = [pdh, pdl] + demand_4h + supply_4h
    key_levels = htf_levels + [rec_h, rec_l]

    res = {
        "signal": "WAIT", "mode": None, "pattern": None, "price": price,
        "rsi_15m": round(rsi_15m, 2), "atr_15m": round(atr_15m, 2),
        "vol_ratio": round(vol_ratio, 2), "vwap": round(vwap, 2),
        "htf_levels": htf_levels, "recent_swing_high": rec_h, "recent_swing_low": rec_l
    }

    if atr_sma > 0 and not (0.55 <= (atr_15m / atr_sma) <= 2.6):
        return res

    side, pattern = check_15m_patterns(df_15m, key_levels, atr_15m)
    if not side or vol_ratio < VOLUME_MULTIPLIER:
        return res

    if (side == "BUY" and not (38 <= rsi_15m <= 70)) or (side == "SELL" and not (30 <= rsi_15m <= 62)):
        return res

    near_htf = any(abs(price - lvl) <= 1.2 * atr_15m for lvl in htf_levels)
    if (side == "BUY" and trend_1h == "UP" and trend_4h != "DOWN") or (side == "SELL" and trend_1h == "DOWN" and trend_4h != "UP"):
        res.update({"signal": side, "mode": "Mode A (Trend Pullback)", "pattern": pattern})
    elif near_htf and pattern in ["Liquidity Sweep", "Pin Bar (Hammer)", "Pin Bar (Shooting Star)"]:
        res.update({"signal": side, "mode": "Mode B (HTF Zone Reversal)", "pattern": pattern})

    return res


def evaluate_trade(analysis: dict, df_15m, funding_rate: float) -> dict:
    side, entry, atr = analysis["signal"], analysis["price"], analysis["atr_15m"]
    last_c = df_15m.iloc[-1]
    min_sl_dist = entry * MIN_SL_PCT

    if side == "BUY":
        raw_sl = min(float(last_c["low"]), analysis["recent_swing_low"]) - (0.3 * atr)
        sl_dist = max(entry - raw_sl, min_sl_dist)
        sl, tp1, tp2 = round(entry - sl_dist, 1), round(entry + 2.0 * sl_dist, 1), round(entry + 3.5 * sl_dist, 1)
    else:
        raw_sl = max(float(last_c["high"]), analysis["recent_swing_high"]) + (0.3 * atr)
        sl_dist = max(raw_sl - entry, min_sl_dist)
        sl, tp1, tp2 = round(entry + sl_dist, 1), round(entry - 2.0 * sl_dist, 1), round(entry - 3.5 * sl_dist, 1)

    fee_usd = entry * ROUND_TRIP_FEE_PCT
    net_rr = round((abs(tp1 - entry) - fee_usd) / (sl_dist + fee_usd), 2)
    if net_rr < MIN_NET_RR:
        return {"approved": False}

    obs_warn = None
    for lvl in analysis["htf_levels"]:
        if side == "BUY" and (entry + 0.2 * sl_dist) < lvl < tp1:
            if (lvl - entry) < 1.5 * sl_dist:
                return {"approved": False}
            obs_warn = f"ध्यान दें: ${lvl:,.0f} पर HTF रेज़िस्टेंस है"
        elif side == "SELL" and tp1 < lvl < (entry - 0.2 * sl_dist):
            if (entry - lvl) < 1.5 * sl_dist:
                return {"approved": False}
            obs_warn = f"ध्यान दें: ${lvl:,.0f} पर HTF सपोर्ट है"

    lots = max(1, int(((RISK_PER_TRADE_INR / USD_TO_INR) / (sl_dist + fee_usd)) / LOT_SIZE_BTC))
    return {
        "approved": True, "side": side, "mode": analysis["mode"], "pattern": analysis["pattern"],
        "entry": round(entry, 1), "sl": sl, "tp1": tp1, "tp2": tp2, "net_rr": net_rr,
        "lots": lots, "risk_inr": RISK_PER_TRADE_INR, "obstacle_warning": obs_warn,
        "funding_rate_pct": round(funding_rate * 100, 4), "tp1_hit": False
    }


def get_atm_options_note(spot_price: float) -> str:
    try:
        res = requests.get(
            f"{REST_BASE_URL}/v2/tickers",
            params={"contract_types": "call_options,put_options", "underlying_asset_symbols": "BTC"},
            timeout=10
        ).json().get("result", [])
        now_ist = datetime.datetime.now(IST)
        today_str = now_ist.strftime("%d%m%y")
        skip_today = now_ist.hour >= 15

        expiries = {}
        for item in res:
            parts = item.get("symbol", "").split("-")
            if len(parts) != 4 or (skip_today and parts[3] == today_str):
                continue
            try:
                expiries.setdefault(datetime.datetime.strptime(parts[3], "%d%m%y"), []).append((parts[0], float(parts[2]), item))
            except ValueError:
                continue

        if not expiries:
            return "ऑप्शन डेटा उपलब्ध नहीं।"
        nearest_exp = min(expiries.keys())
        items = expiries[nearest_exp]
        atm = min(set(x[1] for x in items), key=lambda s: abs(s - spot_price))
        ce = next((x[2] for x in items if x[0] == "C" and x[1] == atm), None)
        pe = next((x[2] for x in items if x[0] == "P" and x[1] == atm), None)

        def fmt(o):
            if not o:
                return "N/A"
            return f"Mark ${float(o.get('mark_price') or 0):.1f} | OI {float(o.get('oi') or 0):,.0f} | Vol {float(o.get('volume') or 0):,.0f}"

        return f"📊 **ATM Options ({int(atm)} | Exp: {nearest_exp.strftime('%d-%b')})**\n• CE: {fmt(ce)}\n• PE: {fmt(pe)}"
    except Exception:
        return "ऑप्शन डेटा उपलब्ध नहीं।"


def create_15m_chart(df_15m, trade: dict, vwap_val: float, filename="signal_chart.png") -> str:
    plot_df = df_15m.tail(60).copy().set_index("datetime")
    hlines = [trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]]
    colors = ["#2962FF", "#F23645", "#089981", "#00E676"]
    if abs(vwap_val - trade["entry"]) / trade["entry"] <= 0.03:
        hlines.append(vwap_val)
        colors.append("#FF9800")

    style = mpf.make_mpf_style(base_mpf_style="nightclouds", gridstyle=":", y_on_right=True)
    mpf.plot(
        plot_df, type="candle", volume=True, style=style,
        title=f"BTCUSD 15M | {trade['side']} ({trade['pattern']})",
        hlines=dict(hlines=hlines, colors=colors, linestyle="--", linewidths=1.3),
        savefig=dict(fname=filename, dpi=130, bbox_inches="tight")
    )
    return filename


# ==============================================================================
# 6. मुख्य One-Shot रनर (GitHub Actions के लिए)
# ==============================================================================
def main():
    st = load_state()
    ltp, mark, fr = get_live_ticker()
    if not ltp:
        print("Live price unavailable.")
        return
    curr_price = mark or ltp

    # --- अवस्था 1: यदि कोई ट्रेड पहले से ACTIVE है ---
    if st["state"] == "ACTIVE_TRADE" and st["trade"]:
        t = st["trade"]
        side = t["side"]

        # TP1 -> Breakeven शिफ़्ट
        if not t.get("tp1_hit") and ((side == "BUY" and curr_price >= t["tp1"]) or (side == "SELL" and curr_price <= t["tp1"])):
            t["tp1_hit"] = True
            t["sl"] = t["entry"]
            send_tg_message(f"🎯 **TP1 हिट (${t['tp1']:,.1f})!** SL को Entry (${t['entry']:,.1f} - Breakeven) पर शिफ़्ट कर दिया है।", reply_to=st["signal_msg_id"])
            save_state(st)
            return

        # TP2 -> ट्रेड सफल + Cooldown
        if (side == "BUY" and curr_price >= t["tp2"]) or (side == "SELL" and curr_price <= t["tp2"]):
            send_tg_message(f"🏆 **TP2 हिट (${t['tp2']:,.1f}) — ट्रेड सफल!** {COOLDOWN_NORMAL_MIN} मिनट Cooldown शुरू।", reply_to=st["signal_msg_id"])
            st.update({"state": "COOLDOWN", "trade": None, "signal_msg_id": None, "cooldown_until": time.time() + COOLDOWN_NORMAL_MIN * 60, "consecutive_sls": 0})
            save_state(st)
            return

        # SL -> ट्रेड बंद + Cooldown
        if (side == "BUY" and curr_price <= t["sl"]) or (side == "SELL" and curr_price >= t["sl"]):
            if t.get("tp1_hit"):
                cd_min, st["consecutive_sls"] = COOLDOWN_NORMAL_MIN, 0
                msg = f"🛡️ **Breakeven SL हिट (${t['sl']:,.1f})** (TP1 बुक हो चुका था)। {cd_min} मिनट Cooldown शुरू।"
            else:
                st["consecutive_sls"] += 1
                cd_min = COOLDOWN_DOUBLE_SL_MIN if st["consecutive_sls"] >= 2 else COOLDOWN_NORMAL_MIN
                msg = f"🛑 **Stop Loss हिट (${t['sl']:,.1f})।** {cd_min} मिनट Cooldown शुरू।"
            send_tg_message(msg, reply_to=st["signal_msg_id"])
            st.update({"state": "COOLDOWN", "trade": None, "signal_msg_id": None, "cooldown_until": time.time() + cd_min * 60})
            save_state(st)
            return

        # लाइव अपडेट (Reply में)
        df_15m = fetch_candles("15m", 60)
        rsi_txt = f"{float(wilders_rsi(df_15m['close'], RSI_PERIOD).iloc[-1]):.1f}" if not df_15m.empty else "N/A"
        diff = (ltp - t["entry"]) if side == "BUY" else (t["entry"] - ltp)
        emoji = "🟢" if diff >= 0 else "🔴"
        send_tg_message(
            f"{emoji} **लाइव अपडेट ({side})**\n• LTP: `${ltp:,.1f}` ({diff:+.1f} USD)\n• 15M RSI: `{rsi_txt}` | SL: `${t['sl']:,.1f}`",
            reply_to=st["signal_msg_id"]
        )
        return

    # --- अवस्था 2: यदि COOLDOWN चल रहा है ---
    if st["state"] == "COOLDOWN":
        if time.time() < st["cooldown_until"]:
            print("In Cooldown. Skipping scan.")
            return
        st["state"] = "SCANNING"

    # --- अवस्था 3: नया 15M मार्केट स्कैन ---
    data = {
        "1d": fetch_candles("1d", 500),
        "4h": fetch_candles("4h", 350),
        "1h": fetch_candles("1h", 350),
        "15m": fetch_candles("15m", 750)
    }
    if any(df.empty for df in data.values()):
        print("Incomplete data.")
        return

    latest_ts = int(data["15m"]["time"].iloc[-1])
    if latest_ts <= st["last_candle_ts"]:
        print("Same candle already scanned.")
        save_state(st)
        return

    st["last_candle_ts"] = latest_ts
    analysis = analyze_market(data)

    if analysis["signal"] == "WAIT":
        print("No setup found.")
        save_state(st)
        return

    trade = evaluate_trade(analysis, data["15m"], fr)
    if not trade["approved"]:
        print("Trade rejected by Risk/Obstacle filter.")
        save_state(st)
        return

    opt_note = get_atm_options_note(trade["entry"])
    chart_file = create_15m_chart(data["15m"], trade, analysis["vwap"])
    mode_tag = "📝 [PAPER]" if PAPER_MODE else "🚨 [LIVE]"
    warn_text = f"\n⚠️ {trade['obstacle_warning']}" if trade["obstacle_warning"] else ""

    caption = (
        f"{mode_tag} **पक्का {trade['side']} सिग्नल — BTCUSD**\n"
        f"• मोड: {trade['mode']}\n"
        f"• पैटर्न: {trade['pattern']} (Vol: {analysis['vol_ratio']}x | RSI: {analysis['rsi_15m']})\n\n"
        f"🎯 **Entry:** `${trade['entry']:,.1f}`\n"
        f"🛑 **Stop Loss:** `${trade['sl']:,.1f}`\n"
        f"✅ **Target 1:** `${trade['tp1']:,.1f}`\n"
        f"🏆 **Target 2:** `${trade['tp2']:,.1f}`\n\n"
        f"⚖️ **Net R:R (फ़ीस+GST बाद):** `1:{trade['net_rr']}`\n"
        f"📦 **सुझावित Lot Size:** `{trade['lots']} Lots` (रिस्क: ₹{int(trade['risk_inr'])})\n"
        f"📌 Weekly VWAP: `${analysis['vwap']:,.1f}` | Funding: `{trade['funding_rate_pct']}%`"
        f"{warn_text}\n\n{opt_note}"
    )

    msg_id = send_tg_photo(chart_file, caption)
    st.update({"state": "ACTIVE_TRADE", "trade": trade, "signal_msg_id": msg_id})
    save_state(st)
    print("Signal sent and state saved!")


if __name__ == "__main__":
    main()
