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
from PIL import Image, ImageOps
import imageio.v2 as imageio

# ==============================================================================
# 1. कॉन्फ़िगरेशन (GitHub Secrets से सीधे नाम मैच किए गए हैं)
# ==============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

REST_BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"

PAPER_MODE = True                 # शुरुआती 7-10 दिन टेस्टिंग के लिए True
RISK_PER_TRADE_INR = 500.0        # प्रति ट्रेड फिक्स्ड रिस्क (₹500)
USD_TO_INR = 85.0                 # Delta India फिक्स्ड कन्वर्ज़न
LOT_SIZE_BTC = 0.001              # 1 Lot = 0.001 BTC
ROUND_TRIP_FEE_PCT = 0.00118      # 0.05% x 2 Taker + 18% GST = 0.118%

RSI_PERIOD = 14
EMA_1H_PERIOD = 20
ATR_PERIOD = 14
VOLUME_MULTIPLIER = 1.35          # 20-SMA से कम से कम 1.35x वॉल्यूम
PINBAR_WICK_RATIO = 2.5           # Wick कम से कम Body से 2.5 गुना
MIN_SL_PCT = 0.0035               # न्यूनतम 0.35% SL
MIN_NET_RR = 1.5                  # फ़ीस+GST के बाद न्यूनतम 1:1.5 R:R

COOLDOWN_MINUTES = 0              # Cooldown बंद (24/7 बिना ब्रेक सिग्नल)
STATE_FILE = "bot_state.json"

IST = pytz.timezone("Asia/Kolkata")
RES_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}


# ==============================================================================
# 2. स्टेट मेमोरी (bot_state.json)
# ==============================================================================
def load_state() -> dict:
    default = {
        "state": "SCANNING",
        "trade": None,
        "signal_msg_id": None,
        "last_candle_ts": 0
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                default.update(json.load(f))
        except Exception:
            pass
    return default


def save_state(st: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


# ==============================================================================
# 3. टेलीग्राम अलर्ट (Text + Photo + 2D Video)
# ==============================================================================
def send_tg_message(text: str, reply_to: int = None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG LOG]", text)
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
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
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG PHOTO]", caption)
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(photo_path, "rb") as f:
            r = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1020], "parse_mode": "Markdown"},
                files={"photo": f},
                timeout=25
            ).json()
        if not r.get("ok"):
            print("TG Photo Response:", r)
        if os.path.exists(photo_path):
            os.remove(photo_path)
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print("TG Photo Error:", e)
        return None


def send_tg_video(video_path: str, caption: str, reply_to: int = None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID or not video_path or not os.path.exists(video_path):
        return None
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendVideo"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption[:1020],
        "parse_mode": "Markdown",
        "supports_streaming": "true"
    }
    if reply_to:
        data["reply_to_message_id"] = reply_to
    try:
        with open(video_path, "rb") as f:
            r = requests.post(url, data=data, files={"video": f}, timeout=45).json()
        if not r.get("ok"):
            print("TG Video Response:", r)
        if os.path.exists(video_path):
            os.remove(video_path)
        return r.get("result", {}).get("message_id")
    except Exception as e:
        print("TG Video Error:", e)
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

            # चल रही अधूरी (Unclosed) कैंडल हटाना
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
        "trend_1d": trend_1d, "trend_4h": trend_4h, "trend_1h": trend_1h,
        "htf_levels": htf_levels,
        "recent_swing_high": rec_h, "recent_swing_low": rec_l
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

    now_ist = datetime.datetime.now(IST)
    funding_warn = ""
    for h in [5, 13, 21]:
        f_time = now_ist.replace(hour=h, minute=30, second=0, microsecond=0)
        if abs((now_ist - f_time).total_seconds()) <= 900:
            funding_warn = "⚠️ Funding सेटलमेंट 15 मिनट के भीतर है!"

    return {
        "approved": True, "side": side, "mode": analysis["mode"], "pattern": analysis["pattern"],
        "entry": round(entry, 1), "sl": sl, "tp1": tp1, "tp2": tp2, "net_rr": net_rr,
        "lots": lots, "risk_inr": RISK_PER_TRADE_INR, "obstacle_warning": obs_warn,
        "funding_rate_pct": round(funding_rate, 4), "funding_warn": funding_warn,
        "tp1_hit": False
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
        skip_today = (now_ist.hour > 14) or (now_ist.hour == 14 and now_ist.minute >= 30)

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
            return "📊 **BTC ATM Options:** डेटा अभी उपलब्ध नहीं।"

        nearest_exp = min(expiries.keys())
        items = expiries[nearest_exp]
        atm = min(set(x[1] for x in items), key=lambda s: abs(s - spot_price))
        ce = next((x[2] for x in items if x[0] == "C" and x[1] == atm), None)
        pe = next((x[2] for x in items if x[0] == "P" and x[1] == atm), None)

        def parse_opt(o):
            if not o:
                return "N/A", 0.0, 0.0
            mp = float(o.get("mark_price") or 0)
            oi = float(o.get("oi") or 0)
            vol = float(o.get("volume") or 0)
            greeks = o.get("greeks") or {}
            delta = float(greeks.get("delta") or 0)
            theta = float(greeks.get("theta") or 0)
            txt = f"प्रीमियम `${mp:.1f}` | Vol `{vol:,.0f}` | OI `{oi:,.0f}` | Δ `{delta:.2f}` | Θ `{theta:.1f}`"
            return txt, oi, vol

        ce_txt, ce_oi, ce_vol = parse_opt(ce)
        pe_txt, pe_oi, pe_vol = parse_opt(pe)

        if (ce_vol + ce_oi) > (pe_vol + pe_oi) * 1.1:
            bias = "Call में ज़्यादा एक्टिविटी → मोमेंटम **Bullish**"
        elif (pe_vol + pe_oi) > (ce_vol + ce_oi) * 1.1:
            bias = "Put में ज़्यादा एक्टिविटी → मोमेंटम **Bearish**"
        else:
            bias = "Call और Put लगभग बराबर → **Neutral / संतुलित**"

        return (
            f"📊 **BTC ATM Options (Strike: {int(atm)} | Exp: {nearest_exp.strftime('%d-%b')})**\n"
            f"• **CALL (CE):** {ce_txt}\n"
            f"• **PUT (PE):** {pe_txt}\n"
            f"• **Option संकेत:** {bias}"
        )
    except Exception:
        return "📊 **BTC ATM Options:** डेटा अभी उपलब्ध नहीं।"


# ==============================================================================
# 6. ग्रीन/रेड चार्ट (IST टाइम) + 2D एनालिसिस वीडियो (MP4)
# ==============================================================================
def get_chart_style():
    mc = mpf.make_marketcolors(
        up="#089981", down="#F23645",
        edge="inherit", wick="inherit", volume="in"
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        gridstyle=":",
        y_on_right=True
    )


def render_chart(df: pd.DataFrame, title: str, filename: str, hlines=None, colors=None, add_ema20=False) -> str:
    plot_df = df.tail(55).copy()
    plot_df["datetime"] = plot_df["datetime"].dt.tz_convert(IST).dt.tz_localize(None)
    plot_df = plot_df.set_index("datetime")

    kwargs = dict(
        type="candle",
        volume=True,
        style=get_chart_style(),
        title=title,
        figsize=(10, 5.6),
        savefig=dict(fname=filename, dpi=115, bbox_inches="tight")
    )
    if hlines:
        kwargs["hlines"] = dict(hlines=hlines, colors=colors, linestyle="--", linewidths=1.4)
    if add_ema20:
        ema_full = df["close"].ewm(span=EMA_1H_PERIOD, adjust=False).mean().tail(55)
        ema_full.index = plot_df.index
        kwargs["addplot"] = mpf.make_addplot(ema_full, color="#FFD600", width=1.4)

    mpf.plot(plot_df, **kwargs)
    return filename


def create_2d_video(data: dict, curr_price: float, analysis: dict, verdict_en: str, trade=None, out_mp4="btc_2d_video.mp4") -> str:
    """1D -> 4H -> 1H (20-EMA) -> 15M चारों टाइमफ्रेम का 8-सेकंड 2D MP4 वीडियो।"""
    frames_files = []
    try:
        frames_files.append(render_chart(
            data["1d"], f"1/4 | BTCUSD 1D (IST) | Trend: {analysis['trend_1d']} | {verdict_en}", "f1_1d.png",
            hlines=[curr_price], colors=["#2962FF"]
        ))
        frames_files.append(render_chart(
            data["4h"], f"2/4 | BTCUSD 4H (IST) | Trend: {analysis['trend_4h']} | {verdict_en}", "f2_4h.png",
            hlines=[curr_price], colors=["#2962FF"]
        ))
        frames_files.append(render_chart(
            data["1h"], f"3/4 | BTCUSD 1H + 20EMA (IST) | Trend: {analysis['trend_1h']} | {verdict_en}", "f3_1h.png",
            hlines=[curr_price], colors=["#2962FF"], add_ema20=True
        ))

        if trade:
            hlines = [trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]]
            colors = ["#2962FF", "#F23645", "#089981", "#00E676"]
            title_15m = f"4/4 | BTCUSD 15M (IST) | {verdict_en} ({trade['pattern']})"
        else:
            hlines = [curr_price]
            colors = ["#2962FF"]
            if abs(analysis["vwap"] - curr_price) / curr_price <= 0.05:
                hlines.append(analysis["vwap"])
                colors.append("#FF9800")
            title_15m = f"4/4 | BTCUSD 15M (IST) | {verdict_en} | RSI: {analysis['rsi_15m']}"

        frames_files.append(render_chart(
            data["15m"], title_15m, "f4_15m.png", hlines=hlines, colors=colors
        ))

        # 1024x576 (16:9) फ़िक्स साइज़ में 2 सेकंड प्रति चार्ट = 8 सेकंड वीडियो
        with imageio.get_writer(
            out_mp4, fps=2, codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        ) as writer:
            for fp in frames_files:
                img = Image.open(fp).convert("RGB")
                img = ImageOps.pad(img, (1024, 576), color=(19, 23, 34))
                arr = np.array(img)
                for _ in range(4):  # 4 frames @ 2fps = 2 seconds per chart
                    writer.append_data(arr)

        for fp in frames_files:
            if os.path.exists(fp):
                os.remove(fp)
        return out_mp4
    except Exception as e:
        print("Video Error:", e)
        for fp in frames_files:
            if os.path.exists(fp):
                os.remove(fp)
        return ""


def send_manual_status(data: dict, curr_price: float, fr: float, analysis: dict, reason: str):
    """मैन्युअल Run workflow दबाने पर फोटो + 2D वीडियो + साफ़ WAIT फैसला।"""
    opt_note = get_atm_options_note(curr_price)
    ema_pos = "1H 20-EMA के ऊपर (UP)" if analysis["trend_1h"] == "UP" else "1H 20-EMA के नीचे (DOWN)"
    now_ist = datetime.datetime.now(IST).strftime("%d-%b %I:%M %p IST")

    caption = (
        f"⏳ **फैसला: WAIT (अभी इंतज़ार करें — न खरीदें, न बेचें)**\n"
        f"🕒 समय: `{now_ist}`\n"
        f"💰 **लाइव प्राइस:** `${curr_price:,.1f}`\n"
        f"📈 **ट्रेंड:** 1D `{analysis['trend_1d']}` | 4H `{analysis['trend_4h']}` | {ema_pos}\n"
        f"📊 **15M RSI:** `{analysis['rsi_15m']}` | **वॉल्यूम:** `{analysis['vol_ratio']}x`\n"
        f"📌 **Weekly VWAP:** `${analysis['vwap']:,.1f}` | **Funding:** `{round(fr, 4)}%`\n"
        f"🔍 **कारण:** {reason}\n\n"
        f"{opt_note}"
    )

    hlines = [curr_price]
    colors = ["#2962FF"]
    if abs(analysis["vwap"] - curr_price) / curr_price <= 0.05:
        hlines.append(analysis["vwap"])
        colors.append("#FF9800")

    photo_file = render_chart(
        data["15m"], f"BTCUSD 15M (IST) | WAIT | Price: ${curr_price:,.1f} | RSI: {analysis['rsi_15m']}",
        "status_chart.png", hlines=hlines, colors=colors
    )
    msg_id = send_tg_photo(photo_file, caption)

    vid_file = create_2d_video(data, curr_price, analysis, "WAIT (NO TRADE)", trade=None)
    send_tg_video(
        vid_file,
        "🎬 **2D मल्टी-टाइमफ्रेम वीडियो (1D → 4H → 1H → 15M)**\n⏳ फैसला: **WAIT (अभी इंतज़ार करें)**",
        reply_to=msg_id
    )


# ==============================================================================
# 7. मुख्य 24/7 One-Shot रनर (Cooldown = 0)
# ==============================================================================
def main():
    is_manual = os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch"
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
        side_hi = "BUY (खरीदा हुआ)" if side == "BUY" else "SELL (बेचा हुआ)"

        if not t.get("tp1_hit") and ((side == "BUY" and curr_price >= t["tp1"]) or (side == "SELL" and curr_price <= t["tp1"])):
            t["tp1_hit"] = True
            t["sl"] = t["entry"]
            send_tg_message(
                f"🎯 **TP1 हिट (${t['tp1']:,.1f})!**\nSL को ऑटोमैटिक Entry (${t['entry']:,.1f} - Breakeven) पर शिफ़्ट कर दिया गया है। अब TP2 ट्रैक हो रहा है।",
                reply_to=st["signal_msg_id"]
            )
            save_state(st)
            return

        if (side == "BUY" and curr_price >= t["tp2"]) or (side == "SELL" and curr_price <= t["tp2"]):
            send_tg_message(
                f"🏆 **TP2 हिट (${t['tp2']:,.1f}) — ट्रेड सफल!**\nबॉट तुरंत अगले नए सेटअप की तलाश में लग गया है।",
                reply_to=st["signal_msg_id"]
            )
            st.update({"state": "SCANNING", "trade": None, "signal_msg_id": None})
            save_state(st)
            return

        if (side == "BUY" and curr_price <= t["sl"]) or (side == "SELL" and curr_price >= t["sl"]):
            if t.get("tp1_hit"):
                msg = f"🛡️ **Breakeven SL हिट (${t['sl']:,.1f})** (TP1 पहले ही बुक हो चुका था)।\nबॉट तुरंत अगले नए सेटअप के लिए तैयार है।"
            else:
                msg = f"🛑 **Stop Loss हिट (${t['sl']:,.1f})।**\nबॉट तुरंत अगले नए सेटअप के लिए तैयार है।"
            send_tg_message(msg, reply_to=st["signal_msg_id"])
            st.update({"state": "SCANNING", "trade": None, "signal_msg_id": None})
            save_state(st)
            return

        df_15m = fetch_candles("15m", 60)
        rsi_txt = f"{float(wilders_rsi(df_15m['close'], RSI_PERIOD).iloc[-1]):.1f}" if not df_15m.empty else "N/A"
        diff = (ltp - t["entry"]) if side == "BUY" else (t["entry"] - ltp)
        emoji = "🟢" if diff >= 0 else "🔴"
        send_tg_message(
            f"{emoji} **लाइव ट्रेड अपडेट — {side_hi}**\n• LTP: `${ltp:,.1f}` ({diff:+.1f} USD)\n• 15M RSI: `{rsi_txt}` | वर्तमान SL: `${t['sl']:,.1f}`",
            reply_to=st["signal_msg_id"]
        )
        return

    # --- अवस्था 2: नया 15M मार्केट स्कैन ---
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
        if is_manual:
            analysis = analyze_market(data)
            send_manual_status(data, curr_price, fr, analysis, "अभी कोई पक्का सेटअप नहीं बना है (24/7 स्कैनिंग चालू है)।")
        print("Same candle already scanned.")
        return

    st["last_candle_ts"] = latest_ts
    analysis = analyze_market(data)

    if analysis["signal"] == "WAIT":
        if is_manual:
            send_manual_status(data, curr_price, fr, analysis, "अभी कोई पक्का सेटअप नहीं बना है (24/7 स्कैनिंग चालू है)।")
        print("No setup found. Staying quiet.")
        save_state(st)
        return

    trade = evaluate_trade(analysis, data["15m"], fr)
    if not trade["approved"]:
        if is_manual:
            send_manual_status(data, curr_price, fr, analysis, "सेटअप बना था, लेकिन Risk/Reward या HTF सपोर्ट/रेज़िस्टेंस फ़िल्टर ने रोक दिया।")
        print("Trade rejected by Risk/Obstacle filter.")
        save_state(st)
        return

    # --- पक्का BUY या SELL सिग्नल (फोटो + 2D वीडियो दोनों) ---
    opt_note = get_atm_options_note(trade["entry"])
    mode_tag = "📝 [PAPER]" if PAPER_MODE else "🚨 [LIVE]"
    ema_pos = "1H 20-EMA के ऊपर" if analysis["trend_1h"] == "UP" else "1H 20-EMA के नीचे"
    action_header = "🟢 **फैसला: BUY (खरीदें) — BTCUSD**" if trade["side"] == "BUY" else "🔴 **फैसला: SELL (बेचें) — BTCUSD**"

    warn_text = f"\n⚠️ {trade['obstacle_warning']}" if trade["obstacle_warning"] else ""
    if trade["funding_warn"]:
        warn_text += f"\n{trade['funding_warn']}"

    caption = (
        f"{mode_tag} {action_header}\n"
        f"• **सेटअप:** {trade['mode']} | {trade['pattern']}\n"
        f"• **पुष्टि:** वॉल्यूम `{analysis['vol_ratio']}x` | 15M RSI `{analysis['rsi_15m']}` | {ema_pos}\n\n"
        f"🎯 **Entry:** `${trade['entry']:,.1f}`\n"
        f"🛑 **Stop Loss:** `${trade['sl']:,.1f}`\n"
        f"✅ **Target 1:** `${trade['tp1']:,.1f}` *(TP1 पर SL ऑटो Entry पर शिफ़्ट)*\n"
        f"🏆 **Target 2:** `${trade['tp2']:,.1f}`\n\n"
        f"⚖️ **Net R:R (फ़ीस+GST बाद):** `1:{trade['net_rr']}`\n"
        f"📦 **सुझावित Lot Size:** `{trade['lots']} Lots` *(रिस्क: ₹{int(trade['risk_inr'])})*\n"
        f"📌 **Weekly VWAP:** `${analysis['vwap']:,.1f}` | **Funding:** `{trade['funding_rate_pct']}%`"
        f"{warn_text}\n\n{opt_note}"
    )

    hlines = [trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]]
    colors = ["#2962FF", "#F23645", "#089981", "#00E676"]
    if abs(analysis["vwap"] - trade["entry"]) / trade["entry"] <= 0.03:
        hlines.append(analysis["vwap"])
        colors.append("#FF9800")

    photo_file = render_chart(
        data["15m"], f"BTCUSD 15M (IST) | {trade['side']} ({trade['pattern']})",
        "signal_chart.png", hlines=hlines, colors=colors
    )
    msg_id = send_tg_photo(photo_file, caption)

    vid_file = create_2d_video(data, trade["entry"], analysis, f"{trade['side']} SIGNAL", trade=trade)
    send_tg_video(
        vid_file,
        f"🎬 **2D मल्टी-टाइमफ्रेम वीडियो (1D → 4H → 1H → 15M)**\n{action_header}",
        reply_to=msg_id
    )

    st.update({"state": "ACTIVE_TRADE", "trade": trade, "signal_msg_id": msg_id})
    save_state(st)
    print("Signal photo + 2D video sent and state saved!")


if __name__ == "__main__":
    main()
