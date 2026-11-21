import os
import json
import time
import asyncio
import subprocess
import datetime
import requests
import pytz
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import mplfinance as mpf
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import edge_tts

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
# 3. टेलीग्राम अलर्ट (Text + Photo + 9:16 Reels Video with Voice)
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
        "width": 720,
        "height": 1280,
        "supports_streaming": "true"
    }
    if reply_to:
        data["reply_to_message_id"] = reply_to
    try:
        with open(video_path, "rb") as f:
            r = requests.post(url, data=data, files={"video": f}, timeout=60).json()
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
# 6. ग्रीन/रेड चार्ट (IST) + 9:16 एनिमेटेड Instagram Reel वीडियो (हिंदी AI वॉइस सहित)
# ==============================================================================
def get_chart_style():
    mc = mpf.make_marketcolors(
        up="#00E676", down="#FF1744",
        edge="inherit", wick="inherit", volume="in"
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=mc,
        facecolor="#0F172A",
        figcolor="#0F172A",
        gridcolor="#1E293B",
        gridstyle=":",
        y_on_right=True
    )


def render_chart(df: pd.DataFrame, title: str, filename: str, hlines=None, colors=None, figsize=(10, 5.6), dpi=115) -> str:
    plot_df = df.copy()
    plot_df["datetime"] = plot_df["datetime"].dt.tz_convert(IST).dt.tz_localize(None)
    plot_df = plot_df.set_index("datetime")

    kwargs = dict(
        type="candle",
        volume=True,
        style=get_chart_style(),
        title=title,
        figsize=figsize,
        savefig=dict(fname=filename, dpi=dpi, bbox_inches="tight", facecolor="#0F172A")
    )
    if hlines:
        kwargs["hlines"] = dict(hlines=hlines, colors=colors, linestyle="--", linewidths=1.5)

    mpf.plot(plot_df, **kwargs)
    return filename


def load_font(size: int, bold: bool = True):
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    ]
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_reel_frame(chart_img_path: str, action: str, curr_price: float, analysis: dict, trade=None, step_pct: float = 1.0) -> np.ndarray:
    """720x1280 (9:16 Instagram Reels) का प्रोफेशनल डार्क ट्रेडिंग कार्ड।"""
    W, H = 720, 1280
    canvas = Image.new("RGB", (W, H), (11, 17, 32))
    draw = ImageDraw.Draw(canvas)

    f_title = load_font(40, True)
    f_badge = load_font(46, True)
    f_med = load_font(30, True)
    f_small = load_font(24, False)

    # रंग और बैज सेटअप
    if action == "BUY":
        badge_bg, badge_txt = (0, 180, 95), "SIGNAL: BUY NOW"
    elif action == "SELL":
        badge_bg, badge_txt = (225, 35, 65), "SIGNAL: SELL NOW"
    else:
        badge_bg, badge_txt = (230, 145, 10), "DECISION: WAIT"

    # टॉप हेडर
    draw.rectangle([(0, 0), (W, 130)], fill=(18, 28, 50))
    draw.text((40, 26), "BTCUSD LIVE ANALYSIS", font=f_title, fill=(255, 255, 255))
    now_str = datetime.datetime.now(IST).strftime("%d %b | %I:%M %p IST")
    draw.text((40, 82), f"Delta Exchange India  •  {now_str}", font=f_small, fill=(148, 163, 184))

    # बड़ा BUY / SELL / WAIT बैज
    draw.rounded_rectangle([(40, 155), (W - 40, 245)], radius=20, fill=badge_bg)
    bbox = draw.textbbox((0, 0), badge_txt, font=f_badge)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 172), badge_txt, font=f_badge, fill=(255, 255, 255))

    # बीच में एनिमेटेड 15M चार्ट
    chart_im = Image.open(chart_img_path).convert("RGB")
    chart_im.thumbnail((W - 40, 540), Image.Resampling.LANCZOS)
    cx = (W - chart_im.width) // 2
    cy = 275 + (540 - chart_im.height) // 2
    canvas.paste(chart_im, (cx, cy))

    # प्रोग्रेस बार (एनीमेशन का फील देने के लिए)
    bar_w = int((W - 80) * min(1.0, max(0.05, step_pct)))
    draw.rounded_rectangle([(40, 835), (W - 40, 847)], radius=6, fill=(30, 41, 59))
    draw.rounded_rectangle([(40, 835), (40 + bar_w, 847)], radius=6, fill=badge_bg)

    # नीचे का इन्फो कार्ड
    draw.rounded_rectangle([(40, 875), (W - 40, 1185)], radius=22, fill=(18, 28, 50), outline=(51, 65, 85), width=2)

    if trade:
        lines = [
            (f"ENTRY :  ${trade['entry']:,.1f}", (56, 189, 248)),
            (f"STOP LOSS :  ${trade['sl']:,.1f}", (248, 113, 113)),
            (f"TARGET 1 :  ${trade['tp1']:,.1f}", (74, 222, 128)),
            (f"TARGET 2 :  ${trade['tp2']:,.1f}", (0, 230, 118)),
            (f"RISK: Rs.500  |  LOTS: {trade['lots']}  |  R:R 1:{trade['net_rr']}", (250, 204, 21)),
        ]
    else:
        lines = [
            (f"LIVE PRICE :  ${curr_price:,.1f}", (56, 189, 248)),
            (f"TREND :  1D {analysis['trend_1d']}  |  4H {analysis['trend_4h']}  |  1H {analysis['trend_1h']}", (248, 250, 252)),
            (f"15M RSI :  {analysis['rsi_15m']}   |   VOL :  {analysis['vol_ratio']}x", (250, 204, 21)),
            (f"WEEKLY VWAP :  ${analysis['vwap']:,.1f}", (251, 146, 60)),
            ("NO TRADE ZONE — WAITING FOR SETUP", (148, 163, 184)),
        ]

    y_pos = 905
    for txt, col in lines:
        draw.text((70, y_pos), txt, font=f_med if y_pos < 1100 else f_small, fill=col)
        y_pos += 54

    # फ़ुटर ब्रांडिंग
    footer = "JAANI 24x7 AUTO TRADING BOT"
    fb = draw.textbbox((0, 0), footer, font=f_small)
    draw.text(((W - (fb[2] - fb[0])) // 2, 1215), footer, font=f_small, fill=(100, 116, 139))

    return np.array(canvas)


async def _make_hindi_voice(script: str, out_mp3: str):
    comm = edge_tts.Communicate(script, voice="hi-IN-MadhurNeural", rate="+6%")
    await comm.save(out_mp3)


def create_animated_reel_with_voice(df_15m: pd.DataFrame, curr_price: float, analysis: dict, trade=None, out_mp4="btc_reel.mp4") -> str:
    """9:16 एनिमेटेड कैंडल वीडियो + असली हिंदी AI वॉइसओवर (Instagram Reels Ready)।"""
    raw_mp4 = "raw_anim.mp4"
    voice_mp3 = "voice_hi.mp3"
    tmp_png = "anim_step.png"

    try:
        action = trade["side"] if trade else "WAIT"
        full_df = df_15m.tail(45).reset_index(drop=True)

        # हिंदी वॉइसओवर स्क्रिप्ट
        p_int = int(round(curr_price))
        if trade:
            side_word = "खरीदने यानी बाय" if trade["side"] == "BUY" else "बेचने यानी सेल"
            script = (
                f"नमस्कार ट्रेडर्स! बिटकॉइन में पक्का {side_word} का सिग्नल बना है। "
                f"अभी प्राइस {p_int} डॉलर है। "
                f"एंट्री {int(trade['entry'])} डॉलर पर, स्टॉप लॉस {int(trade['sl'])} डॉलर, "
                f"और पहला टारगेट {int(trade['tp1'])} डॉलर का है। "
                f"सिर्फ़ 500 रुपये के फिक्स रिस्क के साथ ही ट्रेड करें।"
            )
        else:
            trend_word = "नीचे यानी डाउन" if analysis["trend_1h"] == "DOWN" else "ऊपर यानी अप"
            script = (
                f"नमस्कार ट्रेडर्स! बिटकॉइन अभी {p_int} डॉलर के पास चल रहा है। "
                f"1 घंटे का ट्रेंड {trend_word} है, और 15 मिनट का आर एस आई {int(round(analysis['rsi_15m']))} है। "
                f"अभी कोई पक्का सेटअप नहीं बना है, इसलिए फैसला है वेट। "
                f"अभी न खरीदें, न बेचें, सही मौके का इंतज़ार करें।"
            )

        # 1. हिंदी AI वॉइस जेनरेट करें
        has_audio = False
        try:
            asyncio.run(_make_hindi_voice(script, voice_mp3))
            has_audio = os.path.exists(voice_mp3) and os.path.getsize(voice_mp3) > 1000
        except Exception as ve:
            print("Voice TTS Error:", ve)

        # 2. एक-एक करके कैंडल्स एनिमेट करें (18 से 45 कैंडल्स तक स्मूद ग्रोथ)
        steps = list(range(18, len(full_df) + 1, 2))
        if steps[-1] != len(full_df):
            steps.append(len(full_df))

        frames = []
        for idx, count in enumerate(steps):
            sub_df = full_df.iloc[:count]
            is_last = (idx == len(steps) - 1)
            if is_last and trade:
                hlines = [trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]]
                colors = ["#38BDF8", "#F87171", "#4ADE80", "#00E676"]
            else:
                hlines = [float(sub_df["close"].iloc[-1])]
                colors = ["#38BDF8"]

            render_chart(
                sub_df, f"BTCUSD 15M (IST)  •  ${float(sub_df['close'].iloc[-1]):,.1f}",
                tmp_png, hlines=hlines, colors=colors, figsize=(7.2, 5.4), dpi=100
            )
            frame_arr = build_reel_frame(
                tmp_png, action, curr_price, analysis, trade=trade,
                step_pct=(idx + 1) / len(steps)
            )
            frames.append(frame_arr)

        # वीडियो राइटर (3 FPS: ~5 सेकंड एनीमेशन + 12 सेकंड आखिरी फ्रेम ताकि पूरी आवाज़ कवर हो)
        with imageio.get_writer(
            raw_mp4, fps=3, codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        ) as writer:
            for fr_arr in frames:
                writer.append_data(fr_arr)
            for _ in range(36):  # आखिरी क्लियर चार्ट को 12 सेकंड तक रोकें
                writer.append_data(frames[-1])

        # 3. वीडियो और हिंदी आवाज़ को जोड़ें (आवाज़ खत्म होते ही वीडियो बंद)
        if has_audio:
            cmd = [
                "ffmpeg", "-y",
                "-i", raw_mp4,
                "-i", voice_mp3,
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                "-movflags", "+faststart",
                out_mp4
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0 or not os.path.exists(out_mp4):
                os.replace(raw_mp4, out_mp4)
        else:
            os.replace(raw_mp4, out_mp4)

        for f_clean in [raw_mp4, voice_mp3, tmp_png]:
            if os.path.exists(f_clean):
                os.remove(f_clean)
        return out_mp4
    except Exception as e:
        print("Reel Video Error:", e)
        for f_clean in [raw_mp4, voice_mp3, tmp_png]:
            if os.path.exists(f_clean):
                os.remove(f_clean)
        return ""


def send_manual_status(data: dict, curr_price: float, fr: float, analysis: dict, reason: str):
    """मैन्युअल Run workflow दबाने पर फोटो + 9:16 बोलती हुई हिंदी रील वीडियो।"""
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
    colors = ["#38BDF8"]
    if abs(analysis["vwap"] - curr_price) / curr_price <= 0.05:
        hlines.append(analysis["vwap"])
        colors.append("#FF9800")

    photo_file = render_chart(
        data["15m"].tail(55), f"BTCUSD 15M (IST) | WAIT | Price: ${curr_price:,.1f} | RSI: {analysis['rsi_15m']}",
        "status_chart.png", hlines=hlines, colors=colors
    )
    msg_id = send_tg_photo(photo_file, caption)

    vid_file = create_animated_reel_with_voice(data["15m"], curr_price, analysis, trade=None)
    send_tg_video(
        vid_file,
        "🎬 **Instagram Reels वीडियो (हिंदी वॉइस के साथ)**\n⏳ फैसला: **WAIT (अभी इंतज़ार करें)**",
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

    # --- पक्का BUY या SELL सिग्नल (फोटो + बोलती हुई 9:16 Reels वीडियो दोनों) ---
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
    colors = ["#38BDF8", "#F87171", "#4ADE80", "#00E676"]
    if abs(analysis["vwap"] - trade["entry"]) / trade["entry"] <= 0.03:
        hlines.append(analysis["vwap"])
        colors.append("#FF9800")

    photo_file = render_chart(
        data["15m"].tail(55), f"BTCUSD 15M (IST) | {trade['side']} ({trade['pattern']})",
        "signal_chart.png", hlines=hlines, colors=colors
    )
    msg_id = send_tg_photo(photo_file, caption)

    vid_file = create_animated_reel_with_voice(data["15m"], trade["entry"], analysis, trade=trade)
    send_tg_video(
        vid_file,
        f"🎬 **Instagram Reels वीडियो (हिंदी वॉइस के साथ)**\n{action_header}",
        reply_to=msg_id
    )

    st.update({"state": "ACTIVE_TRADE", "trade": trade, "signal_msg_id": msg_id})
    save_state(st)
    print("Signal photo + Reels Hindi voice video sent!")


if __name__ == "__main__":
    main()
