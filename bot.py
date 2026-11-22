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
import imageio_ffmpeg
import edge_tts
from gtts import gTTS

# ==============================================================================
# 1. कॉन्फ़िगरेशन (100% 15-मिनट सटीक ट्रेडिंग इंजन)
# ==============================================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

REST_BASE_URL = "https://api.india.delta.exchange"
SYMBOL = "BTCUSD"
YOUTUBE_HANDLE = "@Jaanibhai_9"

PAPER_MODE = True                 # शुरुआती टेस्टिंग के लिए True
RISK_PER_TRADE_INR = 500.0        # प्रति ट्रेड फिक्स्ड रिस्क (₹500)
USD_TO_INR = 85.0                 # Delta India फिक्स्ड कन्वर्ज़न
LOT_SIZE_BTC = 0.001              # 1 Lot = 0.001 BTC
ROUND_TRIP_FEE_PCT = 0.00118      # 0.05% x 2 Taker + 18% GST = 0.118%

RSI_PERIOD = 14
EMA_FAST_15M = 20                 # 15M मुख्य ट्रेंड EMA
EMA_SLOW_15M = 50                 # 15M सपोर्टिंग ट्रेंड EMA
ATR_PERIOD = 14
VOLUME_MULTIPLIER = 1.10          # 15M 20-SMA से कम से कम 1.10x वॉल्यूम
PINBAR_WICK_RATIO = 1.8           # Wick कम से कम Body से 1.8 गुना
MIN_SL_PCT = 0.0030               # न्यूनतम 0.30% SL
MIN_NET_RR = 1.5                  # फ़ीस+GST के बाद न्यूनतम 1:1.5 R:R

COOLDOWN_MINUTES = 0              # Cooldown बंद (24/7 बिना ब्रेक सिग्नल)
STATE_FILE = "bot_state.json"

IST = pytz.timezone("Asia/Kolkata")
RES_SECONDS = {"15m": 900}


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
# 3. टेलीग्राम अलर्ट (Text + Photo + 9:16 Reels Video)
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
# 4. Delta India 15M डेटा इंजन
# ==============================================================================
def fetch_candles(resolution: str = "15m", count: int = 750) -> pd.DataFrame:
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

            # चल रही अधूरी (Unclosed) कैंडल हटाना ताकि फ़ेक सिग्नल न आए
            if len(df) > 0 and (df.iloc[-1]["time"] + sec) > int(time.time()):
                df = df.iloc[:-1].reset_index(drop=True)

            df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            if len(df) >= 60:
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
# 5. सटीक 15M प्राइस एक्शन, EMA और गेटकीपर
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


def check_15m_patterns(df_15m: pd.DataFrame, swing_levels: list, atr_val: float, ema20: float, trend_15m: str):
    c, p = df_15m.iloc[-1], df_15m.iloc[-2]
    c_range = c["high"] - c["low"]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["open"], c["close"])
    lower_wick = min(c["open"], c["close"]) - c["low"]

    if c_range < 0.35 * atr_val or c_range == 0:
        return None, None

    close_pos = (c["close"] - c["low"]) / c_range
    eff_body = max(body, 0.05 * c_range)
    tol = 0.12 * atr_val

    # 1. Liquidity Sweep (15M Swing High/Low से रिवर्सल)
    for lvl in swing_levels:
        if c["low"] < lvl and c["close"] > lvl and close_pos >= 0.52:
            return "BUY", "15M Liquidity Sweep"
        if c["high"] > lvl and c["close"] < lvl and close_pos <= 0.48:
            return "SELL", "15M Liquidity Sweep"

    # 2. Pin Bar (Hammer / Shooting Star)
    if lower_wick >= PINBAR_WICK_RATIO * eff_body and close_pos >= 0.60:
        return "BUY", "15M Pin Bar (Hammer)"
    if upper_wick >= PINBAR_WICK_RATIO * eff_body and close_pos <= 0.40:
        return "SELL", "15M Pin Bar (Shooting Star)"

    # 3. Engulfing (24/7 क्रिप्टो के लिए सटीक बॉडी एंगल्फ़िंग)
    p_bull, c_bull = p["close"] > p["open"], c["close"] > c["open"]
    if (not p_bull) and c_bull and (c["close"] > p["open"]) and (c["open"] <= p["close"] + tol):
        return "BUY", "15M Bullish Engulfing"
    if p_bull and (not c_bull) and (c["close"] < p["open"]) and (c["open"] >= p["close"] - tol):
        return "SELL", "15M Bearish Engulfing"

    # 4. 15M EMA Pullback / Momentum Breakout (ट्रेंड की दिशा में मज़बूत कैंडल)
    body_ratio = body / c_range
    if body_ratio >= 0.55:
        if trend_15m == "UP" and c_bull and c["close"] > ema20 and c["low"] <= ema20 + (0.6 * atr_val):
            return "BUY", "15M EMA Pullback Momentum"
        if trend_15m == "DOWN" and (not c_bull) and c["close"] < ema20 and c["high"] >= ema20 - (0.6 * atr_val):
            return "SELL", "15M EMA Pullback Momentum"

    return None, None


def analyze_market_15m(df_15m: pd.DataFrame) -> dict:
    df = df_15m.copy()
    df["rsi"] = wilders_rsi(df["close"], RSI_PERIOD)
    df["atr"] = calc_atr(df, ATR_PERIOD)
    df["ema20"] = df["close"].ewm(span=EMA_FAST_15M, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=EMA_SLOW_15M, adjust=False).mean()
    df["vol_sma"] = df["volume"].shift(1).rolling(20).mean()

    price = float(df["close"].iloc[-1])
    rsi_15m = float(df["rsi"].iloc[-1])
    atr_15m = float(df["atr"].iloc[-1])
    atr_sma = float(df["atr"].rolling(20).mean().iloc[-1])
    ema20 = float(df["ema20"].iloc[-1])
    ema50 = float(df["ema50"].iloc[-1])
    vol_ratio = float(df["volume"].iloc[-1]) / float(df["vol_sma"].iloc[-1] or 1.0)
    vwap = calc_weekly_vwap(df)

    # 15M मुख्य ट्रेंड (20-EMA और 50-EMA के आधार पर)
    if price > ema20 and ema20 >= ema50:
        trend_15m = "UP"
        trend_label = "STRONG UP (20 & 50 EMA के ऊपर)"
    elif price < ema20 and ema20 <= ema50:
        trend_15m = "DOWN"
        trend_label = "STRONG DOWN (20 & 50 EMA के नीचे)"
    elif price > ema20:
        trend_15m = "UP"
        trend_label = "UP (15M 20-EMA के ऊपर)"
    else:
        trend_15m = "DOWN"
        trend_label = "DOWN (15M 20-EMA के नीचे)"

    sw_h, sw_l = find_swings(df.tail(96).reset_index(drop=True), n=3)
    swing_highs = [v for _, v in sw_h[-4:]]
    swing_lows = [v for _, v in sw_l[-4:]]
    rec_h = swing_highs[-1] if swing_highs else float(df["high"].tail(15).max())
    rec_l = swing_lows[-1] if swing_lows else float(df["low"].tail(15).min())
    swing_levels = swing_highs + swing_lows

    res = {
        "signal": "WAIT", "mode": None, "pattern": None, "price": price,
        "rsi_15m": round(rsi_15m, 2), "atr_15m": round(atr_15m, 2),
        "ema20": round(ema20, 1), "ema50": round(ema50, 1),
        "vol_ratio": round(vol_ratio, 2), "vwap": round(vwap, 2),
        "trend_15m": trend_15m, "trend_label": trend_label,
        "swing_levels": swing_levels,
        "recent_swing_high": rec_h, "recent_swing_low": rec_l
    }

    # बहुत ज़्यादा डेड मार्केट फ़िल्टर
    if atr_sma > 0 and not (0.45 <= (atr_15m / atr_sma) <= 3.0):
        return res

    side, pattern = check_15m_patterns(df, swing_levels, atr_15m, ema20, trend_15m)
    if not side or vol_ratio < VOLUME_MULTIPLIER:
        return res

    if (side == "BUY" and not (35 <= rsi_15m <= 72)) or (side == "SELL" and not (28 <= rsi_15m <= 65)):
        return res

    # 15M ट्रेंड के साथ सिग्नल (Mode A) या मज़बूत स्विंग ज़ोन रिवर्सल (Mode B)
    if (side == "BUY" and trend_15m == "UP") or (side == "SELL" and trend_15m == "DOWN"):
        res.update({"signal": side, "mode": "15M Trend Setup", "pattern": pattern})
    elif pattern in ["15M Liquidity Sweep", "15M Pin Bar (Hammer)", "15M Pin Bar (Shooting Star)"]:
        res.update({"signal": side, "mode": "15M Zone Reversal", "pattern": pattern})

    return res


def evaluate_trade(analysis: dict, df_15m: pd.DataFrame, funding_rate: float) -> dict:
    side, entry, atr = analysis["signal"], analysis["price"], analysis["atr_15m"]
    last_c = df_15m.iloc[-1]
    min_sl_dist = entry * MIN_SL_PCT

    if side == "BUY":
        raw_sl = min(float(last_c["low"]), analysis["recent_swing_low"]) - (0.25 * atr)
        sl_dist = max(entry - raw_sl, min_sl_dist)
        # बहुत बड़ा SL होने पर ATR से कैप करें ताकि R:R और Lot सही रहें
        sl_dist = min(sl_dist, max(min_sl_dist, 2.2 * atr))
        sl, tp1, tp2 = round(entry - sl_dist, 1), round(entry + 2.0 * sl_dist, 1), round(entry + 3.5 * sl_dist, 1)
    else:
        raw_sl = max(float(last_c["high"]), analysis["recent_swing_high"]) + (0.25 * atr)
        sl_dist = max(raw_sl - entry, min_sl_dist)
        sl_dist = min(sl_dist, max(min_sl_dist, 2.2 * atr))
        sl, tp1, tp2 = round(entry + sl_dist, 1), round(entry - 2.0 * sl_dist, 1), round(entry - 3.5 * sl_dist, 1)

    fee_usd = entry * ROUND_TRIP_FEE_PCT
    net_rr = round((abs(tp1 - entry) - fee_usd) / (sl_dist + fee_usd), 2)
    if net_rr < MIN_NET_RR:
        return {"approved": False}

    # नजदीकी 15M सपोर्ट/रेज़िस्टेंस की चेतावनी (ट्रेड ब्लॉक नहीं करेगा, सिर्फ़ अलर्ट देगा)
    obs_warn = None
    for lvl in analysis["swing_levels"]:
        if side == "BUY" and (entry + 0.3 * sl_dist) < lvl < tp1:
            obs_warn = f"ध्यान दें: ${lvl:,.0f} के पास 15M रेज़िस्टेंस है"
            break
        elif side == "SELL" and tp1 < lvl < (entry - 0.3 * sl_dist):
            obs_warn = f"ध्यान दें: ${lvl:,.0f} के पास 15M सपोर्ट है"
            break

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
# 6. 2D एनीमे गर्ल + YouTube लोगो (@Jaanibhai_9) + 9:16 Reels वीडियो + हिंदी आवाज़
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


def render_chart(df: pd.DataFrame, title: str, filename: str, hlines=None, colors=None, figsize=(10, 5.6), dpi=115, show_ema=True) -> str:
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
    if show_ema and len(plot_df) >= 20:
        ema20_line = plot_df["close"].ewm(span=EMA_FAST_15M, adjust=False).mean()
        kwargs["addplot"] = mpf.make_addplot(ema20_line, color="#FFD600", width=1.4)

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


def draw_anime_girl_avatar(size: int = 205, mouth_state: int = 0, blink: bool = False, accent=(56, 189, 248)) -> Image.Image:
    S = size * 2
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    d.ellipse([(10, 10), (S - 10, S - 10)], fill=(18, 28, 52, 255), outline=accent + (255,), width=10)

    hair_dark = (32, 24, 62, 255)
    hair_hi = (68, 48, 128, 255)
    d.ellipse([(S * 0.16, S * 0.12), (S * 0.84, S * 0.88)], fill=hair_dark)

    d.chord([(S * 0.14, S * 0.68), (S * 0.86, S * 1.12)], 180, 360, fill=(28, 37, 65, 255), outline=accent + (255,), width=5)
    d.polygon([(S * 0.42, S * 0.72), (S * 0.58, S * 0.72), (S * 0.50, S * 0.86)], fill=(255, 214, 186, 255))

    skin = (255, 218, 192, 255)
    d.rectangle([(S * 0.44, S * 0.62), (S * 0.56, S * 0.75)], fill=(242, 196, 168, 255))
    d.ellipse([(S * 0.28, S * 0.23), (S * 0.72, S * 0.69)], fill=skin)

    d.arc([(S * 0.18, S * 0.13), (S * 0.82, S * 0.72)], 185, 355, fill=(45, 55, 78, 255), width=16)
    d.rounded_rectangle([(S * 0.16, S * 0.38), (S * 0.28, S * 0.57)], radius=18, fill=(30, 41, 59, 255), outline=accent + (255,), width=5)
    d.rounded_rectangle([(S * 0.72, S * 0.38), (S * 0.84, S * 0.57)], radius=18, fill=(30, 41, 59, 255), outline=accent + (255,), width=5)

    d.polygon([(S * 0.24, S * 0.38), (S * 0.35, S * 0.19), (S * 0.50, S * 0.17), (S * 0.65, S * 0.19), (S * 0.76, S * 0.38),
               (S * 0.66, S * 0.31), (S * 0.57, S * 0.39), (S * 0.49, S * 0.29), (S * 0.40, S * 0.39), (S * 0.33, S * 0.31)], fill=hair_dark)
    d.arc([(S * 0.30, S * 0.20), (S * 0.70, S * 0.34)], 200, 340, fill=hair_hi, width=6)

    if blink:
        d.arc([(S * 0.34, S * 0.45), (S * 0.45, S * 0.51)], 190, 350, fill=(25, 20, 40, 255), width=6)
        d.arc([(S * 0.55, S * 0.45), (S * 0.66, S * 0.51)], 190, 350, fill=(25, 20, 40, 255), width=6)
    else:
        for ex in [0.395, 0.605]:
            cx, cy = S * ex, S * 0.475
            d.ellipse([(cx - 24, cy - 22), (cx + 24, cy + 22)], fill=(255, 255, 255, 255))
            d.ellipse([(cx - 17, cy - 17), (cx + 17, cy + 17)], fill=accent + (255,))
            d.ellipse([(cx - 10, cy - 10), (cx + 10, cy + 10)], fill=(20, 20, 45, 255))
            d.ellipse([(cx - 10, cy - 13), (cx - 1, cy - 4)], fill=(255, 255, 255, 255))
            d.ellipse([(cx + 4, cy + 4), (cx + 10, cy + 10)], fill=(255, 255, 255, 255))
            d.arc([(cx - 26, cy - 26), (cx + 26, cy + 10)], 200, 340, fill=(25, 20, 40, 255), width=7)

    d.ellipse([(S * 0.31, S * 0.52), (S * 0.38, S * 0.56)], fill=(255, 145, 165, 140))
    d.ellipse([(S * 0.62, S * 0.52), (S * 0.69, S * 0.56)], fill=(255, 145, 165, 140))
    d.ellipse([(S * 0.493, S * 0.53), (S * 0.507, S * 0.545)], fill=(210, 155, 130, 255))

    if mouth_state == 0:
        d.arc([(S * 0.45, S * 0.56), (S * 0.55, S * 0.62)], 15, 165, fill=(180, 60, 85, 255), width=6)
    elif mouth_state == 1:
        d.ellipse([(S * 0.46, S * 0.575), (S * 0.54, S * 0.618)], fill=(195, 55, 85, 255), outline=(120, 30, 50, 255), width=3)
    else:
        d.ellipse([(S * 0.45, S * 0.568), (S * 0.55, S * 0.635)], fill=(175, 40, 70, 255), outline=(110, 25, 45, 255), width=3)
        d.chord([(S * 0.465, S * 0.595), (S * 0.535, S * 0.632)], 180, 360, fill=(255, 130, 150, 255))

    d.line([(S * 0.22, S * 0.54), (S * 0.38, S * 0.62)], fill=(148, 163, 184, 255), width=6)
    d.ellipse([(S * 0.36, S * 0.60), (S * 0.41, S * 0.64)], fill=accent + (255,))

    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([(10, 10), (S - 10, S - 10)], fill=255)
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)
    return out.resize((size, size), Image.Resampling.LANCZOS)


def draw_youtube_badge(draw: ImageDraw.ImageDraw, W: int, y_top: int, handle: str, font):
    bbox = draw.textbbox((0, 0), handle, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    logo_w, logo_h = 46, 32
    gap = 14
    pad_x, pad_y = 26, 12

    total_w = logo_w + gap + tw
    pill_w = total_w + pad_x * 2
    pill_h = max(logo_h, th) + pad_y * 2

    px1 = (W - pill_w) // 2
    py1 = y_top
    px2 = px1 + pill_w
    py2 = py1 + pill_h

    draw.rounded_rectangle([(px1, py1), (px2, py2)], radius=pill_h // 2, fill=(22, 32, 56), outline=(255, 0, 0), width=2)

    lx1 = px1 + pad_x
    ly1 = py1 + (pill_h - logo_h) // 2
    lx2 = lx1 + logo_w
    ly2 = ly1 + logo_h
    draw.rounded_rectangle([(lx1, ly1), (lx2, ly2)], radius=8, fill=(255, 0, 0))
    tri = [
        (lx1 + 17, ly1 + 8),
        (lx1 + 17, ly2 - 8),
        (lx1 + 32, (ly1 + ly2) // 2)
    ]
    draw.polygon(tri, fill=(255, 255, 255))

    tx = lx2 + gap
    ty = py1 + (pill_h - th) // 2 - 3
    draw.text((tx, ty), handle, font=font, fill=(255, 255, 255))


def build_reel_frame(chart_im: Image.Image, action: str, curr_price: float, analysis: dict,
                     trade=None, step_pct: float = 1.0, mouth_state: int = 0, blink: bool = False) -> np.ndarray:
    W, H = 720, 1280
    canvas = Image.new("RGB", (W, H), (11, 17, 32))
    draw = ImageDraw.Draw(canvas)

    f_title = load_font(38, True)
    f_badge = load_font(44, True)
    f_med = load_font(24, True)
    f_small = load_font(22, False)
    f_yt = load_font(28, True)

    if action == "BUY":
        badge_bg, badge_txt = (0, 180, 95), "15M SIGNAL: BUY NOW"
    elif action == "SELL":
        badge_bg, badge_txt = (225, 35, 65), "15M SIGNAL: SELL NOW"
    else:
        badge_bg, badge_txt = (230, 145, 10), "15M DECISION: WAIT"

    draw.rectangle([(0, 0), (W, 125)], fill=(18, 28, 50))
    draw.text((38, 24), "BTCUSD 15M LIVE ANALYSIS", font=f_title, fill=(255, 255, 255))
    now_str = datetime.datetime.now(IST).strftime("%d %b | %I:%M %p IST")
    draw.text((38, 78), f"Delta Exchange India  •  {now_str}", font=f_small, fill=(148, 163, 184))

    draw.rounded_rectangle([(38, 146), (W - 38, 232)], radius=20, fill=badge_bg)
    bbox = draw.textbbox((0, 0), badge_txt, font=f_badge)
    draw.text(((W - (bbox[2] - bbox[0])) // 2, 162), badge_txt, font=f_badge, fill=(255, 255, 255))

    cx = (W - chart_im.width) // 2
    cy = 256 + (505 - chart_im.height) // 2
    canvas.paste(chart_im, (cx, cy))

    bar_w = int((W - 76) * min(1.0, max(0.05, step_pct)))
    draw.rounded_rectangle([(38, 782), (W - 38, 794)], radius=6, fill=(30, 41, 59))
    draw.rounded_rectangle([(38, 782), (38 + bar_w, 794)], radius=6, fill=badge_bg)

    draw.rounded_rectangle([(38, 818), (W - 38, 1168)], radius=22, fill=(18, 28, 50), outline=(51, 65, 85), width=2)

    if trade:
        lines = [
            (f"ENTRY: ${trade['entry']:,.1f}", (56, 189, 248)),
            (f"SL: ${trade['sl']:,.1f}", (248, 113, 113)),
            (f"TP1: ${trade['tp1']:,.1f}", (74, 222, 128)),
            (f"TP2: ${trade['tp2']:,.1f}", (0, 230, 118)),
            (f"LOTS: {trade['lots']} | R:R 1:{trade['net_rr']}", (250, 204, 21)),
        ]
    else:
        lines = [
            (f"LIVE: ${curr_price:,.1f}", (56, 189, 248)),
            (f"15M TREND: {analysis['trend_15m']}", (248, 250, 252)),
            (f"20-EMA: ${analysis['ema20']:,.0f}", (253, 224, 71)),
            (f"RSI: {analysis['rsi_15m']} | VOL: {analysis['vol_ratio']}x", (250, 204, 21)),
            (f"VWAP: ${analysis['vwap']:,.0f}", (251, 146, 60)),
        ]

    y_pos = 852
    for txt, col in lines:
        draw.text((60, y_pos), txt, font=f_med, fill=col)
        y_pos += 58

    girl = draw_anime_girl_avatar(size=205, mouth_state=mouth_state, blink=blink, accent=badge_bg)
    canvas.paste(girl, (W - 258, 892), girl)

    draw_youtube_badge(draw, W, 1192, YOUTUBE_HANDLE, f_yt)

    return np.array(canvas)


async def _edge_tts_save(script: str, out_mp3: str):
    comm = edge_tts.Communicate(script, voice="hi-IN-SwaraNeural", rate="+5%")
    await comm.save(out_mp3)


def generate_female_hindi_voice(script: str, out_mp3: str) -> bool:
    try:
        asyncio.run(_edge_tts_save(script, out_mp3))
        if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 2000:
            print("Voice generated via edge-tts (Swara Female).")
            return True
    except Exception as e:
        print("edge-tts failed, switching to gTTS backup:", e)

    try:
        tts = gTTS(text=script, lang="hi", slow=False)
        tts.save(out_mp3)
        if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 2000:
            print("Voice generated via gTTS (Hindi Female).")
            return True
    except Exception as e:
        print("gTTS backup also failed:", e)
    return False


def create_animated_reel_with_voice(df_15m: pd.DataFrame, curr_price: float, analysis: dict, trade=None, out_mp4="btc_reel.mp4") -> str:
    raw_mp4 = "raw_anim.mp4"
    voice_mp3 = "voice_hi.mp3"
    tmp_png = "anim_step.png"

    try:
        action = trade["side"] if trade else "WAIT"
        full_df = df_15m.tail(48).reset_index(drop=True)

        p_int = int(round(curr_price))
        if trade:
            side_word = "खरीदने यानी बाय" if trade["side"] == "BUY" else "बेचने यानी सेल"
            script = (
                f"नमस्ते ट्रेडर्स! बिटकॉइन के 15 मिनट चार्ट पर पक्का {side_word} का सिग्नल बना है। "
                f"अभी लाइव प्राइस {p_int} डॉलर है। "
                f"एंट्री {int(trade['entry'])} डॉलर पर, स्टॉप लॉस {int(trade['sl'])} डॉलर, "
                f"और पहला टारगेट {int(trade['tp1'])} डॉलर का है। "
                f"सिर्फ़ 500 रुपये के फिक्स रिस्क के साथ ट्रेड करें। "
                f"ऐसे ही लाइव अपडेट्स के लिए यूट्यूब पर जानी भाई नाइन को सब्सक्राइब करें।"
            )
        else:
            trend_word = "नीचे यानी डाउन" if analysis["trend_15m"] == "DOWN" else "ऊपर यानी अप"
            script = (
                f"नमस्ते ट्रेडर्स! बिटकॉइन अभी {p_int} डॉलर के पास चल रहा है। "
                f"15 मिनट के चार्ट पर ट्रेंड {trend_word} है, और आर एस आई {int(round(analysis['rsi_15m']))} है। "
                f"अभी कोई पक्का सेटअप नहीं बना है, इसलिए फैसला है वेट। "
                f"अभी न खरीदें, न बेचें, सही मौके का इंतज़ार करें। "
                f"ऐसे ही लाइव अपडेट्स के लिए यूट्यूब पर जानी भाई नाइन को सब्सक्राइब करें।"
            )

        has_audio = generate_female_hindi_voice(script, voice_mp3)

        steps = list(range(14, len(full_df) + 1, 2))
        if steps[-1] != len(full_df):
            steps.append(len(full_df))

        mouth_cycle = [0, 1, 2, 1, 2, 0, 1, 2]
        frame_idx = 0
        last_chart_im = None

        with imageio.get_writer(
            raw_mp4, fps=4, codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"]
        ) as writer:
            for idx, count in enumerate(steps):
                sub_df = full_df.iloc[:count]
                is_last = (idx == len(steps) - 1)
                if is_last and trade:
                    hlines = [trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]]
                    colors = ["#38BDF8", "#F87171", "#4ADE80", "#00E676"]
                elif is_last:
                    hlines = [curr_price]
                    colors = ["#38BDF8"]
                else:
                    hlines = [float(sub_df["close"].iloc[-1])]
                    colors = ["#38BDF8"]

                chart_title = f"BTCUSD 15M (IST)  •  LIVE: ${curr_price:,.1f}"
                render_chart(
                    sub_df, chart_title,
                    tmp_png, hlines=hlines, colors=colors, figsize=(7.0, 5.0), dpi=100, show_ema=True
                )
                c_im = Image.open(tmp_png).convert("RGB")
                c_im.thumbnail((680, 500), Image.Resampling.LANCZOS)
                last_chart_im = c_im

                for _ in range(3):
                    fr_arr = build_reel_frame(
                        c_im, action, curr_price, analysis, trade=trade,
                        step_pct=(idx + 1) / len(steps),
                        mouth_state=mouth_cycle[frame_idx % len(mouth_cycle)],
                        blink=(frame_idx % 16 == 0)
                    )
                    writer.append_data(fr_arr)
                    frame_idx += 1

            for _ in range(80):
                fr_arr = build_reel_frame(
                    last_chart_im, action, curr_price, analysis, trade=trade,
                    step_pct=1.0,
                    mouth_state=mouth_cycle[frame_idx % len(mouth_cycle)],
                    blink=(frame_idx % 16 == 0)
                )
                writer.append_data(fr_arr)
                frame_idx += 1

        if has_audio:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", raw_mp4,
                "-i", voice_mp3,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "128k",
                "-shortest",
                "-movflags", "+faststart",
                out_mp4
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and os.path.exists(out_mp4):
                print("Audio + Video merged successfully!")
            else:
                print("FFmpeg merge error:", res.stderr[-300:])
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


def send_manual_status(df_15m: pd.DataFrame, curr_price: float, fr: float, analysis: dict, reason: str):
    opt_note = get_atm_options_note(curr_price)
    now_ist = datetime.datetime.now(IST).strftime("%d-%b %I:%M %p IST")

    caption = (
        f"⏳ **15M फैसला: WAIT (अभी इंतज़ार करें — न खरीदें, न बेचें)**\n"
        f"🕒 समय: `{now_ist}`\n"
        f"💰 **लाइव प्राइस:** `${curr_price:,.1f}`\n"
        f"📈 **15M ट्रेंड:** {analysis['trend_label']}\n"
        f"📊 **15M RSI:** `{analysis['rsi_15m']}` | **वॉल्यूम:** `{analysis['vol_ratio']}x`\n"
        f"📌 **20-EMA:** `${analysis['ema20']:,.1f}` | **Weekly VWAP:** `${analysis['vwap']:,.1f}` | **Funding:** `{round(fr, 4)}%`\n"
        f"🔍 **कारण:** {reason}\n\n"
        f"{opt_note}\n"
        f"▶️ **YouTube:** `{YOUTUBE_HANDLE}`"
    )

    hlines = [curr_price]
    colors = ["#38BDF8"]
    if abs(analysis["vwap"] - curr_price) / curr_price <= 0.05:
        hlines.append(analysis["vwap"])
        colors.append("#FF9800")

    photo_file = render_chart(
        df_15m.tail(55), f"BTCUSD 15M (IST) | WAIT | LIVE: ${curr_price:,.1f} | RSI: {analysis['rsi_15m']}",
        "status_chart.png", hlines=hlines, colors=colors, show_ema=True
    )
    msg_id = send_tg_photo(photo_file, caption)

    vid_file = create_animated_reel_with_voice(df_15m, curr_price, analysis, trade=None)
    send_tg_video(
        vid_file,
        f"🎬 **15M Reels / Shorts वीडियो (हिंदी आवाज़ सहित)**\n⏳ फैसला: **WAIT (अभी इंतज़ार करें)**\n▶️ YouTube: `{YOUTUBE_HANDLE}`",
        reply_to=msg_id
    )


# ==============================================================================
# 7. मुख्य 24/7 15M One-Shot रनर (Cooldown = 0)
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
                f"🏆 **TP2 हिट (${t['tp2']:,.1f}) — ट्रेड सफल!**\nबॉट तुरंत अगले नए 15M सेटअप की तलाश में लग गया है।",
                reply_to=st["signal_msg_id"]
            )
            st.update({"state": "SCANNING", "trade": None, "signal_msg_id": None})
            save_state(st)
            return

        if (side == "BUY" and curr_price <= t["sl"]) or (side == "SELL" and curr_price >= t["sl"]):
            if t.get("tp1_hit"):
                msg = f"🛡️ **Breakeven SL हिट (${t['sl']:,.1f})** (TP1 पहले ही बुक हो चुका था)।\nबॉट तुरंत अगले नए 15M सेटअप के लिए तैयार है।"
            else:
                msg = f"🛑 **Stop Loss हिट (${t['sl']:,.1f})।**\nबॉट तुरंत अगले नए 15M सेटअप के लिए तैयार है।"
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
    df_15m = fetch_candles("15m", 750)
    if df_15m.empty:
        print("15M data unavailable.")
        return

    latest_ts = int(df_15m["time"].iloc[-1])
    if latest_ts <= st["last_candle_ts"]:
        if is_manual:
            analysis = analyze_market_15m(df_15m)
            send_manual_status(df_15m, curr_price, fr, analysis, "अभी कोई पक्का 15M सेटअप नहीं बना है (24/7 स्कैनिंग चालू है)।")
        print("Same 15M candle already scanned.")
        return

    st["last_candle_ts"] = latest_ts
    analysis = analyze_market_15m(df_15m)

    if analysis["signal"] == "WAIT":
        if is_manual:
            send_manual_status(df_15m, curr_price, fr, analysis, "अभी कोई पक्का 15M सेटअप नहीं बना है (24/7 स्कैनिंग चालू है)।")
        print("No 15M setup found. Staying quiet.")
        save_state(st)
        return

    trade = evaluate_trade(analysis, df_15m, fr)
    if not trade["approved"]:
        if is_manual:
            send_manual_status(df_15m, curr_price, fr, analysis, "15M सेटअप बना था, लेकिन Net Risk/Reward 1:1.5 से कम था।")
        print("Trade rejected by Net R:R filter.")
        save_state(st)
        return

    # --- पक्का 15M BUY या SELL सिग्नल ---
    opt_note = get_atm_options_note(trade["entry"])
    mode_tag = "📝 [PAPER]" if PAPER_MODE else "🚨 [LIVE]"
    action_header = "🟢 **15M फैसला: BUY (खरीदें) — BTCUSD**" if trade["side"] == "BUY" else "🔴 **15M फैसला: SELL (बेचें) — BTCUSD**"

    warn_text = f"\n⚠️ {trade['obstacle_warning']}" if trade["obstacle_warning"] else ""
    if trade["funding_warn"]:
        warn_text += f"\n{trade['funding_warn']}"

    caption = (
        f"{mode_tag} {action_header}\n"
        f"• **सेटअप:** {trade['mode']} | {trade['pattern']}\n"
        f"• **पुष्टि:** वॉल्यूम `{analysis['vol_ratio']}x` | 15M RSI `{analysis['rsi_15m']}` | {analysis['trend_label']}\n\n"
        f"🎯 **Entry:** `${trade['entry']:,.1f}`\n"
        f"🛑 **Stop Loss:** `${trade['sl']:,.1f}`\n"
        f"✅ **Target 1:** `${trade['tp1']:,.1f}` *(TP1 पर SL ऑटो Entry पर शिफ़्ट)*\n"
        f"🏆 **Target 2:** `${trade['tp2']:,.1f}`\n\n"
        f"⚖️ **Net R:R (फ़ीस+GST बाद):** `1:{trade['net_rr']}`\n"
        f"📦 **सुझावित Lot Size:** `{trade['lots']} Lots` *(रिस्क: ₹{int(trade['risk_inr'])})*\n"
        f"📌 **15M 20-EMA:** `${analysis['ema20']:,.1f}` | **Weekly VWAP:** `${analysis['vwap']:,.1f}` | **Funding:** `{trade['funding_rate_pct']}%`"
        f"{warn_text}\n\n{opt_note}\n"
        f"▶️ **YouTube:** `{YOUTUBE_HANDLE}`"
    )

    hlines = [trade["entry"], trade["sl"], trade["tp1"], trade["tp2"]]
    colors = ["#38BDF8", "#F87171", "#4ADE80", "#00E676"]
    if abs(analysis["vwap"] - trade["entry"]) / trade["entry"] <= 0.03:
        hlines.append(analysis["vwap"])
        colors.append("#FF9800")

    photo_file = render_chart(
        df_15m.tail(55), f"BTCUSD 15M (IST) | {trade['side']} ({trade['pattern']})",
        "signal_chart.png", hlines=hlines, colors=colors, show_ema=True
    )
    msg_id = send_tg_photo(photo_file, caption)

    vid_file = create_animated_reel_with_voice(df_15m, trade["entry"], analysis, trade=trade)
    send_tg_video(
        vid_file,
        f"🎬 **15M Reels / Shorts वीडियो (हिंदी आवाज़ सहित)**\n{action_header}\n▶️ YouTube: `{YOUTUBE_HANDLE}`",
        reply_to=msg_id
    )

    st.update({"state": "ACTIVE_TRADE", "trade": trade, "signal_msg_id": msg_id})
    save_state(st)
    print("15M Signal photo + Reels/Shorts Hindi voice video sent!")


if __name__ == "__main__":
    main()
