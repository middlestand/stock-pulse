import json
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd

PAIRS = [
    {"key": "USDJPY", "ticker": "USDJPY=X", "name": "ドル円", "code": "USD/JPY", "unit": "円", "digits": 2},
    {"key": "EURUSD", "ticker": "EURUSD=X", "name": "ユーロドル", "code": "EUR/USD", "unit": "ドル", "digits": 4},
    {"key": "GBPJPY", "ticker": "GBPJPY=X", "name": "ポンド円", "code": "GBP/JPY", "unit": "円", "digits": 2},
    {"key": "AUDJPY", "ticker": "AUDJPY=X", "name": "豪ドル円", "code": "AUD/JPY", "unit": "円", "digits": 2},
]

def clamp(val, min_v=-100, max_v=100):
    return max(min_v, min(max_v, int(round(val))))

def get_tnx_diff():
    try:
        df = yf.download("^TNX", period="5d", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        if len(df) >= 2:
            return float(df["Close"].iloc[-1]) - float(df["Close"].iloc[-2])
    except Exception:
        pass
    return 0.0

def analyze_fx(info, dy_1h, now_jst):
    ticker = info["ticker"]
    digits = info["digits"]

    df_1h = yf.download(ticker, period="7d", interval="1h", progress=False)
    df_1d = yf.download(ticker, period="10d", interval="1d", progress=False)

    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.get_level_values(0)

    df_1h = df_1h.dropna(subset=["Close"])
    df_1d = df_1d.dropna(subset=["Close"])

    current_price = float(df_1h["Close"].iloc[-1])
    prev_close = float(df_1d["Close"].iloc[-2]) if len(df_1d) >= 2 else float(df_1d["Open"].iloc[-1])

    today_open = float(df_1d["Open"].iloc[-1])
    today_high = float(df_1d["High"].iloc[-1])
    today_low = float(df_1d["Low"].iloc[-1])

    # 1. 前日終値比（日足の地合い）
    diff = current_price - prev_close
    diff_pct = (diff / prev_close) * 100 if prev_close else 0.0
    score_day = clamp(diff_pct * 95)
    day_label = "買い優勢" if score_day >= 20 else ("売り優勢" if score_day <= -20 else "拮抗")
    day_desc = (
        "前日終値を上回って推移。本日の地合いは買い方向が優勢です。"
        if diff >= 0 else
        "前日終値を下回って推移。本日の地合いは売り方向が優勢です。"
    )

    # 2. 直近1時間の勢い ＋ 移動平均（EMA20）との位置
    p_1h = float(df_1h["Close"].iloc[-2]) if len(df_1h) >= 2 else current_price
    mom_1h = current_price - p_1h
    mom_1h_pct = (mom_1h / p_1h) * 100 if p_1h else 0.0
    ema20 = float(df_1h["Close"].tail(20).mean())
    gap_pct = ((current_price - ema20) / ema20) * 100 if ema20 else 0.0

    # 離れすぎ（過熱）の反落警戒補正
    if gap_pct > 0.45:
        gap_score = -15
    elif gap_pct < -0.45:
        gap_score = 15
    else:
        gap_score = gap_pct * 60

    score_1h = clamp(mom_1h_pct * 160 + gap_score)
    trend_label = "上昇基調" if score_1h >= 20 else ("下落基調" if score_1h <= -20 else "横ばい")
    trend_desc = (
        "短期移動平均を上回り、2時間後まで続きやすい初動が出ています。"
        if current_price >= ema20 else
        "短期移動平均を下回り、上値の重い推移が続いています。"
    )

    # 3. 米国10年債利回りの先行シグナル ＋ 時間帯フィルター
    rate_dir = -1 if info["key"] == "EURUSD" else 1
    score_macro = clamp(dy_1h * 900 * rate_dir)
    macro_label = "追い風" if score_macro >= 20 else ("逆風" if score_macro <= -20 else "中立")

    h = now_jst.hour
    wd = now_jst.weekday()
    is_weekend = (wd == 5 and h >= 7) or (wd == 6) or (wd == 0 and h < 6)

    if is_weekend:
        time_note = "【週末休場中】"
        time_mult = 0.3
    elif h in {9, 16, 17, 21, 22}:
        time_note = "流動性が高くトレンドが出やすい時間帯です。"
        time_mult = 1.2
    elif h in {5, 6, 7, 12, 13, 14}:
        time_note = "値動きが停滞・反転しやすい時間帯です。"
        time_mult = 0.6
    else:
        time_note = "標準的な市場流動性の時間帯です。"
        time_mult = 1.0

    macro_desc = f"米10年債金利の直近変化({dy_1h:+.3f}%)を加味。{time_note}"

    # 総合スコア算出（ダマシを減らすため ±35点以上を明確なシグナルとする）
    raw_total = (score_day * 0.35 + score_1h * 0.40 + score_macro * 0.25) * time_mult
    total_score = clamp(raw_total)

    if total_score >= 35:
        signal = "上昇予測"
    elif total_score <= -35:
        signal = "下落予測"
    else:
        signal = "様子見"

    atr = float((df_1h["High"] - df_1h["Low"]).tail(14).mean())
    if pd.isna(atr) or atr == 0:
        atr = current_price * 0.002
    range_low = round(current_price - atr * 0.7, digits)
    range_high = round(current_price + atr * 1.3 if total_score >= 0 else current_price + atr * 0.7, digits)

    chart_prices = [round(float(x), digits) for x in df_1h["Close"].tail(15).tolist()]

    return {
        "key": info["key"],
        "name": info["name"],
        "code": info["code"],
        "unit": info["unit"],
        "price": round(current_price, digits),
        "diff": round(diff, digits),
        "diff_pct": round(diff_pct, 2),
        "prev_close": round(prev_close, digits),
        "open": round(today_open, digits),
        "high": round(today_high, digits),
        "low": round(today_low, digits),
        "range_low": range_low,
        "range_high": range_high,
        "score": total_score,
        "signal": signal,
        "chart_prices": chart_prices,
        "reasons": [
            {
                "title": "前日終値比の地合い",
                "badge": day_label,
                "desc": day_desc,
                "metric": f"{diff:+.{digits}f} ({diff_pct:+.2f}%)",
                "score": score_day
            },
            {
                "title": "直近1時間のトレンド",
                "badge": trend_label,
                "desc": trend_desc,
                "metric": f"{mom_1h:+.{digits}f} (直近1時間)",
                "score": score_1h
            },
            {
                "title": "米金利・時間帯の追い風",
                "badge": "休場中" if is_weekend else macro_label,
                "desc": macro_desc,
                "metric": f"米10年債 {dy_1h:+.3f}%",
                "score": score_macro
            }
        ]
    }

def main():
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst)
    h_start = (now + timedelta(hours=2)).strftime("%H:%M")
    h_end = (now + timedelta(hours=3)).strftime("%H:%M")

    dy_1h = get_tnx_diff()

    fx_data = {}
    for item in PAIRS:
        try:
            fx_data[item["key"]] = analyze_fx(item, dy_1h, now)
        except Exception as e:
            print(f"Error {item['key']}: {e}")

    output = {
        "updated_at_jst": now.strftime("%Y/%m/%d %H:%M:%S JST"),
        "horizon_range": f"{h_start} 〜 {h_end} (JST)",
        "stocks": fx_data
    }

    with open("stock_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
