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

def fetch_df(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])

def calc_rsi(series, period=14):
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_loss = float(loss.iloc[-1])
    last_gain = float(gain.iloc[-1])
    if pd.isna(last_loss) or last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0
    rs = last_gain / last_loss
    return 100.0 - (100.0 / (1.0 + rs))

def get_macro():
    macro = {"tnx_1h": 0.0, "dxy_1h_pct": 0.0, "sp500_1h_pct": 0.0}
    try:
        df_tnx = fetch_df("^TNX", "5d", "1h")
        if len(df_tnx) >= 2:
            macro["tnx_1h"] = float(df_tnx["Close"].iloc[-1]) - float(df_tnx["Close"].iloc[-2])
    except Exception:
        pass
    try:
        df_dxy = fetch_df("DX-Y.NYB", "5d", "1h")
        if len(df_dxy) >= 2:
            c0, c1 = float(df_dxy["Close"].iloc[-2]), float(df_dxy["Close"].iloc[-1])
            if c0 != 0:
                macro["dxy_1h_pct"] = ((c1 - c0) / c0) * 100
    except Exception:
        pass
    try:
        df_es = fetch_df("ES=F", "5d", "1h")
        if len(df_es) >= 2:
            c0, c1 = float(df_es["Close"].iloc[-2]), float(df_es["Close"].iloc[-1])
            if c0 != 0:
                macro["sp500_1h_pct"] = ((c1 - c0) / c0) * 100
    except Exception:
        pass
    return macro

def analyze_fx(info, macro, now_jst):
    ticker = info["ticker"]
    digits = info["digits"]
    key = info["key"]

    df_1h = fetch_df(ticker, "7d", "1h")
    df_1d = fetch_df(ticker, "10d", "1d")
    try:
        df_15m = fetch_df(ticker, "2d", "15m")
    except Exception:
        df_15m = pd.DataFrame()

    has_15m = len(df_15m) >= 3
    current_price = float(df_15m["Close"].iloc[-1]) if has_15m else float(df_1h["Close"].iloc[-1])
    prev_close = float(df_1d["Close"].iloc[-2]) if len(df_1d) >= 2 else float(df_1d["Open"].iloc[-1])

    today_open = float(df_1d["Open"].iloc[-1])
    today_high = float(df_1d["High"].iloc[-1])
    today_low = float(df_1d["Low"].iloc[-1])

    # 通貨ペアごとの標準値幅（1時間ATR）を算出してボラティリティを均一化
    atr = float((df_1h["High"] - df_1h["Low"]).tail(14).mean())
    if pd.isna(atr) or atr <= 0:
        atr = current_price * 0.002

    # 1. 上位足（日足＋1時間EMA20）の地合い
    diff = current_price - prev_close
    diff_pct = (diff / prev_close) * 100 if prev_close else 0.0
    ema20_1h = float(df_1h["Close"].tail(20).mean())

    day_norm = diff / (atr * 3.0)
    ema_norm = (current_price - ema20_1h) / atr
    score_day = clamp(day_norm * 35 + ema_norm * 35)

    day_label = "買い優勢" if score_day >= 20 else ("売り優勢" if score_day <= -20 else "拮抗")
    day_desc = (
        "日足・1時間移動平均ともに上向き。上位足のトレンドが買いを支えています。"
        if score_day >= 15 else (
            "日足・1時間移動平均ともに下向き。上位足の売り圧力が優勢です。"
            if score_day <= -15 else
            "前日比と短期移動平均の方向が交錯しており、方向感を探る展開です。"
        )
    )

    # 2. 15分足の初動 ＋ RSI過熱ブレーキ
    if has_15m:
        p_prev = float(df_15m["Close"].iloc[-3])  # 直近30分の変化
        mom_short = current_price - p_prev
        mom_norm = mom_short / (atr * 0.6)
        mom_label_txt = "直近30分"
    else:
        p_prev = float(df_1h["Close"].iloc[-2]) if len(df_1h) >= 2 else current_price
        mom_short = current_price - p_prev
        mom_norm = mom_short / atr
        mom_label_txt = "直近1時間"

    rsi_1h = calc_rsi(df_1h["Close"], 14)
    if rsi_1h >= 72:
        rsi_adj = -25
        rsi_note = f"RSI({rsi_1h:.0f})が買われすぎ水準のため反落を警戒。"
    elif rsi_1h <= 28:
        rsi_adj = 25
        rsi_note = f"RSI({rsi_1h:.0f})が売られすぎ水準のため反発を警戒。"
    else:
        rsi_adj = (rsi_1h - 50.0) * 0.45
        rsi_note = f"RSI({rsi_1h:.0f})は過熱感のない健全な水準です。"

    score_1h = clamp(mom_norm * 45 + rsi_adj)
    trend_label = "上昇初動" if score_1h >= 20 else ("下落初動" if score_1h <= -20 else "横ばい")
    trend_desc = f"{mom_label_txt}の値動き({mom_short:+.{digits}f})を分析。{rsi_note}"

    # 3. 3つの外部マクロ指標（米10年債金利・ドル指数DXY・米国株S&P500先物）＋ 時間帯
    tnx = macro["tnx_1h"]
    dxy = macro["dxy_1h_pct"]
    sp500 = macro["sp500_1h_pct"]

    if key == "USDJPY":
        raw_macro = tnx * 600 + dxy * 120 + sp500 * 40
        macro_metric = f"米金利{tnx:+.3f}% / DXY{dxy:+.2f}%"
    elif key == "EURUSD":
        raw_macro = -tnx * 500 - dxy * 150
        macro_metric = f"DXY{dxy:+.2f}% / 米金利{tnx:+.3f}%"
    elif key == "GBPJPY":
        raw_macro = tnx * 500 + sp500 * 75 + dxy * 50
        macro_metric = f"米株{sp500:+.2f}% / 米金利{tnx:+.3f}%"
    else:  # AUDJPY
        raw_macro = sp500 * 115 + tnx * 450
        macro_metric = f"米株{sp500:+.2f}% / 米金利{tnx:+.3f}%"

    score_macro = clamp(raw_macro)
    macro_label = "追い風" if score_macro >= 20 else ("逆風" if score_macro <= -20 else "中立")

    h = now_jst.hour
    wd = now_jst.weekday()
    is_weekend = (wd == 5 and h >= 7) or (wd == 6) or (wd == 0 and h < 6)

    if is_weekend:
        time_note = "【週末休場中】"
        time_mult = 0.25
    elif h in {9, 16, 17, 21, 22, 23}:
        time_note = "大口資金が入りトレンドが続きやすい時間帯です。"
        time_mult = 1.15
    elif h in {5, 6, 7, 12, 13, 14}:
        time_note = "流動性が低くダマシが出やすい時間帯です。"
        time_mult = 0.65
    else:
        time_note = "標準的な市場流動性の時間帯です。"
        time_mult = 1.0

    # 3要素の一致度（コンフルエンス）判定
    scores = [score_day, score_1h, score_macro]
    bull_cnt = sum(1 for s in scores if s >= 15)
    bear_cnt = sum(1 for s in scores if s <= -15)

    if bull_cnt == 3 or bear_cnt == 3:
        conf_mult = 1.25
        conf_note = "3要素すべて同方向へ一致。"
    elif (bull_cnt == 2 and bear_cnt == 0) or (bear_cnt == 2 and bull_cnt == 0):
        conf_mult = 1.0
        conf_note = "2要素が同方向へ一致。"
    else:
        conf_mult = 0.65
        conf_note = "指標間で方向が割れているため慎重判定。"

    macro_desc = f"{conf_note}{time_note}"

    raw_total = (score_day * 0.35 + score_1h * 0.35 + score_macro * 0.30) * conf_mult * time_mult
    total_score = clamp(raw_total)

    if total_score >= 35:
        signal = "上昇予測"
    elif total_score <= -35:
        signal = "下落予測"
    else:
        signal = "様子見"

    range_low = round(current_price - atr * 0.7, digits)
    range_high = round(current_price + atr * 1.3 if total_score >= 0 else current_price + atr * 0.7, digits)
    chart_prices = [round(float(x), digits) for x in df_1h["Close"].tail(15).tolist()]

    return {
        "key": key,
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
                "title": "上位足の地合い（日足・EMA）",
                "badge": day_label,
                "desc": day_desc,
                "metric": f"{diff:+.{digits}f} ({diff_pct:+.2f}%)",
                "score": score_day
            },
            {
                "title": "15分足初動・RSI過熱判定",
                "badge": trend_label,
                "desc": trend_desc,
                "metric": f"{mom_short:+.{digits}f} / RSI {rsi_1h:.0f}",
                "score": score_1h
            },
            {
                "title": "米金利・ドル・株の連動性",
                "badge": "休場中" if is_weekend else macro_label,
                "desc": macro_desc,
                "metric": macro_metric,
                "score": score_macro
            }
        ]
    }

def main():
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst)
    h_start = (now + timedelta(hours=2)).strftime("%H:%M")
    h_end = (now + timedelta(hours=3)).strftime("%H:%M")

    macro = get_macro()

    fx_data = {}
    for item in PAIRS:
        try:
            fx_data[item["key"]] = analyze_fx(item, macro, now)
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
