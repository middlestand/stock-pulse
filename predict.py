import json
from datetime import datetime, timezone, timedelta
import yfinance as yf
import pandas as pd

TARGETS = [
    {"key": "N225", "ticker": "^N225", "name": "日経平均株価", "code": "N225", "unit": "円"},
    {"key": "7203", "ticker": "7203.T", "name": "トヨタ自動車", "code": "7203", "unit": "円"},
    {"key": "9984", "ticker": "9984.T", "name": "ソフトバンクグループ", "code": "9984", "unit": "円"},
    {"key": "NVDA", "ticker": "NVDA", "name": "NVIDIA", "code": "NVDA", "unit": "ドル"},
]

def clamp(val, min_v=-100, max_v=100):
    return max(min_v, min(max_v, int(round(val))))

def analyze_stock(info):
    ticker = info["ticker"]
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

    diff = current_price - prev_close
    diff_pct = (diff / prev_close) * 100
    score_day = clamp(diff_pct * 45)
    day_label = "強い" if score_day >= 25 else ("弱い" if score_day <= -25 else "中立")
    day_desc = (
        "前日終値を上回って推移。買い優勢が続き、勢いは上向きです。"
        if diff_pct >= 0 else
        "前日終値を下回って推移。売り圧力がやや優勢な展開です。"
    )

    p_1h = float(df_1h["Close"].iloc[-2]) if len(df_1h) >= 2 else current_price
    mom_1h_pct = ((current_price - p_1h) / p_1h) * 100
    ema20 = float(df_1h["Close"].tail(20).mean())
    gap_pct = ((current_price - ema20) / ema20) * 100
    score_1h = clamp(mom_1h_pct * 90 + gap_pct * 25)
    trend_label = "上昇基調" if score_1h >= 20 else ("下落基調" if score_1h <= -20 else "横ばい")
    trend_desc = (
        "直近1時間は短期移動平均を上回る形で推移しています。"
        if current_price >= ema20 else
        "直近1時間は短期移動平均を下回り、上値の重い推移です。"
    )

    vol_series = df_1d["Volume"].dropna()
    if len(vol_series) >= 2 and float(vol_series.iloc[-2]) > 0:
        vol_ratio = float(vol_series.iloc[-1]) / float(vol_series.iloc[-2])
    else:
        vol_ratio = 1.0
    direction = 1 if diff_pct >= 0 else -1
    score_vol = clamp((vol_ratio - 0.8) * 60 * direction)
    vol_label = "活発" if vol_ratio >= 1.1 else ("低調" if vol_ratio <= 0.7 else "標準")
    vol_desc = (
        "売買が膨らんでおり、値動きに厚みがあります。"
        if vol_ratio >= 1.1 else
        "出来高は標準的な水準で推移しています。"
    )

    total_score = clamp(score_day * 0.4 + score_1h * 0.4 + score_vol * 0.2)
    if total_score >= 25:
        signal = "上昇予測"
    elif total_score <= -25:
        signal = "下落予測"
    else:
        signal = "様子見"

    atr = float((df_1h["High"] - df_1h["Low"]).tail(14).mean())
    if pd.isna(atr) or atr == 0:
        atr = current_price * 0.005
    range_low = round(current_price - atr * 0.6, 2)
    range_high = round(current_price + atr * 1.2 if total_score >= 0 else current_price + atr * 0.6, 2)

    chart_prices = [round(float(x), 2) for x in df_1h["Close"].tail(15).tolist()]

    return {
        "key": info["key"],
        "name": info["name"],
        "code": info["code"],
        "unit": info["unit"],
        "price": round(current_price, 2),
        "diff": round(diff, 2),
        "diff_pct": round(diff_pct, 2),
        "prev_close": round(prev_close, 2),
        "open": round(today_open, 2),
        "high": round(today_high, 2),
        "low": round(today_low, 2),
        "range_low": range_low,
        "range_high": range_high,
        "score": total_score,
        "signal": signal,
        "chart_prices": chart_prices,
        "reasons": [
            {
                "title": "前日終値比の勢い",
                "badge": day_label,
                "desc": day_desc,
                "metric": f"{diff_pct:+.2f}%",
                "score": score_day
            },
            {
                "title": "直近1時間のトレンド",
                "badge": trend_label,
                "desc": trend_desc,
                "metric": f"{mom_1h_pct:+.2f}% (直近1時間)",
                "score": score_1h
            },
            {
                "title": "出来高の活発度",
                "badge": vol_label,
                "desc": vol_desc,
                "metric": f"前日比 {vol_ratio:.2f}倍",
                "score": score_vol
            }
        ]
    }

def main():
    jst = timezone(timedelta(hours=9))
    now = datetime.now(jst)
    h_start = (now + timedelta(hours=2)).strftime("%H:%M")
    h_end = (now + timedelta(hours=3)).strftime("%H:%M")

    stocks_data = {}
    for item in TARGETS:
        try:
            stocks_data[item["key"]] = analyze_stock(item)
        except Exception as e:
            print(f"Error {item['key']}: {e}")

    output = {
        "updated_at_jst": now.strftime("%Y/%m/%d %H:%M:%S JST"),
        "horizon_range": f"{h_start} 〜 {h_end} (JST)",
        "stocks": stocks_data
    }

    with open("stock_data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
