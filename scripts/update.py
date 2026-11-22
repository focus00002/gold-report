"""
Daily gold report for saving gold in TrueMoney (ออมทอง).

Plan ("daily quota, carry over"):
  - Every day you get a quota (default 100 THB).
  - If today's signal is good enough, buy using ALL quota saved so far.
  - If not, keep the quota and carry it to the next day.
  - Safety cap: if you have waited max_wait_days, buy anyway.
  - When the signal says "expensive" (level 1), suggest selling part of your gold.

What it does every run:
  1. Download 6 years of gold spot (GC=F) and USD/THB from Yahoo Finance.
  2. Convert to Thai gold price per 1 baht-weight (96.5% bar), calibrated
     with the real Gold Traders Association price saved each day.
  3. Compute indicators (MA50, MA200, RSI14, drawdown from 52-week high).
  4. Score every day 1-5 and run the plan from start_date to today.
  5. Backtest the plan vs. buying the same amount every day (1, 3, 5 years).
  6. Write docs/data.json (web page) and docs/report.csv (Google Sheets).

All settings are in CONFIG below. Edit them, or ask Claude to.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

# ----------------------------------------------------------------- CONFIG ---
CONFIG = {
    "daily_quota_thb": 100,        # quota added every day
    "start_date": "2026-11-22",    # day your quota starts counting
    "buy_min_level": 4,            # buy when signal level >= this (4 = น่าซื้อ, 5 = น่าซื้อมาก)
    "max_wait_days": 14,           # buy anyway after waiting this many days
    "sell_pct": 15,                # % of your gold to sell when level = 1
    "rsi_buy": 35,                 # RSI below this = buy point
    "rsi_sell": 70,                # RSI above this = sell point
    "dip_pct": 8,                  # % below 52-week high = buy point
    "stretch_ma200_pct": 15,       # % above MA200 = sell point
    "stretch_ma50_pct": 5,         # % above MA50 = sell point
    "near_high_pct": 2,            # within this % of 52-week high = sell point
}

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = ROOT / "docs"
ACTUAL_CSV = DATA / "thai_gold_actual.csv"     # real Thai price, one row per day
REPORT_CSV = DATA / "daily_report.csv"         # report rows (also copied to docs/)
PURCHASES_CSV = DATA / "my_purchases.csv"      # your real buys/sells (optional)

BKK = timezone(timedelta(hours=7))
GRAMS_PER_BAHT = 15.244
TROY_OZ_G = 31.1035
PURITY = 0.965
OZ_TO_BAHT = GRAMS_PER_BAHT / TROY_OZ_G * PURITY   # USD/oz * THB/USD * this = THB per baht-weight
THAI_API = "https://api.chnwt.dev/thai-gold-api/latest"
LOG_DAYS = 90

LEVELS = {
    5: ("น่าซื้อมาก", "good"),
    4: ("น่าซื้อ", "good"),
    3: ("ปกติ", "neutral"),
    2: ("ราคาแพง", "warning"),
    1: ("พิจารณาขายบางส่วน", "critical"),
}


# --------------------------------------------------------------- helpers ---
def num(s: str) -> float:
    return float(str(s).replace(",", "").strip())


def r(x, n=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), n)


# ------------------------------------------------------------ data fetch ---
def fetch_history() -> pd.DataFrame:
    raw = yf.download(["GC=F", "THB=X"], period="6y", interval="1d",
                      progress=False, auto_adjust=False)["Close"]
    df = raw.rename(columns={"GC=F": "usd_oz", "THB=X": "usdthb"})
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.sort_index().ffill().dropna()
    df["est"] = df["usd_oz"] * df["usdthb"] * OZ_TO_BAHT
    return df


def fetch_thai_today() -> dict | None:
    try:
        j = requests.get(THAI_API, timeout=20).json()
        p = j["response"]["price"]["gold_bar"]
        return {
            "bar_buy": num(p["buy"]),
            "bar_sell": num(p["sell"]),
            "note": f'{j["response"]["update_date"]} {j["response"]["update_time"]}',
        }
    except Exception as e:  # source down: fall back to estimate
        print("Thai price fetch failed:", e)
        return None


def load_actual() -> pd.DataFrame:
    if ACTUAL_CSV.exists():
        a = pd.read_csv(ACTUAL_CSV, parse_dates=["date"])
        return a.set_index("date").sort_index()
    return pd.DataFrame(columns=["bar_buy", "bar_sell", "note"])


def save_actual(a: pd.DataFrame) -> None:
    DATA.mkdir(exist_ok=True)
    a.sort_index().to_csv(ACTUAL_CSV, index_label="date", date_format="%Y-%m-%d")


# -------------------------------------------------------------- analysis ---
def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def build_series(hist: pd.DataFrame, actual: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, float]:
    """Daily calendar series (weekends carry Friday's price) with indicators."""
    # calibration factor: real Thai bar sell price / spot estimate (median of last 30 days)
    factor = 1.0127
    if len(actual):
        est = hist["est"].reindex(actual.index, method="ffill")
        ratios = (actual["bar_sell"].astype(float) / est).dropna().tail(30)
        if len(ratios):
            factor = float(ratios.median())

    trading = hist["est"] * factor
    if len(actual):                   # real prices win where we have them
        real = actual["bar_sell"].astype(float)
        trading = trading.combine_first(real)
        trading.loc[real.index] = real

    # indicators on trading days only, then spread to every calendar day
    t = pd.DataFrame({"price": trading})
    p = t["price"]
    t["ma50"] = p.rolling(50).mean()
    t["ma200"] = p.rolling(200).mean()
    t["rsi"] = rsi(p)
    t["high52"] = p.rolling(252, min_periods=20).max()
    t["low52"] = p.rolling(252, min_periods=20).min()

    days = pd.date_range(t.index[0], max(today, t.index[-1]), freq="D")
    s = t.reindex(days).ffill()
    s["trading_day"] = s.index.isin(t.index)
    return s, factor


def score(row) -> dict:
    c = CONFIG
    price, ma50, ma200, rs, hi = row["price"], row["ma50"], row["ma200"], row["rsi"], row["high52"]
    dd = (hi - price) / hi * 100
    buy, sell = [], []
    if rs < c["rsi_buy"]:
        buy.append(f"RSI {rs:.0f} ต่ำกว่า {c['rsi_buy']} (ขายมากเกินไป)")
    if price < ma50:
        buy.append("ราคาต่ำกว่าเส้นเฉลี่ย 50 วัน")
    if price < ma200:
        buy.append("ราคาต่ำกว่าเส้นเฉลี่ย 200 วัน")
    if dd >= c["dip_pct"]:
        buy.append(f"ลดลง {dd:.1f}% จากจุดสูงสุด 52 สัปดาห์")
    if rs > c["rsi_sell"]:
        sell.append(f"RSI {rs:.0f} สูงกว่า {c['rsi_sell']} (ซื้อมากเกินไป)")
    if price > ma200 * (1 + c["stretch_ma200_pct"] / 100):
        sell.append(f"สูงกว่าเส้น 200 วันเกิน {c['stretch_ma200_pct']}%")
    if price > ma50 * (1 + c["stretch_ma50_pct"] / 100):
        sell.append(f"สูงกว่าเส้น 50 วันเกิน {c['stretch_ma50_pct']}%")
    if dd <= c["near_high_pct"]:
        sell.append("ใกล้จุดสูงสุด 52 สัปดาห์")

    net = len(buy) - len(sell)
    level = 5 if net >= 3 else 4 if net >= 1 else 3 if net == 0 else 2 if net >= -2 else 1
    label, status = LEVELS[level]
    return {"level": level, "label": label, "status": status, "net": net,
            "buy_reasons": buy, "sell_reasons": sell, "drawdown_pct": r(dd, 1)}


def add_levels(s: pd.DataFrame) -> pd.DataFrame:
    ok = s["ma200"].notna()
    s["level"] = float("nan")
    s.loc[ok, "level"] = [score(row)["level"] for _, row in s.loc[ok].iterrows()]
    s["drawdown"] = (s["high52"] - s["price"]) / s["high52"] * 100
    return s


def run_plan(s: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, smart: bool = True) -> pd.DataFrame:
    """Walk every calendar day. smart=False means buy the quota every day (plain DCA)."""
    quota = CONFIG["daily_quota_thb"]
    rows = []
    balance = 0.0
    wait = 0
    invested = baht = 0.0
    for d, row in s.loc[start:end].iterrows():
        balance += quota
        wait += 1
        level = row["level"]
        reason = ""
        if not smart:
            reason = "ซื้อทุกวัน"
        elif not math.isnan(level) and level >= CONFIG["buy_min_level"]:
            reason = "สัญญาณดี"
        elif wait >= CONFIG["max_wait_days"]:
            reason = f"รอครบ {CONFIG['max_wait_days']} วัน"
        buy = balance if reason else 0.0
        if buy:
            invested += buy
            baht += buy / row["price"]
            balance = 0.0
        rows.append({"date": d, "price": row["price"], "level": level,
                     "buy": buy, "reason": reason, "wait_days": wait,
                     "balance": balance, "invested": invested, "baht": baht})
        if buy:
            wait = 0
    return pd.DataFrame(rows).set_index("date")


def summarize_plan(p: pd.DataFrame, price_now: float) -> dict:
    last = p.iloc[-1]
    gold_value = last["baht"] * price_now
    total_in = last["invested"] + last["balance"]           # all quota given so far
    total_now = gold_value + last["balance"]                 # gold + cash waiting
    return {
        "days": int(len(p)), "buys": int((p["buy"] > 0).sum()),
        "quota_total": r(total_in, 0), "invested": r(last["invested"], 0),
        "cash_waiting": r(last["balance"], 0),
        "gold_baht": r(last["baht"], 4), "gold_gram": r(last["baht"] * GRAMS_PER_BAHT, 3),
        "avg_cost": r(last["invested"] / last["baht"] if last["baht"] else None, 0),
        "value": r(total_now, 0), "profit": r(total_now - total_in, 0),
        "profit_pct": r((total_now / total_in - 1) * 100 if total_in else None, 1),
    }


def backtest(s: pd.DataFrame, today: pd.Timestamp, years: int) -> dict:
    start = max(today - pd.DateOffset(years=years), s["ma200"].first_valid_index())
    price_now = s["price"].iloc[-1]
    return {
        "years": years,
        "smart": summarize_plan(run_plan(s, start, today, smart=True), price_now),
        "daily": summarize_plan(run_plan(s, start, today, smart=False), price_now),
    }


def my_portfolio(price_now: float) -> dict | None:
    if not PURCHASES_CSV.exists():
        return None
    rows = pd.read_csv(PURCHASES_CSV).dropna(subset=["amount_thb", "price_per_baht"])
    if rows.empty:
        return None
    rows["baht"] = rows["amount_thb"] / rows["price_per_baht"]
    sign = rows["type"].astype(str).str.lower().map({"buy": 1, "sell": -1}).fillna(1)
    invested = float((rows["amount_thb"] * sign).sum())
    baht = float((rows["baht"] * sign).sum())
    value = baht * price_now
    return {"invested": r(invested, 0), "gold_baht": r(baht, 4),
            "gold_gram": r(baht * GRAMS_PER_BAHT, 3),
            "avg_cost": r(invested / baht if baht else None, 0),
            "value": r(value, 0), "profit": r(value - invested, 0),
            "profit_pct": r((value / invested - 1) * 100 if invested else None, 1),
            "count": int(len(rows))}


def action_text(level: int, today_plan: pd.Series | None) -> tuple[str, str]:
    """Return (main action, sell note)."""
    sell = ""
    if level == 1:
        sell = f"ราคาแพงมาก ถ้ามีทองสะสมอยู่ พิจารณาขายออก {CONFIG['sell_pct']}% ได้"
    if today_plan is None:
        return "แผนยังไม่เริ่ม", sell
    if today_plan["buy"] > 0:
        days = int(today_plan["wait_days"])
        why = today_plan["reason"]
        return f"ซื้อวันนี้ {today_plan['buy']:,.0f} บาท (ยอดทบ {days} วัน, {why})", sell
    return (f"ยังไม่ซื้อ ทบยอดไว้ก่อน ยอดสะสม {today_plan['balance']:,.0f} บาท "
            f"(รอมา {int(today_plan['wait_days'])} วัน)"), sell


# ------------------------------------------------------------------ main ---
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    now = datetime.now(BKK)
    today = pd.Timestamp(now.date())
    start = pd.Timestamp(CONFIG["start_date"])

    hist = fetch_history()
    actual = load_actual()
    thai = fetch_thai_today()
    if thai:
        actual.loc[today, ["bar_buy", "bar_sell", "note"]] = [thai["bar_buy"], thai["bar_sell"], thai["note"]]
        save_actual(actual)

    s, factor = build_series(hist, actual, today)
    s = add_levels(s)
    s = s.loc[:today]
    row = s.iloc[-1]
    prev = s.loc[s["trading_day"] & (s.index < today)].iloc[-1]
    sig = score(row)

    plan = run_plan(s, start, today) if today >= start else None
    today_plan = plan.iloc[-1] if plan is not None else None
    action, sell_note = action_text(sig["level"], today_plan)
    plan_sum = summarize_plan(plan, row["price"]) if plan is not None else None

    summary = {
        "date": today.strftime("%Y-%m-%d"),
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "price_sell": r(row["price"], 0),
        "price_buyback": r(thai["bar_buy"], 0) if thai else None,
        "price_source": "สมาคมค้าทองคำ" if thai else "ประมาณจาก Gold Spot",
        "price_note": thai["note"] if thai else "",
        "change": r(row["price"] - prev["price"], 0),
        "change_pct": r((row["price"] / prev["price"] - 1) * 100, 2),
        "price_per_gram": r(row["price"] / GRAMS_PER_BAHT, 2),
        "usd_oz": r(hist["usd_oz"].iloc[-1], 2),
        "usdthb": r(hist["usdthb"].iloc[-1], 3),
        "ma50": r(row["ma50"], 0), "ma200": r(row["ma200"], 0),
        "rsi": r(row["rsi"], 1),
        "high52": r(row["high52"], 0), "low52": r(row["low52"], 0),
        "signal": sig,
        "buy_today": bool(today_plan is not None and today_plan["buy"] > 0),
        "buy_amount_today": r(today_plan["buy"], 0) if today_plan is not None else 0,
        "balance": r(today_plan["balance"], 0) if today_plan is not None else 0,
        "wait_days": int(today_plan["wait_days"]) if today_plan is not None else 0,
        "action": action, "sell_note": sell_note,
        "plan": plan_sum,
        "calibration": r(factor, 4),
    }

    # daily report: last LOG_DAYS days. Plan columns filled from start_date on.
    log_from = today - pd.Timedelta(days=LOG_DAYS - 1)
    fields = ["date", "price_sell", "change", "rsi", "drawdown_pct", "level", "signal",
              "quota_today", "buy_thb", "balance_after", "action"]
    log = []
    prev_price = None
    for d, x in s.loc[s.index >= log_from - pd.Timedelta(days=7)].iterrows():
        chg = None if prev_price is None else x["price"] - prev_price
        prev_price = x["price"]
        if d < log_from:
            continue
        lvl = int(x["level"])
        p = plan.loc[d] if plan is not None and d in plan.index else None
        if p is None:
            act = "ก่อนเริ่มแผน"
        elif p["buy"] > 0:
            act = f"ซื้อ {p['buy']:,.0f} บาท ({p['reason']})"
        else:
            act = "ทบยอด"
        if lvl == 1:
            act += f" · พิจารณาขาย {CONFIG['sell_pct']}%"
        log.append({
            "date": d.strftime("%Y-%m-%d"), "price_sell": r(x["price"], 0),
            "change": r(chg, 0), "rsi": r(x["rsi"], 1), "drawdown_pct": r(x["drawdown"], 1),
            "level": lvl, "signal": LEVELS[lvl][0],
            "quota_today": CONFIG["daily_quota_thb"] if p is not None else "",
            "buy_thb": r(p["buy"], 0) if p is not None else "",
            "balance_after": r(p["balance"], 0) if p is not None else "",
            "action": act,
        })
    DATA.mkdir(exist_ok=True)
    for path in (REPORT_CSV, DOCS / "report.csv"):
        path.parent.mkdir(exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(reversed(log))

    # chart series: trading days, last 5 years
    t = s.loc[s["trading_day"] | (s.index == today)]
    cut = t.loc[t.index >= today - pd.DateOffset(years=5)]
    series = {
        "date": [d.strftime("%Y-%m-%d") for d in cut.index],
        "price": [r(x, 0) for x in cut["price"]],
        "ma50": [r(x, 0) for x in cut["ma50"]],
        "ma200": [r(x, 0) for x in cut["ma200"]],
        "rsi": [r(x, 1) for x in cut["rsi"]],
    }

    out = {
        "summary": summary,
        "series": series,
        "backtest": [backtest(s, today, y) for y in (1, 3, 5)],
        "portfolio": my_portfolio(row["price"]),
        "log": log[::-1],
        "config": CONFIG,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{summary['date']}  {summary['price_sell']:,} THB  RSI {summary['rsi']}  "
          f"-> {sig['label']}  | {action} {sell_note}")
    for b in out["backtest"]:
        sm, dl = b["smart"], b["daily"]
        print(f"  {b['years']}y  smart: {sm['buys']} buys, {sm['profit_pct']}%  avg {sm['avg_cost']}"
              f"  | daily: {dl['profit_pct']}%  avg {dl['avg_cost']}")


if __name__ == "__main__":
    main()
