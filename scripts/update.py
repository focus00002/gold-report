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
    "max_buy_thb": 3000,           # most money to suggest for one buy
    "level5_multiplier": 1.5,      # level 5: suggest saved quota x this
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


def load_purchases(s: pd.DataFrame) -> pd.DataFrame:
    """Your real buys/sells. Missing price is filled with that day's Thai price."""
    cols = ["date", "type", "amount_thb", "price_per_baht", "note"]
    if not PURCHASES_CSV.exists():
        return pd.DataFrame(columns=cols)
    p = pd.read_csv(PURCHASES_CSV)
    p = p.dropna(subset=["date", "amount_thb"])
    if p.empty:
        return pd.DataFrame(columns=cols)
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["type"] = p["type"].fillna("buy").astype(str).str.lower().str.strip()
    p["amount_thb"] = p["amount_thb"].astype(float)
    p["price_per_baht"] = pd.to_numeric(p["price_per_baht"], errors="coerce")
    day_price = s["price"].reindex(p["date"], method="ffill").to_numpy()
    p["price_per_baht"] = p["price_per_baht"].fillna(pd.Series(day_price, index=p.index))
    return p.sort_values("date")


def my_portfolio(p: pd.DataFrame, price_now: float) -> dict | None:
    if p.empty:
        return None
    p = p.copy()
    p["baht"] = p["amount_thb"] / p["price_per_baht"]
    buys, sells = p[p["type"] != "sell"], p[p["type"] == "sell"]
    bought_thb, bought_baht = buys["amount_thb"].sum(), buys["baht"].sum()
    avg = bought_thb / bought_baht if bought_baht else 0
    sold_baht = sells["baht"].sum()
    sold_thb = sells["amount_thb"].sum()
    baht = bought_baht - sold_baht
    cost_left = baht * avg                                  # cost of gold still held
    value = baht * price_now
    realized = sold_thb - sold_baht * avg                   # profit already taken from sells
    return {"count": int(len(p)), "bought": r(bought_thb, 0), "sold": r(sold_thb, 0),
            "gold_baht": r(baht, 4), "gold_gram": r(baht * GRAMS_PER_BAHT, 3),
            "avg_cost": r(avg or None, 0), "cost_left": r(cost_left, 0),
            "value": r(value, 0), "profit": r(value - cost_left, 0),
            "profit_pct": r((value / cost_left - 1) * 100 if cost_left else None, 1),
            "realized": r(realized, 0)}


def suggest(level: int, balance: float, wait: int) -> tuple[float, str]:
    """Suggested amount for one day (before the max-buy cap) and why."""
    c = CONFIG
    base = max(balance, 0)
    if level == 1:
        return 0.0, "แพงมาก ทบยอดไว้ก่อน"
    if level >= c["buy_min_level"]:
        mult = c["level5_multiplier"] if level == 5 else 1.0
        why = f"สัญญาณดี ซื้อยอดสะสม × {mult:g}" if mult != 1 else "สัญญาณดี ซื้อยอดสะสม"
        return round(base * mult / 10) * 10, why
    if wait >= c["max_wait_days"]:
        return round(base / 10) * 10, f"รอครบ {c['max_wait_days']} วัน ซื้อเลย"
    return 0.0, "ยังไม่ถึงจังหวะ ทบยอดไว้ก่อน"


def real_ledger(s: pd.DataFrame, purchases: pd.DataFrame, start: pd.Timestamp, today: pd.Timestamp) -> pd.DataFrame:
    """Quota account from start_date using your REAL buys.
    balance = quota given so far - money you actually spent on buys."""
    quota, cap = CONFIG["daily_quota_thb"], CONFIG["max_buy_thb"]
    buys = purchases[purchases["type"] != "sell"].groupby("date")["amount_thb"].sum()
    rows, carried, wait = [], 0.0, 0
    for d, row in s.loc[start:today].iterrows():
        before = carried + quota
        wait += 1
        lvl = int(row["level"])
        raw, why = suggest(lvl, before, wait)
        bought = float(buys.get(d, 0.0))
        carried = before - bought
        rows.append({"date": d, "level": lvl, "balance_before": before, "wait_days": wait,
                     "suggest_raw": raw, "suggest": min(raw, cap), "over_cap": raw > cap,
                     "why": why, "bought": bought, "balance_after": carried})
        if bought > 0:
            wait = 0
    return pd.DataFrame(rows).set_index("date")


# ------------------------------------------------------------------ main ---
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    now = datetime.now(BKK)
    today = pd.Timestamp(now.date())
    start = pd.Timestamp(CONFIG["start_date"])
    cap = CONFIG["max_buy_thb"]

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

    purchases = load_purchases(s)
    ledger = real_ledger(s, purchases, start, today) if today >= start else None
    L = ledger.iloc[-1] if ledger is not None else None

    if L is None:
        action = "แผนยังไม่เริ่ม"
    elif L["bought"] > 0:
        action = f"วันนี้บันทึกการซื้อแล้ว {L['bought']:,.0f} บาท"
    elif L["suggest"] > 0:
        action = f"แนะนำซื้อวันนี้ {L['suggest']:,.0f} บาท"
        if L["over_cap"]:
            action += f" (สูตรคำนวณได้ {L['suggest_raw']:,.0f} บาท เกินงบ {cap:,} บาท)"
    else:
        action = f"ยังไม่ซื้อ ทบยอดไว้ก่อน (ยอดสะสม {L['balance_before']:,.0f} บาท)"
    sell_note = (f"ราคาแพงมาก ถ้ามีทองสะสมอยู่ พิจารณาขายออก {CONFIG['sell_pct']}% ได้"
                 if sig["level"] == 1 else "")

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
        "signal": sig, "action": action, "sell_note": sell_note,
        "calibration": r(factor, 4),
    }
    if L is not None:
        summary.update({
            "balance": r(L["balance_before"], 0),
            "suggest": r(L["suggest"], 0), "suggest_raw": r(L["suggest_raw"], 0),
            "over_cap": bool(L["over_cap"]), "suggest_why": L["why"],
            "bought_today": r(L["bought"], 0), "balance_after": r(L["balance_after"], 0),
            "wait_days": int(L["wait_days"]),
            "plan_days": int(len(ledger)),
            "quota_total": r(len(ledger) * CONFIG["daily_quota_thb"], 0),
            "spent_total": r(ledger["bought"].sum(), 0),
        })

    # daily report: last LOG_DAYS days. Quota columns filled from start_date on.
    log_from = today - pd.Timedelta(days=LOG_DAYS - 1)
    fields = ["date", "price_sell", "change", "rsi", "drawdown_pct", "level", "signal",
              "balance", "suggest_thb", "bought_thb", "balance_after", "action"]
    log = []
    prev_price = None
    for d, x in s.loc[s.index >= log_from - pd.Timedelta(days=7)].iterrows():
        chg = None if prev_price is None else x["price"] - prev_price
        prev_price = x["price"]
        if d < log_from:
            continue
        lvl = int(x["level"])
        g = ledger.loc[d] if ledger is not None and d in ledger.index else None
        if g is None:
            act = "ก่อนเริ่มแผน"
        else:
            act = g["why"]
            if g["over_cap"]:
                act += f" (สูตรได้ {g['suggest_raw']:,.0f} เกินงบ)"
        if lvl == 1:
            act += f" · พิจารณาขาย {CONFIG['sell_pct']}%"
        log.append({
            "date": d.strftime("%Y-%m-%d"), "price_sell": r(x["price"], 0),
            "change": r(chg, 0), "rsi": r(x["rsi"], 1), "drawdown_pct": r(x["drawdown"], 1),
            "level": lvl, "signal": LEVELS[lvl][0],
            "balance": r(g["balance_before"], 0) if g is not None else "",
            "suggest_thb": r(g["suggest"], 0) if g is not None else "",
            "bought_thb": r(g["bought"], 0) if g is not None else "",
            "balance_after": r(g["balance_after"], 0) if g is not None else "",
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
        "portfolio": my_portfolio(purchases, row["price"]),
        "purchases": [
            {"date": x["date"].strftime("%Y-%m-%d"), "type": x["type"],
             "amount_thb": r(x["amount_thb"], 2), "price_per_baht": r(x["price_per_baht"], 0),
             "gram": r(x["amount_thb"] / x["price_per_baht"] * GRAMS_PER_BAHT, 4),
             "note": "" if pd.isna(x["note"]) else str(x["note"])}
            for _, x in purchases.iloc[::-1].iterrows()],
        "log": log[::-1],
        "config": CONFIG,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"{summary['date']}  {summary['price_sell']:,} THB  RSI {summary['rsi']}  "
          f"-> {sig['label']}  | {action} {sell_note}")


if __name__ == "__main__":
    main()
