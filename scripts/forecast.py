"""
Price forecast for 1, 7, 15 and 30 days ahead.

How it works (kept simple on purpose):
  1. Volatility: how much the Thai gold price usually moves per day
     (EWMA of daily log returns). This sets the WIDTH of the range.
  2. Lean: a small tilt up or down from
       - trend  (last 20 trading days return vs. normal volatility)
       - news   (headline keywords from Google News, English + Thai)
     The lean only shifts the middle of the range a little.
  3. Output per horizon: middle price, 50% range, 80% range,
     probability of being higher than today.

Nobody can predict gold reliably. The range is the useful part:
"80% range" means that in the past, the real price ended inside it
about 80% of the time (see backtest()).
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

HORIZONS = [1, 7, 15, 30]            # calendar days ahead
Z50, Z80 = 0.6745, 1.2816
TILT = 0.3                           # lean moves the middle by at most 0.3 x range width (1 sigma)
TREND_DAYS = 20
NEWS_HOURS = 48
BKK = timezone(timedelta(hours=7))

FEEDS = [
    ("en", "https://news.google.com/rss/search?q=" + quote("gold price when:2d") + "&hl=en-US&gl=US&ceid=US:en"),
    ("th", "https://news.google.com/rss/search?q=" + quote("ราคาทอง when:2d") + "&hl=th&gl=TH&ceid=TH:th"),
]

BULL_EN = [r"\brise[sn]?\b", r"\brising\b", r"\bgains?\b", r"\bjump", r"\bsurge", r"\bsoar", r"\brall(y|ies)\b",
           r"\bclimb", r"record high", r"\bhigher\b", r"\brebound", r"safe[- ]haven", r"rate cuts?\b",
           r"weak(er)? dollar", r"dollar (falls|slips|weakens)", r"\bbullish\b", r"\bbuy(ing)? the dip",
           r"\badvanc", r"\bupbeat\b", r"strong(er)? demand"]
BEAR_EN = [r"\bfall(s|ing)?\b", r"\bdrops?\b", r"\bslides?\b", r"\bslump", r"\bplunge", r"\btumble", r"\bcrash",
           r"\bsinks?\b", r"\blower\b", r"\blows?\b", r"\bloss(es)?\b", r"rate hikes?", r"strong(er)? dollar",
           r"dollar (rises|gains|strengthens)", r"\bbearish\b", r"sell-?off", r"\bcrushed\b", r"\bpressure",
           r"\bplummet", r"\bdeclin", r"\bretreat", r"gold (weakens|slips)", r"\bslips?\b", r"profit[- ]taking"]
BULL_TH = [r"ขึ้น\s*\d", r"ปรับขึ้น", r"พุ่ง", r"ทะยาน", r"ฟื้น", r"สูงสุด", r"บวก", r"แรงซื้อ"]
BEAR_TH = [r"ลง\s*\d", r"ปรับลง", r"ลดลง", r"ร่วง", r"ดิ่ง", r"ต่ำสุด", r"กดดัน", r"แรงขาย", r"ทรุด"]


# ------------------------------------------------------------------ news ---
def headline_score(title: str, lang: str) -> int:
    t = title.lower()
    bull, bear = (BULL_TH, BEAR_TH) if lang == "th" else (BULL_EN, BEAR_EN)
    b = sum(bool(re.search(p, t)) for p in bull)
    s = sum(bool(re.search(p, t)) for p in bear)
    return (b > s) - (s > b)


def fetch_news(now: datetime) -> list[dict]:
    items, seen = [], set()
    for lang, url in FEEDS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (gold-report)"})
            root = ET.fromstring(r.content)
        except Exception as e:
            print("news fetch failed:", lang, e)
            continue
        for it in root.findall(".//item"):
            title = (it.findtext("title") or "").strip()
            source = (it.findtext("source") or "").strip()
            if source and title.endswith(" - " + source):
                title = title[: -len(source) - 3]
            key = re.sub(r"\W+", "", title.lower())[:60]
            if not title or key in seen:
                continue
            seen.add(key)
            try:
                when = parsedate_to_datetime(it.findtext("pubDate"))
            except Exception:
                continue
            age_h = (now - when).total_seconds() / 3600
            if age_h > NEWS_HOURS or age_h < -1:
                continue
            items.append({"lang": lang, "title": title, "source": source,
                          "link": it.findtext("link") or "", "time": when.astimezone(BKK).strftime("%Y-%m-%d %H:%M"),
                          "age_h": round(age_h, 1), "score": headline_score(title, lang)})
    items.sort(key=lambda x: x["age_h"])
    return items


def news_lean(items: list[dict]) -> dict:
    """Average headline score, newer headlines count more. Range -1..1."""
    if not items:
        return {"score": 0.0, "count": 0, "bull": 0, "bear": 0, "neutral": 0}
    w = np.array([math.exp(-x["age_h"] / 24) for x in items])
    sc = np.array([x["score"] for x in items], dtype=float)
    return {"score": float((w * sc).sum() / w.sum()), "count": len(items),
            "bull": int((sc > 0).sum()), "bear": int((sc < 0).sum()), "neutral": int((sc == 0).sum())}


# ----------------------------------------------------------- statistics ---
def trading_prices(s: pd.DataFrame) -> pd.Series:
    return s.loc[s["trading_day"], "price"].astype(float)


def ewma_vol(logret: pd.Series, lam: float = 0.94) -> pd.Series:
    return np.sqrt((logret ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def trend_lean(p: pd.Series, vol: pd.Series) -> pd.Series:
    """20-day log return measured in 'normal moves', squashed to -1..1."""
    r = np.log(p / p.shift(TREND_DAYS))
    return np.tanh(r / (vol * math.sqrt(TREND_DAYS)))


def n_trading(h: int) -> int:
    return max(1, round(h * 5 / 7))


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def band(price: float, sigma_d: float, lean: float, h: int) -> dict:
    sig = sigma_d * math.sqrt(n_trading(h))
    mu = lean * TILT * sig
    q = lambda z: price * math.exp(mu + z * sig)
    return {"center": q(0), "lo50": q(-Z50), "hi50": q(Z50), "lo80": q(-Z80), "hi80": q(Z80),
            "p_up": norm_cdf(mu / sig), "sigma_pct": sig * 100}


# ------------------------------------------------------------- backtest ---
def backtest(s: pd.DataFrame, years: int = 3) -> list[dict]:
    """Check past ranges (trend lean only: there is no news history)."""
    p = trading_prices(s)
    lr = np.log(p / p.shift(1))
    vol = ewma_vol(lr.dropna()).reindex(p.index)
    tl = trend_lean(p, vol)
    cal = s["price"]                                     # calendar series for target lookup
    start = p.index[-1] - pd.DateOffset(years=years)
    out = []
    for h in HORIZONS:
        hit80 = hit50 = dir_ok = n = ups = 0
        for d in p.index[(p.index >= start)]:
            target = d + pd.Timedelta(days=h)
            if target > cal.index[-1] or math.isnan(vol[d]) or math.isnan(tl[d]):
                continue
            b = band(p[d], vol[d], tl[d], h)
            actual = cal[target]
            n += 1
            ups += actual > p[d]
            hit80 += b["lo80"] <= actual <= b["hi80"]
            hit50 += b["lo50"] <= actual <= b["hi50"]
            if abs(tl[d]) > 0.1:
                dir_ok += (actual > p[d]) == (tl[d] > 0)
        dir_n = sum(1 for d in p.index[p.index >= start]
                    if d + pd.Timedelta(days=h) <= cal.index[-1] and not math.isnan(tl[d]) and abs(tl[d]) > 0.1)
        out.append({"h": h, "n": n, "cover80": round(hit80 / n * 100, 1) if n else None,
                    "cover50": round(hit50 / n * 100, 1) if n else None,
                    "direction": round(dir_ok / dir_n * 100, 1) if dir_n else None,
                    "up_rate": round(ups / n * 100, 1) if n else None})   # % of times price simply went up
    return out


# ------------------------------------------------------- forecast log ---
def update_log(path: Path, today: pd.Timestamp, rows: list[dict], cal: pd.Series) -> dict:
    """Save today's forecasts and score the ones whose target date has passed."""
    cols = ["made", "h", "target", "base", "center", "lo80", "hi80", "p_up"]
    log = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=cols)
    log = log[log["made"] != today.strftime("%Y-%m-%d")]
    new = pd.DataFrame([{"made": today.strftime("%Y-%m-%d"), "h": r["h"],
                         "target": (today + pd.Timedelta(days=r["h"])).strftime("%Y-%m-%d"),
                         "base": round(r["base"]), "center": round(r["center"]),
                         "lo80": round(r["lo80"]), "hi80": round(r["hi80"]), "p_up": round(r["p_up"], 3)}
                        for r in rows])
    log = pd.concat([log, new], ignore_index=True).sort_values(["made", "h"])
    path.parent.mkdir(exist_ok=True)
    log.to_csv(path, index=False)

    done = log[pd.to_datetime(log["target"]) <= today].copy()
    if done.empty:
        return {"count": 0, "rows": []}
    done["actual"] = [cal.get(pd.Timestamp(t), float("nan")) for t in done["target"]]
    done = done.dropna(subset=["actual"])
    done["in80"] = (done["lo80"] <= done["actual"]) & (done["actual"] <= done["hi80"])
    done["dir_ok"] = (done["actual"] > done["base"]) == (done["p_up"] > 0.5)
    recent = done.sort_values("target", ascending=False).head(12)
    return {
        "count": int(len(done)),
        "cover80": round(done["in80"].mean() * 100, 1),
        "direction": round(done["dir_ok"].mean() * 100, 1),
        "rows": [{"made": x.made, "h": int(x.h), "target": x.target, "lo80": int(x.lo80), "hi80": int(x.hi80),
                  "center": int(x.center), "actual": round(float(x.actual)), "in80": bool(x.in80),
                  "dir_ok": bool(x.dir_ok)} for x in recent.itertuples()],
    }


# ------------------------------------------------------------------ main ---
def make_forecast(s: pd.DataFrame, today: pd.Timestamp, now: datetime, log_path: Path) -> dict:
    p = trading_prices(s)
    lr = np.log(p / p.shift(1)).dropna()
    vol = ewma_vol(lr)
    sigma_d = float(vol.iloc[-1])
    tl = float(trend_lean(p, vol.reindex(p.index)).iloc[-1])
    news = fetch_news(now)
    nl = news_lean(news)
    lean = 0.5 * tl + 0.5 * nl["score"] if nl["count"] else tl
    price = float(s["price"].iloc[-1])

    rows = []
    for h in HORIZONS:
        b = band(price, sigma_d, lean, h)
        rows.append({"h": h, "target": (today + pd.Timedelta(days=h)).strftime("%Y-%m-%d"),
                     "base": price, **b})

    def word(x):
        return "เอียงขึ้น" if x > 0.15 else "เอียงลง" if x < -0.15 else "ทรงตัว"

    return {
        "price": round(price),
        "sigma_daily_pct": round(sigma_d * 100, 2),
        "trend": round(tl, 2), "trend_word": word(tl),
        "news": {**{k: (round(v, 2) if isinstance(v, float) else v) for k, v in nl.items()},
                 "word": word(nl["score"]) if nl["count"] else "ไม่มีข่าว"},
        "lean": round(lean, 2), "lean_word": word(lean),
        "rows": [{k: (round(v, 3) if k == "p_up" else round(v, 2) if k == "sigma_pct" else
                      round(v) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
        "backtest": backtest(s),
        "track": update_log(log_path, today, rows, s["price"]),
        "headlines_th": [{k: x[k] for k in ("title", "source", "link", "time", "score")}
                         for x in news if x["lang"] == "th"][:8],
        "headlines_en": [{k: x[k] for k in ("title", "source", "link", "time", "score")}
                         for x in news if x["lang"] == "en"][:8],
    }
