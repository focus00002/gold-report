"""
Add one buy/sell row to data/my_purchases.csv from a GitHub issue form.

Reads the issue body from the ISSUE_BODY environment variable.
Writes a short Thai result message to result.md (used as the issue comment).
Exit code 1 = bad input (nothing saved).
"""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PURCHASES_CSV = ROOT / "data" / "my_purchases.csv"
RESULT = ROOT / "result.md"
GRAMS_PER_BAHT = 15.244
BKK = timezone(timedelta(hours=7))

LABELS = {
    "ประเภท": "type",
    "จำนวนเงิน (บาท)": "amount_thb",
    "วันที่ (ปี ค.ศ.-เดือน-วัน)": "date",
    "ได้ทองกี่กรัม (ไม่บังคับ)": "gram",
    "ราคาต่อบาททองคำ (ไม่บังคับ)": "price_per_baht",
    "หมายเหตุ (ไม่บังคับ)": "note",
}


def parse(body: str) -> dict:
    fields = {}
    for block in re.split(r"^###\s+", body, flags=re.M)[1:]:
        label, _, value = block.partition("\n")
        value = value.strip()
        key = LABELS.get(label.strip())
        if key:
            fields[key] = "" if value == "_No response_" else value
    return fields


def number(text: str, name: str, required: bool = False) -> float | None:
    text = (text or "").replace(",", "").replace("บาท", "").replace("กรัม", "").strip()
    if not text:
        if required:
            raise ValueError(f"ต้องใส่{name}")
        return None
    if not re.fullmatch(r"\d+(\.\d+)?", text):
        raise ValueError(f"{name} ต้องเป็นตัวเลข (ได้รับ: {text})")
    v = float(text)
    if not 0 < v < 10_000_000:
        raise ValueError(f"{name} ไม่ถูกต้อง: {text}")
    return v


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    f = parse(os.environ.get("ISSUE_BODY", ""))
    try:
        kind = "sell" if f.get("type", "").strip() == "ขาย" else "buy"
        amount = number(f.get("amount_thb"), "จำนวนเงิน", required=True)
        gram = number(f.get("gram"), "จำนวนกรัม")
        price = number(f.get("price_per_baht"), "ราคาต่อบาททองคำ")
        date_text = (f.get("date") or "").strip()
        if date_text:
            day = datetime.strptime(date_text, "%Y-%m-%d").date()
            if day.year > 2400:                       # Buddhist year typed by mistake
                day = day.replace(year=day.year - 543)
        else:
            day = datetime.now(BKK).date()
        if day > datetime.now(BKK).date():
            raise ValueError(f"วันที่ {day} อยู่ในอนาคต")
    except ValueError as e:
        msg = str(e)
        if "does not match format" in msg:
            msg = "รูปแบบวันที่ต้องเป็น ปี ค.ศ.-เดือน-วัน เช่น 2026-11-30"
        RESULT.write_text(f"❌ บันทึกไม่สำเร็จ: {msg}\n\nกรุณาเปิดฟอร์มใหม่แล้วกรอกอีกครั้ง", encoding="utf-8")
        print(msg)
        return 1

    if gram and not price:
        price = amount / (gram / GRAMS_PER_BAHT)
    note = re.sub(r"[\r\n,]+", " ", f.get("note", ""))[:100]

    new_file = not PURCHASES_CSV.exists() or PURCHASES_CSV.stat().st_size == 0
    with PURCHASES_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["date", "type", "amount_thb", "price_per_baht", "note"])
        w.writerow([day.isoformat(), kind, f"{amount:g}", f"{price:.0f}" if price else "", note])

    word = "ขาย" if kind == "sell" else "ซื้อ"
    price_txt = f"ราคา {price:,.0f} บาท/บาททองคำ" if price else "ใช้ราคาสมาคมค้าทองคำของวันนั้น"
    RESULT.write_text(
        f"✅ บันทึกการ{word}แล้ว: {day.isoformat()} จำนวน {amount:,.2f} บาท ({price_txt})\n\n"
        "หน้าเว็บจะอัปเดตภายใน 1-2 นาที", encoding="utf-8")
    print("saved", day, kind, amount, price)
    return 0


if __name__ == "__main__":
    sys.exit(main())
