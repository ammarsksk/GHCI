from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

RANDOM_SEED = 20240523
random.seed(RANDOM_SEED)

DATA_DIR = Path("data")
TAXONOMY_PATH = Path("config/taxonomy.yaml")

@dataclass
class Label:
    id: str
    display_name: str
    description: str
    keywords: List[str]
    example_merchants: List[str]


def _parse_taxonomy(path: Path) -> List[Label]:
    """Parse a minimal subset of the YAML taxonomy without external deps."""
    labels: List[Label] = []
    current: Optional[Dict[str, object]] = None
    section: Optional[str] = None

    def push_current() -> None:
        nonlocal current
        if not current:
            return
        labels.append(
            Label(
                id=current.get("id", "").strip(),
                display_name=current.get("display_name", "").strip(),
                description=current.get("description", "").strip(),
                keywords=current.get("keywords", []),
                example_merchants=current.get("example_merchants", []),
            )
        )
        current = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("fallback:"):
                break
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()

            if indent == 2 and stripped.startswith("- id:"):
                push_current()
                current = {
                    "id": stripped.split(":", 1)[1].strip(),
                    "keywords": [],
                    "example_merchants": [],
                }
                section = None
                continue

            if current is None:
                continue

            if indent == 4 and ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                if value:
                    current[key] = value
                    section = None
                else:
                    section = key
                continue

            if indent >= 6 and stripped.startswith("- ") and section in {"keywords", "example_merchants"}:
                value = stripped[2:].strip()
                if value.startswith("\"") and value.endswith("\""):
                    value = value[1:-1]
                current.setdefault(section, []).append(value)
                continue

        push_current()

    return labels


CITY_POOL = [
    "Delhi",
    "Mumbai",
    "Bengaluru",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Kolkata",
    "Ahmedabad",
    "Kochi",
    "Jaipur",
]

UPI_HANDLES = ["@ybl", "@oksbi", "@okhdfcbank", "@paytm", "@ibl", "@upi"]
BANK_CODES = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YES"]
CARD_NETWORKS = ["VISA", "MASTERCARD", "RUPAY", "AMEX"]
ACCOUNT_TYPES = ["SAVINGS", "CURRENT", "CREDIT_CARD"]
CHANNEL_MAP = {
    "DINING": ["CARD", "UPI", "POS"],
    "GROCERIES": ["UPI", "CARD", "POS"],
    "FUEL": ["POS", "CARD"],
    "UTILITIES_POWER": ["NETBANKING", "AUTOPAY"],
    "UTILITIES_TELECOM": ["UPI", "AUTOPAY", "NETBANKING"],
    "UTILITIES_WATER_GAS": ["NETBANKING", "AUTOPAY"],
    "SHOPPING_ECOM": ["CARD", "NETBANKING", "UPI"],
    "SHOPPING_ELECTRONICS": ["CARD", "EMI", "NETBANKING"],
    "TRAVEL": ["CARD", "UPI", "NETBANKING"],
    "MOBILITY": ["UPI", "CARD"],
    "HEALTH": ["CARD", "UPI", "NETBANKING"],
    "ENTERTAINMENT": ["UPI", "CARD", "SUBSCRIPTION"],
    "EDUCATION": ["NETBANKING", "UPI", "AUTOPAY"],
    "FINANCIAL_SERVICES": ["NETBANKING", "NEFT", "CARD"],
    "FEES": ["CARD", "NETBANKING"],
    "HOME_IMPROVEMENT": ["CARD", "NETBANKING"],
    "SUBSCRIPTIONS": ["SUBSCRIPTION", "CARD", "UPI"],
    "CHARITY": ["NETBANKING", "UPI"],
    "INCOME": ["NEFT", "IMPS", "SALARY"],
}

AMOUNT_RULES = {
    "DINING": (120, 2500),
    "GROCERIES": (150, 4500),
    "FUEL": (400, 5500),
    "UTILITIES_POWER": (600, 4000),
    "UTILITIES_TELECOM": (199, 2500),
    "UTILITIES_WATER_GAS": (300, 2200),
    "SHOPPING_ECOM": (250, 12000),
    "SHOPPING_ELECTRONICS": (1200, 65000),
    "TRAVEL": (800, 55000),
    "MOBILITY": (150, 1800),
    "HEALTH": (250, 48000),
    "ENTERTAINMENT": (120, 4500),
    "EDUCATION": (500, 75000),
    "FINANCIAL_SERVICES": (350, 85000),
    "FEES": (50, 2500),
    "HOME_IMPROVEMENT": (500, 90000),
    "SUBSCRIPTIONS": (99, 15000),
    "CHARITY": (250, 25000),
    "INCOME": (15000, 250000),
}

CREDIT_LABELS = {"INCOME"}


def _triangular_amount(cat_id: str) -> float:
    low, high = AMOUNT_RULES.get(cat_id, (150, 5000))
    mode = (low + high) / 2.0 * random.uniform(0.85, 1.15)
    amount = random.triangular(low, high, max(low, min(mode, high)))
    return round(amount, 2)


def _random_dates() -> tuple[str, str]:
    start = datetime(2023, 1, 1)
    end = datetime(2024, 4, 30)
    day_offset = random.randint(0, (end - start).days)
    value_date = start + timedelta(days=day_offset)
    posted_delta = random.randint(0, 3)
    posted_at = value_date + timedelta(days=posted_delta)
    return value_date.strftime("%Y-%m-%d"), posted_at.strftime("%Y-%m-%d")


def _reference(channel: str) -> str:
    if channel == "UPI":
        return f"UPI{random.randint(10**7, 10**8 - 1)}"
    if channel == "NEFT":
        return f"NEFT{random.randint(10**6, 10**7 - 1)}"
    if channel == "IMPS":
        return f"IMPS{random.randint(10**6, 10**7 - 1)}"
    if channel == "NETBANKING":
        return f"NB{random.randint(10**6, 10**7 - 1)}"
    if channel == "POS":
        return f"POS{random.randint(100000, 999999)}"
    if channel == "CARD":
        return f"CARD{random.randint(100000, 999999)}"
    if channel == "AUTOPAY":
        return f"AUTO{random.randint(10**5, 10**6 - 1)}"
    if channel == "EMI":
        return f"EMI{random.randint(10**5, 10**6 - 1)}"
    if channel == "SUBSCRIPTION":
        return f"SUB{random.randint(10**5, 10**6 - 1)}"
    if channel == "SALARY":
        return f"SAL{random.randint(10**5, 10**6 - 1)}"
    return f"TX{random.randint(10**6, 10**7 - 1)}"


def _narrative(channel: str, merchant: str, keyword: str, city: str, reference: str) -> str:
    merchant_token = merchant.upper() if channel in {"POS", "CARD"} else merchant
    if channel == "UPI":
        handle = random.choice(UPI_HANDLES)
        return f"UPI/{reference} {merchant_token}{handle} {keyword} {city.upper()}"
    if channel == "POS":
        return f"POS {city.upper()} {merchant_token} REF {reference}"
    if channel == "CARD":
        network = random.choice(CARD_NETWORKS)
        return f"{network} {merchant_token} {city.upper()} REF {reference}"
    if channel == "NETBANKING":
        bank = random.choice(BANK_CODES)
        return f"{bank} NETBANK {merchant} {keyword} {reference}"
    if channel == "NEFT":
        bank = random.choice(BANK_CODES)
        return f"NEFT {reference} {bank} CREDIT {merchant} {keyword}"
    if channel == "IMPS":
        return f"IMPS {reference} TO {merchant} {city.upper()}"
    if channel == "AUTOPAY":
        return f"AUTOPAY {merchant} {keyword} REF {reference}"
    if channel == "EMI":
        return f"EMI {merchant} INST {reference}"
    if channel == "SUBSCRIPTION":
        return f"SUBSCRIPTION {merchant} {keyword} {reference}"
    if channel == "SALARY":
        return f"SALARY CREDIT {merchant} {reference}"
    return f"TXN {merchant} {keyword} {reference}"


def _choose(iterable: Iterable[str], fallback: str) -> str:
    items = [item for item in iterable if item]
    return random.choice(items) if items else fallback


def _generate_rows(labels: List[Label], per_label: int = 220) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label in labels:
        merchants = label.example_merchants or [label.display_name or label.id]
        for _ in range(per_label):
            channel = random.choice(CHANNEL_MAP.get(label.id, ["CARD", "UPI", "NETBANKING"]))
            city = random.choice(CITY_POOL)
            keyword = _choose(label.keywords, label.id.lower())
            merchant = _choose(merchants, label.display_name or label.id.title())
            amount = _triangular_amount(label.id)
            credit_or_debit = "CR" if label.id in CREDIT_LABELS else "DR"
            value_date, posted_at = _random_dates()
            reference = _reference(channel)
            rows.append(
                {
                    "transaction_id": f"TX{random.randint(10**9, 10**10 - 1)}",
                    "value_date": value_date,
                    "posted_at": posted_at,
                    "amount": f"{amount:.2f}",
                    "currency": "INR",
                    "dr_cr": credit_or_debit,
                    "merchant_name": merchant,
                    "reference": reference,
                    "narrative": _narrative(channel, merchant, keyword, city, reference),
                    "channel": channel,
                    "account_type": random.choice(ACCOUNT_TYPES),
                    "city": city,
                    "keyword_hit": keyword,
                    "category_id": label.id,
                    "category_display_name": label.display_name or label.id.title(),
                }
            )
    random.shuffle(rows)
    return rows


def write_dataset(rows: List[Dict[str, object]], train_ratio: float = 0.8) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    split_index = int(len(rows) * train_ratio)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]
    fieldnames = list(train_rows[0].keys())

    with (DATA_DIR / "train.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    with (DATA_DIR / "test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    print(f"Wrote {len(train_rows)} training rows and {len(test_rows)} test rows")


def main() -> None:
    labels = [label for label in _parse_taxonomy(TAXONOMY_PATH) if label.id]
    rows = _generate_rows(labels)
    write_dataset(rows)


if __name__ == "__main__":
    main()
