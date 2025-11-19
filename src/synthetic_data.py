from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    group: str = "EXPENSE"
    budget_bucket: str = "DISCRETIONARY"


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
                id=str(current.get("id", "")).strip(),
                display_name=str(current.get("display_name", "")).strip(),
                description=str(current.get("description", "")).strip(),
                keywords=current.get("keywords", []),
                example_merchants=current.get("example_merchants", []),
                group=str(current.get("group", "EXPENSE")).strip() or "EXPENSE",
                budget_bucket=str(current.get("budget_bucket", "")).strip() or "UNKNOWN",
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
    # Core spending categories
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
    "HOME_IMPROVEMENT": ["CARD", "NETBANKING"],
    "PERSONAL_CARE_BEAUTY": ["CARD", "UPI", "WALLET"],
    "PETS": ["CARD", "UPI", "NETBANKING"],
    "GIFTS_OCCASIONS": ["CARD", "UPI", "WALLET"],
    "HOBBIES_SPORTS": ["CARD", "UPI"],
    "ALCOHOL_BARS": ["CARD", "UPI"],
    "KIDS_BABY": ["CARD", "UPI", "WALLET"],
    # Financial / utilities / fees
    "FEES": ["CARD", "NETBANKING"],
    "SUBSCRIPTIONS": ["SUBSCRIPTION", "CARD", "UPI"],
    "CHARITY": ["NETBANKING", "UPI"],
    # Housing & loans
    "HOUSING_RENT": ["NEFT", "NETBANKING", "UPI"],
    "HOUSING_MAINTENANCE": ["NEFT", "NETBANKING"],
    "HOME_SERVICES": ["UPI", "CARD", "NETBANKING"],
    "DEBT_LOAN_HOME": ["AUTOPAY", "NACH", "NETBANKING"],
    "DEBT_LOAN_PERSONAL": ["AUTOPAY", "NACH", "NETBANKING"],
    "DEBT_LOAN_VEHICLE": ["AUTOPAY", "NACH", "NETBANKING"],
    "DEBT_LOAN_EDUCATION": ["AUTOPAY", "NACH", "NETBANKING"],
    "DEBT_LOAN_BNPL": ["UPI", "NETBANKING", "AUTOPAY"],
    "DEBT_CREDIT_CARD_BILL": ["NETBANKING", "UPI", "AUTOPAY"],
    "DEBT_COLLECTION_AGENCY": ["NETBANKING", "UPI"],
    # Investments & insurance
    "INVEST_MF_SIP": ["AUTOPAY", "NETBANKING", "UPI"],
    "INVEST_STOCK_BROKERAGE": ["NETBANKING"],
    "INVEST_GOLD_SILVER": ["CARD", "UPI", "NETBANKING"],
    "INVEST_RETIREMENT": ["NETBANKING", "AUTOPAY"],
    "INSURANCE_PREMIUM_LIFE": ["NETBANKING", "AUTOPAY", "CARD"],
    "INSURANCE_PREMIUM_HEALTH": ["NETBANKING", "AUTOPAY", "CARD"],
    "INSURANCE_PREMIUM_MOTOR": ["NETBANKING", "AUTOPAY", "CARD"],
    # Government / taxes
    "GOVT_TAXES_DIRECT": ["NETBANKING"],
    "GOVT_TAXES_INDIRECT": ["NETBANKING"],
    "GOVT_FEES_PENALTIES": ["NETBANKING", "CARD", "UPI"],
    # Transfers & income
    "TRANSFER_P2P_FAMILY_FRIENDS": ["UPI", "IMPS", "NEFT"],
    "TRANSFER_SELF_INTERNAL": ["UPI", "NEFT", "NETBANKING"],
    "TRANSFER_CASH_WITHDRAWAL": ["ATM"],
    "TRANSFER_FOREX": ["NETBANKING", "NEFT", "RTGS"],
    "INCOME_SALARY": ["NEFT", "SALARY"],
    "INCOME_BUSINESS_SELF_EMPLOYED": ["NEFT", "RTGS", "IMPS"],
    "INCOME_INVESTMENT": ["NEFT", "NETBANKING"],
    "INCOME_REFUNDS": ["CARD", "UPI", "NETBANKING"],
    "INCOME_GOVT_BENEFIT": ["NEFT"],
}

AMOUNT_RULES = {
    # Food & daily spend
    "DINING": (120, 2500),
    "GROCERIES": (150, 4500),
    "FUEL": (400, 5500),
    "MOBILITY": (150, 1800),
    # Utilities
    "UTILITIES_POWER": (600, 4000),
    "UTILITIES_TELECOM": (199, 2500),
    "UTILITIES_WATER_GAS": (300, 2200),
    # Shopping & lifestyle
    "SHOPPING_ECOM": (250, 12000),
    "SHOPPING_ELECTRONICS": (1200, 65000),
    "HOME_IMPROVEMENT": (500, 90000),
    "PERSONAL_CARE_BEAUTY": (300, 6000),
    "PETS": (300, 8000),
    "GIFTS_OCCASIONS": (250, 9000),
    "HOBBIES_SPORTS": (300, 12000),
    "ALCOHOL_BARS": (400, 7000),
    "KIDS_BABY": (300, 10000),
    # Travel / health / education / entertainment
    "TRAVEL": (800, 55000),
    "HEALTH": (250, 48000),
    "ENTERTAINMENT": (120, 4500),
    "DIGITAL_GOODS_GAMING": (50, 3000),
    "EDUCATION": (500, 75000),
    # Financial & debt / investments
    "FEES": (50, 2500),
    "HOUSING_RENT": (5000, 65000),
    "HOUSING_MAINTENANCE": (1000, 8000),
    "DEBT_LOAN_HOME": (8000, 85000),
    "DEBT_LOAN_PERSONAL": (3000, 55000),
    "DEBT_LOAN_VEHICLE": (3000, 45000),
    "DEBT_LOAN_EDUCATION": (2000, 35000),
    "DEBT_LOAN_BNPL": (500, 15000),
    "DEBT_CREDIT_CARD_BILL": (3000, 150000),
    "DEBT_COLLECTION_AGENCY": (1000, 60000),
    "INVEST_MF_SIP": (1000, 50000),
    "INVEST_STOCK_BROKERAGE": (1000, 150000),
    "INVEST_GOLD_SILVER": (2000, 200000),
    "INVEST_RETIREMENT": (1000, 100000),
    "INSURANCE_PREMIUM_LIFE": (500, 50000),
    "INSURANCE_PREMIUM_HEALTH": (1000, 80000),
    "INSURANCE_PREMIUM_MOTOR": (800, 45000),
    # Government & transfers
    "GOVT_TAXES_DIRECT": (2000, 250000),
    "GOVT_TAXES_INDIRECT": (1000, 100000),
    "GOVT_FEES_PENALTIES": (200, 25000),
    "TRANSFER_P2P_FAMILY_FRIENDS": (200, 75000),
    "TRANSFER_SELF_INTERNAL": (200, 250000),
    "TRANSFER_CASH_WITHDRAWAL": (500, 20000),
    "TRANSFER_FOREX": (10000, 500000),
    # Income (credit)
    "INCOME_SALARY": (15000, 250000),
    "INCOME_BUSINESS_SELF_EMPLOYED": (5000, 500000),
    "INCOME_INVESTMENT": (500, 150000),
    "INCOME_REFUNDS": (100, 50000),
    "INCOME_GOVT_BENEFIT": (500, 150000),
    # Legacy / fallback labels (if still present)
    "FINANCIAL_SERVICES": (350, 85000),
    "SUBSCRIPTIONS": (99, 15000),
    "CHARITY": (250, 25000),
    "INCOME": (15000, 250000),
}

# Realistic class frequency priors (normalized at runtime for labels present)
LABEL_WEIGHTS: Dict[str, float] = {
    # Everyday spend tends to dominate
    "GROCERIES": 0.14,
    "DINING": 0.13,
    "MOBILITY": 0.11,
    "UTILITIES_TELECOM": 0.10,
    "SHOPPING_ECOM": 0.06,
    "SHOPPING_ELECTRONICS": 0.03,
    "FUEL": 0.07,
    "HEALTH": 0.06,
    "TRAVEL": 0.06,
    "ENTERTAINMENT": 0.05,
    "UTILITIES_POWER": 0.04,
    "UTILITIES_WATER_GAS": 0.03,
    "HOME_IMPROVEMENT": 0.02,
    "SUBSCRIPTIONS": 0.02,
    "CHARITY": 0.01,
    # New lifestyle buckets
    "PERSONAL_CARE_BEAUTY": 0.02,
    "PETS": 0.005,
    "GIFTS_OCCASIONS": 0.01,
    "HOBBIES_SPORTS": 0.008,
    "ALCOHOL_BARS": 0.01,
    "KIDS_BABY": 0.01,
    # Housing / debt / investments
    "HOUSING_RENT": 0.03,
    "DEBT_CREDIT_CARD_BILL": 0.025,
    "DEBT_LOAN_HOME": 0.02,
    "DEBT_LOAN_PERSONAL": 0.015,
    "INVEST_MF_SIP": 0.02,
    "INVEST_STOCK_BROKERAGE": 0.015,
    "INVEST_GOLD_SILVER": 0.015,
    "INVEST_RETIREMENT": 0.01,
    "INSURANCE_PREMIUM_LIFE": 0.01,
    "INSURANCE_PREMIUM_HEALTH": 0.008,
    # Transfers & income
    "TRANSFER_P2P_FAMILY_FRIENDS": 0.03,
    "TRANSFER_SELF_INTERNAL": 0.04,
    "TRANSFER_CASH_WITHDRAWAL": 0.015,
    "INCOME_SALARY": 0.02,
    "INCOME_BUSINESS_SELF_EMPLOYED": 0.01,
    "INCOME_INVESTMENT": 0.01,
    "INCOME_REFUNDS": 0.02,
}

# Account type distribution
ACCOUNT_TYPE_WEIGHTS: List[Tuple[str, float]] = [
    ("SAVINGS", 0.58),
    ("CURRENT", 0.22),
    ("CREDIT_CARD", 0.20),
]

# Noise knobs (can be overridden via CLI)
DEFAULT_NOISE_RATIO = 0.18  # fraction of rows to perturb
LABEL_NOISE_RATIO = 0.02    # fraction to flip to another label
CHANNEL_NOISE_RATIO = 0.04  # fraction to choose non-canonical channel
DATE_ANOMALY_RATIO = 0.02   # fraction to invert or delay posted_at unusually
MISSING_VALUE_RATIO = 0.03  # individual field missingness probability (on noisy rows)
CURRENCY_ALT_RATIO = 0.03   # on noisy rows, chance to use non-INR currency


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

def _weighted_account_type() -> str:
    r = random.random()
    acc = 0.0
    for name, w in ACCOUNT_TYPE_WEIGHTS:
        acc += w
        if r <= acc:
            return name
    return ACCOUNT_TYPE_WEIGHTS[-1][0]


def _apply_row_noise(row: Dict[str, object], label_ids: List[str], noise_ratio: float) -> None:
    if random.random() >= noise_ratio:
        return

    # Minor casing/spacing glitches in merchant and narrative
    def _glitch(s: object) -> str:
        t = s if isinstance(s, str) else ("" if s is None else str(s))
        r = random.random()
        if r < 0.25:
            t = t.upper()
        elif r < 0.50:
            t = t.lower()
        if random.random() < 0.20:
            t = t.replace(" ", "  ")
        if random.random() < 0.10 and len(t) > 6:
            i = random.randint(1, len(t) - 2)
            t = t[:i] + random.choice("qxz") + t[i + 1 :]
        return t

    # Some rows use alternative currency or unusual amounts
    if random.random() < CURRENCY_ALT_RATIO:
        row["currency"] = random.choice(["USD", "EUR"])  # rare FX
    # Outliers: occasionally 2-5x typical amounts (positive values only)
    if random.random() < 0.05:
        try:
            amt = float(str(row.get("amount", "0")).replace(",", ""))
            bump = random.uniform(2.0, 5.0)
            row["amount"] = f"{amt * bump:.2f}"
        except Exception:
            pass

    # Channel noise: choose any channel sometimes
    if random.random() < CHANNEL_NOISE_RATIO:
        row["channel"] = random.choice(
            list({c for lst in CHANNEL_MAP.values() for c in lst} | {"NEFT", "IMPS", "SALARY"})
        )

    # Date anomalies: posted_at before value_date or large delay
    if random.random() < DATE_ANOMALY_RATIO:
        try:
            v = datetime.strptime(str(row["value_date"]), "%Y-%m-%d")
            if random.random() < 0.5:
                p = v - timedelta(days=random.randint(1, 2))
            else:
                p = v + timedelta(days=random.randint(7, 14))
            row["posted_at"] = p.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Refund-like credit: flip some debits into labelled refunds
    if (
        row.get("dr_cr") == "DR"
        and "INCOME_REFUNDS" in label_ids
        and random.random() < 0.05
    ):
        row["dr_cr"] = "CR"
        row["narrative"] = f"REFUND {row['narrative']}"
        row["category_id"] = "INCOME_REFUNDS"
        row["category_display_name"] = "Income \u2013 Refunds & Chargebacks"

    # Missingness for some non-critical fields
    for key in ["merchant_name", "city", "reference", "keyword_hit"]:
        if random.random() < MISSING_VALUE_RATIO:
            row[key] = "" if random.random() < 0.7 else None

    # Label noise: flip to another category id occasionally
    if random.random() < LABEL_NOISE_RATIO and label_ids:
        new_id = random.choice(label_ids)
        row["category_id"] = new_id
        # do not change display name often to simulate inconsistencies

    # Glitchy text fields last
    row["merchant_name"] = _glitch(row.get("merchant_name", ""))
    row["narrative"] = _glitch(row.get("narrative", ""))


def _label_weight_vector(labels: List[Label]) -> List[float]:
    ws = [LABEL_WEIGHTS.get(lbl.id, 0.03) for lbl in labels]
    s = sum(ws)
    if s <= 0:
        return [1.0 / len(labels)] * len(labels)
    return [w / s for w in ws]


def _generate_rows(labels: List[Label], total_rows: int, noise_ratio: float = DEFAULT_NOISE_RATIO) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not labels:
        return rows

    weights = _label_weight_vector(labels)
    # Compute integer allocation per label, distribute remainder
    alloc = [int(total_rows * w) for w in weights]
    remainder = total_rows - sum(alloc)
    for i in random.sample(range(len(alloc)), k=remainder):
        alloc[i] += 1

    # Track used transaction_ids to avoid collisions
    used_ids: set[str] = set()

    def _unique_txid() -> str:
        while True:
            tx = f"TX{random.randint(10**9, 10**10 - 1)}"
            if tx not in used_ids:
                used_ids.add(tx)
                return tx

    all_label_ids = [lbl.id for lbl in labels]

    for label, n_rows in zip(labels, alloc):
        merchants = label.example_merchants or [label.display_name or label.id]
        for _ in range(n_rows):
            # Occasionally pick a non-canonical channel
            canonical_channels = CHANNEL_MAP.get(label.id, ["CARD", "UPI", "NETBANKING"])
            if random.random() < CHANNEL_NOISE_RATIO:
                channel = random.choice(list({c for lst in CHANNEL_MAP.values() for c in lst} | {"NEFT", "IMPS", "SALARY"}))
            else:
                channel = random.choice(canonical_channels)

            city = random.choice(CITY_POOL)
            keyword = _choose(label.keywords, label.id.lower())
            merchant = _choose(merchants, label.display_name or label.id.title())
            amount = _triangular_amount(label.id)
            if random.random() < 0.03:
                # Introduce occasional tiny or zero amounts
                amount = round(random.choice([0.0, random.uniform(1.0, 25.0)]), 2)

            credit_or_debit = "CR" if label.group.upper() == "INCOME" else "DR"
            # refund-like flips handled in noise function
            value_date, posted_at = _random_dates()
            reference = _reference(channel)
            row = {
                "transaction_id": _unique_txid(),
                "value_date": value_date,
                "posted_at": posted_at,
                "amount": f"{amount:.2f}",
                "currency": "INR",
                "dr_cr": credit_or_debit,
                "merchant_name": merchant,
                "reference": reference,
                "narrative": _narrative(channel, merchant, keyword, city, reference),
                "channel": channel,
                "account_type": _weighted_account_type(),
                "city": city,
                "keyword_hit": keyword,
                "category_id": label.id,
                "category_display_name": label.display_name or label.id.title(),
            }
            _apply_row_noise(row, all_label_ids, noise_ratio)
            rows.append(row)

    random.shuffle(rows)
    return rows


def write_dataset(rows: List[Dict[str, object]], train_size: int, test_size: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = train_size + test_size
    if len(rows) < total:
        raise ValueError(f"Not enough rows generated ({len(rows)}) for requested total {total}")

    train_rows = rows[:train_size]
    test_rows = rows[train_size : train_size + test_size]
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
    ap = argparse.ArgumentParser(description="Generate synthetic transaction datasets with realistic noise")
    ap.add_argument("--train-size", type=int, default=300_000, help="Number of rows for train.csv")
    ap.add_argument("--test-size", type=int, default=150_000, help="Number of rows for test.csv")
    ap.add_argument("--noise-ratio", type=float, default=DEFAULT_NOISE_RATIO, help="Fraction of rows to perturb with noise [0-1]")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)
    labels = [label for label in _parse_taxonomy(TAXONOMY_PATH) if label.id]
    total = args.train_size + args.test_size
    rows = _generate_rows(labels, total_rows=total, noise_ratio=args.noise_ratio)
    write_dataset(rows, train_size=args.train_size, test_size=args.test_size)


if __name__ == "__main__":
    main()
