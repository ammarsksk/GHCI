# src/make_dataset_rich.py
import random, csv, argparse, math, itertools, re
from pathlib import Path
from datetime import datetime, timedelta

random.seed(42)

CITIES = ["DELHI","MUMBAI","BLR","KOLKATA","PUNE","HYD","CHENNAI","AHMEDABAD","JAIPUR","LUCKNOW","SURAT","INDORE"]
BANKS  = ["HDFC","ICICI","SBI","AXIS","KOTAK","YES","BOB","PNB","IDFC"]
VPAS   = ["@okaxis","@okhdfcbank","@ybl","@apl","@icici","@oksbi","@okboi","@okkotak","@idfcbank"]
CURR   = ["INR","INR","INR","USD","EUR"]  # mostly INR, sometimes FX

# Merchant catalog (base names). We'll auto-create aliases/variants.
MERCHANTS = {
    "DINING": [
        "Starbucks","Dominos","McDonalds","KFC","Barista","Cafe Coffee Day","Haldiram","Bikanervala","Faasos",
        "Burger King","Subway","Behrouz Biryani","Pizza Hut","Chai Point","Chayos","Wow Momo","Theobroma",
        "Mad Over Donuts","Giani","Biryani Blues","Sagar Ratna","Punjab Grill","Mainland China","Taco Bell"
    ],
    "GROCERIES": [
        "DMart","BigBasket","Lidl","Aldi","More","Reliance Fresh","Spencers","Big Bazaar","Natures Basket",
        "Spar","Metro Cash and Carry","Kirana King","FreshToHome","JioMart","Milk Basket","Star Bazaar"
    ],
    "FUEL": [
        "Shell","HPCL","BPCL","IOCL","IndianOil","Essar","Nayara","Total","Reliance Petro"
    ],
    "UTILITIES": [
        "BSES","Tata Power","MSEB","Adani Electricity","BWSSB","Delhi Jal Board","Indraprastha Gas",
        "Airtel","Jio","Vodafone Idea","BSNL","Telekom","MTNL","Torrent Power","CESC"
    ],
    "SHOPPING": [
        "Amazon","Flipkart","Myntra","IKEA","Decathlon","Ajio","Nykaa","Croma","Reliance Digital",
        "Tanishq","H&M","Zara","Pepperfry","FirstCry","Boat","Noise","Apple Store","Mi Store"
    ],
    "OTHER": [
        "IRCTC","INDIGO","SPICEJET","Air India","Uber","Ola","Rapido","Delhi Metro","Max Hospital",
        "Apollo Hospital","BookMyShow","MakeMyTrip","Cleartrip","Paytm","CPA FEE","FASTag","Axis Rewards"
    ]
}

# Hindi/hinglish tokens to sprinkle into strings
HI_TOK = {
    "bill": ["बिल","बिजली बिल","पानी बिल"],
    "recharge": ["रीचार्ज","रिचार्ज"],
    "gas": ["गैस"],
    "fuel": ["फ्यूल","पेट्रोल"],
    "paid": ["भुगतान","पेड"],
    "ticket": ["टिकट"],
}

EMOJIS = ["💳","🧾","🛒","🍔","☕","⛽","⚡","💡","📶","🧼","📦"]

def rnd(a, b): return random.randint(a, b)
def pick(lst): return random.choice(lst)

def amount():
    base = pick([49,79,89,99,129,149,199,249,299,349,499,799,999,1299,1499,1999,2499,3599,4999,7999,12999])
    # occasional paise/cents
    if random.random() < 0.3: return f"{base}.{rnd(0,99):02d}"
    return f"{base}.00"

def dt_str():
    base = datetime(2024, 7, 1)
    dt = base + timedelta(days=rnd(0, 480), seconds=rnd(0, 86399))
    # 80% yyyy-mm-dd, 20% dd/mm/yy
    if random.random() < 0.8:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.strftime("%d/%m/%y %H:%M")

def mask_account():
    return f"XXXXXXXX{rnd(1000,9999)}"

def utr():
    return f"{rnd(10**10,10**11-1)}{rnd(1000,9999)}"

def rrn():
    return f"{rnd(10**6,10**7-1)}{rnd(1000,9999)}"

def gstin():
    return f"{rnd(10,99)}ABCDE{rnd(1000,9999)}A1Z{rnd(0,9)}"

def mcc(label):
    # rough MCC mapping
    return {"DINING":"5812","GROCERIES":"5411","FUEL":"5541","UTILITIES":"4900","SHOPPING":"5399","OTHER":"7399"}[label]

def variantize_name(name):
    """Create real-world variants: spacing, apostrophes, abbreviations, leetspeak."""
    variants = set()
    base = name
    variants.add(base)
    variants.add(base.upper())
    variants.add(base.lower())
    variants.add(base.replace(" ", ""))
    variants.add(base.replace(" and ", " & "))
    variants.add(base.replace("'", ""))
    variants.add(base.replace("Cafe", "Café"))
    variants.add(re.sub(r"a", "@", base, flags=re.I))
    if "Dominos" in base:
        variants.update(["Domino's","Dominos Pizza","Domino s","DOMINOS"])
    if "Starbucks" in base:
        variants.update(["Starbux","Starbucks Coffee","STARBUCKS BLR","SBX"])
    if "IndianOil" in base or "IOCL" in base:
        variants.update(["INDIAN OIL","IOCL","Indian Oil"])
    if "Vodafone" in base:
        variants.update(["Vi","Vodafone Idea","Voda-Idea"])
    return list(variants)

def code_mix(text, prob=0.15):
    """Insert some Hindi/hinglish tokens with small prob."""
    if random.random() > prob: return text
    replacements = [
        ("BILL", pick(HI_TOK["bill"])),
        ("bill", pick(HI_TOK["bill"])),
        ("FUEL", pick(HI_TOK["fuel"])),
        ("fuel", pick(HI_TOK["fuel"])),
        ("GAS", pick(HI_TOK["gas"])),
        ("gas", pick(HI_TOK["gas"])),
        ("TICKET", pick(HI_TOK["ticket"])),
        ("ticket", pick(HI_TOK["ticket"])),
        ("PAID", pick(HI_TOK["paid"])),
        ("paid", pick(HI_TOK["paid"])),
        ("RECHARGE", pick(HI_TOK["recharge"])),
        ("recharge", pick(HI_TOK["recharge"])),
    ]
    for a, b in replacements:
        if a in text and random.random() < 0.5:
            text = text.replace(a, b)
    return text

def noise_text(s: str):
    """Add realistic noise: case, spaces, emoji, punctuation, typos."""
    r = random.random()
    if r < 0.25: s = s.upper()
    elif r < 0.5: s = s.lower()
    # double spaces / random hyphens / slashes
    if random.random() < 0.25: s = s.replace(" ", "  ")
    if random.random() < 0.2: s = s.replace(" ", "-")
    if random.random() < 0.15: s = s.replace("-", "/")
    # append emoji or extra refs
    if random.random() < 0.2: s += " " + pick(EMOJIS)
    # tiny typo
    if random.random() < 0.2 and len(s) > 6:
        i = rnd(1, len(s)-2)
        s = s[:i] + pick("qxz") + s[i+1:]
    return s

def make_variants_catalog():
    cat = {}
    for lab, names in MERCHANTS.items():
        vlist = []
        for n in names:
            v = variantize_name(n)
            vlist.append({"base": n.lower(), "variants": v})
        cat[lab] = vlist
    return cat

def split_merchants(train_ratio=0.75):
    """Split base merchants per label; all variants inherit the split ⇒ merchant-disjoint."""
    train, test = {}, {}
    for lab, items in make_variants_catalog().items():
        random.shuffle(items)
        k = max(1, int(len(items)*train_ratio))
        train[lab] = items[:k]
        test[lab]  = items[k:] or items[-1:]
    return train, test

def patterns(label, merch_variant, amt, city, bank, vpa, last4, ref, direction, currency, date):
    """Return multiple textual patterns for the same logical transaction."""
    mcc_code = mcc(label)
    base = [
        f"{merch_variant} POS {city} {amt}",
        f"{bank} UPI/{rnd(1000,9999)} {merch_variant} {amt} {vpa}",
        f"IMPS RRN {rrn()} {merch_variant} {amt}",
        f"NEFT {merch_variant} UTR {utr()} {amt}",
        f"CARD {last4} ECOM {merch_variant} {amt}",
        f"AUTOPAY {merch_variant} {amt}",
        f"{merch_variant} ONLINE {amt}",
        f"{merch_variant} BILL PAYMENT {amt}",
        f"{merch_variant} TxnID {ref} MCC {mcc_code} {amt}",
        f"{bank} {direction} {merch_variant} {amt} {currency}",
        f"{merch_variant} GSTIN {gstin()} INV#{rnd(10000,99999)} {amt}",
        f"{merch_variant} {city} {date} {amt} AUTH {rnd(100000,999999)}"
    ]
    # category tweaks
    if label == "FUEL":
        base.append(f"{merch_variant} FUEL SURCHARGE {round(float(amt.split('.')[0])*0.025,2)} REVERSAL")
    if label == "UTILITIES":
        base.append(f"{merch_variant} BILL DESK {amt}")
        base.append(f"{merch_variant} AUTO PAY {amt}")
    if label == "SHOPPING":
        base.append(f"{merch_variant} CASHBACK -{round(float(amt.split('.')[0]) * 0.1,2)}")
        base.append(f"{merch_variant} REFUND -{amt}")
    if label == "DINING":
        base.append(f"{merch_variant} TIP {round(float(amt.split('.')[0])*0.05,2)}")
    if label == "OTHER":
        base.append(f"{merch_variant} TICKET {ref} {amt}")
    # EMI / Fees / Charges (generic)
    base.extend([
        f"CONVENIENCE FEE {merch_variant} {round(float(amt.split('.')[0])*0.02,2)}",
        f"NO COST EMI {merch_variant} {amt} 3/6",
        f"SPLIT BILL {merch_variant} PART {rnd(1,3)}/3 {amt}"
    ])
    # Add code-mix and noise variants
    out = []
    for s in base:
        s = s.replace("  ", " ")
        s = code_mix(s, prob=0.2)
        s = noise_text(s)
        out.append(s)
    return out

def make_rows(items, n_per_cat, label):
    rows=[]
    # flatten all merchant variants for this label
    variants = []
    for rec in items:
        for v in rec["variants"]:
            variants.append((rec["base"], v))
    if not variants:
        return rows
    for _ in range(n_per_cat):
        base, var = pick(variants)
        amt = amount()
        city, bank, vpa = pick(CITIES), pick(BANKS), pick(VPAS)
        last4 = rnd(1000,9999)
        ref   = rnd(10**6,10**7-1)
        direction = pick(["DEBIT","CREDIT","DR","CR"])
        currency = pick(CURR)
        date = dt_str()
        pats = patterns(label, var, amt, city, bank, vpa, last4, ref, direction, currency, date)
        raw = pick(pats)
        # occasionally add refund/chargeback markers
        if random.random() < 0.05:
            raw += " CHARGEBACK"
        if random.random() < 0.05:
            raw = "REVERSAL " + raw
        rows.append({
            "id": f"r{rnd(10**8,10**9-1)}",
            "raw_text": raw,
            "date": date,
            "amount": amt,
            "currency": currency,
            "direction": direction,
            "merchant_base": base,       # used only for analysis/splitting
            "merchant_display": var,     # human-facing variant
            "channel": pick(["POS","UPI","IMPS","NEFT","ECOM","AUTOPAY","ONLINE","ATM"]),
            "city": city,
            "bank": bank,
            "account_mask": mask_account(),
            "label": label
        })
    return rows

def build_dataset(train_ratio=0.75, train_per_cat=10000, test_per_cat=3000):
    train_out, test_out = [], []
    train_items, test_items = split_merchants(train_ratio=train_ratio)
    for lab in MERCHANTS.keys():
        train_rows = make_rows(train_items[lab], train_per_cat, lab)
        test_rows  = make_rows(test_items[lab],  test_per_cat,  lab)
        train_out.extend(train_rows)
        test_out.extend(test_rows)
    random.shuffle(train_out)
    random.shuffle(test_out)
    return train_out, test_out

def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-cat", type=int, default=10000)
    ap.add_argument("--test-per-cat",  type=int, default=3000)
    ap.add_argument("--train-merchant-ratio", type=float, default=0.75)
    args = ap.parse_args()

    train, test = build_dataset(
        train_ratio=args.train_merchant_ratio,
        train_per_cat=args.train_per_cat,
        test_per_cat=args.test_per_cat
    )
    write_csv("data/train.csv", train)
    write_csv("data/test.csv", test)

    from collections import Counter
    print("Train counts:", Counter([r["label"] for r in train]))
    print("Test counts:",  Counter([r["label"] for r in test]))
    print(f"Wrote data/train.csv ({len(train)} rows), data/test.csv ({len(test)} rows)")

if __name__ == "__main__":
    main()
