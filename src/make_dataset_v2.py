# src/make_dataset_v2.py
import random, csv, argparse, re
from pathlib import Path
from datetime import datetime, timedelta
random.seed(42)

# ---------- Utility helpers ----------
def rnd(a,b): return random.randint(a,b)
def pick(x):  return random.choice(x)
def amount():
    base = pick([49,79,89,99,129,149,199,249,299,349,499,799,999,1299,1499,1999,2499,3599,4999,7999,12999,19999])
    return f"{base}.{rnd(0,99):02d}" if random.random()<0.3 else f"{base}.00"
def dt_str():
    base = datetime(2024, 1, 1)
    dt = base + timedelta(days=rnd(0, 640), seconds=rnd(0, 86399))
    return dt.strftime("%Y-%m-%d %H:%M:%S") if random.random()<0.8 else dt.strftime("%d/%m/%y %H:%M")
def mask_ac(): return f"XXXXXXXX{rnd(1000,9999)}"
def utr(): return f"{rnd(10**10,10**11-1)}{rnd(1000,9999)}"
def rrn(): return f"{rnd(10**6,10**7-1)}{rnd(1000,9999)}"
def gstin(): return f"{rnd(10,99)}ABCDE{rnd(1000,9999)}A1Z{rnd(0,9)}"
CITIES = ["DELHI","MUMBAI","BLR","KOLKATA","PUNE","HYD","CHENNAI","AHMEDABAD","JAIPUR","LUCKNOW","SURAT","INDORE"]
BANKS  = ["HDFC","ICICI","SBI","AXIS","KOTAK","YES","BOB","PNB","IDFC"]
VPAS   = ["@okaxis","@okhdfcbank","@ybl","@apl","@icici","@oksbi","@okboi","@okkotak","@idfcbank"]
CURR   = ["INR"]*9 + ["USD","EUR"]
EMOJI  = ["💳","🧾","🛒","🍔","☕","⛽","⚡","💡","📶","🧼","📦","🎫","🎮"]

# ---------- Label schema & MCCs ----------
MCC = {
  "DINING":"5812","GROCERIES":"5411","FUEL":"5541",
  "UTILITIES_POWER":"4900","UTILITIES_TELECOM":"4814","UTILITIES_WATER_GAS":"4900",
  "SHOPPING_ECOM":"5399","SHOPPING_ELECTRONICS":"5732",
  "TRAVEL":"4112","MOBILITY":"4121","HEALTH":"8062","ENTERTAINMENT":"7841",
  "FEES":"6011","OTHER":"7399"
}

# ---------- Merchant catalog per class (base names) ----------
CATALOG = {
  "DINING": ["Starbucks","Dominos","McDonalds","KFC","Barista","Cafe Coffee Day","Haldiram","Bikanervala","Faasos","Burger King","Subway","Behrouz Biryani","Pizza Hut","Chai Point","Chayos","Wow Momo","Theobroma","Mad Over Donuts","Giani","Biryani Blues","Sagar Ratna","Punjab Grill","Mainland China","Taco Bell"],
  "GROCERIES": ["DMart","BigBasket","Lidl","Aldi","More","Reliance Fresh","Spencers","Big Bazaar","Natures Basket","Spar","Metro Cash and Carry","Kirana King","FreshToHome","JioMart","Milk Basket","Star Bazaar"],
  "FUEL": ["Shell","HPCL","BPCL","IOCL","IndianOil","Essar","Nayara","Total","Reliance Petro"],
  "UTILITIES_POWER": ["BSES","Tata Power","MSEB","Adani Electricity","Torrent Power","CESC"],
  "UTILITIES_TELECOM": ["Airtel","Jio","Vodafone Idea","BSNL","MTNL","Telekom"],
  "UTILITIES_WATER_GAS": ["BWSSB","Delhi Jal Board","Indraprastha Gas","Mahanagar Gas","GAIL Gas","IGL"],
  "SHOPPING_ECOM": ["Amazon","Flipkart","Myntra","Ajio","Nykaa","Pepperfry","FirstCry","Tanishq Online"],
  "SHOPPING_ELECTRONICS": ["Croma","Reliance Digital","Apple Store","Mi Store","Boat","Noise","Samsung Store","OnePlus"],
  "TRAVEL": ["IRCTC","INDIGO","SPICEJET","Air India","Vistara","Delhi Metro","MakeMyTrip","Cleartrip","Goibibo"],
  "MOBILITY": ["Uber","Ola","Rapido"],
  "HEALTH": ["Max Hospital","Apollo Hospital","Fortis","MedPlus","1mg","Pharmeasy"],
  "ENTERTAINMENT": ["BookMyShow","Sony Liv","Hotstar","Netflix","Zee5","Steam","PlayStation Store"],
  "FEES": ["FASTag","Convenience Fee","Processing Fee","Chargeback","Surcharge"],
  "OTHER": ["Axis Rewards","Generic Merchant","Local Vendor","Neighborhood Store"]
}

# Variants / noise
def variantize(name):
    v=set([name, name.upper(), name.lower(), name.replace(" ", ""), name.replace("Cafe","Café")])
    if "Dominos" in name: v.update(["Domino's","Domino s","DOMINOS"])
    if "Starbucks" in name: v.update(["Starbucks Coffee","SBX","Starbux"])
    if name in ["Vodafone Idea"]: v.update(["Vodafone","Vi","Voda-Idea"])
    if "IndianOil" in name: v.update(["Indian Oil","IOCL"])
    return list(v)

def noise_text(s):
    if random.random()<0.25: s=s.upper()
    elif random.random()<0.5: s=s.lower()
    if random.random()<0.2: s=s.replace(" ", "  ")
    if random.random()<0.15: s=s.replace(" ", "-")
    if random.random()<0.15: s=s.replace("-", "/")
    if random.random()<0.2: s += " " + pick(EMOJI)
    if random.random()<0.2 and len(s)>6:
        i=rnd(1,len(s)-2); s=s[:i]+pick("qxz")+s[i+1:]
    return s

def patterns(label, merch, amt, city, bank, vpa, last4, ref, direction, currency, date):
    mcc = MCC[label]
    base = [
      f"{merch} POS {city} {amt}",
      f"{bank} UPI/{rnd(1000,9999)} {merch} {amt} {vpa}",
      f"IMPS RRN {rrn()} {merch} {amt}",
      f"NEFT {merch} UTR {utr()} {amt}",
      f"CARD {last4} ECOM {merch} {amt}",
      f"AUTOPAY {merch} {amt}",
      f"{merch} ONLINE {amt}",
      f"{merch} TxnID {ref} MCC {mcc} {amt}",
      f"{bank} {direction} {merch} {amt} {currency}",
      f"{merch} {city} {date} {amt} AUTH {rnd(100000,999999)}",
    ]

    def add_hint(ok, txt):
        if ok: base.append(txt)

    # --- add class-unique hints with ~60% probability for weaker classes ---
    p = 0.6
    if label == "SHOPPING_ECOM":
        add_hint(random.random()<p, f"{merch} ORDER ID {ref}")
        add_hint(random.random()<p, f"{merch} FULFILLED BY AMAZON")
        add_hint(random.random()<p, f"{merch} MARKETPLACE")
    if label == "FUEL":
        add_hint(random.random()<p, f"{merch} PETROL PUMP {city}")
        add_hint(random.random()<p, f"FUEL SURCHARGE {round(float(amt.split('.')[0])*0.025,2)} REVERSAL")
    if label == "UTILITIES_POWER":
        add_hint(random.random()<p, f"{merch} ELECTRICITY BILL PAYMENT")
        add_hint(random.random()<p, f"{merch} EB BILL {city}")
    if label == "UTILITIES_TELECOM":
        add_hint(random.random()<p, f"{merch} PREPAID RECHARGE {amt}")
        add_hint(random.random()<p, f"{merch} DATA PACK {rnd(1,5)}GB")
    if label == "UTILITIES_WATER_GAS":
        add_hint(random.random()<p, f"{merch} WATER BILL {amt}")
        add_hint(random.random()<p, f"{merch} GAS BILL {amt}")
    if label == "FEES":
        add_hint(True,               f"{merch} CONVENIENCE FEE")   # always one explicit fee cue
        add_hint(random.random()<p, f"{merch} SERVICE CHARGE")
        add_hint(random.random()<p, f"{merch} PROCESSING FEE")

    # class-specific extras already present
    if label=="DINING": base += [f"{merch} TIP {round(float(amt.split('.')[0])*0.05,2)}"]
    if label.startswith("SHOPPING"): base += [f"{merch} CASHBACK -{round(float(amt.split('.')[0]) * 0.1,2)}", f"{merch} REFUND -{amt}"]
    if label=="TRAVEL": base += [f"{merch} TICKET {ref} {amt}"]
    if label=="MOBILITY": base += [f"{merch} RIDE {ref} {amt}"]
    if label=="HEALTH": base += [f"{merch} OPD {ref} {amt}", f"{merch} PHARMACY {amt}"]
    if label=="ENTERTAINMENT": base += [f"{merch} SUBSCRIPTION {amt}"]

    return [noise_text(s) for s in base]


def split_merchants(train_ratio=0.75):
    train, test = {}, {}
    for lab, names in CATALOG.items():
        items = [{"base": n.lower(), "variants": variantize(n)} for n in names]
        random.shuffle(items)
        k = max(1, int(len(items)*train_ratio))
        train[lab] = items[:k]
        test[lab]  = items[k:] or items[-1:]
    return train, test

def make_rows(items_for_label, label, n_rows):
    variants=[]
    for rec in items_for_label:
        for v in rec["variants"]:
            variants.append((rec["base"], v))
    rows=[]
    for _ in range(n_rows):
        base, var = pick(variants)
        amt = amount(); city=pick(CITIES); bank=pick(BANKS); vpa=pick(VPAS)
        last4=rnd(1000,9999); ref=rnd(10**6,10**7-1)
        direction = pick(["DEBIT","CREDIT","DR","CR"])
        currency = pick(CURR); date = dt_str()
        raw = pick(patterns(label, var, amt, city, bank, vpa, last4, ref, direction, currency, date))
        if random.random()<0.05: raw += " CHARGEBACK"
        if random.random()<0.05: raw = "REVERSAL " + raw
        rows.append({
            "id": f"r{rnd(10**8,10**9-1)}",
            "raw_text": raw, "date": date, "amount": amt, "currency": currency, "direction": direction,
            "merchant_base": base, "merchant_display": var, "channel": pick(["POS","UPI","IMPS","NEFT","ECOM","AUTOPAY","ONLINE","ATM"]),
            "city": city, "bank": bank, "account_mask": mask_ac(), "label": label
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-class", type=int, default=8000)   # 14 classes → ~112k
    ap.add_argument("--test-per-class",  type=int, default=2500)   # 14 classes → ~35k
    ap.add_argument("--train-merchant-ratio", type=float, default=0.75)
    args = ap.parse_args()

    train_split, test_split = split_merchants(args.train_merchant_ratio)
    train_rows=[]; test_rows=[]
    for lab in CATALOG.keys():
        train_rows += make_rows(train_split[lab], lab, args.train_per_class)
        test_rows  += make_rows(test_split[lab],  lab, args.test_per_class)

    random.shuffle(train_rows); random.shuffle(test_rows)

    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/train.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(train_rows[0].keys())); w.writeheader(); w.writerows(train_rows)
    with open("data/test.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(test_rows[0].keys()));  w.writeheader(); w.writerows(test_rows)

    from collections import Counter
    print("Train counts:", Counter([r["label"] for r in train_rows]))
    print("Test counts:",  Counter([r["label"] for r in test_rows]))
    print(f"Wrote train:{len(train_rows)} test:{len(test_rows)}")

if __name__=="__main__":
    main()
