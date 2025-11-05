import random, csv
from pathlib import Path

random.seed(42)

CATS = {
    "DINING":    ["Starbucks","Dominos","McDonalds","KFC","Barista","Cafe Coffee Day"],
    "GROCERIES": ["DMart","BigBasket","Lidl","Aldi","More","Reliance Fresh"],
    "FUEL":      ["Shell","HPCL","BPCL","IOCL","IndianOil"],
    "UTILITIES": ["Vodafone","Airtel","Jio","Telekom","BSES","MSEB","Water Board"],
    "SHOPPING":  ["Amazon","Flipkart","Myntra","IKEA","Decathlon"]
}

UPI_VPAS = ["@okaxis","@okhdfcbank","@ybl","@apl","@icici"]
CITIES    = ["DELHI","MUMBAI","BLR","KOLKATA","PUNE","HYD"]

def noised(s: str) -> str:
    s = s if random.random()<0.6 else s.upper()
    s = s if random.random()<0.6 else s.lower()
    if random.random()<0.25:  # tiny typo
        i = random.randrange(len(s))
        s = s[:i] + random.choice("xq") + s[i+1:]
    if random.random()<0.35:  # add digits/refs
        s += f" {random.randint(10,99)}{random.randint(10,99)}"
    return s

def make_row(cat, merchant):
    amt = random.choice([89,149,299,499,799,1299,2199,3599])
    flavor = random.random()
    if flavor < 0.25:
        raw = f"{merchant} POS {random.choice(CITIES)} {amt}.00"
    elif flavor < 0.5:
        raw = f"HDFC UPI/{random.randint(1000,9999)} {merchant} {amt}.00 {random.choice(UPI_VPAS)}"
    elif flavor < 0.75:
        raw = f"NEFT {merchant} REF{random.randint(100000,999999)} {amt}"
    else:
        raw = f"{merchant} ONLINE {amt}"
    return {
        "id": f"r{random.randint(10**7,10**8-1)}",
        "raw_text": noised(raw),
        "merchant_base": merchant.lower(),
        "label": cat
    }

def main(n_per_cat=400):
    rows=[]
    for cat, merchants in CATS.items():
        for _ in range(n_per_cat):
            m = random.choice(merchants)
            rows.append(make_row(cat, m))
    # add some OTHER/noise rows
    for _ in range(200):
        rows.append(make_row("OTHER", random.choice(["IRCTC","INDIGO","SBI ATM","UBER","OLACABS"])))
    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/train.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["id","raw_text","merchant_base","label"])
        w.writeheader(); w.writerows(rows)
    print(f"Wrote data/train.csv with {len(rows)} rows")

if __name__=="__main__":
    main()
