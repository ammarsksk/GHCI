import re, yaml, unicodedata
from typing import Optional, Tuple

def load_taxonomy(path="config/taxonomy.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)  # fold accents (café -> cafe)
    s = re.sub(r"\s+", " ", s)
    return s

# Phrase/regex patterns that generalize beyond merchant names
PATTERNS = [
    # UTILITIES
    (r"\b(bill( desk)?|bill payment|recharge|electric|electricity|water bill|gas bill|telekom|bsnl|vodafone|vi|jio|mtnl)\b", "UTILITIES"),
    (r"\bmcc\s*4900\b", "UTILITIES"),
    (r"\bautopay\b", "UTILITIES"),

    # FUEL
    (r"\b(fuel|petrol|diesel|surcharge)\b", "FUEL"),
    (r"\bmcc\s*5541\b", "FUEL"),

    # SHOPPING
    (r"\b(cashback|refund|no cost emi|emi|ecom|online store)\b", "SHOPPING"),
    (r"\bmcc\s*5399\b", "SHOPPING"),

    # DINING
    (r"\b(cafe|café|coffee|pizza|biryani|donuts?|burger|momo|theobroma|chai|tip)\b", "DINING"),
    (r"\bmcc\s*5812\b", "DINING"),

    # GROCERIES
    (r"\b(grocery|kirana|basket|bazaar|mart|fresh to home|freshtohome|milk basket|milkbasket|nature'?s?\s*basket)\b", "GROCERIES"),
    (r"\bmcc\s*5411\b", "GROCERIES"),

    # OTHER (travel, mobility, tickets, hospitals, metro)
    (r"\b(irctc|indigo|air\s*india|spicejet|ticket|metro|apollo hospital|max hospital|bookmyshow|fastag)\b", "OTHER"),
]

def build_keyword_map(tax):
    kmap={}
    for item in tax["labels"]:
        lab=item["id"]
        for kw in item.get("keywords", []):
            kmap[kw.lower()]=lab
    return kmap, tax.get("fallback","OTHER")

_COMPILED = [(re.compile(pat), lab) for pat, lab in PATTERNS]

def rules_predict(text: str, tax) -> Optional[Tuple[str,float,dict]]:
    nrm = normalize(text)
    # 1) explicit keywords from taxonomy.yaml
    kmap, fallback = build_keyword_map(tax)
    hits=[]
    for kw,lab in kmap.items():
        if kw in nrm:
            hits.append((lab, kw))
    if hits:
        lab, kw = max(hits, key=lambda x: len(x[1]))
        conf = 0.97 if len(kw)>=4 else 0.93
        return lab, conf, {"rule":"keyword", "hit":kw}

    # 2) regex phrase patterns
    for rgx, lab in _COMPILED:
        if rgx.search(nrm):
            return lab, 0.94, {"rule":"pattern", "hit":rgx.pattern}

    # 3) simple UPI+fuel hint as last resort
    if "upi" in nrm and any(x in nrm for x in ["shell","iocl","hpcl","bpcl","indianoil"]):
        return "FUEL", 0.90, {"rule":"upi+fuelhint"}
    return None
