# src/rules_v2.py
import re, yaml, unicodedata
from typing import Optional, Tuple

def load_taxonomy(path="config/taxonomy.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s

# broad phrase/MCC rules by our expanded schema
REGEX_RULES = [
  (r"\bmcc\s*5812\b|\b(cafe|café|coffee|pizza|biryani|donuts?|burger|momo|theobroma|chai|tip)\b", "DINING"),
  (r"\bmcc\s*5411\b|\b(grocery|kirana|basket|bazaar|mart|milkbasket|nature'?s?\s*basket|fresh ?to ?home)\b", "GROCERIES"),
  (r"\bmcc\s*5541\b|\b(fuel|petrol|diesel|surcharge)\b", "FUEL"),
  (r"\bmcc\s*4900\b|\b(bill( desk)?|bill payment)\b", "UTILITIES_POWER"),
  (r"\bmcc\s*4814\b|\b(recharge|telekom|bsnl|vodafone|vi|airtel|jio|mtnl)\b", "UTILITIES_TELECOM"),
  (r"\b(water bill|jal board|gas bill|png|igl|indraprastha gas|mahanagar gas)\b", "UTILITIES_WATER_GAS"),
  (r"\bmcc\s*5399\b|\b(cashback|refund|ecom|online store|ajio|myntra|flipkart|amazon)\b", "SHOPPING_ECOM"),
  (r"\bmcc\s*5732\b|\b(croma|reliance digital|apple store|mi store|boat|noise|samsung|oneplus)\b", "SHOPPING_ELECTRONICS"),
  (r"\bmcc\s*4112\b|\b(irctc|indigo|air\s*india|spicejet|vistara|metro|ticket|makemytrip|cleartrip|goibibo)\b", "TRAVEL"),
  (r"\bmcc\s*4121\b|\b(uber|ola|rapido|ride)\b", "MOBILITY"),
  (r"\bmcc\s*8062\b|\b(hospital|opd|pharmacy|pharmeasy|1mg|medplus|fortis|apollo|max)\b", "HEALTH"),
  (r"\bmcc\s*7841\b|\b(bookmyshow|netflix|sony liv|hotstar|zee5|steam|playstation)\b", "ENTERTAINMENT"),
  (r"\bmcc\s*6011\b|\b(fastag|convenience fee|processing fee|chargeback|reversal|surcharge)\b", "FEES"),
]

_COMPILED = [(re.compile(pat), lab) for pat, lab in REGEX_RULES]

def build_keyword_map(tax):
    kmap={}
    for item in tax["labels"]:
        lab=item["id"]
        for kw in item.get("keywords", []):
            kmap[kw.lower()] = lab
    return kmap, tax.get("fallback","OTHER")

def rules_predict(text: str, tax) -> Optional[Tuple[str,float,dict]]:
    nrm = normalize(text)

    # 1) explicit keywords from taxonomy.yaml (your merchant dictionary)
    kmap, fallback = build_keyword_map(tax)
    hits=[]
    for kw, lab in kmap.items():
        if kw in nrm:
            hits.append((lab, kw))
    if hits:
        lab, kw = max(hits, key=lambda x: len(x[1]))
        conf = 0.98 if len(kw)>=4 else 0.94
        return lab, conf, {"rule":"keyword", "hit":kw}

    # 2) regex/MCC/phrase patterns
    for rgx, lab in _COMPILED:
        if rgx.search(nrm):
            return lab, 0.95, {"rule":"pattern", "hit":rgx.pattern}

    # 3) fallback heuristic
    if "upi" in nrm and any(x in nrm for x in ["shell","iocl","hpcl","bpcl","indianoil"]):
        return "FUEL", 0.90, {"rule":"upi+fuelhint"}
    return None
