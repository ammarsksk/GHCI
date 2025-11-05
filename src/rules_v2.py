# src/rules_v2.py
import re, yaml, unicodedata
from typing import Optional, Tuple

def load_taxonomy(path="config/taxonomy.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = re.sub(r"\s+", " ", s)
    return s

# Expanded, conservative patterns. Weak classes need mild model support (handled in trainer).
REGEX_RULES = [
  # DINING
  (r"\bmcc\s*5812\b|\b(cafe|café|coffee|pizza|biryani|donuts?|burger|momo|chai|theobroma|tip|restaurant|kitchen)\b", "DINING"),

  # GROCERIES
  (r"\bmcc\s*5411\b|\b(grocery|kirana|basket|bazaar|mart|milkbasket|nature'?s?\s*basket|fresh\s*to\s*home|supermarket)\b", "GROCERIES"),

  # FUEL (many aliases + surcharge combos)
  (r"\bmcc\s*5541\b|\b(fuel|petrol|diesel|gas\s*station|petrol\s*pump|pump|fuel\s*surcharge|surcharge\s*reversal|iocl|indian\s*oil|bpcl|hpcl|hindustan\s*petroleum|gulf)\b", "FUEL"),

  # UTILITIES_POWER (discoms + electricity terms)
  (r"\bmcc\s*4900\b|\b(bses|tata\s*power|adani\s*(electric|electricity)|torrent\s*power|cesc|bescom|kseb|pspcl|tpddl|wbsedcl|ugvcl|mgvcl|dgvcl|mseb|mppkvvcl|dvvnl|cesc|apdcl)\b|\belectric(ity)?\s*(bill|payment)\b", "UTILITIES_POWER"),

  # UTILITIES_TELECOM (operators + recharge/data)
  (r"\bmcc\s*4814\b|\b(recharge|telekom|bsnl|vodafone|vi\b|airtel|jio|mtnl|data\s*pack|prepaid|postpaid|fiber|broadband)\b", "UTILITIES_TELECOM"),

  # UTILITIES_WATER_GAS
  (r"\b(water\s*bill|jal\s*board|png\b|gas\s*bill|igl|indraprastha\s*gas|mahanagar\s*gas|gail\s*gas|bwssb|djb)\b", "UTILITIES_WATER_GAS"),

  # SHOPPING_ECOM (marketplace & order artefacts)
  (r"\bmcc\s*5399\b|\b(amazon|amzn|azn|flipkart|fkrt|ajio|myntra|nykaa|fulfilled\s*by|marketplace|order\s*(id|no|#)|shipment|delivered|return\s*initiated|prime)\b", "SHOPPING_ECOM"),

  # SHOPPING_ELECTRONICS
  (r"\bmcc\s*5732\b|\b(croma|reliance\s*digital|apple\s*store|mi\s*store|boat|noise|samsung|oneplus|electronics|gadget)\b", "SHOPPING_ELECTRONICS"),

  # TRAVEL
  (r"\bmcc\s*4112\b|\b(irctc|indigo|air\s*india|spicejet|vistara|metro|ticket|pnr|makemytrip|cleartrip|goibibo|boarding)\b", "TRAVEL"),

  # MOBILITY
  (r"\bmcc\s*4121\b|\b(uber|ola|rapido|ride|driver\s*tip)\b", "MOBILITY"),

  # HEALTH
  (r"\bmcc\s*8062\b|\b(hospital|opd|pharmacy|chemist|pharmeasy|1mg|medplus|fortis|apollo|max)\b", "HEALTH"),

  # ENTERTAINMENT
  (r"\bmcc\s*7841\b|\b(bookmyshow|netflix|sony\s*liv|hotstar|zee5|steam|playstation|subscription|season\s*pass)\b", "ENTERTAINMENT"),

  # FEES
  (r"\bmcc\s*6011\b|\b(fastag|convenience\s*fee|processing\s*fee|platform\s*fee|chargeback|reversal|surcharge|service\s*charge)\b", "FEES"),
]
_COMPILED = [(re.compile(p), lab) for p, lab in REGEX_RULES]

def build_keyword_map(tax):
    kmap={}
    for item in tax["labels"]:
        lab=item["id"]
        for kw in item.get("keywords", []):
            kmap[kw.lower()] = lab
    return kmap, tax.get("fallback","OTHER")

def rules_predict(text: str, tax) -> Optional[Tuple[str,float,dict]]:
    nrm = normalize(text)
    kmap, fallback = build_keyword_map(tax)

    hits=[]
    for kw, lab in kmap.items():
        if kw in nrm:
            hits.append((lab, kw))
    if hits:
        lab, kw = max(hits, key=lambda x: len(x[1]))
        conf = 0.98 if len(kw)>=4 else 0.94
        return lab, conf, {"rule":"keyword", "hit":kw}

    for rgx, lab in _COMPILED:
        if rgx.search(nrm):
            return lab, 0.95, {"rule":"pattern", "hit":rgx.pattern}

    # weak composite heuristic for fuel-like UPI text
    if "upi" in nrm and any(x in nrm for x in ["shell","iocl","hpcl","bpcl","indian oil","fuel","petrol","diesel","pump"]):
        return "FUEL", 0.90, {"rule":"upi+fuelhint"}
    return None
