# lexicon_patch.py
# Brand lexicon + probability bumps for the COARSE router and fine overrides.
import re

# Coarse keys your v5 uses
COARSE_KEYS = ['FOOD','HEALTH','SHOPPING','TRAVEL','UTILITIES','OTHER_MISC']

# Brand patterns -> semantic buckets we map to a COARSE key
LEX = {
  'UTILITIES': [
     r'\bjio\b', r'\bvi\b', r'\bvodafone\b', r'\bairtel\b', r'\bbsnl\b',
     r'\btelekom\b', r'\bo2\b', r'\b(prepaid|postpaid|recharge|bill)\b'
  ],
  'SHOPPING': [
     r'\bajio\b', r'\bnykaa\b', r'\bmyntra\b', r'\bflipkart\b', r'\bamazon\b',
     r'\breliance\s*digital\b', r'\bcroma\b', r'\bvijay\s*sales\b'
  ],
  'FOOD': [
     r'\bzomato\b', r'\bswiggy\b', r'\bfaasos\b', r'\bkfc\b', r'\bmc(d|donald)\b',
     r'\bdominos\b', r'\bstarbucks\b', r'\bbarista\b', r'\bcafe\b', r'\brestaurant\b'
  ],
  'FUEL': [
     r'\bpetro\b', r'\bhpcl\b', r'\bbpcl\b', r'\biocl\b', r'\bindian\s*oil\b', r'\bessar\b'
  ],
  'TRAVEL': [
     r'\birctc\b', r'\bindigo\b', r'\bair\s*india\b', r'\bspicejet\b', r'\blufthansa\b',
     r'\buber\b', r'\bola\b', r'\brapido\b', r'\bolacabs\b'
  ],
  'HEALTH': [
     r'\b1mg\b', r'\bpharmeasy\b', r'\bapollo\b', r'\brandd\b', r'\bmedplus\b'
  ],
  'GROCERIES': [
     r'\bblinkit\b', r'\bzepto\b', r'\bbigbasket\b', r'\bdmart\b', r'\blidl\b', r'\baldi\b',
     r'\breliance\s*(smart|fresh)\b', r'\bmore\b'
  ],
}

# Where to bump a matched bucket at the COARSE layer
BUCKET_TO_COARSE = {
  'FOOD':'FOOD',
  'FUEL':'TRAVEL',
  'HEALTH':'HEALTH',
  'SHOPPING':'SHOPPING',
  'TRAVEL':'TRAVEL',
  'UTILITIES':'UTILITIES',
  'GROCERIES':'SHOPPING',   # groceries sits under shopping in coarse
}

def coarse_boost(coarse_proba, raw_text):
    """Small, safe probability bumps before argmax coarse routing."""
    if coarse_proba is None: 
        return None
    txt = (raw_text or "")
    bumps = dict.fromkeys(COARSE_KEYS, 0.0)
    for bucket, pats in LEX.items():
        for p in pats:
            if re.search(p, txt, flags=re.I):
                ck = BUCKET_TO_COARSE.get(bucket, 'OTHER_MISC')
                bumps[ck] = max(bumps[ck], 0.08)  # up to +8pp per bucket
    out = coarse_proba.copy()
    # apply in index order
    for i, key in enumerate(COARSE_KEYS):
        out[i] += bumps[key]
    s = out.sum()
    if s > 0:
        out /= s
    return out

def fine_hard_override(pred_label, raw_text):
    """Surgical fixes for worst confusions (applied once, post fine-model)."""
    t = (raw_text or "").lower()
    # Fuel family
    if re.search(r'reliance.*petro|hpcl|bpcl|iocl|indian\s*oil|essar', t): 
        return 'FUEL'
    # Telecom vs Ajio
    if re.search(r'\bjio\b', t) and not re.search(r'\bajio\b', t):
        return 'UTILITIES_TELECOM'
    if re.search(r'\bajio\b|\breliance\s*digital\b|\bcroma\b|\bvijay\s*sales\b|\bflipkart\b|\bamazon\b', t):
        return 'SHOPPING_ECOM'
    # Food brands
    if re.search(r'\bzomato\b|\bswiggy\b|\bfaasos\b|\bcafe\b|\brestaurant\b|\bkfc\b|mc(d|donald)', t):
        return 'DINING'
    # Groceries
    if re.search(r'blinkit|zepto|bigbasket|dmart|lidl|aldi|reliance\s*(smart|fresh)\b', t):
        return 'GROCERIES'
    # Mobility
    if re.search(r'\buber\b|\bola\b|\brapido\b', t):
        return 'MOBILITY'
    return pred_label
