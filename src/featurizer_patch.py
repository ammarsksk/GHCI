# featurizer_patch.py
# Shared TF-IDF featurizer with noise scrubbing and a char-gram branch.
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

# extremely common bank artifacts & generic noise
NOISE_TOKENS = set("""
upi imps neft rtgs rrn ref txn txid pos auth otp balance avail avl stmt utr ifsc
ac a c xx xxxx xxxx0000 xxxxxx0000 masked ####
transfer trf iban swift bic online merchant generic ecom ecommerce e-commerce
wallet bank ib internet net banking pending success failed chargeback reversal
debit credit cr dr crdt drcr
""".split())

CITY_STOP = set("""
mumbai delhi noida gurgaon bengaluru bangalore hyderabad pune kolkata chennai
berlin magdeburg munich hamburg frankfurt stuttgart dresden leipzig
""".split())

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r'[_\-:/|.,]+', ' ', s)             # unify separators
    s = re.sub(r'\b\d{3,}\b', ' ', s)              # nuke long digit runs
    toks = re.findall(r'[a-z0-9]+', s)
    toks = [t for t in toks if t not in NOISE_TOKENS and t not in CITY_STOP]
    return ' '.join(toks)

def build_text_union():
    # word(1-2) + char_wb(3-5) — separates near-homographs (ajio vs jio, hpcl etc.)
    word = TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=300_000
    )
    char = TfidfVectorizer(
        preprocessor=normalize_text,
        analyzer='char_wb',
        ngram_range=(3, 5),
        min_df=2,
        max_features=200_000
    )
    return FeatureUnion([('w', word), ('c', char)])
