from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import sys
import importlib

import numpy as np
from joblib import load

from .lexicon_patch import coarse_boost, fine_hard_override


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "txcat_fastcpu_v7.joblib"

DEFAULT_LOW_CONF = 0.60


def _patch_sgd_log() -> None:
    """
    Guard against sklearn pickle version drift: ensure _sgd_fast.Log exists
    so joblib can unpickle SGDClassifier artifacts trained elsewhere.
    For inference we do not need the original Cython loss implementation,
    so a lightweight Python stand-in is sufficient.
    """
    try:
        import sklearn.linear_model._sgd_fast as _sgd_fast  # type: ignore

        class _DummyLog:
            def __init__(self, *args, **kwargs) -> None:
                # Accept any signature used by older pickles.
                pass

            def __call__(self, *args, **kwargs):
                # Loss value is irrelevant at inference time.
                return 0.0

        setattr(_sgd_fast, "Log", _DummyLog)
    except Exception:
        # If the patch fails, the loader will surface the real error later.
        pass


def _ensure_featurizer_module() -> None:
    """
    Ensure that the module name `featurizer_patch` is importable.
    Older pickles reference this top-level module; in this repo the
    implementation lives under src.featurizer_patch, so we alias it.
    """
    if "featurizer_patch" in sys.modules:
        return
    try:
        # If a top-level shim exists, this will succeed.
        importlib.import_module("featurizer_patch")
        return
    except ModuleNotFoundError:
        try:
            fp = importlib.import_module("src.featurizer_patch")
            sys.modules["featurizer_patch"] = fp
        except Exception:
            # Leave resolution to the normal import mechanism.
            pass


def _get_feature_names(vec) -> List[str]:
    """Best-effort feature name extraction, mirroring the trainer."""
    try:
        return list(vec.get_feature_names_out())
    except Exception:
        names: List[str] = []
        try:
            # Expect a FeatureUnion([('w', word_vec), ('c', char_vec)])
            w = vec.transformer_list[0][1]
            c = vec.transformer_list[1][1]
            names = list(getattr(w, "get_feature_names_out", lambda: [])()) + list(
                getattr(c, "get_feature_names_out", lambda: [])()
            )
        except Exception:
            pass
        return names


@lru_cache(maxsize=1)
def load_model() -> Dict[str, object]:
    """
    Load the txcat v7 model artifact once per process.
    """
    _patch_sgd_log()
    _ensure_featurizer_module()
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    obj = load(MODEL_PATH)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict model artifact, got {type(obj)}")
    return obj


def _reconstruct_fine_solo(
    fine_models: Dict[str, object],
    fine_thresholds: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """
    Rebuild the 'fine_solo' mapping used during training:
    buckets with exactly one fine label and no trained classifier.
    """
    fine_solo: Dict[str, str] = {}
    for coarse_key, thr_map in fine_thresholds.items():
        labels = list(thr_map.keys())
        if len(labels) == 1 and (coarse_key not in fine_models or fine_models[coarse_key] is None):
            fine_solo[coarse_key] = labels[0]
    return fine_solo


def predict_batch(texts: List[str], low_conf: float = DEFAULT_LOW_CONF) -> List[Dict[str, object]]:
    """
    Run hierarchical prediction on a batch of raw texts using the
    saved faststack model (currently v7). Mirrors the
    training-time hierarchy logic used in the trainer.
    """
    if not texts:
        return []

    mdl = load_model()
    vec = mdl["vectorizer"]
    coarse_lr = mdl["coarse_lr"]
    fine_models = mdl["fine_models"]
    fine_encoders = mdl["fine_encoders"]
    fine_thresholds = mdl["fine_thresholds"]
    coarse_thresholds = mdl["coarse_thresholds"]
    le_coarse = mdl["le_coarse"]

    fine_solo = _reconstruct_fine_solo(fine_models, fine_thresholds)

    # Vectorize
    X = vec.transform(texts)

    # Coarse routing
    coarse_proba = coarse_lr.predict_proba(X)
    coarse_classes_idx = getattr(coarse_lr, "classes_", None)
    if coarse_classes_idx is None:
        coarse_classes_idx = np.arange(coarse_proba.shape[1])
    coarse_classes_keys = list(le_coarse.inverse_transform(coarse_classes_idx))
    coarse_idx_to_key = {i: k for i, k in enumerate(coarse_classes_keys)}

    routed_coarse: List[str] = []
    coarse_conf: List[float] = []
    for i, row_p in enumerate(coarse_proba):
        boosted = coarse_boost(row_p.copy(), texts[i])
        top_idx = int(np.argmax(boosted))
        top_key = coarse_idx_to_key[top_idx]
        thr = coarse_thresholds.get(top_key, 0.2)
        if boosted[top_idx] >= thr:
            routed_coarse.append(top_key)
        else:
            routed_coarse.append("OTHER_MISC")
        coarse_conf.append(float(boosted[top_idx]))

    # Fine predictions per bucket
    feat_names = _get_feature_names(vec)
    bucket_to_indices: Dict[str, List[int]] = {}
    for i, ck in enumerate(routed_coarse):
        bucket_to_indices.setdefault(ck, []).append(i)

    results: List[Dict[str, object]] = [{} for _ in range(len(texts))]

    for ck, idxs in bucket_to_indices.items():
        if ck in fine_solo:
            solo_label = fine_solo[ck]
            for i in idxs:
                needs_review = coarse_conf[i] < low_conf
                results[i] = {
                    "text": texts[i],
                    "coarse_label": ck,
                    "coarse_confidence": coarse_conf[i],
                    "label": solo_label,
                    "confidence": 0.99,
                    "explanation": "",
                    "needs_review": needs_review,
                }
            continue

        if ck not in fine_models or fine_models[ck] is None:
            for i in idxs:
                needs_review = coarse_conf[i] < low_conf
                results[i] = {
                    "text": texts[i],
                    "coarse_label": ck,
                    "coarse_confidence": coarse_conf[i],
                    "label": "OTHER",
                    "confidence": 0.0,
                    "explanation": "",
                    "needs_review": needs_review,
                }
            continue

        model = fine_models[ck]
        enc = fine_encoders[ck]
        probas = model.predict_proba(X[idxs])

        model_class_idx = getattr(model, "classes_", None)
        if model_class_idx is None:
            model_class_idx = np.arange(probas.shape[1])
        labels_for_cols = list(enc.inverse_transform(model_class_idx))

        # For explanations, reuse the LR head when present
        nem = getattr(model, "named_estimators_", {})
        lr_part = nem.get("lr") if isinstance(nem, dict) else None

        for local_idx, row_i in enumerate(idxs):
            proba = np.asarray(probas[local_idx])
            j = int(np.argmax(proba))
            lbl = labels_for_cols[j]
            pmax = float(proba[j])

            thr_map = fine_thresholds.get(ck, {})
            thr = thr_map.get(lbl, 0.2)
            if pmax < thr and "OTHER" in labels_for_cols:
                other_idx = labels_for_cols.index("OTHER")
                lbl = "OTHER"
                pmax = float(proba[other_idx])

            lbl = fine_hard_override(lbl, texts[row_i])
            needs_review = pmax < low_conf

            expl = ""
            try:
                if lr_part is not None and needs_review:
                    xrow = X[row_i]
                    coef = lr_part.coef_[j]
                    idx = xrow.indices
                    data = xrow.data
                    pairs = []
                    for jj, val in zip(idx, data):
                        if jj < len(coef):
                            fname = feat_names[jj] if jj < len(feat_names) else f"f{jj}"
                            pairs.append((fname, float(coef[jj] * val)))
                    pairs.sort(key=lambda t: t[1], reverse=True)
                    expl = "; ".join(f"{a}:{b:.2f}" for a, b in pairs[:6])
            except Exception:
                expl = ""

            results[row_i] = {
                "text": texts[row_i],
                "coarse_label": ck,
                "coarse_confidence": coarse_conf[row_i],
                "label": lbl,
                "confidence": pmax,
                "explanation": expl,
                "needs_review": needs_review,
            }

    return results
