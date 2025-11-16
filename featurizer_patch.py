"""
Compatibility shim for unpickling existing txcat model artifacts.

The trained pipeline was saved with references to a top-level module
named `featurizer_patch`. In this repository the actual implementation
lives under `src/featurizer_patch.py`, so we re-export it here to make
joblib/pickle able to resolve those references.
"""

from src.featurizer_patch import *  # noqa: F401,F403

