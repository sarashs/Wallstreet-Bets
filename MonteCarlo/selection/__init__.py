"""Model selection via WAIC (Widely Applicable Information Criterion)."""
from .waic import compute_waic, select_model

__all__ = ["compute_waic", "select_model"]
