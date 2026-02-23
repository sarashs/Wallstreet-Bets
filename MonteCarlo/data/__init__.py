"""Data ingestion and cleaning module."""
from .fetcher import fetch_returns, compute_spearman_correlation

__all__ = ["fetch_returns", "compute_spearman_correlation"]
