"""MCMC sampling module — Gibbs sampler with Metropolis-within-Gibbs steps."""
from .sampler import GibbsSampler
from .diagnostics import compute_rhat, compute_ess

__all__ = ["GibbsSampler", "compute_rhat", "compute_ess"]
