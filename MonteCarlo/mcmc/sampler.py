"""
mcmc/sampler.py
---------------
Gibbs sampler with Metropolis-within-Gibbs steps for both Model A and Model B.

Chain configuration (defaults per spec):
    n_chains   = 4
    warmup     = 2000   (burn-in + adaptation; spec says 5000 but user-configurable)
    n_samples  = 5000   (post-warmup draws per chain; spec says 10000)
    thinning   = 2      (store every 2nd sample)

Adaptation:
    Every `adapt_interval` iterations during warmup, acceptance rates are checked
    and step sizes are scaled by ±10 % to nudge toward the target (35 % for scalars,
    25 % for vector blocks).  Step sizes are frozen after warmup.

Output:
    GibbsSampler.run() returns a dict of numpy arrays:
        "samples_A" : shape (n_chains × n_stored, param_dim)   [for model A]
        "samples_B" : shape (n_chains × n_stored, param_dim)   [for model B]
        "acceptance" : dict of per-parameter acceptance rates
"""

import logging
import numpy as np
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adaptation targets
# ---------------------------------------------------------------------------
TARGET_ACCEPT_SCALAR = 0.35     # single-parameter Metropolis
TARGET_ACCEPT_VECTOR = 0.25     # block / vectorised Metropolis


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class GibbsSampler:
    """Run Gibbs + Metropolis-within-Gibbs for Model A or B.

    Parameters
    ----------
    model : ModelA | ModelB
        An initialised model object.
    returns : np.ndarray, shape (N, T)
        Cleaned log-returns matrix.
    n_chains : int
        Number of independent chains to run (for R-hat diagnostics).
    warmup : int
        Number of burn-in / adaptation iterations per chain.
    n_samples : int
        Number of post-warmup iterations per chain.
    thinning : int
        Store every k-th post-warmup sample.
    seed : int
        Base random seed (each chain gets seed + chain_idx).
    progress_callback : callable, optional
        Called with a progress-fraction (0.0 → 1.0) during sampling.
    """

    def __init__(
        self,
        model,
        returns: np.ndarray,
        n_chains: int = 4,
        warmup: int = 2000,
        n_samples: int = 5000,
        thinning: int = 2,
        seed: int = 42,
        progress_callback: Optional[Callable] = None,
    ):
        self.model    = model
        self.returns  = returns
        self.N, self.T = returns.shape
        self.n_chains = n_chains
        self.warmup   = warmup
        self.n_samples = n_samples
        self.thinning  = thinning
        self.seed      = seed
        self.progress_callback = progress_callback

        # n_stored = samples stored per chain (post-warmup, after thinning)
        self.n_stored = n_samples // thinning

        logger.info(
            "Sampler created: model=%s, N=%d, T=%d, chains=%d, warmup=%d, samples=%d, thin=%d",
            model.name, self.N, self.T, n_chains, warmup, n_samples, thinning,
        )

    # ------------------------------------------------------------------
    # Public: run all chains
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Run all chains and return combined posterior samples.

        Returns
        -------
        dict with:
            "samples"     : np.ndarray, shape (n_chains * n_stored, param_dim)
            "param_names" : list[str]
            "acceptance"  : dict  — mean acceptance rate per parameter group
            "chain_samples": list of per-chain sample arrays (needed for R-hat)
        """
        chain_samples = []
        chain_accepts = []
        total_iters   = self.n_chains * (self.warmup + self.n_samples)
        done_iters    = 0

        for chain_idx in range(self.n_chains):
            logger.info("Starting chain %d / %d", chain_idx + 1, self.n_chains)
            samples, accepts = self._run_chain(
                chain_idx,
                self.seed + chain_idx,
                done_iters,
                total_iters,
            )
            chain_samples.append(samples)
            chain_accepts.append(accepts)
            done_iters += self.warmup + self.n_samples

        combined = np.concatenate(chain_samples, axis=0)   # (n_chains * n_stored, D)
        mean_accepts = {
            k: float(np.mean([ca[k] for ca in chain_accepts]))
            for k in chain_accepts[0]
        }

        logger.info(
            "Sampling complete: %d total draws. Mean accepts: %s",
            combined.shape[0],
            {k: f"{v:.2%}" for k, v in mean_accepts.items()},
        )

        return {
            "samples":       combined,
            "param_names":   self.model.param_names(self.N),
            "acceptance":    mean_accepts,
            "chain_samples": chain_samples,   # list of (n_stored, D) per chain
        }

    # ------------------------------------------------------------------
    # Single chain
    # ------------------------------------------------------------------

    def _run_chain(
        self,
        chain_idx: int,
        seed: int,
        start_progress: int,
        total_progress: int,
    ) -> tuple[np.ndarray, dict]:
        rng    = np.random.default_rng(seed)
        state  = self.model.initialize(self.returns, rng, chain_idx)
        steps  = self._initial_step_sizes()
        counts = self._zero_counts()

        # Pre-allocate storage
        param_dim = len(self.model.state_to_vector(state))
        storage   = np.zeros((self.n_stored, param_dim))
        stored    = 0

        total_chain = self.warmup + self.n_samples

        # ---- Warmup (with adaptation) ----
        for it in range(self.warmup):
            state, accept = self._step(state, rng, steps)
            self._accumulate(counts, accept)

            if (it + 1) % 100 == 0:
                self._adapt_steps(steps, counts)
                self._reset_counts(counts)

            self._maybe_callback(start_progress + it, total_progress)

        # Freeze step sizes
        def _fmt_step(v):
            if isinstance(v, np.ndarray):
                return f"[{v.mean():.4f}]"
            return f"{float(v):.4f}"
        logger.debug("Chain %d warmup done. Steps: %s", chain_idx, {k: _fmt_step(v) for k, v in steps.items()})

        # ---- Sampling ----
        sample_counts = self._zero_counts()

        for it in range(self.n_samples):
            state, accept = self._step(state, rng, steps)
            self._accumulate(sample_counts, accept)

            if it % self.thinning == 0 and stored < self.n_stored:
                storage[stored] = self.model.state_to_vector(state)
                stored += 1

            self._maybe_callback(start_progress + self.warmup + it, total_progress)

        # Acceptance rates during sampling
        total_s = max(self.n_samples, 1)
        accepts = {k: float(v) / total_s for k, v in sample_counts.items()}

        logger.info(
            "Chain %d complete: stored=%d samples. Sampling accepts: %s",
            chain_idx, stored, {k: f"{v:.2%}" for k, v in accepts.items()},
        )
        return storage[:stored], accepts

    # ------------------------------------------------------------------
    # One MCMC iteration (dispatches to model-specific logic)
    # ------------------------------------------------------------------

    def _step(self, state: dict, rng: np.random.Generator, steps: dict) -> tuple[dict, dict]:
        accepts = {}

        if self.model.name == "A":
            # 1. λ_it  (Gibbs)
            state = self.model.update_lambda(state, self.returns, rng)

            # 2. μ_i   (Gibbs)
            state = self.model.update_mu_i(state, self.returns, rng)

            # 3. σ_i   (Metropolis, vectorised)
            state, acc = self.model.update_sigma_i(state, self.returns, steps["sigma_i"], rng)
            accepts["sigma_i"] = acc.mean()

            # 4. ν     (Metropolis, scalar)
            state, acc = self.model.update_nu(state, steps["nu"], rng)
            accepts["nu"] = acc

            # 5. μ_0   (Gibbs)
            state = self.model.update_mu_0(state, rng)

            # 6. τ     (Metropolis, scalar)
            state, acc = self.model.update_tau(state, steps["tau"], rng)
            accepts["tau"] = acc

            # 7. ν_s   (Metropolis, scalar)
            state, acc = self.model.update_nu_s(state, steps["nu_s"], rng)
            accepts["nu_s"] = acc

            # 8. ξ     (Metropolis, scalar)
            state, acc = self.model.update_xi(state, steps["xi"], rng)
            accepts["xi"] = acc

        elif self.model.name == "B":
            # 1. z_it  (Gibbs, categorical)
            state = self.model.update_z(state, self.returns, rng)

            # 2. μ_ik  (Gibbs, conjugate, per component)
            state = self.model.update_mu_ik(state, self.returns, rng)

            # 3. σ_ik  (Metropolis, vectorised per component)
            state, acc = self.model.update_sigma_ik(state, self.returns, steps["sigma_ik"], rng)
            accepts["sigma_ik"] = acc.mean()

            # 4. π_i   (Gibbs, Beta conjugate)
            state = self.model.update_pi(state, rng)

            # 5. μ_0k  (Gibbs, per component)
            state = self.model.update_mu_0k(state, rng)

            # 6. τ_k   (Metropolis, per component)
            state, acc = self.model.update_tau_k(state, steps["tau_k"], rng)
            accepts["tau_k"] = acc.mean()

            # 7. ν_k   (Metropolis, per component)
            state, acc = self.model.update_nu_k(state, steps["nu_k"], rng)
            accepts["nu_k"] = acc.mean()

            # 8. ξ_k   (Metropolis, per component)
            state, acc = self.model.update_xi_k(state, steps["xi_k"], rng)
            accepts["xi_k"] = acc.mean()

            # 9. α_π, β_π  (Metropolis)
            state, acc = self.model.update_alpha_beta_pi(state, steps["alpha_beta_pi"], rng)
            accepts["alpha_beta_pi"] = acc.mean()

        return state, accepts

    # ------------------------------------------------------------------
    # Step-size management
    # ------------------------------------------------------------------

    def _initial_step_sizes(self) -> dict:
        if self.model.name == "A":
            return {
                "sigma_i" : np.full(self.N, 0.1),
                "nu"      : 0.2,
                "tau"     : 0.3,
                "nu_s"    : 0.2,
                "xi"      : 0.3,
            }
        else:  # B
            return {
                "sigma_ik"      : np.array([0.1, 0.1]),
                "tau_k"         : np.array([0.3, 0.3]),
                "nu_k"          : np.array([0.2, 0.2]),
                "xi_k"          : np.array([0.3, 0.3]),
                "alpha_beta_pi" : np.array([0.3, 0.3]),
            }

    def _zero_counts(self) -> dict:
        if self.model.name == "A":
            return {"sigma_i": 0.0, "nu": 0.0, "tau": 0.0, "nu_s": 0.0, "xi": 0.0}
        else:
            return {"sigma_ik": 0.0, "tau_k": 0.0, "nu_k": 0.0, "xi_k": 0.0, "alpha_beta_pi": 0.0}

    def _accumulate(self, counts: dict, accept: dict):
        for k, v in accept.items():
            counts[k] = counts.get(k, 0.0) + float(v)

    def _reset_counts(self, counts: dict):
        for k in counts:
            counts[k] = 0.0

    def _adapt_steps(self, steps: dict, counts: dict, interval: int = 100):
        """Scale step sizes ±10 % based on empirical acceptance rates."""
        for key, rate in counts.items():
            observed = rate / interval
            target   = TARGET_ACCEPT_SCALAR

            if observed > target:
                factor = 1.1
            elif observed < target * 0.5:
                factor = 0.8
            elif observed < target:
                factor = 0.9
            else:
                factor = 1.0

            if isinstance(steps[key], np.ndarray):
                steps[key] = np.clip(steps[key] * factor, 1e-4, 5.0)
            else:
                steps[key] = float(np.clip(steps[key] * factor, 1e-4, 5.0))

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def _maybe_callback(self, done: int, total: int):
        if self.progress_callback is not None and total > 0:
            self.progress_callback(done / total)
