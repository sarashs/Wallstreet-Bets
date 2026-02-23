"""
models/model_a.py
-----------------
Hierarchical Student-t model (Model A).

Model specification
-------------------
Likelihood (via scale-mixture of Normals data augmentation):
    λ_it ~ Gamma(ν/2, ν/2)
    r_it | μ_i, σ_i, λ_it ~ Normal(μ_i, σ_i² / λ_it)

Stock-level priors:
    μ_i ~ Normal(μ_0, τ²)
    σ_i ~ LogNormal(log(ν_s), ξ²)

Hyperpriors:
    μ_0 ~ Normal(0, 0.005²)
    τ   ~ HalfCauchy(0.002)
    ν_s ~ LogNormal(-4, 1)
    ξ   ~ HalfCauchy(1.0)
    ν   ~ Gamma(2, 0.1) + 2   (shifted so ν > 2)

All log-density helpers return raw values (without Jacobian).  The Jacobian
for log-scale Metropolis proposals (+log θ' − log θ) is added by the sampler.
"""

import logging
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prior hyper-parameters (fixed constants from the spec)
# ---------------------------------------------------------------------------
PRIOR = dict(
    mu_0_scale   = 0.005,    # σ for μ_0 prior
    tau_cauchy   = 0.002,    # HalfCauchy scale for τ
    nu_s_mu      = -4.0,     # log-normal mean for ν_s (log-space)
    nu_s_sigma   = 1.0,      # log-normal std for ν_s (log-space)
    xi_cauchy    = 1.0,      # HalfCauchy scale for ξ
    nu_gamma_a   = 2.0,      # Gamma shape for ν (shifted)
    nu_gamma_b   = 0.1,      # Gamma rate for ν (shifted)
)


# ---------------------------------------------------------------------------
# Log-density helpers — all in *natural* parameter space
# ---------------------------------------------------------------------------

def _log_normal_pdf(x, mu, sigma):
    """Log-density of Normal(mu, sigma²)."""
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def _log_half_cauchy(x, scale):
    """Log-density of HalfCauchy(scale) for x > 0."""
    return np.log(2.0 / np.pi) - np.log(scale) - np.log(1.0 + (x / scale) ** 2)


def _log_lognormal(x, log_mu, sigma):
    """Log-density of LogNormal(log_mu, sigma²) for x > 0."""
    return (
        -np.log(x)
        - 0.5 * np.log(2 * np.pi)
        - np.log(sigma)
        - 0.5 * ((np.log(x) - log_mu) / sigma) ** 2
    )


def _log_gamma_pdf(x, alpha, beta):
    """Log-density of Gamma(alpha, beta) with rate parameterisation."""
    from scipy.special import gammaln
    return alpha * np.log(beta) - gammaln(alpha) + (alpha - 1) * np.log(x) - beta * x


# ---------------------------------------------------------------------------
# Model A class
# ---------------------------------------------------------------------------

class ModelA:
    """Hierarchical Student-t model.

    This class provides:
      - `initialize(returns, rng)` → initial state dict
      - `log_lik_pointwise(state, returns)` → (N, T) for WAIC
      - Methods for each Gibbs/MH update step (called by the sampler)
    """

    name = "A"

    def __init__(self):
        self.prior = PRIOR

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, returns: np.ndarray, rng: np.random.Generator, chain_idx: int = 0) -> dict:
        """Create a valid starting state for one chain.

        Strategy (per the spec):
        1. Estimate sample mean and variance of each stock.
        2. Initialise ν at 5.
        3. Add small per-chain perturbation to break symmetry.
        """
        N, T = returns.shape
        perturb = 0.05 * (chain_idx + 1)

        mu_i   = returns.mean(axis=1) * (1 + perturb * rng.standard_normal(N))
        sigma_i = returns.std(axis=1).clip(1e-4) * np.exp(perturb * rng.standard_normal(N))

        nu     = 5.0 + perturb
        mu_0   = mu_i.mean()
        tau    = max(mu_i.std(), 1e-5)
        nu_s   = sigma_i.mean().clip(1e-5)
        xi     = 1.0

        # Initialise λ_it from Gamma(ν/2, ν/2)
        lambda_it = rng.gamma(nu / 2, 2 / nu, size=(N, T))

        state = dict(
            mu_i     = mu_i,
            sigma_i  = sigma_i,
            nu       = nu,
            mu_0     = mu_0,
            tau      = tau,
            nu_s     = nu_s,
            xi       = xi,
            lambda_it= lambda_it,
        )
        logger.debug("Model A chain %d initialised: ν=%.2f, mean σ=%.4f", chain_idx, nu, sigma_i.mean())
        return state

    # ------------------------------------------------------------------
    # Gibbs update: λ_it (data augmentation weights)
    # ------------------------------------------------------------------

    def update_lambda(self, state: dict, returns: np.ndarray, rng: np.random.Generator) -> dict:
        """λ_it | rest ~ Gamma((ν+1)/2, (ν + (r-μ)²/σ²)/2)."""
        nu       = state["nu"]
        mu_i     = state["mu_i"]
        sigma_i  = state["sigma_i"]

        residuals = returns - mu_i[:, None]               # (N, T)
        shape = (nu + 1.0) / 2.0
        rate  = (nu + residuals**2 / sigma_i[:, None]**2) / 2.0

        state["lambda_it"] = rng.gamma(shape, 1.0 / rate)  # (N, T)
        return state

    # ------------------------------------------------------------------
    # Gibbs update: μ_i (conjugate Normal)
    # ------------------------------------------------------------------

    def update_mu_i(self, state: dict, returns: np.ndarray, rng: np.random.Generator) -> dict:
        """μ_i | rest ~ Normal(m_i, V_i)  [conjugate update via augmentation]."""
        lambda_it = state["lambda_it"]
        sigma_i   = state["sigma_i"]
        mu_0      = state["mu_0"]
        tau       = state["tau"]

        prec_lik   = np.sum(lambda_it / sigma_i[:, None]**2, axis=1)   # (N,)
        prec_prior = 1.0 / tau**2

        V_i = 1.0 / (prec_lik + prec_prior)
        m_i = V_i * (
            prec_prior * mu_0
            + np.sum(lambda_it * returns, axis=1) / sigma_i**2
        )

        state["mu_i"] = rng.normal(m_i, np.sqrt(V_i))
        return state

    # ------------------------------------------------------------------
    # Metropolis update: σ_i  (log-scale proposal)
    # ------------------------------------------------------------------

    def update_sigma_i(
        self,
        state: dict,
        returns: np.ndarray,
        step_size: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray]:
        """Metropolis step for σ_i with log-Normal proposal (vectorised over stocks)."""
        N          = returns.shape[0]
        lambda_it  = state["lambda_it"]
        mu_i       = state["mu_i"]
        nu_s       = state["nu_s"]
        xi         = state["xi"]
        sigma_curr = state["sigma_i"]

        log_sigma_curr = np.log(sigma_curr)
        log_sigma_prop = log_sigma_curr + step_size * rng.standard_normal(N)
        sigma_prop     = np.exp(log_sigma_prop)

        def log_lik(s):
            # Augmented normal: r | λ ~ Normal(μ, σ²/λ)
            return np.sum(
                -np.log(s[:, None]) - 0.5 * lambda_it * (returns - mu_i[:, None])**2 / s[:, None]**2,
                axis=1,
            )  # (N,)

        def log_prior(s):
            return _log_lognormal(s, np.log(nu_s), xi)

        log_accept = (
            log_lik(sigma_prop) - log_lik(sigma_curr)
            + log_prior(sigma_prop) - log_prior(sigma_curr)
            + log_sigma_prop - log_sigma_curr   # Jacobian for log-scale MH
        )
        log_accept = np.minimum(0.0, log_accept)

        accept = np.log(rng.random(N)) < log_accept
        state["sigma_i"] = np.where(accept, sigma_prop, sigma_curr)
        return state, accept.astype(float)

    # ------------------------------------------------------------------
    # Metropolis update: ν  (proposal on log(ν − 2) scale to ensure ν > 2)
    # ------------------------------------------------------------------

    def update_nu(
        self,
        state: dict,
        step_size: float,
        rng: np.random.Generator,
    ) -> tuple[dict, float]:
        """Metropolis for degrees-of-freedom ν.  Proposal on log(ν-2) scale."""
        lambda_it = state["lambda_it"]
        nu_curr   = state["nu"]

        log_nu_m2_curr = np.log(nu_curr - 2.0)
        log_nu_m2_prop = log_nu_m2_curr + step_size * rng.standard_normal()
        nu_prop        = np.exp(log_nu_m2_prop) + 2.0

        def log_lik_nu(nu):
            # p(λ_it | ν) = Gamma(ν/2, ν/2) — contribution of the latent weights
            return np.sum(
                _log_gamma_pdf(lambda_it, nu / 2.0, nu / 2.0)
            )

        def log_prior_nu(nu):
            # ν ~ Gamma(2, 0.1) + 2  →  (ν - 2) ~ Gamma(2, 0.1)
            return _log_gamma_pdf(nu - 2.0, PRIOR["nu_gamma_a"], PRIOR["nu_gamma_b"])

        log_accept = (
            log_lik_nu(nu_prop) - log_lik_nu(nu_curr)
            + log_prior_nu(nu_prop) - log_prior_nu(nu_curr)
            + log_nu_m2_prop - log_nu_m2_curr   # Jacobian
        )
        accept = rng.random() < np.exp(min(0.0, log_accept))
        if accept:
            state["nu"] = nu_prop
        return state, float(accept)

    # ------------------------------------------------------------------
    # Gibbs update: μ_0  (conjugate Normal hyperprior on μ_i)
    # ------------------------------------------------------------------

    def update_mu_0(self, state: dict, rng: np.random.Generator) -> dict:
        """μ_0 | rest ~ Normal(m_0, V_0)."""
        mu_i = state["mu_i"]
        tau  = state["tau"]
        N    = len(mu_i)

        prior_prec = 1.0 / PRIOR["mu_0_scale"]**2
        lik_prec   = N / tau**2
        V_0 = 1.0 / (prior_prec + lik_prec)
        m_0 = V_0 * lik_prec * mu_i.mean()

        state["mu_0"] = rng.normal(m_0, np.sqrt(V_0))
        return state

    # ------------------------------------------------------------------
    # Metropolis update: τ  (HalfCauchy prior, log-scale proposal)
    # ------------------------------------------------------------------

    def update_tau(
        self,
        state: dict,
        step_size: float,
        rng: np.random.Generator,
    ) -> tuple[dict, float]:
        """Metropolis for τ (cross-stock mean dispersion)."""
        mu_i   = state["mu_i"]
        mu_0   = state["mu_0"]
        tau_c  = state["tau"]
        N      = len(mu_i)

        log_tau_curr = np.log(tau_c)
        log_tau_prop = log_tau_curr + step_size * rng.standard_normal()
        tau_prop     = np.exp(log_tau_prop)

        def log_lik_tau(tau):
            return np.sum(_log_normal_pdf(mu_i, mu_0, tau))

        def log_prior_tau(tau):
            return _log_half_cauchy(tau, PRIOR["tau_cauchy"])

        log_accept = (
            log_lik_tau(tau_prop) - log_lik_tau(tau_c)
            + log_prior_tau(tau_prop) - log_prior_tau(tau_c)
            + log_tau_prop - log_tau_curr
        )
        accept = rng.random() < np.exp(min(0.0, log_accept))
        if accept:
            state["tau"] = tau_prop
        return state, float(accept)

    # ------------------------------------------------------------------
    # Metropolis update: ν_s  (population volatility scale)
    # ------------------------------------------------------------------

    def update_nu_s(
        self,
        state: dict,
        step_size: float,
        rng: np.random.Generator,
    ) -> tuple[dict, float]:
        """Metropolis for ν_s (population-level volatility center in log-space)."""
        sigma_i   = state["sigma_i"]
        xi        = state["xi"]
        nu_s_curr = state["nu_s"]

        log_nu_s_curr = np.log(nu_s_curr)
        log_nu_s_prop = log_nu_s_curr + step_size * rng.standard_normal()
        nu_s_prop     = np.exp(log_nu_s_prop)

        def log_lik(ns):
            return np.sum(_log_lognormal(sigma_i, np.log(ns), xi))

        def log_prior(ns):
            return _log_lognormal(ns, PRIOR["nu_s_mu"], PRIOR["nu_s_sigma"])

        log_accept = (
            log_lik(nu_s_prop) - log_lik(nu_s_curr)
            + log_prior(nu_s_prop) - log_prior(nu_s_curr)
            + log_nu_s_prop - log_nu_s_curr
        )
        accept = rng.random() < np.exp(min(0.0, log_accept))
        if accept:
            state["nu_s"] = nu_s_prop
        return state, float(accept)

    # ------------------------------------------------------------------
    # Metropolis update: ξ  (vol dispersion across stocks)
    # ------------------------------------------------------------------

    def update_xi(
        self,
        state: dict,
        step_size: float,
        rng: np.random.Generator,
    ) -> tuple[dict, float]:
        """Metropolis for ξ (dispersion of stock volatilities around ν_s)."""
        sigma_i  = state["sigma_i"]
        nu_s     = state["nu_s"]
        xi_curr  = state["xi"]

        log_xi_curr = np.log(xi_curr)
        log_xi_prop = log_xi_curr + step_size * rng.standard_normal()
        xi_prop     = np.exp(log_xi_prop)

        def log_lik(xi):
            return np.sum(_log_lognormal(sigma_i, np.log(nu_s), xi))

        def log_prior(xi):
            return _log_half_cauchy(xi, PRIOR["xi_cauchy"])

        log_accept = (
            log_lik(xi_prop) - log_lik(xi_curr)
            + log_prior(xi_prop) - log_prior(xi_curr)
            + log_xi_prop - log_xi_curr
        )
        accept = rng.random() < np.exp(min(0.0, log_accept))
        if accept:
            state["xi"] = xi_prop
        return state, float(accept)

    # ------------------------------------------------------------------
    # WAIC support: pointwise log-likelihood on HELD OUT returns
    # ------------------------------------------------------------------

    def log_lik_pointwise(self, state: dict, returns: np.ndarray) -> np.ndarray:
        """Compute log p(r_it | θ) for every observation.

        Returns
        -------
        np.ndarray, shape (N, T) — log-likelihoods under Student-t marginal.
        """
        from scipy.stats import t as student_t

        mu_i    = state["mu_i"]
        sigma_i = state["sigma_i"]
        nu      = state["nu"]

        log_ll = np.zeros(returns.shape)
        for i in range(returns.shape[0]):
            log_ll[i] = student_t.logpdf(
                returns[i], df=nu, loc=mu_i[i], scale=sigma_i[i]
            )
        return log_ll

    # ------------------------------------------------------------------
    # Utility: flatten state to a 1-D vector for storage
    # ------------------------------------------------------------------

    def state_to_vector(self, state: dict) -> np.ndarray:
        """Flatten the non-latent parameters to a 1-D float64 array."""
        return np.concatenate([
            state["mu_i"],
            state["sigma_i"],
            [state["nu"], state["mu_0"], state["tau"], state["nu_s"], state["xi"]],
        ])

    def param_names(self, N: int) -> list[str]:
        """Return parameter names matching state_to_vector order."""
        names  = [f"mu_{i}" for i in range(N)]
        names += [f"sigma_{i}" for i in range(N)]
        names += ["nu", "mu_0", "tau", "nu_s", "xi"]
        return names
