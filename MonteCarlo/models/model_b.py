"""
models/model_b.py
-----------------
Hierarchical 2-component Normal-mixture model (Model B).

Model specification
-------------------
Likelihood:
    z_it | π_i    ~ Bernoulli(π_i)          [regime: 0=calm, 1=stress]
    r_it | z=k   ~ Normal(μ_ik, σ_ik²)

Marginal (integrating out z):
    p(r_it) = π_i·N(μ_i1,σ_i1²) + (1−π_i)·N(μ_i0,σ_i0²)

Stock-level priors (k ∈ {0, 1}):
    μ_ik ~ Normal(μ_0k, τ_k²)
    σ_ik ~ LogNormal(log(ν_k), ξ_k²)
    π_i  ~ Beta(α_π, β_π)

Hyperpriors (per component):
    μ_0k ~ Normal(0, 0.005²)
    τ_k  ~ HalfCauchy(0.002)
    ν_k  ~ LogNormal(-4, 1)
    ξ_k  ~ HalfCauchy(1.0)

Mixing hyperpriors:
    α_π ~ Gamma(2, 0.1)
    β_π ~ Gamma(2, 0.1)

Identifiability constraint:
    σ_i0 < σ_i1  for all i  (component 0 = calm/low-vol, component 1 = stress/high-vol)
"""

import logging
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared log-density helpers  (copy from model_a to keep modules self-contained)
# ---------------------------------------------------------------------------

def _log_normal_pdf(x, mu, sigma):
    return -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * ((x - mu) / sigma) ** 2


def _log_half_cauchy(x, scale):
    return np.log(2.0 / np.pi) - np.log(scale) - np.log(1.0 + (x / scale) ** 2)


def _log_lognormal(x, log_mu, sigma):
    return (
        -np.log(x)
        - 0.5 * np.log(2 * np.pi)
        - np.log(sigma)
        - 0.5 * ((np.log(x) - log_mu) / sigma) ** 2
    )


def _log_gamma_pdf(x, alpha, beta):
    from scipy.special import gammaln
    return alpha * np.log(beta) - gammaln(alpha) + (alpha - 1) * np.log(x) - beta * x


# ---------------------------------------------------------------------------
# Prior constants
# ---------------------------------------------------------------------------
PRIOR = dict(
    mu_0_scale  = 0.005,
    tau_cauchy  = 0.002,
    nu_k_mu     = -4.0,
    nu_k_sigma  = 1.0,
    xi_cauchy   = 1.0,
    alpha_gamma_a = 2.0,
    alpha_gamma_b = 0.1,
    beta_gamma_a  = 2.0,
    beta_gamma_b  = 0.1,
)


# ---------------------------------------------------------------------------
# Model B class
# ---------------------------------------------------------------------------

class ModelB:
    """Hierarchical 2-component Normal-mixture model with regime-switching.

    Components:
        k=0 : calm regime  (lower volatility)
        k=1 : stress regime (higher volatility)
    """

    name = "B"

    def __init__(self):
        self.prior = PRIOR

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, returns: np.ndarray, rng: np.random.Generator, chain_idx: int = 0) -> dict:
        """Initialise state via K-means on |returns| for calm/stress split."""
        N, T  = returns.shape
        perturb = 0.05 * (chain_idx + 1)

        # K-means on absolute returns to separate calm / stress days
        abs_r = np.abs(returns).T  # (T, N)
        avg_abs = abs_r.mean(axis=1, keepdims=True)   # (T, 1)
        try:
            km = KMeans(n_clusters=2, random_state=42 + chain_idx, n_init=5)
            labels = km.fit_predict(avg_abs)
        except Exception:
            labels = (avg_abs.ravel() > avg_abs.mean()).astype(int)

        # Ensure label 0 = calm (lower vol), label 1 = stress (higher vol)
        vol_0 = returns[:, labels == 0].std()
        vol_1 = returns[:, labels == 1].std()
        if vol_0 > vol_1:
            labels = 1 - labels   # swap

        z_it = labels[None, :].repeat(N, axis=0).astype(bool)  # (N, T)

        mu_ik    = np.zeros((N, 2))
        sigma_ik = np.zeros((N, 2))
        pi_i     = np.zeros(N)

        for i in range(N):
            for k in range(2):
                mask = (z_it[i] == bool(k))
                obs  = returns[i, mask]
                if obs.size > 1:
                    mu_ik[i, k]    = obs.mean()
                    sigma_ik[i, k] = obs.std().clip(1e-4)
                else:
                    mu_ik[i, k]    = 0.0
                    sigma_ik[i, k] = returns[i].std().clip(1e-4) * (1 + k)

            # Enforce ordering constraint σ_i0 < σ_i1
            if sigma_ik[i, 0] >= sigma_ik[i, 1]:
                sigma_ik[i, 0] *= 0.8
                sigma_ik[i, 1] *= 1.2

            pi_i[i] = z_it[i].mean().clip(0.05, 0.95)

        # Add per-chain perturbation
        mu_ik    += perturb * rng.standard_normal((N, 2)) * 0.001
        sigma_ik *= np.exp(perturb * rng.standard_normal((N, 2)) * 0.05)

        # Hyper-parameters
        mu_0k   = mu_ik.mean(axis=0)
        tau_k   = mu_ik.std(axis=0).clip(1e-5)
        nu_k    = sigma_ik.mean(axis=0).clip(1e-5)
        xi_k    = np.array([1.0, 1.0])
        alpha_pi = 2.0
        beta_pi  = 2.0

        state = dict(
            mu_ik    = mu_ik,
            sigma_ik = sigma_ik,
            pi_i     = pi_i,
            z_it     = z_it,
            mu_0k    = mu_0k,
            tau_k    = tau_k,
            nu_k     = nu_k,
            xi_k     = xi_k,
            alpha_pi = alpha_pi,
            beta_pi  = beta_pi,
        )
        logger.debug(
            "Model B chain %d initialised: mean π=%.2f, σ_calm=%.4f, σ_stress=%.4f",
            chain_idx, pi_i.mean(), sigma_ik[:, 0].mean(), sigma_ik[:, 1].mean()
        )
        return state

    # ------------------------------------------------------------------
    # Gibbs: z_it  (regime indicators)
    # ------------------------------------------------------------------

    def update_z(self, state: dict, returns: np.ndarray, rng: np.random.Generator) -> dict:
        """z_it | rest ~ Bernoulli(softmax([log_p0, log_p1]))."""
        mu_ik    = state["mu_ik"]
        sigma_ik = state["sigma_ik"]
        pi_i     = state["pi_i"]

        log_p0 = (
            np.log(1.0 - pi_i[:, None])
            + _log_normal_pdf(returns, mu_ik[:, 0:1], sigma_ik[:, 0:1])
        )  # (N, T)
        log_p1 = (
            np.log(pi_i[:, None])
            + _log_normal_pdf(returns, mu_ik[:, 1:2], sigma_ik[:, 1:2])
        )  # (N, T)

        # Numerically stable normalisation
        log_Z    = np.logaddexp(log_p0, log_p1)
        prob_1   = np.exp(log_p1 - log_Z)

        state["z_it"] = rng.random(returns.shape) < prob_1   # (N, T) bool
        return state

    # ------------------------------------------------------------------
    # Gibbs: μ_ik  (conjugate Normal per component)
    # ------------------------------------------------------------------

    def update_mu_ik(self, state: dict, returns: np.ndarray, rng: np.random.Generator) -> dict:
        """μ_ik | rest ~ Normal(m_ik, V_ik)."""
        z_it     = state["z_it"]
        sigma_ik = state["sigma_ik"]
        mu_0k    = state["mu_0k"]
        tau_k    = state["tau_k"]
        N        = returns.shape[0]

        for k in range(2):
            mask   = (z_it == bool(k))                    # (N, T)
            T_k    = mask.sum(axis=1).astype(float)       # (N,)
            sum_r  = (returns * mask).sum(axis=1)          # (N,)

            prec_lik   = T_k / sigma_ik[:, k]**2
            prec_prior = 1.0 / tau_k[k]**2

            V_ik = 1.0 / (prec_lik + prec_prior)
            m_ik = V_ik * (prec_prior * mu_0k[k] + sum_r / sigma_ik[:, k]**2)

            state["mu_ik"][:, k] = rng.normal(m_ik, np.sqrt(V_ik))

        return state

    # ------------------------------------------------------------------
    # Metropolis: σ_ik  (log-scale, with ordering constraint)
    # ------------------------------------------------------------------

    def update_sigma_ik(
        self,
        state: dict,
        returns: np.ndarray,
        step_sizes: np.ndarray,      # shape (2,)
        rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray]:
        """Metropolis for σ_ik with σ_i0 < σ_i1 hard constraint."""
        z_it     = state["z_it"]
        mu_ik    = state["mu_ik"]
        nu_k     = state["nu_k"]
        xi_k     = state["xi_k"]
        sigma_ik = state["sigma_ik"].copy()
        N        = returns.shape[0]

        accepts = np.zeros((N, 2))

        for k in range(2):
            mask = (z_it == bool(k))                      # (N, T) bool
            sigma_curr     = sigma_ik[:, k]
            log_sigma_curr = np.log(sigma_curr)
            log_sigma_prop = log_sigma_curr + step_sizes[k] * rng.standard_normal(N)
            sigma_prop     = np.exp(log_sigma_prop)

            # Ordering constraint: σ_i0 < σ_i1
            if k == 0:
                valid = sigma_prop < sigma_ik[:, 1]
            else:
                valid = sigma_prop > sigma_ik[:, 0]

            def log_lik(s, k_=k, mask_=mask):
                # Sum Normal log-density over component-k observations
                resid = returns - mu_ik[:, k_:k_+1]       # (N, T)
                return np.sum(
                    mask_ * (_log_normal_pdf(returns, mu_ik[:, k_:k_+1], s[:, None])),
                    axis=1,
                )  # (N,)

            def log_prior(s, k_=k):
                return _log_lognormal(s, np.log(nu_k[k_]), xi_k[k_])

            log_accept = (
                log_lik(sigma_prop) - log_lik(sigma_curr)
                + log_prior(sigma_prop) - log_prior(sigma_curr)
                + log_sigma_prop - log_sigma_curr
            )
            accept = valid & (np.log(rng.random(N)) < log_accept)
            sigma_ik[:, k] = np.where(accept, sigma_prop, sigma_curr)
            accepts[:, k]  = accept.astype(float)

        state["sigma_ik"] = sigma_ik
        return state, accepts

    # ------------------------------------------------------------------
    # Gibbs: π_i  (Beta conjugate)
    # ------------------------------------------------------------------

    def update_pi(self, state: dict, rng: np.random.Generator) -> dict:
        """π_i | rest ~ Beta(α_π + n_i1, β_π + n_i0)."""
        z_it     = state["z_it"]
        alpha_pi = state["alpha_pi"]
        beta_pi  = state["beta_pi"]

        n_i1 = z_it.sum(axis=1)            # count of stress days per stock (N,)
        n_i0 = (~z_it).sum(axis=1)         # count of calm days

        state["pi_i"] = rng.beta(alpha_pi + n_i1, beta_pi + n_i0).clip(1e-4, 1 - 1e-4)
        return state

    # ------------------------------------------------------------------
    # Gibbs: μ_0k  (conjugate Normal hyperprior on μ_ik)
    # ------------------------------------------------------------------

    def update_mu_0k(self, state: dict, rng: np.random.Generator) -> dict:
        """μ_0k | rest ~ Normal(m_0k, V_0k)."""
        mu_ik = state["mu_ik"]
        tau_k = state["tau_k"]
        N     = mu_ik.shape[0]

        for k in range(2):
            prior_prec = 1.0 / PRIOR["mu_0_scale"]**2
            lik_prec   = N / tau_k[k]**2
            V_0k = 1.0 / (prior_prec + lik_prec)
            m_0k = V_0k * lik_prec * mu_ik[:, k].mean()
            state["mu_0k"][k] = rng.normal(m_0k, np.sqrt(V_0k))

        return state

    # ------------------------------------------------------------------
    # Metropolis: τ_k  (HalfCauchy, log-scale)
    # ------------------------------------------------------------------

    def update_tau_k(
        self,
        state: dict,
        step_sizes: np.ndarray,   # (2,)
        rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray]:
        accepts = np.zeros(2)

        for k in range(2):
            mu_ik  = state["mu_ik"][:, k]
            mu_0k  = state["mu_0k"][k]
            tau_c  = state["tau_k"][k]

            log_t_curr = np.log(tau_c)
            log_t_prop = log_t_curr + step_sizes[k] * rng.standard_normal()
            tau_prop   = np.exp(log_t_prop)

            log_accept = (
                np.sum(_log_normal_pdf(mu_ik, mu_0k, tau_prop))
                - np.sum(_log_normal_pdf(mu_ik, mu_0k, tau_c))
                + _log_half_cauchy(tau_prop, PRIOR["tau_cauchy"])
                - _log_half_cauchy(tau_c,    PRIOR["tau_cauchy"])
                + log_t_prop - log_t_curr
            )
            accept = rng.random() < np.exp(min(0.0, log_accept))
            if accept:
                state["tau_k"][k] = tau_prop
            accepts[k] = float(accept)

        return state, accepts

    # ------------------------------------------------------------------
    # Metropolis: ν_k  (log-scale)
    # ------------------------------------------------------------------

    def update_nu_k(
        self,
        state: dict,
        step_sizes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray]:
        accepts = np.zeros(2)

        for k in range(2):
            sigma_ik = state["sigma_ik"][:, k]
            xi_k     = state["xi_k"][k]
            nu_c     = state["nu_k"][k]

            log_n_curr = np.log(nu_c)
            log_n_prop = log_n_curr + step_sizes[k] * rng.standard_normal()
            nu_prop    = np.exp(log_n_prop)

            log_accept = (
                np.sum(_log_lognormal(sigma_ik, np.log(nu_prop), xi_k))
                - np.sum(_log_lognormal(sigma_ik, np.log(nu_c),  xi_k))
                + _log_lognormal(nu_prop, PRIOR["nu_k_mu"], PRIOR["nu_k_sigma"])
                - _log_lognormal(nu_c,    PRIOR["nu_k_mu"], PRIOR["nu_k_sigma"])
                + log_n_prop - log_n_curr
            )
            accept = rng.random() < np.exp(min(0.0, log_accept))
            if accept:
                state["nu_k"][k] = nu_prop
            accepts[k] = float(accept)

        return state, accepts

    # ------------------------------------------------------------------
    # Metropolis: ξ_k  (log-scale)
    # ------------------------------------------------------------------

    def update_xi_k(
        self,
        state: dict,
        step_sizes: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray]:
        accepts = np.zeros(2)

        for k in range(2):
            sigma_ik = state["sigma_ik"][:, k]
            nu_k     = state["nu_k"][k]
            xi_c     = state["xi_k"][k]

            log_x_curr = np.log(xi_c)
            log_x_prop = log_x_curr + step_sizes[k] * rng.standard_normal()
            xi_prop    = np.exp(log_x_prop)

            log_accept = (
                np.sum(_log_lognormal(sigma_ik, np.log(nu_k), xi_prop))
                - np.sum(_log_lognormal(sigma_ik, np.log(nu_k), xi_c))
                + _log_half_cauchy(xi_prop, PRIOR["xi_cauchy"])
                - _log_half_cauchy(xi_c,    PRIOR["xi_cauchy"])
                + log_x_prop - log_x_curr
            )
            accept = rng.random() < np.exp(min(0.0, log_accept))
            if accept:
                state["xi_k"][k] = xi_prop
            accepts[k] = float(accept)

        return state, accepts

    # ------------------------------------------------------------------
    # Metropolis: α_π and β_π  (Beta mixing hyperpriors)
    # ------------------------------------------------------------------

    def update_alpha_beta_pi(
        self,
        state: dict,
        step_sizes: np.ndarray,   # (2,) for [alpha, beta]
        rng: np.random.Generator,
    ) -> tuple[dict, np.ndarray]:
        """Metropolis for α_π and β_π."""
        from scipy.special import betaln

        pi_i  = state["pi_i"]
        alpha = state["alpha_pi"]
        beta  = state["beta_pi"]
        N     = len(pi_i)
        accepts = np.zeros(2)

        def log_lik(a, b):
            # product of Beta(pi_i; a, b) over all stocks
            return N * (-betaln(a, b)) + (a - 1) * np.log(pi_i).sum() + (b - 1) * np.log(1 - pi_i).sum()

        for idx, param in enumerate(["alpha_pi", "beta_pi"]):
            curr = state[param]
            log_c = np.log(curr)
            log_p = log_c + step_sizes[idx] * rng.standard_normal()
            prop  = np.exp(log_p)

            a_c = alpha if param == "alpha_pi" else alpha
            b_c = beta  if param == "beta_pi"  else beta
            a_p = prop  if param == "alpha_pi" else alpha
            b_p = prop  if param == "beta_pi"  else beta

            if param == "alpha_pi":
                a_p, b_p = prop, beta
            else:
                a_p, b_p = alpha, prop

            log_accept = (
                log_lik(a_p, b_p) - log_lik(a_c, b_c)
                + _log_gamma_pdf(prop, PRIOR[f"{param[:-3]}_gamma_a"], PRIOR[f"{param[:-3]}_gamma_b"])
                - _log_gamma_pdf(curr, PRIOR[f"{param[:-3]}_gamma_a"], PRIOR[f"{param[:-3]}_gamma_b"])
                + log_p - log_c
            )
            accept = rng.random() < np.exp(min(0.0, log_accept))
            if accept:
                state[param] = prop
                if param == "alpha_pi":
                    alpha = prop
                else:
                    beta = prop
            accepts[idx] = float(accept)

        return state, accepts

    # ------------------------------------------------------------------
    # WAIC: pointwise log-likelihood (marginalising over z)
    # ------------------------------------------------------------------

    def log_lik_pointwise(self, state: dict, returns: np.ndarray) -> np.ndarray:
        """Compute log p(r_it | θ) for every observation using log-sum-exp.

        Returns
        -------
        np.ndarray, shape (N, T)
        """
        mu_ik    = state["mu_ik"]
        sigma_ik = state["sigma_ik"]
        pi_i     = state["pi_i"]

        log_p0 = (
            np.log(1.0 - pi_i[:, None])
            + _log_normal_pdf(returns, mu_ik[:, 0:1], sigma_ik[:, 0:1])
        )
        log_p1 = (
            np.log(pi_i[:, None])
            + _log_normal_pdf(returns, mu_ik[:, 1:2], sigma_ik[:, 1:2])
        )
        return np.logaddexp(log_p0, log_p1)   # (N, T)

    # ------------------------------------------------------------------
    # Utility: flatten state for storage
    # ------------------------------------------------------------------

    def state_to_vector(self, state: dict) -> np.ndarray:
        return np.concatenate([
            state["mu_ik"].ravel(),           # (2N,)
            state["sigma_ik"].ravel(),        # (2N,)
            state["pi_i"],                    # (N,)
            state["mu_0k"],                   # (2,)
            state["tau_k"],                   # (2,)
            state["nu_k"],                    # (2,)
            state["xi_k"],                    # (2,)
            [state["alpha_pi"], state["beta_pi"]],
        ])

    def param_names(self, N: int) -> list[str]:
        names  = [f"mu_{i}_{k}" for k in range(2) for i in range(N)]
        names += [f"sigma_{i}_{k}" for k in range(2) for i in range(N)]
        names += [f"pi_{i}" for i in range(N)]
        names += ["mu_01", "mu_02", "tau_1", "tau_2", "nu_1", "nu_2", "xi_1", "xi_2"]
        names += ["alpha_pi", "beta_pi"]
        return names
