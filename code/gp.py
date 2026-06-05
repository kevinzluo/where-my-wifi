import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import numpy as np
from functools import partial
from statistics import NormalDist
from sklearn.exceptions import NotFittedError


def _coverage_level_label(level):
    return f"{int(round(100 * level))}"


def _as_2d_prediction_array(name, value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (posterior_samples, n_points).")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _flatten_cov_chains(cov_chains, n_samples):
    cov_chains = np.asarray(cov_chains, dtype=float)
    if cov_chains.ndim == 3:
        cov_chains = cov_chains.reshape(-1, cov_chains.shape[-1])
    if cov_chains.ndim != 2 or cov_chains.shape[1] != 2:
        raise ValueError("cov_chains must have shape (samples, 2) or (chains, samples, 2).")
    if cov_chains.shape[0] != n_samples:
        raise ValueError(
            "cov_chains must flatten to the same number of posterior samples as pred_means."
        )
    if not np.all(np.isfinite(cov_chains)):
        raise ValueError("cov_chains must contain only finite values.")
    return cov_chains


def posterior_predictive_interval_summary(
    y_true,
    pred_means,
    pred_vars,
    *,
    cov_chains=None,
    X_test=None,
    obs_count_test=None,
    levels=(0.5, 0.8, 0.9, 0.95),
    target="observed",
    random_state=305,
    return_intervals=False,
):
    """
    Summarize held-out coverage for GP posterior predictive intervals.

    pred_means and pred_vars should be the arrays returned by
    GaussianProcess.predict. They are interpreted as conditional latent
    posterior moments for each retained variance sample. The default target is
    the held-out aggregated observation, so indoor/outdoor observation noise is
    added from cov_chains and scaled by obs_count_test.
    """
    if target not in {"observed", "latent"}:
        raise ValueError("target must be 'observed' or 'latent'.")

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("y_true must contain only finite values.")

    pred_means = _as_2d_prediction_array("pred_means", pred_means)
    pred_vars = _as_2d_prediction_array("pred_vars", pred_vars)
    if pred_vars.shape != pred_means.shape:
        raise ValueError("pred_vars must have the same shape as pred_means.")

    n_samples, n_points = pred_means.shape
    if y_true.shape[0] != n_points:
        raise ValueError("y_true length must match the prediction point dimension.")

    levels = tuple(float(level) for level in levels)
    if not levels or any(level <= 0 or level >= 1 for level in levels):
        raise ValueError("levels must be probabilities strictly between 0 and 1.")

    total_vars = np.maximum(pred_vars, 0.0)

    if target == "observed":
        cov_chains = _flatten_cov_chains(cov_chains, n_samples)

        if X_test is None:
            raise ValueError("X_test is required when target='observed'.")
        X_test = np.asarray(X_test)
        if X_test.ndim != 2 or X_test.shape[0] != n_points or X_test.shape[1] < 3:
            raise ValueError("X_test must have shape (n_points, d) with d >= 3.")

        if obs_count_test is None:
            obs_count_test = np.ones(n_points)
        obs_count_test = np.asarray(obs_count_test, dtype=float).reshape(-1)
        if obs_count_test.shape[0] != n_points:
            raise ValueError("obs_count_test length must match y_true.")
        if np.any(obs_count_test <= 0) or not np.all(np.isfinite(obs_count_test)):
            raise ValueError("obs_count_test must contain positive finite counts.")

        is_indoor = X_test[:, 2].astype(bool)
        obs_vars = np.where(is_indoor[None, :], cov_chains[:, 1:2], cov_chains[:, 0:1])
        total_vars = total_vars + np.maximum(obs_vars, 0.0) / obs_count_test[None, :]

    if hasattr(random_state, "normal"):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    draws = rng.normal(pred_means, np.sqrt(total_vars))

    pred_mean = pred_means.mean(axis=0)
    mixture_var = total_vars.mean(axis=0) + pred_means.var(axis=0)
    normal_dist = NormalDist()

    summary = {
        "target": target,
        "n_points": int(n_points),
        "n_posterior_samples": int(n_samples),
        "mean_predictive_sd": float(np.sqrt(mixture_var).mean()),
    }
    intervals = {}

    for level in levels:
        label = _coverage_level_label(level)
        alpha = (1.0 - level) / 2.0

        lower = np.quantile(draws, alpha, axis=0)
        upper = np.quantile(draws, 1.0 - alpha, axis=0)
        covered = (y_true >= lower) & (y_true <= upper)
        summary[f"coverage_{label}"] = float(covered.mean())
        summary[f"avg_width_{label}"] = float((upper - lower).mean())

        z = normal_dist.inv_cdf(0.5 + level / 2.0)
        normal_lower = pred_mean - z * np.sqrt(mixture_var)
        normal_upper = pred_mean + z * np.sqrt(mixture_var)
        normal_covered = (y_true >= normal_lower) & (y_true <= normal_upper)
        summary[f"normal_coverage_{label}"] = float(normal_covered.mean())
        summary[f"normal_avg_width_{label}"] = float((normal_upper - normal_lower).mean())

        if return_intervals:
            intervals[label] = {
                "lower": lower,
                "upper": upper,
                "normal_lower": normal_lower,
                "normal_upper": normal_upper,
            }

    if return_intervals:
        summary["intervals"] = intervals

    return summary


@partial(jax.jit, static_argnames=['posterior_mean_cov', 'update_IG_in', 'update_IG_out', 'calibration'])
def _gibbs_step(key, X_train, y_train, variances, obs_count,
               posterior_mean_cov, update_IG_in, update_IG_out, jitter=0, calibration=False):
    '''
    Helper function for a single Gibbs step. To be called by the Gibbs sampler, not intended for direct calling by user.

    key: jax.random key used for update sampling
    X_train, y_train: training data arrays
    variances: last variances sample
    obs_count: number of raw observations averaged into each y_train entry
    posterior_mean_cov: function to compute posterior mean and covariance at fixed variances
    update_IG_in: function to compute posterior update for var_in inverse Gamma parameters
    update_IG_out: function to compute posterior update for var_out inverse Gamma parameters
    '''
    key_f, key_var_in, key_var_out = jr.split(key, 3)

    m_post, cov_post = posterior_mean_cov(variances=variances)
    if not calibration:
        f_hat = jr.multivariate_normal(key_f, m_post, cov_post + jnp.eye(cov_post.shape[0]) * jitter, method='cholesky')
    else:
        f_hat = m_post

    S = obs_count * (y_train - f_hat)**2
    mask_in = X_train[:,2] == 1
    S_in  = jnp.sum(jnp.where(mask_in, S, 0.0))
    S_out = jnp.sum(jnp.where(mask_in, 0.0, S))

    an_in, bn_in = update_IG_in(S_in)
    an_out, bn_out = update_IG_out(S_out)

    sigma2_in = bn_in / jr.gamma(key_var_in, an_in)
    sigma2_out = bn_out / jr.gamma(key_var_out, an_out)

    return f_hat, jnp.array([sigma2_out, sigma2_in])

class GaussianProcess():
    def __init__(self, m, K):
        '''
        Initialize a Gaussian Process instance with a mean function and covariance kernel.

        m : x_i -> mean, (d,) -> (1,)
        K : x_i, x_j-> cov, (d,) x (d,) -> (1,)
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))

        self.trained = False

    def fit(self, X_train, y_train, obs_count=None, obs_sse=None, variances=None, priors=jnp.full(4, jnp.nan)):
        '''
        X_train: (n, d) array
        y_train: (n,) array
        obs_count: (n,) array, number of raw observations averaged into each y_train entry (default: 1 each)
        obs_sse: (n,) array, within-location sum of squared errors for each aggregated y_train entry (deflat: 0 each)

        variances : (2,) array, initial condition for sampling variances, (default: Var(y_train)/2 for both)
        priors : (4,) array, inverse gamma hyperparameters for indoor/outdoor components
            - default: [a0_out=1.001, b0_out=0.001, a0_in=1.001, b0_in=0.001]
            - var_out ~ IG(a0_out, b0_out)
            - var_in ~ IG(a0_in, b0_in)
        '''
        if variances is None:
            variances = jnp.ones(2) * y_train.var() / 2
        self.variances = variances

        self.X_train = X_train
        self.y_train = y_train
        self.n, self.d = X_train.shape

        priors = jnp.array(priors, dtype=X_train.dtype)
        default_priors = jnp.array([
            1+1e-3, 1e-3,
            1+1e-3, 1e-3,
        ], dtype=X_train.dtype)
        IG_priors = priors.at[no_prior := jnp.isnan(priors)].set(default_priors[no_prior])
        self.IG_priors = IG_priors

        self.mn = self.m(self.X_train)
        self.Kn = self.K(self.X_train, self.X_train)
        self.Kn = 0.5 * (self.Kn + self.Kn.T)

        self.mask_in = (self.X_train[:,2] == 1)
        self.obs_count = jnp.ones(X_train.shape[0]) if (obs_count is None) else obs_count

        self.obs_sse = jnp.zeros(X_train.shape[0]) if (obs_sse is None) else obs_sse
        self.obs_sse_in = jnp.sum(jnp.where(self.mask_in, self.obs_sse, 0.0))
        self.obs_sse_out = jnp.sum(jnp.where(self.mask_in, 0.0, self.obs_sse))

        self.n_out = ((1 - self.mask_in) * self.obs_count).sum()
        self.n_in = (self.mask_in * self.obs_count).sum()

        def update_IG_out(S_out):
            return self.IG_priors[0] + self.n_out/2, self.IG_priors[1] + (S_out + self.obs_sse_out)/2
        def update_IG_in(S_in):
            return self.IG_priors[2] + self.n_in/2, self.IG_priors[3] + (S_in + self.obs_sse_in)/2

        @jax.jit
        def posterior_mean_cov(X_new, variances, K_new, Kn_new):
            '''
            Conditional posterior mean and covariance of X_new at fixed variances.
            '''
            noise_diag = (variances[0] + (variances[1] - variances[0]) * self.mask_in) / self.obs_count
            L = jnp.linalg.cholesky(self.Kn + jnp.diag(noise_diag))

            A = jsp.linalg.cho_solve((L, True), self.y_train - self.mn)
            m_post = self.m(X_new) + Kn_new.T @ A

            V = jsp.linalg.solve_triangular(L, Kn_new, lower=True)
            cov_post = K_new - V.T @ V
            cov_post = 0.5 * (cov_post + cov_post.T)

            return m_post, cov_post

        self.update_IG_in = update_IG_in
        self.update_IG_out = update_IG_out
        self.posterior_mean_cov = posterior_mean_cov

        self.trained = True

    def gibbs(self, key=jr.PRNGKey(305), chains=1, samples=100, calibration_iters=10, jitter_factor=1.5):
        '''
        Run a Gibbs sampler to sample from the posterior distribution of the GP instance.

        key: jax.random key, key used for random sampling in Gibbs chain
        chains: int, number of Gibbs chains to simulate in parallel
        samples: int, length of each Gibbs chain
        calibration_iters: int, iterations to run calibration -- Gibbs without sampling f_hat,
            used to set Cholesky jitter and good Gibbs starting point
        jitter_factor: float, set Cholesky jitter to be jitter_factor * |min(eigenvalue(cov_pot))|

        Returns: (f_chains, var_chains)
            - f_chains: (chains, samples, n_train) array, posterior samples of f at training data
            - var_chains: (chains, samples, 2), posterior samples of group sampling variances
        '''
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")

        key_calib, key_chains = jr.split(key)

        ### calibrate the jitter for cholesky decomposition
        # REMARK: set jitter to be just larger than the magnitude of the smallest negative eigenvalue of cov_post
        # ---> For fp32, approximately 1e-3. For fp64, approximately 1e-10 or 1e-11
        calib_step = partial(_gibbs_step, X_train=self.X_train, y_train=self.y_train, obs_count=self.obs_count,
            posterior_mean_cov=partial(self.posterior_mean_cov, X_new=self.X_train, K_new=self.Kn, Kn_new=self.Kn),
                            update_IG_in=self.update_IG_in, update_IG_out=self.update_IG_out, jitter=0, calibration=True)

        def calib_loop(i, val):
            variances, key = val
            f_hat_new, variances_new = calib_step(key=key, variances=variances)
            return (variances_new, jr.fold_in(key, i))
        sigma_calib, _ = jax.lax.fori_loop(0, calibration_iters, calib_loop, (self.variances, key_calib))
        self.variances = sigma_calib

        mu_post, cov_post = self.posterior_mean_cov(self.X_train, sigma_calib, self.Kn, self.Kn)
        eigvals, eigvecs = jnp.linalg.eigh(cov_post)
        self.jitter = (-jnp.where(eigvals <= 0, eigvals, 0)).max() * jitter_factor

        ### run gibbs sampler
        gibbs_step = partial(_gibbs_step, X_train=self.X_train, y_train=self.y_train, obs_count=self.obs_count,
                posterior_mean_cov=partial(self.posterior_mean_cov, X_new=self.X_train, K_new=self.Kn, Kn_new=self.Kn),
                                update_IG_in=self.update_IG_in, update_IG_out=self.update_IG_out, jitter=self.jitter, calibration=False)

        def single_chain_scan(variances, key):
            f_hat_new, variances_new = gibbs_step(key=key, variances=variances)
            return variances_new, (f_hat_new, variances_new)

        keys = jr.split(key_chains, chains)
        carry, gibbs_chains = jax.vmap(
            lambda key: jax.lax.scan(single_chain_scan, init=self.variances, xs=jr.split(key, samples))
        )(keys)

        self.f_hat = gibbs_chains[0][-1, -1]
        self.variances = gibbs_chains[1][-1, -1]

        S = self.obs_count * (self.y_train - self.f_hat)**2
        S_in  = jnp.sum(jnp.where(self.mask_in, S, 0.0))
        S_out = jnp.sum(jnp.where(self.mask_in, 0.0, S))

        an_in, bn_in = self.update_IG_in(S_in)
        an_out, bn_out = self.update_IG_out(S_out)
        self.IG_posteriors= jnp.array([an_out, bn_out, an_in, bn_in])

        return gibbs_chains

    def predict(self, X_new, cov_chains, method='sequential'):
        '''
        Predict on a set of new points, automatically enforcing temporal uncorrelatedness with training data.

        X_new: (n_new, d) array, new observation points
        cov_chains: (n_batches, chain_length, 2) array, posterior variance samples, usually from gibbs output

        method: string, method for computing the predictions.
            From fastest to slowest (also from most to least VRAM-intensive):
            - full_parallel: parallel over both batches and chains
            - parallel: sequential over batches, parallel over chains
            - sequential: parallel over batches, sequential over chains <- default
            - full_sequential: sequential over both batches and chains
            Generally, choose the first one that runs without OOM errors. However, note that memory shenanigans can cause slowdowns even when not OOM.

        Returns: (means, vars):
            - means: (n_batches * chain_length, n_new) array, conditional posterior means at each variance pair in cov_chains, for each point in X_new
            - vars: (n_batches * chain_length, n_new) array, conditional posterior variance at each variance pair in cov_chains, for each point in X_new
        '''
        # Set t_new to infinity to remove temporal covariance among new points
        # and between new points and training data.
        K_new = self.K(X_new, X_new.at[:,3].set(jnp.inf))
        Kn_new = self.K(self.X_train, X_new.at[:,3].set(jnp.inf))

        @jax.jit
        def single_posterior(variances):
            m, cov = self.posterior_mean_cov(X_new, variances, K_new, Kn_new)
            return m, jnp.diag(cov)  # (n,) instead of (n, n)

        if method == 'full_parallel':
            batch_means, batch_vars = jax.vmap(jax.vmap(single_posterior))(cov_chains)

        elif method == 'parallel':
            batch_means, batch_vars = jax.lax.map(lambda chain: jax.vmap(single_posterior)(chain), cov_chains)

        elif method == 'sequential':
            batch_means, batch_vars = jax.vmap(lambda chain: jax.lax.map(single_posterior, chain))(cov_chains)

        elif method == 'full_sequential':
            batch_means, batch_vars = jax.lax.map(lambda chain: jax.lax.map(single_posterior, chain), cov_chains)

        else:
            raise ValueError("Please choose a valid prediction method, ['full_parallel', 'parallel', 'sequential', 'full_sequential']. See docstring for details.")

        return jnp.concatenate(batch_means, axis=0), jnp.concatenate(batch_vars, axis=0)
