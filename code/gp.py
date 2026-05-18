import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from functools import partial
from sklearn.exceptions import NotFittedError

@partial(jax.jit, static_argnames=['p_m_cov', 'update_IG_in', 'update_IG_out'])
def _gibbs_step(key, X_train, y_train, variances, obs_count,
               p_m_cov, update_IG_in, update_IG_out):
    '''
    key: jax.random key
    '''
    key_f, key_var_in, key_var_out = jr.split(key, 3)

    m_post, cov_post = p_m_cov(X_train, variances)
    eigvals, eigvecs = jnp.linalg.eigh(cov_post)
    eigvals_safe = jnp.maximum(eigvals, 0.0)   # clip negative eigenvalues
    f_hat = eigvecs @ (jnp.sqrt(eigvals_safe) * jr.normal(key_f, shape=eigvals.shape, dtype=X_train.dtype)) + m_post

    S = (y_train - f_hat)**2
    S_in  = jnp.sum(jnp.where(X_train[:,2], S, 0.0) * obs_count)
    S_out = jnp.sum(jnp.where(1-X_train[:,2], 0.0, S) * obs_count)

    an_in, bn_in = update_IG_in(S_in)
    an_out, bn_out = update_IG_out(S_out)

    sigma2_in = bn_in / jr.gamma(key_var_in, an_in)
    sigma2_out = bn_out / jr.gamma(key_var_out, an_out)

    return f_hat, jnp.array([sigma2_out, sigma2_in])

class GaussianProcess():
    def __init__(self, m, K):
        '''
        m : x_i -> mean
        K : x_i, x_j-> cov
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))

        self.trained = False

    def fit(self, X_train, y_train, obs_count=None, variances=None, priors=jnp.full(4, jnp.nan)):
        '''
        X_train : n x d array
        y_train : n x 1 array

        Sigma_0 : x_i -> error variance initial setting
        priors : inverse gamma hyperparameters for indoor/outdoor components, a0_out, b0_out, a0_in, b0_in
        '''
        if variances is None:
            variances = jnp.ones(2) * y_train.var() / 2
        self.variances = variances

        self.X_train = X_train
        self.y_train = y_train
        self.n, self.d = X_train.shape

        priors = jnp.array(priors, dtype=X_train.dtype)
        default_priors = jnp.array([
            1+1e-3, 1e3,
            1+1e-3, 1e3,
        ], dtype=X_train.dtype)
        IG_priors = priors.at[no_prior := jnp.isnan(priors)].set(default_priors[no_prior])
        self.IG_priors = IG_priors

        self.mn = self.m(self.X_train)
        self.Kn = self.K(self.X_train, self.X_train)
        self.Kn = 0.5 * (self.Kn + self.Kn.T)

        self.mask_in = (self.X_train[:,2] == 1)
        self.obs_count = jnp.ones(X_train.shape[0]) if (obs_count is None) else obs_count
        self.n_out = ((self.X_train[:,2] == 0) * self.obs_count).sum()
        self.n_in = ((self.X_train[:,2] == 1) * self.obs_count).sum()

        def update_IG_out(S_out):
            return self.IG_priors[0] + self.n_out/2, self.IG_priors[3] + S_out/2
        def update_IG_in(S_in):
            return self.IG_priors[2] + self.n_in/2, self.IG_priors[1] + S_in/2

        @jax.jit
        def posterior_mean_cov(X_new, variances, K_new, Kn_new):
            noise_diag = (variances[0] + (variances[1] - variances[0]) * self.mask_in) / self.obs_count
            L = jnp.linalg.cholesky(self.Kn + jnp.diag(noise_diag))

            A = jsp.linalg.cho_solve((L, True), self.y_train - self.mn)
            m_post = self.m(X_new) + Kn_new.T @ A

            V = jsp.linalg.solve_triangular(L, Kn_new, lower=True)
            cov_post = K_new - V.T @ V

            return m_post, cov_post

        self.update_IG_in = update_IG_in
        self.update_IG_out = update_IG_out
        self.posterior_mean_cov = posterior_mean_cov

        self.trained = True

    def gibbs(self, key=jr.PRNGKey(305), chains=1, samples=20):
        '''
        key: jax.random key
        '''
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")

        gibbs_step = partial(_gibbs_step, X_train=self.X_train, y_train=self.y_train, obs_count=self.obs_count,
                p_m_cov=partial(self.posterior_mean_cov, K_new=self.Kn, Kn_new=self.Kn),
                                update_IG_in=self.update_IG_in, update_IG_out=self.update_IG_out)

        def single_chain_scan(variances, key):
            f_hat_new, variances_new = gibbs_step(key=key, variances=variances)
            return variances_new, (f_hat_new, variances_new)

        keys = jr.split(key, chains)
        carry, gibbs_chains = jax.vmap(
            lambda key: jax.lax.scan(single_chain_scan, init=self.variances, xs=jr.split(key, samples))
        )(keys)

        self.f_hat = gibbs_chains[0][-1, -1]
        self.variances = gibbs_chains[1][-1, -1]

        S = (self.y_train - self.f_hat)**2
        S_in  = jnp.sum(jnp.where(self.X_train[:,2], S, 0.0) * self.obs_count)
        S_out = jnp.sum(jnp.where(self.X_train[:,2], 0.0, S) * self.obs_count)

        an_in, bn_in = self.update_IG_in(S_in)
        an_out, bn_out = self.update_IG_out(S_out)
        self.IG_posteriors= jnp.array([an_in, bn_in, an_out, bn_out])

        return gibbs_chains

    def predict(self, X_new, cov_chains, method='parallel'):
        # note: set t_new to be infinity to kill temporal covariance
        K_new = self.K(X_new, X_new.at[:,3].set(jnp.inf))
        Kn_new = self.K(self.X_train, X_new.at[:,3].set(jnp.inf))

        @jax.jit
        def single_posterior(variances):
            m, cov = self.posterior_mean_cov(X_new, variances, K_new, Kn_new)
            return m, jnp.diag(cov)  # (n,) instead of (n, n)

        flat_chains = jnp.concatenate(cov_chains, axis=0)
        if method == 'parallel':
            means, vars = jax.vmap(single_posterior)(flat_chains)
        elif method == 'sequential':
            means, vars = jax.lax.map(single_posterior, flat_chains)
        return means, vars