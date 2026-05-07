import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from functools import partial
from sklearn.exceptions import NotFittedError

import time

STABILIZER = 1e-3

# Timing utitlities

# Global store for timestamps
_timings = {}

def _record_time(label_bytes):
    label = label_bytes.tobytes().decode().rstrip('\x00')
    _timings[label] = time.perf_counter()

def checkpoint(label: str):
    """Drop a timing checkpoint inside a JIT function."""
    # Encode label as a fixed-length array so JAX can trace it
    encoded = jnp.frombuffer(label.encode().ljust(64, b'\x00'), dtype=jnp.uint8)
    jax.debug.callback(_record_time, encoded)

def print_timings():
    labels = list(_timings.keys())
    for a, b in zip(labels, labels[1:]):
        dt = (_timings[b] - _timings[a]) * 1000
        print(f"  {a} -> {b}: {dt:.3f} ms")


@partial(jax.jit, static_argnames=['stabilizer'])
def stable_cholesky(M, stabilizer=STABILIZER):
    '''
    Numerically stable Cholesky decomposition
    '''
    # M = 0.5 * (M + M.T)

    # def has_nan(state):
    #     _, chol = state
    #     return jnp.any(jnp.isnan(chol))

    # def recompute(state):
    #     stabilizer, _ = state
    #     stabilizer = stabilizer * 10
    #     chol = jnp.linalg.cholesky(M + jnp.eye(M.shape[0]) * stabilizer)
    #     return stabilizer, chol
    # chol = jnp.linalg.cholesky(M)
    # valid_stabilizer, chol = jax.lax.while_loop(has_nan, recompute, (stabilizer, chol))
    # return chol
    return jnp.linalg.cholesky(0.5 * (M + M.T) + jnp.eye(M.shape[0]) * stabilizer)

@partial(jax.jit, static_argnames=['stabilizer'])
def fast_inverse_cov(M, stabilizer=STABILIZER):
    '''
    Faster inverse of a positive semi-definite matrix
    '''
    L = stable_cholesky(M, stabilizer)
    return jsp.linalg.cho_solve((L, True), jnp.eye(M.shape[0]))

@partial(jax.jit, static_argnames=['p_m_cov', 'update_IG_in', 'update_IG_out', 'stabilizer'])
def _gibbs_step(key, X_train, y_train, Sigma_0,
               p_m_cov, update_IG_in, update_IG_out, stabilizer=STABILIZER):
    '''
    key: jax.random key
    cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
        multiply stabilizer by 10 until cholesky decomposition works
    '''
    # checkpoint("start")
    
    key_f, key_var_in, key_var_out = jr.split(key, 3)

    m_post, cov_post = p_m_cov(X_train, Sigma_0)
    # checkpoint("f posterior update")
    sqrtCov = stable_cholesky(cov_post, stabilizer)
    f_hat = sqrtCov @ jr.normal(key_f, shape=m_post.shape, dtype=X_train.dtype) + m_post
    # checkpoint("f posterior sample")

    S = (y_train - f_hat)**2
    S_in  = jnp.sum(jnp.where(X_train[:,-1], S, 0.0))
    S_out = jnp.sum(jnp.where(1-X_train[:,-1], 0.0, S))

    an_in, bn_in = update_IG_in(S_in)
    an_out, bn_out = update_IG_out(S_out)
    # checkpoint("sigma2 posterior update")

    sigma2_in = bn_in / jr.gamma(key_var_in, an_in)
    sigma2_out = bn_out / jr.gamma(key_var_out, an_out)
    Sigma_0 = jnp.where(X_train[:,-1] == 1, sigma2_in, sigma2_out)
    # checkpoint("sigma2 posterior sample")

    return f_hat, Sigma_0

class GaussianProcess():
    def __init__(self, m, K, cov_stabilizer=STABILIZER):
        '''
        m : x_i -> mean
        K : x_i, x_j-> cov
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))
        self.cov_stabilizer = cov_stabilizer

        self.trained = False

    def fit(self, X_train, y_train, Sigma_0=None, priors=jnp.full(4, jnp.nan)):
        '''
        X_train : n x d array
        y_train : n x 1 array

        Sigma_0 : x_i -> error variance initial setting
        priors : inverse gamma hyperparameters for indoor/outdoor components, a0_in, b0_in, a0_out, b0_out
        '''
        if Sigma_0 is None:
            Sigma_0 = jnp.ones(X_train.shape[0]) * y_train.var()
        self.Sigma_0 = Sigma_0

        self.X_train = X_train
        self.y_train = y_train
        self.n, self.d = X_train.shape

        priors = jnp.array(priors, dtype=X_train.dtype)
        default_priors = jnp.array([
            1+1e-3, 1e3, #y_train[X_train[:,-1].astype(bool)].var(),
            1+1e-3, 1e3, #y_train[~X_train[:,-1].astype(bool)].var(),
        ], dtype=X_train.dtype)
        IG_priors = priors.at[no_prior := jnp.isnan(priors)].set(default_priors[no_prior])
        self.IG_priors = IG_priors

        self.mn = self.m(self.X_train)
        self.Kn = self.K(self.X_train, self.X_train)
        self.Kn = 0.5 * (self.Kn + self.Kn.T)

        def update_IG_in(S_in):
            return self.IG_priors[0] + (self.X_train[:,-1] == 1).sum()/2, self.IG_priors[1] + S_in/2
        def update_IG_out(S_out):
            return self.IG_priors[2] + (self.X_train[:,-1] == 0).sum()/2, self.IG_priors[3] + S_out/2

        @jax.jit
        def posterior_mean_cov(X_new, Sigma_0, K_new):
            L = stable_cholesky(self.Kn + jnp.diag(Sigma_0), self.cov_stabilizer)
            Kn_new = self.K(self.X_train, X_new)

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
        cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
            multiply stabilizer by 10 until cholesky decomposition works
        '''
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")

        gibbs_step = partial(_gibbs_step, X_train=self.X_train, y_train=self.y_train,
                                p_m_cov=partial(self.posterior_mean_cov, K_new=self.Kn),
                                update_IG_in=self.update_IG_in, update_IG_out=self.update_IG_out,
                                stabilizer=self.cov_stabilizer)

        def single_chain_scan(Sigma_0, key):
            f_hat_new, Sigma_0_new = gibbs_step(key=key, Sigma_0=Sigma_0)
            return Sigma_0_new, (f_hat_new, Sigma_0_new)

        keys = jr.split(key, chains)
        carry, gibbs_chains = jax.vmap(
            lambda key: jax.lax.scan(single_chain_scan, init=self.Sigma_0, xs=jr.split(key, samples))
        )(keys)

        self.f_hat = gibbs_chains[0][0][-1]
        self.Sigma_0 = gibbs_chains[1][0][-1]

        S = (self.y_train - self.f_hat)**2
        S_in  = jnp.sum(jnp.where(self.X_train[:,-1], S, 0.0))
        S_out = jnp.sum(jnp.where(1-self.X_train[:,-1], 0.0, S))

        an_in, bn_in = self.update_IG_in(S_in)
        an_out, bn_out = self.update_IG_out(S_out)
        self.IG_posteriors= jnp.array([an_in, bn_in, an_out, bn_out])

        return gibbs_chains

    def posterior(self, X_new, cov_chains):
        K_new = self.K(X_new, X_new)

        @jax.jit
        def single_posterior(sigma2_vec):
            m, cov = self.posterior_mean_cov(X_new, sigma2_vec, K_new)
            return m, jnp.diag(cov)  # (n,) instead of (n, n)

        flat_chains = jnp.concatenate(cov_chains, axis=0)
        means, vars = jax.lax.map(single_posterior, flat_chains)
        return means, vars

    def simulate(self, X_new, cov_chains, key=jr.PRNGKey(305)):
        K_new = self.K(X_new, X_new)
        def sample_conditional(args):
            Sigma_0, key_f = args
            m_post, cov_post = self.posterior_mean_cov(X_new, Sigma_0, K_new)
            sqrtCov = stable_cholesky(cov_post)
            return sqrtCov @ jr.normal(key_f, shape=m_post.shape) + m_post

        return jax.vmap(lambda Sigma_0, key_c:
            jax.lax.map(sample_conditional, (Sigma_0, jr.split(key_c, Sigma_0.shape[0])))
                       )(cov_chains, jr.split(key, cov_chains.shape[0]))

