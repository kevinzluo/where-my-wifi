import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from functools import partial
from sklearn.exceptions import NotFittedError


@partial(jax.jit, static_argnames=['stabilizer'])
def stable_cholesky(M, stabilizer=1e-8):
    '''
    Numerically stable Cholesky decomposition
    '''
    M = 0.5 * (M + M.T)

    def has_nan(state):
        _, chol = state
        return jnp.any(jnp.isnan(chol))

    def recompute(state):
        stabilizer, _ = state
        stabilizer = stabilizer * 10
        chol = jnp.linalg.cholesky(M + jnp.eye(M.shape[0]) * stabilizer)
        return stabilizer, chol
    chol = jnp.linalg.cholesky(M)
    valid_stabilizer, chol = jax.lax.while_loop(has_nan, recompute, (stabilizer, chol))
    return chol


@partial(jax.jit, static_argnames=['stabilizer'])
def fast_inverse_cov(M, stabilizer=1e-8):
    '''
    Faster inverse of a positive semi-definite matrix
    '''
    L = stable_cholesky(M, stabilizer)
    return jsp.linalg.cho_solve((L, True), jnp.eye(M.shape[0]))


@partial(jax.jit, static_argnames=['stabilizer'])
def sample_inv_wishart(key, Psi, nu, stabilizer=1e-8):
    '''
    Samples from Inv-Wishart(Psi, nu).
    If W ~ Wishart(Psi^{-1}, nu), then W^{-1} Inv-Wishart(Psi, nu).
    '''
    d = Psi.shape[0]
    key_chi, key_norm = jr.split(key)

    # sample wishart
    L = stable_cholesky(fast_inverse_cov(Psi, stabilizer), stabilizer)
    # Bartlett decomposition
    # Diagonal: sqrt chi-square
    chi2 = jr.chisquare(key_chi, df=nu - jnp.arange(d))
    # Full matrix of normals
    Z = jr.normal(key_norm, shape=(d, d))
    # Keep strictly lower triangle
    A = jnp.tril(Z, k=-1) + jnp.diag(jnp.sqrt(chi2))

    # note L @ A is the Cholesky decomp. of a Wishart sample
    # directly invert without reconstructing W
    return jsp.linalg.cho_solve((L @ A, True), jnp.eye(d))


@partial(jax.jit, static_argnames=['p_m_cov', 'pnu', 'pPsi', 'stabilizer'])
def gibbs_step(key, X_train, Sigma_0, p_m_cov, pPsi, pnu, stabilizer=1e-8):
    '''
    key: jax.random key
    cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
        multiply stabilizer by 10 until cholesky decomposition works
    '''
    key_f, key_W = jr.split(key)

    m_post, cov_post = p_m_cov(X_train, Sigma_0)
    sqrtCov = stable_cholesky(cov_post, stabilizer)
    f_hat = sqrtCov @ jr.normal(key_f, shape=m_post.shape) + m_post

    Psi_post = pPsi(f_hat)
    Sigma_0 = sample_inv_wishart(key_W, Psi_post, pnu, stabilizer)

    return f_hat, Sigma_0

class GaussianProcess():
    def __init__(self, m, K, cov_stabilizer=1e-8):
        '''
        m : x_i -> mean
        K : x_i, x_j-> cov
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))
        self.cov_stabilizer = cov_stabilizer

        self.trained = False

    def fit(self, X_train, y_train, Sigma_0=None, Psi_0=None, nu_0=None):
        '''
        X_train : n x d array
        y_train : n x 1 array

        Sigma_0 : x_i -> error variance initial setting
        Psi_0 : n x n -> Sigma_0 prior scale matrix
        nu_0 : float -> Sigma_0 prior degrees of freedom
        '''
        if Sigma_0 is None:
            Sigma_0 = jnp.eye(X_train.shape[0]) * 1e-3 #jnp.var(y_train)
        self.Sigma_0 = Sigma_0

        self.X_train = X_train
        self.y_train = y_train
        self.n, self.d = X_train.shape

        self.mn = self.m(self.X_train)
        self.Kn = self.K(self.X_train, self.X_train)

        if nu_0 is None: nu_0 = self.n + 2
        if Psi_0 is None: Psi_0 = jnp.eye(self.n) * 1e-3 #jnp.var(y_train) * (nu_0 - self.n - 1)

        self.Psi_0 = Psi_0
        self.nu_0 = nu_0

        @jax.jit
        def posterior_mean_cov(X_new, Sigma_0):
            L = stable_cholesky(self.Kn + Sigma_0, self.cov_stabilizer)
            Kn_new = self.K(self.X_train, X_new)

            A = jsp.linalg.cho_solve((L, True), self.y_train - self.mn)
            m_post = self.m(X_new) + Kn_new.T @ A

            V = jsp.linalg.solve_triangular(L, Kn_new, lower=True)
            cov_post = self.K(X_new, X_new) - V.T @ V

            return m_post, cov_post

        @jax.jit
        def posterior_Psi(f_hat):
            Psi = self.Psi_0 + (self.y_train - f_hat)[:,None] @ (self.y_train - f_hat)[None,:]
            return Psi

        self.posterior_mean_cov = posterior_mean_cov
        self.posterior_Psi = posterior_Psi
        self.nu_n = self.nu_0 + self.n

        self.trained = True

    def gibbs(self, key):
        '''
        key: jax.random key
        cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
            multiply stabilizer by 10 until cholesky decomposition works
        '''
        new_key, gibbs_key = jr.split(key)
        f_hat, Sigma_0 = gibbs_step(gibbs_key,
            self.X_train, self.Sigma_0, self.posterior_mean_cov,
            self.posterior_Psi, self.nu_n, self.cov_stabilizer)
        self.f_hat = f_hat
        self.Sigma_0 = Sigma_0
        return new_key

    def predict(self, X_new, candidates=None):
        '''
        X_new : n_new x (d - k)
        candidates : m x k
            for each row in X_new fill the last k columns with argmax_{j=1,...m} E[X_new, candidate_j | y]
        '''
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")

        if candidates is not None:
            candidates = jnp.array(candidates)
            m, k = candidates.shape
            n, l = X_new.shape
            if k > self.d:
                raise ValueError("max_over array has incorrect shape to be maximized over.")
            X_new = jnp.concatenate([X_new, jnp.zeros((n, self.d - l))], axis=1)

            X_new_rep = jnp.repeat(X_new, m, axis=0)
            candidates_tile = jnp.tile(candidates, (n,1))

            X_new_filled = X_new_rep.at[:,-k:].set(candidates_tile)
            scores, _ = self.posterior_mean_cov(X_new_filled, self.Sigma_0).reshape(n, m)

            best_candidates =  candidates[jnp.argmax(scores, axis=1)]
            X_new = X_new.at[:,-k:].set(best_candidates)

        return X_new, *self.posterior_mean_cov(X_new, self.Sigma_0)