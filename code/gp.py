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

    valid_stabilizer, chol = jax.lax.while_loop(has_nan, recompute, (stabilizer, jnp.full(M.shape, jnp.nan)))
    return chol


@partial(jax.jit, static_argnames=['stabilizer'])
def fast_inverse_cov(M, stabilizer=1e-8):
    '''
    Faster inverse of a positive semi-definite matrix
    '''
    L = stable_cholesky(M, stabilizer)
    return jsp.linalg.cho_solve((L, True), jnp.eye(M.shape[0]))


@partial(jax.jit, static_argnames=['stabilizer'])
def update_f(Sigma_0, Kn, stabilizer=1e-8):
    '''
    Compute a Gibbs update for f, given Sigma_0.
    '''
    Kn_inv = fast_inverse_cov(Kn + Sigma_0, stabilizer)
    return Kn_inv


@partial(jax.jit, static_argnames=['stabilizer'])
def sample_inv_wishart(key, Psi, nu, stabilizer=1e-8):
    '''
    Samples from Inv-Wishart(Psi, nu).
    If W ~ Wishart(Psi^{-1}, nu), then W^{-1} Inv-Wishart(Psi, nu).
    '''
    d = Psi.shape[0]
    key_chi, key_norm = jr.split(key)

    # sample wishart
    Psi_inv = fast_inverse_cov(Psi, stabilizer)
    L = stable_cholesky(Psi_inv, stabilizer)
    # Bartlett decomposition
    # Diagonal: sqrt chi-square
    chi2 = jr.chisquare(key_chi, df=nu - jnp.arange(d))
    diag = jnp.sqrt(chi2)
    # Full matrix of normals
    Z = jr.normal(key_norm, shape=(d, d))
    # Keep strictly lower triangle
    A = jnp.tril(Z, k=-1)
    # Set diagonal
    A = A + jnp.diag(diag)

    # Wishart sample
    LA = L @ A
    W = LA @ LA.T

    return fast_inverse_cov(W, stabilizer)


@partial(jax.jit, static_argnames=['pm', 'pcov', 'pnu', 'pPsi', 'stabilizer'])
def gibbs_step(key, X_train, Kn_inv, pm, pcov, pPsi, pnu, Kn,
               stabilizer=1e-8):
    '''
    key: jax.random key
    cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
        multiply stabilizer by 10 until cholesky decomposition works
    '''
    key_f, key_W = jr.split(jr.PRNGKey(key))

    m_post = pm(X_train, Kn_inv)
    cov_post = pcov(X_train, Kn_inv)

    sqrtCov = stable_cholesky(cov_post, stabilizer)
    f_hat = sqrtCov @ jr.normal(key_f, shape=m_post.shape) + m_post

    Psi_post = pPsi(f_hat)
    Sigma_0 = sample_inv_wishart(key_W, Psi_post, pnu, stabilizer)

    Kn_inv = update_f(Sigma_0, Kn, stabilizer)
    return f_hat, Sigma_0, Kn_inv

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
            Sigma_0 = jnp.eye(X_train.shape[0]) * jnp.var(y_train)
        self.Sigma_0 = Sigma_0

        self.X_train = X_train
        self.y_train = y_train
        self.n, self.d = X_train.shape

        if Psi_0 is None: Psi_0 = jnp.eye(self.n) * 1e-3
        if nu_0 is None: nu_0 = self.d + 1
        self.Psi_0 = Psi_0
        self.nu_0 = nu_0

        @jax.jit
        def posterior_mean(X_new, Kn_inv):
            return self.m(X_new) + self.K(X_new, self.X_train) @ Kn_inv @ (self.y_train - self.mn)

        @jax.jit
        def posterior_cov(X_new, Kn_inv):
            return self.K(X_new, X_new) - self.K(X_new, self.X_train) @ Kn_inv @ self.K(self.X_train, X_new)

        @jax.jit
        def posterior_Psi(f_hat):
            Psi = self.Psi_0 + (self.y_train - f_hat)[:,None] @ (self.y_train - f_hat)[None,:]
            return Psi

        self.posterior_mean = posterior_mean
        self.posterior_cov = posterior_cov
        self.posterior_Psi = posterior_Psi
        self.nu_n = self.nu_0 + self.n
        self.mn = self.m(self.X_train)
        self.Kn = self.K(self.X_train, self.X_train)

        self.Kn_inv = update_f(self.Sigma_0, self.Kn, self.cov_stabilizer)

        self.trained = True

    def gibbs(self, key):
        '''
        key: jax.random key
        cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
            multiply stabilizer by 10 until cholesky decomposition works
        '''
        f_hat, Sigma_0, Kn_inv = gibbs_step(key,
            self.X_train, self.Kn_inv, self.posterior_mean, self.posterior_cov,
            self.posterior_Psi, self.nu_n, self.Kn, self.cov_stabilizer)

        self.f_hat = f_hat
        self.Sigma_0 = Sigma_0
        self.Kn_inv = Kn_inv

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
            scores = self.posterior_mean(X_new_filled, self.Kn_inv).reshape(n, m)

            best_candidates =  candidates[jnp.argmax(scores, axis=1)]
            X_new = X_new.at[:,-k:].set(best_candidates)

        return X_new, self.posterior_mean(X_new, self.Kn_inv), self.posterior_cov(X_new, self.Kn_inv)