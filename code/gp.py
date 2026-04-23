import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
from functools import partial
from sklearn.exceptions import NotFittedError

def stable_cholesky(M):
    '''
    Numerically stable Cholesky decomposition
    '''
    return jnp.linalg.cholesky(0.5 * (M + M.T))

def fast_inverse_cov(M):
    '''
    Faster inverse of a positive semi-definite matrix
    '''
    L = stable_cholesky(M)
    return jsp.linalg.cho_solve((L, True), jnp.eye(M.shape[0]))

@partial(jax.jit, static_argnames=['n', 'm', 'K'])
def precomputes(X_train, Sigma_0, n, m, K):
    mn = m(X_train)
    Kn = K(X_train, X_train)
    Kn_inv = fast_inverse_cov(Kn + Sigma_0)
    return mn, Kn, Kn_inv

@partial(jax.jit, static_argnames=['d'])
def sample_inv_wishart(key, Psi, nu, d):
    '''
    Samples from Inv-Wishart(Psi, nu).
    If W ~ Wishart(Psi^{-1}, nu), then W^{-1} Inv-Wishart(Psi, nu).
    '''
    p = Psi.shape[0]
    key_chi, key_norm = jr.split(key)

    # sample wishart
    Psi_inv = fast_inverse_cov(Psi)
    L = stable_cholesky(Psi_inv)
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

    return fast_inverse_cov(W)

class GaussianProcess():
    def __init__(self, m, K):
        '''
        m : x_i -> mean
        K : x_i, x_j-> cov
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))

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

        self.update_f()

        self.trained = True

    def update_f(self, use_mean_Cov_0=False):
        '''
        Compute a Gibbs update for f, given Sigma_0. Sets the posterior mean and covariance functions for the MVN distribution.
        '''
        if use_mean_Cov_0: self.Sigma_0 = self.posterior_mean_Sigma_0
        self.mn, self.Kn, self.Kn_inv = precomputes(self.X_train, self.Sigma_0, self.n, self.m, self.K)

        def posterior_mean(X_new):
            return self.m(X_new) + self.K(X_new, self.X_train) @ self.Kn_inv @ (self.y_train - self.mn)
        def posterior_cov(X_new):
            return self.K(X_new, X_new) - self.K(X_new, self.X_train) @ self.Kn_inv @ self.K(self.X_train, X_new)

        self.posterior_mean = posterior_mean
        self.posterior_cov = posterior_cov

    def update_Sigma(self):
        '''
        Compute a Gibbs update for Sigma_0, given f. Sets a posterior scale and df for Inv-Wishart distribution.
        '''
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")
        self.posterior_Psi = self.Psi_0 + (self.y_train - self.f_hat)[:,None] @ (self.y_train - self.f_hat)[None,:]
        self.posterior_nu = self.nu_0 + self.n
        self.posterior_mean_Sigma_0 = self.posterior_Psi / (self.posterior_nu - self.n - 1)

    def gibbs_step(self, key, cov_stabilizer=1e-8):
        '''
        key: jax.random key
        cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
            multiply stabilizer by 10 until cholesky decomposition works
        '''
        key_f, key_W = jr.split(jr.PRNGKey(key))

        m_post = self.posterior_mean(self.X_train)
        cov_post = self.posterior_cov(self.X_train)
        sqrtCov = stable_cholesky(cov_post)
        while jnp.any(jnp.isnan(sqrtCov)):
            sqrtCov = stable_cholesky(cov_post + jnp.eye(cov_post.shape[0]) * cov_stabilizer)
            cov_stabilizer *= 10
        self.f_hat = sqrtCov @ jr.normal(key_f, shape=self.n) + m_post

        self.update_Sigma()
        self.Sigma_0 = sample_inv_wishart(key_W, self.posterior_Psi, self.posterior_nu, self.n)

        self.update_f()

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
            scores = self.posterior_mean(X_new_filled).reshape(n, m)

            best_candidates =  candidates[jnp.argmax(scores, axis=1)]
            X_new = X_new.at[:,-k:].set(best_candidates)

        return X_new, self.posterior_mean(X_new), self.posterior_cov(X_new)