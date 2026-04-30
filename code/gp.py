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

@partial(jax.jit, static_argnames=['p_m_cov', 'update_IG_in', 'update_IG_out', 'stabilizer'])
def gibbs_step(key, X_train, y_train, Sigma_0,
               p_m_cov, update_IG_in, update_IG_out, stabilizer=1e-8):
    '''
    key: jax.random key
    cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
        multiply stabilizer by 10 until cholesky decomposition works
    '''
    key_f, key_var_in, key_var_out = jr.split(key, 3)

    m_post, cov_post = p_m_cov(X_train, Sigma_0)
    sqrtCov = stable_cholesky(cov_post, stabilizer)
    f_hat = sqrtCov @ jr.normal(key_f, shape=m_post.shape) + m_post

    S = (y_train - f_hat)**2
    S_in  = jnp.sum(jnp.where(X_train[:,-1], S, 0.0))
    S_out = jnp.sum(jnp.where(1-X_train[:,-1], 0.0, S))

    an_in, bn_in = update_IG_in(S_in)
    an_out, bn_out = update_IG_out(S_out)

    sigma2_in = bn_in / jr.gamma(key_var_in, an_in)
    sigma2_out = bn_out / jr.gamma(key_var_out, an_out)
    Sigma_0 = jnp.diag(jnp.where(X_train[:,-1] == 1, sigma2_in, sigma2_out))

    return f_hat, Sigma_0, jnp.stack([an_in, bn_in, an_out, bn_out])

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

    def fit(self, X_train, y_train, Sigma_0=None, priors=jnp.full(4, jnp.nan)):
        '''
        X_train : n x d array
        y_train : n x 1 array

        Sigma_0 : x_i -> error variance initial setting
        priors : inverse gamma hyperparameters for indoor/outdoor components, a0_in, b0_in, a0_out, b0_out
        '''
        if Sigma_0 is None:
            Sigma_0 = jnp.eye(X_train.shape[0]) * 1e-3 #jnp.var(y_train)
        self.Sigma_0 = Sigma_0

        self.X_train = X_train
        self.y_train = y_train
        self.n, self.d = X_train.shape

        priors = jnp.array(priors)
        default_priors = jnp.array([
            1+1e-3, 1e3, #y_train[X_train[:,-1].astype(bool)].var(),
            1+1e-3, 1e3, #y_train[~X_train[:,-1].astype(bool)].var(),
        ])
        IG_priors = priors.at[no_prior := jnp.isnan(priors)].set(default_priors[no_prior])
        self.IG_priors = IG_priors

        self.mn = self.m(self.X_train)
        self.Kn = self.K(self.X_train, self.X_train)

        def update_IG_in(S_in):
            return self.IG_priors[0] + (self.X_train[:,-1] == 1).sum()/2, self.IG_priors[1] + S_in/2
        def update_IG_out(S_out):
            return self.IG_priors[2] + (self.X_train[:,-1] == 0).sum()/2, self.IG_priors[3] + S_out/2

        @jax.jit
        def posterior_mean_cov(X_new, Sigma_0):
            L = stable_cholesky(self.Kn + Sigma_0, self.cov_stabilizer)
            Kn_new = self.K(self.X_train, X_new)

            A = jsp.linalg.cho_solve((L, True), self.y_train - self.mn)
            m_post = self.m(X_new) + Kn_new.T @ A

            V = jsp.linalg.solve_triangular(L, Kn_new, lower=True)
            cov_post = self.K(X_new, X_new) - V.T @ V

            return m_post, cov_post

        self.update_IG_in = update_IG_in
        self.update_IG_out = update_IG_out
        self.posterior_mean_cov = posterior_mean_cov

        @jax.jit
        def sample_f_cond(key, X_new, Sigma_0):
            mu_new, Sigma_new = self.posterior_mean_cov(X_new, Sigma_0)
            sqrtCov = stable_cholesky(Sigma_new, self.cov_stabilizer)
            return sqrtCov @ jr.normal(key, shape=mu_new.shape) + mu_new

        @jax.jit
        def sample_f_marg(key, X_new, E_Sigma, IG_posteriors):
            # approximate since we use E_Sigma instead of integrating out
            # partly vibe coded, needs double-checking later
            mu_new, Sigma_new = self.posterior_mean_cov(X_new, E_Sigma)
            indoor = X_new[:,-1]
            sigma2_star = jnp.where(indoor,
                            IG_posteriors[1] / (IG_posteriors[0] - 1),
                            IG_posteriors[3] / (IG_posteriors[2] - 1))
            df_new = jnp.where(indoor, 2*IG_posteriors[0], 2*IG_posteriors[2])
            scale = jnp.sqrt(jnp.diag(Sigma_new) + sigma2_star)
            key_n, key_c = jr.split(key)
            Z = jr.normal(key_n, shape=mu_new.shape)
            X2 = jr.chisquare(key_c, df=df_new, shape=mu_new.shape)
            t = Z / jnp.sqrt(X2 / df_new)
            return mu_new + scale*t

        self._sample_f_cond = sample_f_cond
        self._sample_f_marg = sample_f_marg

        self.trained = True

    def gibbs(self, key):
        '''
        key: jax.random key
        cov_stabilizer: if cholesky decomp fails due to numerical issues, add diagonal matrix with cov_stabilizer
            multiply stabilizer by 10 until cholesky decomposition works
        '''
        new_key, gibbs_key = jr.split(key)
        f_hat, Sigma_0, IG_posteriors = gibbs_step(gibbs_key,
            self.X_train, self.y_train, self.Sigma_0, self.posterior_mean_cov,
            self.update_IG_in, self.update_IG_out, self.cov_stabilizer)

        self.f_hat = f_hat
        self.Sigma_0 = Sigma_0
        self.IG_posteriors = IG_posteriors
        return new_key

    def sample_f(self, key, X_new, conditional=False, Sigma_0=None):
        if conditional:
            if Sigma_0 is None: Sigma_0 = self.Sigma_0
            return self._sample_f_cond(key, X_new, Sigma_0)
        else:
            sigma2_in = self.IG_posteriors[1] / (self.IG_posteriors[0] - 1)
            sigma2_out = self.IG_posteriors[3] / (self.IG_posteriors[2] - 1)
            E_Sigma = jnp.diag(jnp.where(self.X_train[:,-1] == 1, sigma2_in, sigma2_out))
            return self._sample_f_marg(key, X_new, E_Sigma, self.IG_posteriors)

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