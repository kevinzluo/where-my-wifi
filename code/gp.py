import jax
import jax.numpy as jnp
import jax.scipy as jsp
from sklearn.exceptions import NotFittedError

class GaussianProcess():
    def __init__(self, m, K, Sigma=None):
        '''
        m : x_i -> mean
        K : x_i, x_j-> cov
        Sigma : x_i -> error variance
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))
        if Sigma is None:
            Sigma = lambda x: 1.0
        self.Sigma = jax.jit(jax.vmap(Sigma))

        self.trained = False

    def fit(self, X_train, y_train):
        '''
        X_train : n x d array
        y_train : n x 1 array
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.d = X_train.shape[1]

        self.mn = self.m(X_train)
        self.Kn = self.K(X_train, X_train)

        # faster matrix inversion
        L = jnp.linalg.cholesky(self.Kn + jnp.diag(self.Sigma(X_train)))
        self.Kn_inv = jsp.linalg.cho_solve((L, True), jnp.eye(self.Kn.shape[0]))

        self.trained = True

    def predict(self, X_new, candidates=None):
        '''
        X_new : n_new x (d - k)
        candidates : m x k
            for each row in X_new fill the last k columns with argmax_{j=1,...m} E[X_new, candidate_j | y]
        '''
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")

        def posterior_mean(X_new):
            return self.m(X_new) + self.K(X_new, self.X_train) @ self.Kn_inv @ (self.y_train - self.mn)
        def posterior_cov(X_new):
            return self.K(X_new, X_new) - self.K(X_new, self.X_train) @ self.Kn_inv @ self.K(self.X_train, X_new)

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
            scores = posterior_mean(X_new_filled).reshape(n, m)

            best_candidates =  candidates[jnp.argmax(scores, axis=1)]
            X_new = X_new.at[:,-k:].set(best_candidates)

        return X_new, posterior_mean(X_new), posterior_cov(X_new)