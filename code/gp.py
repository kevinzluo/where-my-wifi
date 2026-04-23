import jax
import jax.numpy as jnp
import jax.scipy as jsp
from sklearn.exceptions import NotFittedError

class GaussianProcess():
    def __init__(self, m, K, Sigma=None):
        '''
        m : array(x, y, z) -> mean
        K : array(x_i, y_i, z_i), array(x_j, y_j, z_j) -> cov
        m : array(x, y, z) -> error variance
        '''
        self.m = jax.jit(jax.vmap(m))
        self.K = jax.jit(jax.vmap(jax.vmap(K, in_axes=(None, 0)), in_axes=(0, None)))
        if Sigma is None:
            Sigma = lambda xyz: 1.0
        self.Sigma = jax.jit(jax.vmap(Sigma))

        self.trained = False

    def fit(self, X_train, y_train):
        '''
        X_train : n x d array
        y_train : n x 1 array
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.mn = self.m(X_train)
        self.Kn = self.K(X_train, X_train)

        # faster matrix inversion
        L = jnp.linalg.cholesky(self.Kn + jnp.diag(self.Sigma(X_train)))
        self.Kn_inv = jsp.linalg.cho_solve((L, True), jnp.eye(self.Kn.shape[0]))

        self.trained = True

    def predict(self, X_new):
        if not self.trained:
            raise NotFittedError("This Gaussian Process instance has not been fitted.")
        y_mean = self.m(X_new) + self.K(X_new, self.X_train) @ self.Kn_inv @ (self.y_train - self.mn)
        y_cov = self.K(X_new, X_new) - self.K(X_new, self.X_train) @ self.Kn_inv @ self.K(self.X_train, X_new)
        return y_mean, y_cov