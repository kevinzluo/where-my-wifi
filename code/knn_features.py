import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class WifiKNNFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, indoor_scale=1.0, ap_scale=1.0, n_ap=None):
        self.indoor_scale = indoor_scale
        self.ap_scale = ap_scale
        self.n_ap = n_ap

    def fit(self, X, y=None):
        X = np.asarray(X)
        if self.n_ap is None:
            self.n_ap_ = int(np.nanmax(X[:, 4])) + 1
        else:
            self.n_ap_ = int(self.n_ap)
        return self

    def transform(self, X):
        X = np.asarray(X)
        transformed = X[:, [0, 1, 2]].astype(float).copy()
        transformed[:, 2] = transformed[:, 2] / self.indoor_scale

        ap_idx = X[:, 4].astype(int)
        ap_one_hot = np.zeros((X.shape[0], self.n_ap_), dtype=transformed.dtype)
        valid_ap = (ap_idx >= 0) & (ap_idx < self.n_ap_)
        ap_one_hot[np.flatnonzero(valid_ap), ap_idx[valid_ap]] = 1.0
        ap_one_hot = ap_one_hot / self.ap_scale

        return np.concatenate([transformed, ap_one_hot], axis=1)
