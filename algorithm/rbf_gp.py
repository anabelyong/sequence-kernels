from typing import NamedTuple
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, cholesky, solve_triangular
from jax.nn import softplus

LOWER = True

class RBFParams(NamedTuple):
    raw_amplitude: jnp.ndarray
    raw_noise: jnp.ndarray
    lengthscale: float

def rbf_kernel(X1, X2=None, lengthscale=0.1):
    if X2 is None:
        X2 = X1
    sqdist = jnp.sum((X1[:, None] - X2[None, :]) ** 2, axis=-1)
    return jnp.exp(-0.5 * sqdist / lengthscale ** 2)

class ZeroMeanRBFGP:
    def __init__(self, X_train, y_train):
        self.set_training_data(X_train, y_train)
        self._kernel_cache = {}

    def set_training_data(self, X_train, y_train):
        self._X_train = jnp.asarray(X_train)
        self._y_train = jnp.asarray(y_train)

    def _kernel_matrix(self, A, B, params: RBFParams):
        return rbf_kernel(jnp.asarray(A), jnp.asarray(B), params.lengthscale)

    def marginal_log_likelihood(self, params: RBFParams):
        K_train = self._kernel_matrix(self._X_train, self._X_train, params)
        return kgp.mll_train(
            a=jnp.exp(params.raw_amplitude),
            s=jnp.exp(params.raw_noise),
            k_train_train=K_train,
            y_train=self._y_train
        )

    def _predict_with_precomputed(self, params: RBFParams,
                                  k_train_train, k_test_train, k_test_test,
                                  y_train, full_covar=True):
        return kgp.noiseless_predict(
            a=jnp.exp(params.raw_amplitude),
            s=jnp.exp(params.raw_noise),
            k_train_train=k_train_train,
            k_test_train=k_test_train,
            k_test_test=k_test_test,
            y_train=y_train,
            full_covar=full_covar
        )