from typing import NamedTuple
import jax.numpy as jnp
from jax.nn import softplus
import supplementary_info.kern_gp as kgp
from supplementary_info.grantham_dist import contextual_grantham_distance_matrix


def contextual_imq_kernel(contexts, actions, contexts_y=None, actions_y=None, scale=1.0, beta=0.5):
    if contexts_y is None:
        contexts_y = contexts
    if actions_y is None:
        actions_y = actions

    # Compute contextual Grantham distances
    joint_dist = contextual_grantham_distance_matrix(contexts, actions, contexts_y, actions_y)

    # Apply inverse multiquadric kernel
    return (1 + scale) ** beta / (scale + joint_dist) ** beta

TRANSFORM = softplus

class ContextualGP_Params(NamedTuple):
    raw_amplitude: jnp.ndarray
    raw_noise: jnp.ndarray
    scale: float
    beta: float


class ZeroMeanContextualGP:
    def __init__(self, contexts_train: list[str], actions_train: list[str], y_train: jnp.ndarray):
        self.set_training_data(contexts_train, actions_train, y_train)
        self._kernel_cache = {}

    def set_training_data(self, contexts_train: list[str], actions_train: list[str], y_train: jnp.ndarray):
        self._contexts_train = contexts_train
        self._actions_train = actions_train
        self._y_train = jnp.asarray(y_train)

    def _kernel_matrix(self, C1, A1, C2, A2, params: ContextualGP_Params):
        key = (tuple(C1), tuple(A1), tuple(C2), tuple(A2), params.scale, params.beta)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = jnp.asarray(
                contextual_imq_kernel(C1, A1, C2, A2, scale=params.scale, beta=params.beta)
            )
        return self._kernel_cache[key]

    def marginal_log_likelihood(self, params: ContextualGP_Params) -> jnp.ndarray:
        K_train = self._kernel_matrix(
            self._contexts_train, self._actions_train,
            self._contexts_train, self._actions_train,
            params
        )
        return kgp.mll_train(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=K_train,
            y_train=self._y_train
        )

    def predict_f(self, params: ContextualGP_Params,
                  contexts_test: list[str], actions_test: list[str],
                  full_covar=True):
        k_train_train = self._kernel_matrix(
            self._contexts_train, self._actions_train,
            self._contexts_train, self._actions_train,
            params
        )
        k_test_train = self._kernel_matrix(
            contexts_test, actions_test,
            self._contexts_train, self._actions_train,
            params
        )
        k_test_test = self._kernel_matrix(
            contexts_test, actions_test,
            contexts_test, actions_test,
            params
        ) if full_covar else jnp.diag(
            self._kernel_matrix(
                contexts_test, actions_test,
                contexts_test, actions_test,
                params
            )
        )
        return kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=k_train_train,
            k_test_train=k_test_train,
            k_test_test=k_test_test,
            y_train=self._y_train,
            full_covar=full_covar
        )

    def predict_y(self, params: ContextualGP_Params,
                  contexts_test: list[str], actions_test: list[str],
                  full_covar=True):
        mean, covar = self.predict_f(params, contexts_test, actions_test, full_covar)
        noise = TRANSFORM(params.raw_noise)
        if full_covar:
            covar += jnp.eye(len(actions_test)) * noise
        else:
            covar += noise
        return mean, covar

    def _predict_with_precomputed(self, params: ContextualGP_Params,
                                  k_train_train, k_test_train, k_test_test,
                                  y_train, full_covar=True):
        return kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=k_train_train,
            k_test_train=k_test_train,
            k_test_test=k_test_test,
            y_train=y_train,
            full_covar=full_covar
        )

    def get_y_train(self):
        return self._y_train
