from typing import NamedTuple
import jax.numpy as jnp
from jax.nn import softplus
import kern_gp as kgp
from seq_tools import hamming_dist

def imq_hamming_kernel(seqs_x, seqs_y=None, alphabet_name='prot', scale=1.0, beta=0.5, lag=1):
    if seqs_y is None:
        seqs_y = seqs_x
    h_dists = hamming_dist(seqs_x, seqs_y, alphabet_name=alphabet_name, lag=lag)
    return (1 + scale) ** beta / (scale + h_dists) ** beta


TRANSFORM = softplus

class IMQHGP_Params(NamedTuple):
    raw_amplitude: jnp.ndarray
    raw_noise: jnp.ndarray
    scale: float
    beta: float
    lag: int
    alphabet_name: str = 'prot'


class ZeroMeanIMQHGP:
    def __init__(self, seqs_train: list[str], y_train: jnp.ndarray):
        self.set_training_data(seqs_train, y_train)

    def set_training_data(self, seqs_train: list[str], y_train: jnp.ndarray):
        self._seqs_train = seqs_train
        self._y_train = jnp.asarray(y_train)

    def _kernel_matrix(self, A: list[str], B: list[str], params: IMQHGP_Params):
        return jnp.asarray(imq_hamming_kernel(
            A, B,
            alphabet_name=params.alphabet_name,
            scale=params.scale,
            beta=params.beta,
            lag=params.lag
        ))

    def marginal_log_likelihood(self, params: IMQHGP_Params) -> jnp.ndarray:
        K_train = self._kernel_matrix(self._seqs_train, self._seqs_train, params)
        return kgp.mll_train(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=K_train,
            y_train=self._y_train
        )

    def predict_f(self, params: IMQHGP_Params, seqs_test: list[str], full_covar: bool = True):
        return kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._kernel_matrix(self._seqs_train, self._seqs_train, params),
            k_test_train=self._kernel_matrix(seqs_test, self._seqs_train, params),
            k_test_test=self._kernel_matrix(seqs_test, seqs_test, params) if full_covar else jnp.diag(self._kernel_matrix(seqs_test, seqs_test, params)),
            y_train=self._y_train,
            full_covar=full_covar
        )

    def predict_y(self, params: IMQHGP_Params, seqs_test: list[str], full_covar: bool = True):
        mean, covar = self.predict_f(params, seqs_test, full_covar)
        noise = TRANSFORM(params.raw_noise)
        if full_covar:
            covar += jnp.eye(len(seqs_test)) * noise
        else:
            covar += noise
        return mean, covar
