import jax.numpy as jnp
import numpy as np

class LSE:
    def __init__(self, model, params, X_pool, y_pool, init_indices,
                 h=0.5, omega=None, epsilon=0.05, delta=0.01,
                 rule='amb', verbose=True):
        self.model = model
        self.params = params
        self.h = h
        self.omega = omega  # if set, overrides fixed h threshold
        self.epsilon = epsilon
        self.delta = delta
        self.rule = rule
        self.verbose = verbose
        self.t = 1
        self.X_pool = X_pool
        self.y_pool = y_pool

        self.vt = list(init_indices)
        self.ut = list(set(range(len(X_pool))) - set(self.vt))
        self.ht = []
        self.lt = []
        self.history = []

        # Precompute full kernel blocks involving test pool
        self.K_pool_pool = model._kernel_matrix(X_pool, X_pool, params)

        self._update_gp()

    def _beta(self):
        return 2 * np.log((np.pi**2 * (self.t+1)**2) / (6 * self.delta))

    def _update_gp(self):
        X_train = [self.X_pool[i] for i in self.vt]
        y_train = self.y_pool[self.vt]
        self.model.set_training_data(X_train, y_train)

    def _threshold(self, mu):
        return self.omega * jnp.max(mu) if self.omega is not None else self.h

    def step(self):
        if not self.ut:
            return False

        # Use cached kernel blocks
        k_tt = self.K_pool_pool[np.ix_(self.ut, self.ut)]
        k_tn = self.K_pool_pool[np.ix_(self.ut, self.vt)]
        k_nn = self.K_pool_pool[np.ix_(self.vt, self.vt)]

        mu, cov = self.model._predict_with_precomputed(
            self.params, k_nn, k_tn, k_tt, self.y_pool[self.vt], full_covar=False
        )

        std = jnp.sqrt(cov)
        b = np.sqrt(self._beta())
        lower = mu - b * std
        upper = mu + b * std

        h_current = self._threshold(mu)

        H_mask = (lower + self.epsilon > h_current)
        L_mask = (upper - self.epsilon <= h_current)

        idx_H = [self.ut[i] for i in np.where(H_mask)[0]]
        idx_L = [self.ut[i] for i in np.where(L_mask)[0]]

        self.ht.extend(idx_H)
        self.lt.extend(idx_L)
        newly_classified = set(idx_H + idx_L)
        self.ut = [i for i in self.ut if i not in newly_classified]

        if not self.ut:
            return False

        if self.rule == 'amb':
            ambiguity = np.minimum(upper - h_current, h_current - lower)
        elif self.rule == 'var':
            ambiguity = std
        elif self.rule == 'random':
            next_idx_local = np.random.randint(len(self.ut))
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

        if self.rule != 'random':
            next_idx_local = int(np.argmax(ambiguity))
        next_global_idx = self.ut[next_idx_local]

        mu_val = float(np.ravel(np.asarray(mu[next_idx_local]))[0])
        std_val = float(np.ravel(np.asarray(std[next_idx_local]))[0])
        lower_val = float(np.ravel(np.asarray(lower[next_idx_local]))[0])
        upper_val = float(np.ravel(np.asarray(upper[next_idx_local]))[0])

        log_entry = {
            "iteration": self.t,
            "query_index": next_global_idx,
            "mu": mu_val,
            "std": std_val,
            "lower": lower_val,
            "upper": upper_val,
            "label": int(self.y_pool[next_global_idx]),
            "remaining": len(self.ut),
            "classified_H": len(self.ht),
            "classified_L": len(self.lt)
        }

        self.history.append(log_entry)

        if self.verbose:
            print(f"[t={self.t:02d}] Queried idx={next_global_idx}, y={log_entry['label']}, "
                  f"μ={log_entry['mu']:.3f}, σ={log_entry['std']:.3f}, CI=({log_entry['lower']:.3f}, {log_entry['upper']:.3f})")

        self.vt.append(next_global_idx)
        self.ut.remove(next_global_idx)
        self._update_gp()
        self.t += 1
        return True

    def run(self, max_iter=100):
        for _ in range(max_iter):
            if not self.step():
                break
