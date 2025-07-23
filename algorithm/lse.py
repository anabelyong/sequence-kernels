import jax.numpy as jnp
import numpy as np

class LSE:
    def __init__(self, model, params, X_pool, y_pool, init_indices,
                 h=0.5, omega=None, epsilon=0.05, delta=0.01,
                 rule='amb', verbose=True, precomputed_kernel=None):
        self.model = model
        self.params = params
        self.h = h
        self.omega = omega
        self.epsilon = epsilon
        self.delta = delta
        self.rule = rule
        self.verbose = verbose
        self.t = 1
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.full_kernel = precomputed_kernel

        self.vt = list(init_indices)
        self.ut = list(set(range(len(X_pool))) - set(self.vt))
        self.ht = []
        self.lt = []
        self.history = []

        if self.full_kernel is not None:
            self.K_pool_pool = self.full_kernel
        else:
            self.K_pool_pool = model._kernel_matrix(X_pool, X_pool, params)

        self._update_gp()

    def _beta(self):
        return 2 * np.log((np.pi**2 * (self.t + 1)**2) / (6 * self.delta))

    def _update_gp(self):
        X_train = [self.X_pool[i] for i in self.vt]
        y_train = self.y_pool[self.vt]
        self.model.set_training_data(X_train, y_train)

    def _threshold(self, mu):
        return self.omega * jnp.max(mu) if self.omega is not None else self.h

    def step(self):
        if not self.ut:
            return False
        
        k_tt = self.K_pool_pool[np.ix_(self.ut, self.ut)]
        k_tn = self.K_pool_pool[np.ix_(self.ut, self.vt)]
        k_nn = self.K_pool_pool[np.ix_(self.vt, self.vt)]

        mu, cov = self.model._predict_with_precomputed(
            self.params, k_nn, k_tn, k_tt, self.y_pool[self.vt], full_covar=True
        )

        mu = mu.ravel()
        std = jnp.sqrt(jnp.diag(cov)).ravel()
        b = np.sqrt(self._beta())
        lower = mu - b * std
        upper = mu + b * std

        h_current = self._threshold(mu)

        # Boolean masks for classification
        H_mask = (lower + self.epsilon > h_current)
        L_mask = (upper - self.epsilon <= h_current)

        # Classify based on masks
        idx_H = [self.ut[i] for i in np.where(np.array(H_mask))[0]]
        idx_L = [self.ut[i] for i in np.where(np.array(L_mask))[0]]

        self.ht.extend(idx_H)
        self.lt.extend(idx_L)
        newly_classified = set(idx_H + idx_L)

        # Update ut
        ut_new = [i for i in self.ut if i not in newly_classified]
        if not ut_new:
            return False

        # Keep only unclassified predictions
        keep_mask = ~(H_mask | L_mask)

        mu = mu[keep_mask]
        std = std[keep_mask]
        lower = lower[keep_mask]
        upper = upper[keep_mask]

        if mu.shape[0] != len(ut_new):
            raise RuntimeError("Mismatch between unclassified indices and prediction outputs.")

        # Select next query
        if self.rule == 'amb':
            ambiguity = np.minimum(upper - h_current, h_current - lower)
        elif self.rule == 'var':
            ambiguity = std
        elif self.rule == 'random':
            next_idx_local = np.random.randint(len(ut_new))
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

        if self.rule != 'random':
            next_idx_local = int(np.argmax(ambiguity))

        next_global_idx = ut_new[next_idx_local]

        # Log
        log_entry = {
            "iteration": self.t,
            "query_index": next_global_idx,
            "mu": float(mu[next_idx_local]),
            "std": float(std[next_idx_local]),
            "lower": float(lower[next_idx_local]),
            "upper": float(upper[next_idx_local]),
            "label": int(self.y_pool[next_global_idx]),
            "remaining": len(ut_new),
            "classified_H": len(self.ht),
            "classified_L": len(self.lt)
        }

        self.history.append(log_entry)

        if self.verbose:
            print(f"[t={self.t:02d}] Queried idx={next_global_idx}, y={log_entry['label']}, "
                  f"μ={log_entry['mu']:.3f}, σ={log_entry['std']:.3f}, "
                  f"CI=({log_entry['lower']:.3f}, {log_entry['upper']:.3f})")

        self.vt.append(next_global_idx)
        self.ut = [i for i in ut_new if i != next_global_idx]
        self._update_gp()
        self.t += 1
        return True

    def run(self, max_iter=100):
        for _ in range(max_iter):
            if not self.step():
                break
