import jax.numpy as jnp
import numpy as np

class LSE:
    def __init__(self, model, params, X_pool, y_pool, init_indices, h=0.5, epsilon=0.05, delta=0.01, verbose=True):
        self.model = model
        self.params = params
        self.h = h
        self.epsilon = epsilon
        self.delta = delta
        self.verbose = verbose
        self.t = 1
        self.X_pool = X_pool
        self.y_pool = y_pool

        self.vt = list(init_indices)                    # Visited (labeled)
        self.ut = list(set(range(len(X_pool))) - set(self.vt))  # Unlabeled pool
        self.ht = []  # High (driver)
        self.lt = []  # Low (passenger)

        self.history = []  # <- store logs per iteration

        self._update_gp()  # Fit GP on initial seed

    def _beta(self):
        return 2 * np.log((np.pi**2 * (self.t+1)**2) / (6 * self.delta))

    def _update_gp(self):
        X_train = [self.X_pool[i] for i in self.vt]
        y_train = self.y_pool[self.vt]
        self.model.set_training_data(X_train, y_train)

    def step(self):
        if not self.ut:
            return False  # Done

        X_test = [self.X_pool[i] for i in self.ut]
        mu, std = self.model.predict_f(self.params, X_test, full_covar=False)
        b = np.sqrt(self._beta())

        lower = mu - b * std
        upper = mu + b * std

        H_mask = (lower + self.epsilon > self.h)
        L_mask = (upper - self.epsilon <= self.h)

        idx_H = [self.ut[i] for i in np.where(H_mask)[0]]
        idx_L = [self.ut[i] for i in np.where(L_mask)[0]]

        self.ht.extend(idx_H)
        self.lt.extend(idx_L)

        newly_classified = set(idx_H + idx_L)
        self.ut = [i for i in self.ut if i not in newly_classified]

        if not self.ut:
            return False

        # Ambiguity-based selection
        ambiguity = np.minimum(upper - self.h, self.h - lower)
        next_idx_local = int(np.argmax(ambiguity))
        next_global_idx = self.ut[next_idx_local]

        # Logging info
        log_entry = {
            "iteration": self.t,
            "query_index": next_global_idx,
            "mu": float(mu[next_idx_local]),
            "std": float(std[next_idx_local]),
            "lower": float(lower[next_idx_local]),
            "upper": float(upper[next_idx_local]),
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
