import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from algorithm.rbf_gp import ZeroMeanRBFGP, RBFParams, rbf_kernel
from algorithm.lse import LSE

# Toy function and data
f = lambda x: np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x)
X_pool = np.linspace(0, 1, 100).reshape(-1, 1)
y_real = f(X_pool).ravel()
y_class = (y_real >= 0).astype(int)
init_indices = [10, 50, 90]

# GP model
params = RBFParams(raw_amplitude=jnp.log(1.0), raw_noise=jnp.log(1e-2), lengthscale=0.1)
model = ZeroMeanRBFGP(X_train=X_pool[init_indices], y_train=y_class[init_indices])
K_full = rbf_kernel(X_pool, X_pool, lengthscale=params.lengthscale)

# Run LSE
lse = LSE(
    model=model,
    params=params,
    X_pool=X_pool,
    y_pool=y_class,
    init_indices=init_indices,
    h=0.5,
    epsilon=0.05,
    delta=0.01,
    rule="amb",
    verbose=True,
    precomputed_kernel=K_full
)
lse.run(max_iter=20)

# Plotting Level Set Estimation Results
x_vals = X_pool.ravel()
plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_real, label='f(x)', color='black')
plt.axhline(0.5, color='gray', linestyle='--', label='Level h=0.5')

plt.scatter(x_vals[lse.ht], y_real[lse.ht], color='red', label='Classified H (>= h)', s=50)
plt.scatter(x_vals[lse.lt], y_real[lse.lt], color='blue', label='Classified L (< h)', s=50)
plt.scatter(x_vals[init_indices], y_real[init_indices], color='green', label='Initial Queries', marker='x', s=100)

plt.legend()
plt.title("LSE Estimation on Sum of Sin & Cos Function")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.tight_layout()
plt.savefig("lse_benchmark.pdf", dpi=300)
print("Plot saved to: lse_benchmark.pdf")

# 2. Quantitative Check
true_H = np.where(y_class == 1)[0]
true_L = np.where(y_class == 0)[0]

correct_H = len(set(lse.ht) & set(true_H))
correct_L = len(set(lse.lt) & set(true_L))

print(f"[✔] True positives (classified H): {correct_H} / {len(lse.ht)}")
print(f"[✔] True negatives (classified L): {correct_L} / {len(lse.lt)}")

# Refit GP on all labelled data (H ∪ L)
labelled_indices = lse.ht + lse.lt
X_labeled = X_pool[labelled_indices]
y_labeled = y_class[labelled_indices]
model = ZeroMeanRBFGP(X_train=X_labeled, y_train=y_labeled)

# Predictive mean and variance on full grid
mu_pred, var_pred = model.predict(X_pool, params=params)
std_pred = np.sqrt(var_pred)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_real, label='True $f(x)$', color='black')
plt.plot(x_vals, mu_pred, label='GP mean $\mu(x)$', color='orange', linestyle='--')
plt.fill_between(
    x_vals,
    mu_pred - 2 * std_pred,
    mu_pred + 2 * std_pred,
    alpha=0.3,
    color='orange',
    label='GP $\pm 2\sigma$'
)
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold $h=0.5$')

# Mark labelled points
plt.scatter(x_vals[lse.ht], y_real[lse.ht], color='red', label='Classified H (≥ h)', s=50)
plt.scatter(x_vals[lse.lt], y_real[lse.lt], color='blue', label='Classified L (< h)', s=50)
plt.scatter(x_vals[init_indices], y_real[init_indices], color='green', marker='x', s=100, label='Initial Queries')

plt.title("GP Posterior Mean & Uncertainty After LSE")
plt.xlabel("x")
plt.ylabel("f(x), GP $\mu(x)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("gp_posterior_lse.pdf")
plt.show()

