import pandas as pd
import numpy as np
import jax.numpy as jnp
import logging

from supplementary_info.imq_gp import imq_hamming_kernel, ZeroMeanIMQHGP, IMQHGP_Params
from supplementary_info.lse import LSE

# Configuration
DATA_PATH = 'supplementary_info/mutation_with_sequences.csv'
INIT_SIZE = 10
KERNEL_PARAMS = {
    'scale': 1.0,
    'beta': 0.5,
    'lag': 1,
    'alphabet_name': 'prot'
}
EPSILON = 0.05
OMEGA = 0.40

def load_data(path):
    df = pd.read_csv(path).dropna(subset=["mut_seq", "Type"])
    df = df.sample(n=1400, random_state=42).reset_index(drop=True)
    seqs = df["mut_seq"].tolist()
    labels = df["Type"].apply(lambda x: 1.0 if x.lower() == "driver" else 0.0).values
    return seqs, labels, df  

def precompute_kernel(seqs):
    return imq_hamming_kernel(seqs, seqs, **KERNEL_PARAMS)

def evaluate_accuracy(lse, df):
    true_labels = df["Type"].apply(lambda x: 1 if x.lower() == "driver" else 0).values
    queried_indices = lse.vt
    pred_labels = [1 if i in lse.ht else 0 for i in queried_indices]
    true_labels_queried = [true_labels[i] for i in queried_indices]
    correct = [int(p == t) for p, t in zip(pred_labels, true_labels_queried)]
    return sum(correct) / len(queried_indices) if queried_indices else 0.0

def main():
    logging.basicConfig(level=logging.INFO)
    print("Loading data...")
    seqs, labels, df = load_data(DATA_PATH)
    full_kernel = precompute_kernel(seqs)

    results = []

    for max_iter in range(20, 1401, 10):
        print(f"\n>> Running LSE algorithm with MAX_ITER = {max_iter}")
        init_idx = np.random.choice(len(seqs), size=INIT_SIZE, replace=False)

        gp_model = ZeroMeanIMQHGP([seqs[i] for i in init_idx], jnp.array(labels[init_idx]))

        params = IMQHGP_Params(
            raw_amplitude=jnp.array(1.0),
            raw_noise=jnp.array(1e-2),
            **KERNEL_PARAMS
        )

        lse = LSE(
            model=gp_model,
            params=params,
            X_pool=seqs,
            y_pool=labels,
            init_indices=init_idx,
            omega=OMEGA,
            epsilon=EPSILON,
            rule="amb",
            verbose=False,
            precomputed_kernel=full_kernel
        )

        lse.run(max_iter=max_iter)
        accuracy = evaluate_accuracy(lse, df)
        results.append({"MAX_ITER": max_iter, "Accuracy": accuracy})
        print(f"  → Accuracy: {accuracy:.2%}")

    pd.DataFrame(results).to_csv("lse_accuracy_vs_iterations.csv", index=False)
    print("Saved to: lse_accuracy_vs_iterations.csv")

if __name__ == "__main__":
    main()
