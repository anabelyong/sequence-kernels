import pandas as pd
import numpy as np
import jax.numpy as jnp
import logging

from supplementary_info.imq_gp import imq_hamming_kernel, ZeroMeanIMQHGP, IMQHGP_Params
from supplementary_info.lse import LSE

# Configuration
DATA_PATH = 'supplementary_info/mutation_with_sequences.csv'
INIT_SIZE = 10
MAX_ITER = 300
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
    df = df.sample(n=300, random_state=42).reset_index(drop=True)
    seqs = df["mut_seq"].tolist()
    labels = df["Type"].apply(lambda x: 1.0 if x.lower() == "driver" else 0.0).values
    return seqs, labels, df  

def precompute_kernel(seqs):
    return imq_hamming_kernel(seqs, seqs, **KERNEL_PARAMS)

def evaluate_lse_results(lse, df):
    true_labels = df["Type"].apply(lambda x: 1 if x.lower() == "driver" else 0).values
    mutations = df["Mutation"].values if "Mutation" in df.columns else [f"mut_{i}" for i in range(len(df))]

    queried_indices = lse.vt
    pred_labels = [1 if i in lse.ht else 0 for i in queried_indices]
    true_labels_queried = [true_labels[i] for i in queried_indices]

    correct = [int(p == t) for p, t in zip(pred_labels, true_labels_queried)]
    correct_mutations = [mutations[i] for i, is_correct in zip(queried_indices, correct) if is_correct]

    print("\nQueried mutation predictions (index, mutation, true, predicted, correct?):")
    for idx, pred, true in zip(queried_indices, pred_labels, true_labels_queried):
        tag = "✓" if pred == true else "✗"
        print(f"  idx={idx:3d}, {mutations[idx]:>10}, true={true}, pred={pred}  {tag}")

    num_correct = sum(correct)
    print(f"\nCorrect predictions: {num_correct} / {len(queried_indices)}")
    print(f"Accuracy on queried points: {num_correct / len(queried_indices):.2%}")

def main():
    logging.basicConfig(level=logging.INFO)
    print("Loaded data.")
    seqs, labels, df = load_data(DATA_PATH)
    init_idx = np.random.choice(len(seqs), size=INIT_SIZE, replace=False)

    print("Precomputing full kernel matrix")
    full_kernel = precompute_kernel(seqs)

    print("Initializing GP model...")
    gp_model = ZeroMeanIMQHGP([seqs[i] for i in init_idx], jnp.array(labels[init_idx]))

    params = IMQHGP_Params(
        raw_amplitude=jnp.array(1.0),
        raw_noise=jnp.array(1e-2),  
        **KERNEL_PARAMS
    )

    print(">> Running LSE algorithm...")
    lse = LSE(
        model=gp_model,
        params=params,
        X_pool=seqs,
        y_pool=labels,
        init_indices=init_idx,
        omega=OMEGA,
        epsilon=EPSILON,
        rule="amb",
        verbose=True,
        precomputed_kernel=full_kernel 
    )

    lse.run(max_iter=MAX_ITER)
    print("LSE algorithm completed.")

    evaluate_lse_results(lse, df)

if __name__ == "__main__":
    main()
