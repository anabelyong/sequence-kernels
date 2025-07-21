import jax
import jax.numpy as jnp

ALPHA = 1.833
BETA = 0.1018
GAMMA = 0.000399

# Amino acid properties: [c, p, v]
_amino_acids_list = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"
]

_props_array = jnp.array([
    [0.00,  8.1,  31],
    [0.65, 10.5, 124],
    [1.33, 11.6,  56],
    [1.38, 13.0,  54],
    [2.75,  5.5,  55],
    [0.89, 10.5,  85],
    [0.92, 12.3,  83],
    [0.74,  9.0,   3],
    [0.58, 10.4,  96],
    [0.00,  5.2, 111],
    [0.00,  4.9, 111],
    [0.33, 11.3, 119],
    [0.00,  5.7, 105],
    [0.00,  5.2, 132],
    [0.39,  8.0,  32.5],
    [1.42,  9.2,  32],
    [0.71,  8.6,  61],
    [0.13,  5.4, 170],
    [0.20,  6.2, 136],
    [0.00,  5.9,  84],
    [0.00,  0.0,   0],  # padding
])

# Create lookup: aa -> index
aa_to_idx = {aa: i for i, aa in enumerate(_amino_acids_list)}

def _pad_sequences(seqs, pad_char="-"):
    max_len = max(len(seq) for seq in seqs)
    return [seq.ljust(max_len, pad_char) for seq in seqs]

def _encode_sequence(seq: str):
    return jnp.array([aa_to_idx.get(aa, aa_to_idx["-"]) for aa in seq], dtype=jnp.int32)

def _batch_encode(seqs):
    return jnp.stack([_encode_sequence(seq) for seq in seqs], axis=0)

@jax.jit
def grantham_distance_matrix(c_x, a_x, c_y, a_y):
    """
    Args:
        c_x, a_x: (N1, L) int32
        c_y, a_y: (N2, L) int32

    Returns:
        (N1, N2) distance matrix
    """
    props = _props_array  # shape (21, 3)

    def seq_props(idx_seq):
        return props[idx_seq]  # (L, 3)

    cx_props = jax.vmap(seq_props)(c_x)
    ax_props = jax.vmap(seq_props)(a_x)
    cy_props = jax.vmap(seq_props)(c_y)
    ay_props = jax.vmap(seq_props)(a_y)

    cxp = cx_props[:, None, :, :]  # (N1, 1, L, 3)
    axp = ax_props[:, None, :, :]
    cyp = cy_props[None, :, :, :]  # (1, N2, L, 3)
    ayp = ay_props[None, :, :, :]

    # Total shape: (N1, N2, L, 3)
    dc = cxp[..., 0] - cyp[..., 0]
    dp = cxp[..., 1] - cyp[..., 1]
    dv = cxp[..., 2] - cyp[..., 2]

    dac = axp[..., 0] - ayp[..., 0]
    dap = axp[..., 1] - ayp[..., 1]
    dav = axp[..., 2] - ayp[..., 2]

    # Compute pairwise sum of distances for context + action
    dist_context = jnp.sqrt(ALPHA * dc**2 + BETA * dp**2 + GAMMA * dv**2)
    dist_action = jnp.sqrt(ALPHA * dac**2 + BETA * dap**2 + GAMMA * dav**2)

    return jnp.sum(dist_context + dist_action, axis=-1)  # sum over L

# Wrapper
def contextual_grantham_distance_matrix(contexts, actions, contexts_y, actions_y):
    contexts = _pad_sequences(contexts)
    actions = _pad_sequences(actions)
    contexts_y = _pad_sequences(contexts_y)
    actions_y = _pad_sequences(actions_y)

    c_x = _batch_encode(contexts)  # (N1, L)
    a_x = _batch_encode(actions)
    c_y = _batch_encode(contexts_y)  # (N2, L)
    a_y = _batch_encode(actions_y)

    return grantham_distance_matrix(c_x, a_x, c_y, a_y)

"""
#Test cases!
if __name__ == "__main__":
    a1 = "R"  
    a2 = "H"  
    d = grantham_distance(a1, a2)
    print(f"Grantham distance between {a1} ({amino_acids[a1]['name']}) and {a2} ({amino_acids[a2]['name']}): {d:.4f}")
"""
if __name__ == "__main__":
    c_x = ["AAA", "AAB"]
    a_x = ["VKTA", "MRT"]
    dmat = contextual_grantham_distance_matrix(c_x, a_x, c_x, a_x)
    print("Contextual Grantham matrix:\n", dmat)
