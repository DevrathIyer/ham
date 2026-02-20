import jax
import jax.numpy as jnp

from models import HNL, HNM, HopfieldHNL, HopfieldHNM


def compute_hopfield_weights(binary_memories: jax.Array) -> jax.Array:
    num_memories, binary_dim = binary_memories.shape
    # Hebbian learning: W = (1/N) * M^T @ M
    W = binary_memories.T @ binary_memories / num_memories
    # Zero diagonal (no self-connections in classic Hopfield)
    W = W - jnp.diag(jnp.diag(W))
    return W


def hopfield_update(state: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sign(weights @ state)


def hopfield_iterate(
    initial_state: jax.Array, weights: jax.Array, num_iterations: int
) -> jax.Array:
    def body_fn(state, _):
        new_state = hopfield_update(state, weights)
        return new_state, None

    final_state, _ = jax.lax.scan(body_fn, initial_state, None, length=num_iterations)
    return final_state


# =============================================================================
# Conversion Functions
# =============================================================================


def convert_hnl_to_hopfield(
    hnl: HNL,
    key: jax.Array,
    binary_dim: int,
    active_dims: int,
    num_iterations: int = 5,
) -> HopfieldHNL:
    # Create binary projection for each head: (num_heads, binary_dim, head_dim)
    keys = jax.random.split(key, hnl.num_heads)

    d = hnl.head_dim
    Ws = []
    for key in keys:
        """
        def generate_row(row_key):
            # Pick w unique indices from D for each row
            indices = jax.random.choice(row_key, d, shape=(active_dims,), replace=False)
            row = jnp.zeros(d).at[indices].set(1.0)
            return row

        key, weight_key = jax.random.split(key)
        row_keys = jax.random.split(key, binary_dim)
        #Ws.append(jax.vmap(generate_row)(row_keys) * weights)
        """
        key, weight_key, row_key = jax.random.split(key, 3)
        weights = jax.random.normal(weight_key, shape=(binary_dim, d))
        rows = jax.random.bernoulli(
            row_key, active_dims / binary_dim, shape=(binary_dim, d)
        )
        Ws.append(weights * rows)

    bin_proj = jnp.asarray(Ws)

    binary_memories = hnl.memories

    # Project each memory through bin_proj to get patterns in binary_dim space
    # For each head: (num_memories, head_dim) @ (binary_dim, head_dim).T -> (num_memories, binary_dim)
    # Then binarize the projected patterns
    def compute_weights_for_head(bin_proj_h, mem_h):
        mem_h = mem_h / jnp.linalg.norm(mem_h, axis=-1, keepdims=True)
        bin_scores = jnp.einsum("kd,md->mk", bin_proj_h, mem_h)
        vals, indices = jax.lax.top_k(bin_scores, active_dims, axis=-1)

        mask = jnp.zeros_like(bin_scores)
        mem_idx = jnp.arange(mem_h.shape[0])[:, None]
        mem_h_bits = mask.at[mem_idx, indices].set(1.0)

        true_sims = jnp.einsum("j,ij->i", mem_h[0], mem_h)
        ham_sims = jnp.einsum("j,ij->i", mem_h_bits[0], mem_h_bits) / active_dims
        jac_sims = (
            ham_sims
            * active_dims
            / jnp.sum(jnp.maximum(mem_h_bits[0][None, :], mem_h_bits), axis=-1)
        )
        # jax.debug.print("True sims: {}", true_sims)
        # jax.debug.print("Ham sims: {}", ham_sims)
        # jax.debug.print("Jac sims: {}", jnp.sin(jac_sims * jnp.pi / 2))
        """
        mem_h_bin = jnp.einsum("mk,kd->md", mem_h_bits, bin_proj_h)
        norm = jnp.linalg.norm(mem_h_bin, axis=-1, keepdims=True)
        mem_h_bin = jnp.where(norm > 0, mem_h_bin / norm, mem_h_bin)

        jax.debug.print(
            "Error: {}", jnp.mean(jnp.linalg.norm(mem_h_bin - mem_h, axis=-1), axis=-1)
        )
        """

        return mem_h_bits

    weight_matrix = jax.vmap(compute_weights_for_head)(bin_proj, binary_memories)

    return HopfieldHNL(
        in_feats=hnl.in_feats,
        out_feats=hnl.out_feats,
        num_mems=hnl.num_mems,
        num_heads=hnl.num_heads,
        head_dim=hnl.head_dim,
        is_class=hnl.is_class,
        query_proj=hnl.query_proj,
        binary_dim=binary_dim,
        active_dims=active_dims,
        num_iterations=num_iterations,
        bin_proj=bin_proj,
        weight_matrix=weight_matrix,
        memories=hnl.memories,
    )


def convert_hnm_to_hopfield(
    hnm: HNM,
    key: jax.Array,
    binary_dim: int,
    active_dims: int,
    num_iterations: int = 5,
) -> HopfieldHNM:
    keys = jax.random.split(key, len(hnm.layers))
    hopfield_layers = [
        convert_hnl_to_hopfield(layer, k, binary_dim, active_dims, num_iterations)
        for layer, k in zip(hnm.layers, keys)
    ]
    return HopfieldHNM(*hopfield_layers)
