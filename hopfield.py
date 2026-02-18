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
        blocks = []
        for _ in range(int(jnp.ceil(binary_dim / d))):
            n_key, key = jax.random.split(key)
            Q, _ = jnp.linalg.qr(jax.random.normal(n_key, (d, d)))
            blocks.append(Q)
        Ws.append(jnp.vstack(blocks)[:binary_dim, :])
        """
        W = jax.random.normal(key, (binary_dim, d))
        W /= jnp.linalg.norm(W, axis=-1, keepdims=True)
        Ws.append(W)

    bin_proj = jnp.asarray(Ws)

    binary_memories = hnl.memories

    # Project each memory through bin_proj to get patterns in binary_dim space
    # For each head: (num_memories, head_dim) @ (binary_dim, head_dim).T -> (num_memories, binary_dim)
    # Then binarize the projected patterns
    def compute_weights_for_head(bin_proj_h, mem_h):
        mem_h = mem_h / jnp.linalg.norm(mem_h, axis=-1, keepdims=True)
        bin_scores = jnp.einsum("kd,md->mk", bin_proj_h, mem_h)
        _, indices = jax.lax.top_k(bin_scores, active_dims, axis=-1)
        mask = jnp.zeros_like(bin_scores)
        mem_h_bits = mask.at[jnp.arange(mem_h.shape[0])[:, None], indices].set(1.0)

        mem_h_bin = jnp.einsum("mk,kd->md", mem_h_bits, bin_proj_h)
        mem_h_bin /= jnp.linalg.norm(mem_h_bin, axis=-1, keepdims=True)
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
