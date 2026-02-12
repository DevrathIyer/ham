"""Hopfield network conversion utilities.

This module provides functions to convert trained soft-attention HNL/HCL layers
into binary Hopfield networks for inference.
"""

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
    num_iterations: int = 5,
) -> HopfieldHNL:
    # Create binary projection for each head: (num_heads, binary_dim, head_dim)
    keys = jax.random.split(key, hnl.num_heads)
    bin_proj = [
        jnp.linalg.qr(jax.random.normal(key, (binary_dim, hnl.head_dim)))[0]
        for key in keys
    ]

    bin_inv = jnp.stack([proj.T for proj in bin_proj])
    bin_proj = jnp.stack(bin_proj)

    # Binarize memories and project to binary space
    # memories: (num_heads, num_memories, head_dim)
    # binary_memories = binarize_memories(hnl.memories)
    binary_memories = hnl.memories

    # Project each memory through bin_proj to get patterns in binary_dim space
    # For each head: (num_memories, head_dim) @ (binary_dim, head_dim).T -> (num_memories, binary_dim)
    # Then binarize the projected patterns
    def compute_weights_for_head(bin_proj_h, mem_h):
        # mem_h = mem_h / jnp.linalg.norm(mem_h, axis=-1, keepdims=True)
        binary_patterns = jnp.sign(mem_h @ bin_proj_h.T)  # (num_memories, binary_dim)

        """
        binary_patterns = binary_patterns / jnp.linalg.norm(
            binary_patterns, axis=-1, keepdims=True
        )
        """
        # return compute_hopfield_weights(binary_patterns)
        return binary_patterns

    weight_matrix = jax.vmap(compute_weights_for_head)(bin_proj, binary_memories)

    memories = hnl.memories / jnp.linalg.norm(hnl.memories, axis=-1, keepdims=True)

    mem_back = jnp.einsum("hmb,hdb->hmd", weight_matrix, bin_inv)
    mem_back /= jnp.linalg.norm(mem_back, axis=-1, keepdims=True)
    err = jnp.mean(jnp.linalg.norm(memories - mem_back, axis=-1))

    return HopfieldHNL(
        in_feats=hnl.in_feats,
        out_feats=hnl.out_feats,
        num_heads=hnl.num_heads,
        head_dim=hnl.head_dim,
        is_class=hnl.is_class,
        query_proj=hnl.query_proj,
        binary_dim=binary_dim,
        num_iterations=num_iterations,
        bin_proj=bin_proj,
        bin_inv=bin_inv,
        weight_matrix=weight_matrix,
        memories=hnl.memories,
    )


def convert_hnm_to_hopfield(
    hnm: HNM,
    key: jax.Array,
    binary_dim: int,
    num_iterations: int = 5,
) -> HopfieldHNM:
    keys = jax.random.split(key, len(hnm.layers))
    hopfield_layers = [
        convert_hnl_to_hopfield(layer, k, binary_dim, num_iterations)
        for layer, k in zip(hnm.layers, keys)
    ]
    return HopfieldHNM(*hopfield_layers)
