"""Hopfield network conversion utilities.

This module provides functions to convert trained soft-attention HNL/HCL layers
into binary Hopfield networks for inference.
"""

import jax
import jax.numpy as jnp

from models import (HCL, HCM, HNL, HNM, HopfieldHCL, HopfieldHCM, HopfieldHNL,
                    HopfieldHNM)


def create_binary_projection(
    key: jax.Array, input_dim: int, binary_dim: int
) -> jax.Array:
    # Random binary matrix scaled for variance preservation
    proj = jax.random.normal(key, (binary_dim, input_dim))
    proj = proj / jnp.linalg.norm(proj)  # Scale for variance preservation
    return proj  #


def binarize_memories(memories: jax.Array) -> jax.Array:
    return jnp.sign(memories)


def compute_hopfield_weights(binary_memories: jax.Array) -> jax.Array:
    """Compute Hopfield weight matrix using Hebbian learning.

    W = (1/N) * sum_i(m_i @ m_i^T)

    Args:
        binary_memories: Binary patterns of shape (num_memories, binary_dim)

    Returns:
        Weight matrix of shape (binary_dim, binary_dim)
    """
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
    """Run Hopfield network for fixed number of iterations.

    Args:
        initial_state: Initial binary state of shape (binary_dim,)
        weights: Hopfield weight matrix of shape (binary_dim, binary_dim)
        num_iterations: Number of update iterations

    Returns:
        Final binary state after iterations
    """

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
    binary_dim: int | None = None,
    num_iterations: int = 5,
) -> HopfieldHNL:
    """Convert a trained HNL layer to a binary Hopfield layer.

    Args:
        hnl: Trained HNL layer
        key: JAX random key for generating projection matrices
        binary_dim: Dimension of binary space (defaults to num_memories)
        num_iterations: Number of Hopfield update iterations

    Returns:
        HopfieldHNL layer with binary dynamics
    """

    if binary_dim is None:
        binary_dim = hnl.num_memories

    # Create binary projection for each head: (num_heads, binary_dim, head_dim)
    keys = jax.random.split(key, hnl.num_heads)
    bin_proj = jnp.stack(
        [create_binary_projection(k, hnl.head_dim, binary_dim) for k in keys]
    )

    # Binarize memories and project to binary space
    # memories: (num_heads, num_memories, head_dim)
    binary_memories = binarize_memories(hnl.memories)

    # Project each memory through bin_proj to get patterns in binary_dim space
    # For each head: (num_memories, head_dim) @ (binary_dim, head_dim).T -> (num_memories, binary_dim)
    # Then binarize the projected patterns
    def compute_weights_for_head(bin_proj_h, mem_h):
        # Project memories to binary space: (num_memories, head_dim) @ (head_dim, binary_dim)
        binary_patterns = jnp.sign(mem_h @ bin_proj_h.T)  # (num_memories, binary_dim)
        # return compute_hopfield_weights(binary_patterns)
        return binary_patterns

    weight_matrix = jax.vmap(compute_weights_for_head)(bin_proj, binary_memories)

    return HopfieldHNL(
        in_features=hnl.in_features,
        out_features=hnl.out_features,
        num_heads=hnl.num_heads,
        head_dim=hnl.head_dim,
        binary_dim=binary_dim,
        num_iterations=num_iterations,
        use_activation=hnl.use_activation,
        query_proj=hnl.query_proj,
        layer_norm=hnl.layer_norm,
        bin_proj=bin_proj,
        weight_matrix=weight_matrix,
    )


def convert_hcl_to_hopfield(
    hcl: HCL,
    key: jax.Array,
    binary_dim: int | None = None,
    num_iterations: int = 5,
) -> HopfieldHCL:
    """Convert a trained HCL layer to a binary Hopfield layer.

    Args:
        hcl: Trained HCL layer
        key: JAX random key for generating projection matrices
        binary_dim: Dimension of binary space (defaults to num_memories)
        num_iterations: Number of Hopfield update iterations

    Returns:
        HopfieldHCL layer with binary dynamics
    """
    if binary_dim is None:
        binary_dim = hcl.num_memories

    # Create binary projection for each channel: (out_channels, binary_dim, 1)
    keys = jax.random.split(key, hcl.out_channels)
    bin_proj = jnp.stack(
        [create_binary_projection(k, 1, binary_dim)[:, :, None] for k in keys]
    )
    bin_proj = bin_proj.squeeze(-1)  # (out_channels, binary_dim, 1)

    # For HCL, memories are (out_channels, num_memories)
    # Each channel has scalar memories, so we treat them as patterns directly
    binary_memories = binarize_memories(hcl.memories)  # (out_channels, num_memories)

    # Compute Hopfield weights for each channel
    # Treat each memory slot as a dimension of the pattern space
    def compute_weights_for_channel(mem_c):
        # mem_c: (num_memories,) - treat as a single pattern or multiple 1D patterns
        # For scalar attention, we interpret: num_memories patterns of dimension 1
        # Or: 1 pattern of dimension num_memories
        # Let's use: binary_dim patterns (projected from memories)
        # Actually, for HCL we just use the binarized memories directly
        # Memory shape is (num_memories,), treat as (num_memories, 1) patterns
        patterns = mem_c.reshape(-1, 1)  # (num_memories, 1)
        # But we need binary_dim x binary_dim weight matrix
        # Use outer product of binarized memory vector with itself
        return (
            jnp.outer(mem_c, mem_c) / mem_c.shape[0]
            - jnp.eye(mem_c.shape[0])
            * (jnp.outer(mem_c, mem_c) / mem_c.shape[0]).diagonal()[:, None]
        )

    weight_matrix = jax.vmap(compute_weights_for_channel)(binary_memories)

    # Adjust bin_proj shape to match expected dimensions
    # For HCL, we project scalar queries to binary_dim space
    keys = jax.random.split(key, hcl.out_channels)
    bin_proj = jnp.stack(
        [create_binary_projection(k, 1, binary_dim) for k in keys]
    )  # (out_channels, binary_dim, 1)

    return HopfieldHCL(
        in_channels=hcl.in_channels,
        out_channels=hcl.out_channels,
        binary_dim=binary_dim,
        num_iterations=num_iterations,
        kernel_size=hcl.kernel_size,
        use_activation=hcl.use_activation,
        query_conv=hcl.query_conv,
        layer_norm=hcl.layer_norm,
        bin_proj=bin_proj,
        weight_matrix=weight_matrix,
    )


def convert_hnm_to_hopfield(
    hnm: HNM,
    key: jax.Array,
    binary_dim: int | None = None,
    num_iterations: int = 5,
) -> HopfieldHNM:
    """Convert a trained HNM model to a binary Hopfield model.

    Args:
        hnm: Trained HNM model
        key: JAX random key
        binary_dim: Dimension of binary space for each layer
        num_iterations: Number of Hopfield iterations per layer

    Returns:
        HopfieldHNM model
    """
    keys = jax.random.split(key, len(hnm.layers))
    hopfield_layers = [
        convert_hnl_to_hopfield(layer, k, binary_dim, num_iterations)
        for layer, k in zip(hnm.layers, keys)
    ]
    return HopfieldHNM(*hopfield_layers)


def convert_hcm_to_hopfield(
    hcm: HCM,
    key: jax.Array,
    binary_dim: int | None = None,
    num_iterations: int = 5,
) -> HopfieldHCM:
    """Convert a trained HCM model to a binary Hopfield model.

    Args:
        hcm: Trained HCM model
        key: JAX random key
        binary_dim: Dimension of binary space for each layer
        num_iterations: Number of Hopfield iterations per layer

    Returns:
        HopfieldHCM model
    """
    n_conv = len(hcm.conv_layers)
    n_fc = len(hcm.fc_layers)
    keys = jax.random.split(key, n_conv + n_fc)

    hopfield_conv_layers = [
        convert_hcl_to_hopfield(layer, k, binary_dim, num_iterations)
        for layer, k in zip(hcm.conv_layers, keys[:n_conv])
    ]
    hopfield_fc_layers = [
        convert_hnl_to_hopfield(layer, k, binary_dim, num_iterations)
        for layer, k in zip(hcm.fc_layers, keys[n_conv:])
    ]

    return HopfieldHCM(hopfield_conv_layers, hopfield_fc_layers, hcm.pool)
