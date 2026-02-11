"""Model definitions using Equinox."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable

    def __init__(self, *layers: eqx.nn.Linear, activation: Callable = jax.nn.relu):
        self.layers = layers
        self.activation = activation

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    @classmethod
    def create(
        cls,
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        *,
        activation: Callable = jax.nn.relu,
        key: jax.Array,
    ) -> "MLP":
        """Factory method for creating standard architectures."""
        dims = [in_features] + hidden_dims + [out_features]
        keys = jax.random.split(key, len(dims) - 1)
        layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)
        ]
        return cls(*layers, activation=activation)


class HNL(eqx.Module):
    """Hopfield Network Layer with Multi-Head External Attention."""

    in_feats: int
    out_feats: int
    num_mems: int
    num_heads: int
    head_dim: int
    temperature: float
    query_proj: eqx.nn.Linear
    memories: jax.Array
    layer_norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_mems: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        temp: float = 1e-2,
        *,
        key: jax.Array,
    ):
        assert (
            out_feats % num_heads == 0
        ), f"out_features ({out_feats}) must be divisible by num_heads ({num_heads})"

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_mems = num_mems
        self.num_heads = num_heads
        self.head_dim = out_feats // num_heads
        self.temperature = temp
        print(f"N_heads is {self.num_heads}")

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.query_proj = eqx.nn.Linear(in_feats, out_feats, use_bias=False, key=k1)
        self.memories = (
            jax.random.normal(k3, (num_heads, num_mems, self.head_dim)) * 0.02
        )
        self.layer_norm = eqx.nn.LayerNorm(out_feats)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jax.Array, hard: bool, key: jax.Array) -> jax.Array:
        q = self.query_proj(x)
        q = q.reshape(self.num_heads, self.head_dim)

        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        # mem_norm = self.memories / jnp.sum(self.memories, axis=-1, keepdims=True)
        mem_norm = self.memories / jnp.linalg.norm(
            self.memories, axis=-1, keepdims=True
        )

        attn_scores = jnp.einsum("hd,hmd->hm", q_norm, mem_norm)
        if hard:
            top_mems = jnp.argmax(attn_scores, axis=-1)
            out = self.memories[jnp.arange(self.num_heads), top_mems]
        else:
            attn_weights = jax.nn.softmax(attn_scores / self.temperature, axis=-1)
            out = jnp.einsum("hm,hmd->hd", attn_weights, mem_norm)

        out = out.reshape(self.out_feats)
        out = self.layer_norm(out)
        out = self.dropout(out, key=key)
        return out


class HNM(eqx.Module):
    layers: tuple[HNL, ...]

    def __init__(self, layers: HNL):
        self.layers = layers

    def __call__(self, x: jax.Array, key: jax.Array, hard: bool = False) -> jax.Array:
        keys = jax.random.split(key, len(self.layers))
        for layer, k in zip(self.layers, keys):
            x = layer(x, hard, key=k)
        return x


class CNN(eqx.Module):
    conv_layers: tuple[eqx.nn.Conv2d, ...]
    fc_layers: tuple[eqx.nn.Linear, ...]
    pool: eqx.nn.MaxPool2d

    def __init__(
        self,
        conv_layers: list[eqx.nn.Conv2d],
        fc_layers: list[eqx.nn.Linear],
        pool: eqx.nn.MaxPool2d | None = None,
    ):
        self.conv_layers = tuple(conv_layers)
        self.fc_layers = tuple(fc_layers)
        self.pool = (
            pool if pool is not None else eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        for conv in self.conv_layers:
            x = self.pool(jax.nn.relu(conv(x)))
        x = x.reshape(-1)
        for fc in self.fc_layers[:-1]:
            x = jax.nn.relu(fc(x))
        return self.fc_layers[-1](x)

    @classmethod
    def create(
        cls,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
        *,
        key: jax.Array,
    ) -> "CNN":
        """Factory method for creating standard architectures."""
        keys = jax.random.split(key, 5)
        conv_layers = [
            eqx.nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, key=keys[0]),
            eqx.nn.Conv2d(64, 64, kernel_size=3, padding=1, key=keys[1]),
        ]
        final_size = image_size // 4  # After 2 max pools
        fc_layers = [
            eqx.nn.Linear(64 * final_size * final_size, 128, key=keys[3]),
            eqx.nn.Linear(128, num_classes, key=keys[4]),
        ]
        return cls(conv_layers, fc_layers)


class ResidualBlock(eqx.Module):
    """Residual block with skip connection."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm
    shortcut: eqx.nn.Conv2d | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 3)

        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, key=keys[0]
        )
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, key=keys[1]
        )
        self.norm1 = eqx.nn.BatchNorm(out_channels, axis_name="batch")
        self.norm2 = eqx.nn.BatchNorm(out_channels, axis_name="batch")

        if in_channels != out_channels:
            self.shortcut = eqx.nn.Conv2d(
                in_channels, out_channels, kernel_size=1, key=keys[2]
            )
        else:
            self.shortcut = None

    def __call__(
        self, x: jax.Array, state: eqx.nn.State
    ) -> tuple[jax.Array, eqx.nn.State]:
        """Forward pass with residual connection."""
        identity = x

        out = self.conv1(x)
        out, state = self.norm1(out, state)
        out = jax.nn.relu(out)

        out = self.conv2(out)
        out, state = self.norm2(out, state)

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = jax.nn.relu(out + identity)
        return out, state


def count_parameters(model: eqx.Module) -> int:
    """Count total trainable parameters in a model."""
    params, _ = eqx.partition(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


# =============================================================================
# Hopfield Network Layers (Binary, for inference after training)
# =============================================================================


class HopfieldHNL(eqx.Module):
    """Binary Hopfield version of HNL for inference.

    Replaces soft attention with discrete Hopfield dynamics:
    1. Project query to binary space via bin_proj
    2. Run Hopfield iterations
    3. Project back to real space via bin_proj.T
    """

    in_features: int
    out_features: int
    num_heads: int
    head_dim: int
    binary_dim: int
    num_iterations: int
    query_proj: eqx.nn.Linear
    layer_norm: eqx.nn.LayerNorm
    bin_proj: jax.Array  # (num_heads, binary_dim, head_dim)
    bin_inv: jax.Array  # (num_heads, binary_dim, head_dim)
    weight_matrix: jax.Array  # (num_heads, num_memories, binary_dim)
    memories: jax.Array  # (num_heads, num_memories, head_dim)
    temperature: float = 1e-2
    # weight_matrix: jax.Array  # (num_heads, binary_dim, binary_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        q = self.query_proj(x)
        q = q.reshape(self.num_heads, self.head_dim)

        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        bq = jnp.sign(jnp.einsum("hd,hbd->hb", q_norm, self.bin_proj))

        q_back = jnp.einsum("hb,hdb->hd", bq, self.bin_inv)
        q_back /= jnp.linalg.norm(q_back, axis=-1, keepdims=True)
        err = jnp.mean(jnp.linalg.norm(q_norm - q_back, axis=-1))
        """
        jax.debug.print(
            "err: {err}",
            err=err,
        )
        """

        attn_scores = jnp.cos(
            jnp.pi
            / 2
            * (1 - (jnp.einsum("hb,hmb->hm", bq, self.weight_matrix) / self.binary_dim))
        )
        # attn_scores = jnp.einsum("hb,hmb->hm", bq, self.weight_matrix) / self.binary_dim
        # attn_scores = jnp.sign(attn_scores) * jnp.sqrt(jnp.abs(attn_scores))

        # jax.debug.print("{}", jnp.linalg.norm(self.weight_matrix, axis=-1))
        # jax.debug.print("{}", jnp.linalg.norm(bq, axis=-1))
        memories = self.memories / jnp.linalg.norm(
            self.memories, axis=-1, keepdims=True
        )

        # jax.debug.print("{}", jnp.linalg.norm(self.weight_matrix, axis=-1))
        # jax.debug.print("{}", jnp.linalg.norm(bq, axis=-1))

        c_scores = jnp.einsum("hd,hmd->hm", q_norm, memories)
        # jax.debug.print("{} {}", jnp.max(attn_scores), jnp.max(c_scores))
        err = jnp.mean((attn_scores - c_scores))
        """
        jax.debug.print(
            "err: {err}",
            err=err,
        )
        """

        attn_weights = jax.nn.softmax(attn_scores / self.temperature, axis=-1)
        out = jnp.einsum("hm,hmb->hb", attn_weights, self.weight_matrix)
        out = jnp.einsum("hb,hdb->hd", out, self.bin_inv)
        out /= jnp.linalg.norm(out, axis=-1, keepdims=True)

        out = out.reshape(self.out_features)
        out = self.layer_norm(out)
        # jax.debug.print("{out}", out=out)
        return out


class HopfieldHNM(eqx.Module):
    """Hopfield Network Model - container for HopfieldHNL layers."""

    layers: tuple[HopfieldHNL, ...]

    def __init__(self, *layers: HopfieldHNL):
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x
