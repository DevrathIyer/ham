from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp


class MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable

    def __init__(self, *layers: eqx.nn.Linear, activation: Callable = jax.nn.relu):
        self.layers = layers
        self.activation = activation

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        hard: bool = False,
    ) -> jax.Array:
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
        dims = [in_features] + hidden_dims + [out_features]
        keys = jax.random.split(key, len(dims) - 1)
        layers = [
            eqx.nn.Linear(d_in, d_out, key=k)
            for d_in, d_out, k in zip(dims[:-1], dims[1:], keys)
        ]
        return cls(*layers, activation=activation)


class HNL(eqx.Module):
    in_feats: int
    out_feats: int
    num_mems: int
    num_heads: int
    head_dim: int
    is_class: bool
    query_proj: eqx.nn.Linear
    memories: jax.Array
    dropout: eqx.nn.Dropout
    temperature: jax.Array

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_mems: int,
        num_heads: int,
        is_class: bool = False,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
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
        self.is_class = is_class

        k1, k2 = jax.random.split(key)

        self.query_proj = eqx.nn.Linear(in_feats, out_feats, use_bias=False, key=k1)
        self.memories = (
            jax.random.normal(k2, (num_heads, num_mems, self.head_dim)) * 0.02
        )
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.temperature = jnp.array(temperature, dtype=jnp.float32)

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        hard: bool = False,
    ) -> jax.Array:
        q = self.query_proj(x)
        q = q.reshape(self.num_heads, self.head_dim)

        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        mem_norm = self.memories / jnp.linalg.norm(
            self.memories, axis=-1, keepdims=True
        )

        attn_scores = jnp.einsum("hd,hmd->hm", q_norm, mem_norm)
        if self.is_class:
            return attn_scores.flatten() * 10
        else:
            if hard:
                top_mems = jnp.argmax(attn_scores, axis=-1)
                mem_probs = jax.nn.one_hot(top_mems, num_classes=self.num_mems, axis=-1)
            else:
                mem_probs = jax.nn.softmax(attn_scores / self.temperature, axis=-1)
        out = jnp.einsum("hm,hmd->hd", mem_probs, mem_norm) * jnp.sqrt(self.head_dim)
        out = out.reshape(self.out_feats)
        return out


class HNM(eqx.Module):
    layers: tuple[HNL, ...]

    def __init__(self, layers: HNL):
        self.layers = layers

    def __call__(
        self,
        x: jax.Array,
        key: jax.Array,
        hard: bool = False,
    ) -> jax.Array:
        keys = jax.random.split(key, len(self.layers))
        for layer, k in zip(self.layers, keys):
            x = layer(x, key=k, hard=hard)
        return x

    def set_layer_temperature(self, layer_idx: int, temperature: float) -> "HNM":
        return eqx.tree_at(
            lambda m: m.layers[layer_idx].temperature,
            self,
            jnp.array(temperature, dtype=jnp.float32),
        )

    def set_temperature(self, temperature: float) -> "HNM":
        model = self
        for i, layer in enumerate(self.layers):
            if not layer.is_class:
                model = eqx.tree_at(
                    lambda m, idx=i: m.layers[idx].temperature,
                    model,
                    jnp.array(temperature, dtype=jnp.float32),
                )
        return model


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


def count_parameters(model: eqx.Module) -> int:
    params, _ = eqx.partition(model, eqx.is_array)
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


class HopfieldHNL(eqx.Module):
    in_feats: int
    out_feats: int
    num_mems: int
    num_heads: int
    head_dim: int
    is_class: bool
    query_proj: eqx.nn.Linear
    binary_dim: int
    active_dims: int
    num_iterations: int
    bin_proj: jax.Array  # (num_heads, binary_dim, head_dim)
    weight_matrix: jax.Array  # (num_heads, num_memories, binary_dim)
    memories: jax.Array  # (num_heads, num_memories, head_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        q = self.query_proj(x)
        q = q.reshape(self.num_heads, self.head_dim)

        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        bin_scores = jnp.einsum("hbd,hd->hb", self.bin_proj, q_norm)
        _, indices = jax.lax.top_k(bin_scores, self.active_dims, axis=-1)
        mask = jnp.zeros_like(bin_scores)
        bq = mask.at[jnp.arange(self.num_heads)[:, None], indices].set(1.0)

        # Normalize by active_dims (fraction of query bits that match each memory),
        # giving scores in [0, 1] regardless of binary_dim.
        # Normalizing by binary_dim instead would shrink logits as binary_dim grows,
        # collapsing the class-layer softmax toward uniform and increasing error.
        attn_scores = (
            jnp.einsum("hb,hmb->hm", bq, self.weight_matrix) / self.active_dims
        )

        attn_scores = 2 * attn_scores / (1 + attn_scores)

        # attn_scores /= jnp.sum(jnp.maximum(bq[:, None], self.weight_matrix), axis=-1)
        # attn_scores = jnp.sin(attn_scores * jnp.pi / 2)
        # attn_scores = 1 - jnp.arccos(attn_scores) / jnp.pi

        if self.is_class:
            return attn_scores.flatten() * 10
        # top_mems = jnp.argmax(attn_scores, axis=-1)
        # mem_probs = jax.nn.one_hot(top_mems, num_classes=self.num_mems, axis=-1)

        mem_probs = jax.nn.softmax((attn_scores * 10), axis=-1)

        # out = jnp.einsum("hm,hmb->hb", mem_probs, self.weight_matrix)
        # out = jnp.einsum("hb,hbd->hd", out, self.bin_proj)

        mem_norm = self.memories / jnp.linalg.norm(
            self.memories, axis=-1, keepdims=True
        )
        out = jnp.einsum("hm,hmd->hd", mem_probs, mem_norm) * jnp.sqrt(self.head_dim)
        # out *= jnp.sqrt(self.head_dim) / jnp.linalg.norm(out, axis=-1, keepdims=True)

        out = out.reshape(self.out_feats)
        return out


class HopfieldHNM(eqx.Module):
    layers: tuple[HopfieldHNL, ...]

    def __init__(self, *layers: HopfieldHNL):
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x
