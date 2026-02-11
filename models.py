"""Model definitions using Equinox."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class MLP(eqx.Module):
    """Simple Multi-Layer Perceptron - pass layers directly.

    Examples:
        # Define layers explicitly
        keys = jax.random.split(key, 3)
        model = MLP(
            eqx.nn.Linear(784, 128, key=keys[0]),
            eqx.nn.Linear(128, 64, key=keys[1]),
            eqx.nn.Linear(64, 10, key=keys[2]),
        )

        # Or use the factory
        model = MLP.create(784, [128, 64], 10, key=key)
    """

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

    in_features: int
    out_features: int
    num_memories: int
    num_heads: int
    head_dim: int
    temperature: float
    use_activation: bool
    query_proj: eqx.nn.Linear
    memories: jax.Array
    layer_norm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_memories: int,
        num_heads: int = 8,
        use_activation: bool = True,
        dropout_rate: float = 0.0,
        temperature: float = 1e-2,
        *,
        key: jax.Array,
    ):
        assert (
            out_features % num_heads == 0
        ), f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"

        self.in_features = in_features
        self.out_features = out_features
        self.num_memories = num_memories
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.temperature = temperature
        print(f"N_heads is {self.num_heads}")
        self.use_activation = use_activation

        k1, k2, k3, k4, k5 = jax.random.split(key, 5)

        self.query_proj = eqx.nn.Linear(
            in_features, out_features, use_bias=False, key=k1
        )
        self.memories = (
            jax.random.normal(k3, (num_heads, num_memories, self.head_dim)) * 0.02
        )
        self.layer_norm = eqx.nn.LayerNorm(out_features)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        q = self.query_proj(x)
        q = q.reshape(self.num_heads, self.head_dim)

        q_norm = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        # mem_norm = self.memories / jnp.sum(self.memories, axis=-1, keepdims=True)
        mem_norm = self.memories / jnp.linalg.norm(
            self.memories, axis=-1, keepdims=True
        )

        attn_scores = jnp.einsum("hd,hmd->hm", q_norm, mem_norm)
        attn_weights = jax.nn.softmax(attn_scores / self.temperature, axis=-1)
        out = jnp.einsum("hm,hmd->hd", attn_weights, mem_norm)

        out = out.reshape(self.out_features)
        out = self.layer_norm(out)
        out = self.dropout(out, key=key)
        return out


class HNM(eqx.Module):
    layers: tuple[HNL, ...]

    def __init__(self, *layers: HNL):
        self.layers = layers

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        keys = jax.random.split(key, len(self.layers))
        for layer, k in zip(self.layers, keys):
            x = layer(x, key=k)
        return x

    @classmethod
    def create(
        cls,
        in_features: int,
        hidden_dims: list[int],
        out_features: int,
        *,
        num_memories: int = 1024,
        num_heads: int = 4,
        key: jax.Array,
    ) -> "HNM":
        dims = [in_features] + hidden_dims + [out_features]
        keys = jax.random.split(key, len(dims) - 1)

        layers = []
        for i, (d_in, d_out, k) in enumerate(zip(dims[:-1], dims[1:], keys)):
            is_last = i == len(dims) - 2
            layers.append(
                HNL(
                    d_in,
                    d_out,
                    num_memories=num_memories,
                    num_heads=1 if is_last else num_heads,
                    use_activation=not is_last,
                    key=k,
                )
            )
        return cls(*layers)


class HCL(eqx.Module):
    in_channels: int
    out_channels: int
    num_memories: int
    kernel_size: int
    temperature: float
    use_activation: bool
    query_conv: eqx.nn.Conv2d
    memories: jax.Array
    layer_norm: eqx.nn.GroupNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_memories: int = 32,
        use_activation: bool = True,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        *,
        key: jax.Array,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_memories = num_memories
        self.kernel_size = kernel_size
        self.temperature = temperature
        self.use_activation = use_activation

        k1, k2, k3 = jax.random.split(key, 3)

        # Query projection via conv (captures local spatial context)
        self.query_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            key=k1,
        )

        # External memory (shared across all spatial positions)
        # Each channel has its own memory bank
        self.memories = jax.random.normal(k3, (out_channels, num_memories)) * 0.02

        # GroupNorm works better than LayerNorm for conv (normalize over channels)
        self.layer_norm = eqx.nn.GroupNorm(min(8, out_channels), out_channels)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tensor of shape (C_in, H, W)
            key: Random key for dropout

        Returns:
            Output tensor of shape (C_out, H, W)
        """
        _, H, W = x.shape

        # Project to queries via conv: (out_channels, H, W)
        q = self.query_conv(x)

        # Attention over memory at each spatial position
        # q: (C, H, W), keys: (C, M) -> (C, H, W, M)
        attn_scores = jnp.einsum("chw,cm->chwm", q, self.memories)
        attn_scores = attn_scores / self.temperature
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)

        # Retrieve from memory: (C, H, W, M) @ (C, M) -> (C, H, W)
        out = jnp.einsum("chwm,cm->chw", attn_weights, self.memories)

        # Residual from query projection + norm
        out = self.layer_norm(out + q)
        out = self.dropout(out, key=key)

        if self.use_activation:
            out = jax.nn.gelu(out)

        return out


class HCM(eqx.Module):
    conv_layers: tuple[HCL, ...]
    fc_layers: tuple[HNL, ...]
    pool: eqx.nn.MaxPool2d

    def __init__(
        self,
        conv_layers: list[HCL],
        fc_layers: list[HNL],
        pool: eqx.nn.MaxPool2d | None = None,
    ):
        self.conv_layers = tuple(conv_layers)
        self.fc_layers = tuple(fc_layers)
        self.pool = (
            pool if pool is not None else eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        n_layers = len(self.conv_layers) + len(self.fc_layers)
        keys = jax.random.split(key, n_layers)
        key_idx = 0
        for conv in self.conv_layers:
            x = self.pool(conv(x, key=keys[key_idx]))
            key_idx += 1
        x = x.reshape(-1)
        for fc in self.fc_layers:
            x = fc(x, key=keys[key_idx])
            key_idx += 1
        return x

    @classmethod
    def create(
        cls,
        in_channels: int = 1,
        num_classes: int = 10,
        image_size: int = 28,
        *,
        num_memories: int = 32,
        key: jax.Array,
    ) -> "HCM":
        """Factory method for creating standard architectures."""
        keys = jax.random.split(key, 4)

        conv_layers = [
            HCL(
                in_channels,
                16,
                kernel_size=3,
                num_memories=num_memories,
                key=keys[0],
            ),
            HCL(
                16,
                32,
                kernel_size=3,
                num_memories=num_memories,
                key=keys[1],
            ),
            HCL(
                32,
                64,
                kernel_size=3,
                num_memories=num_memories,
                key=keys[2],
            ),
        ]

        final_size = image_size // 8  # After 3 max pools
        fc_layers = [
            HNL(
                64 * final_size * final_size,
                128,
                num_memories=num_memories,
                key=keys[3],
            ),
            HNL(
                128,
                num_classes,
                num_memories=num_memories,
                num_heads=1,
                key=keys[3],
            ),
        ]

        return cls(conv_layers, fc_layers)


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
    use_activation: bool
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

        """
        # For each head, run Hopfield retrieval
        def hopfield_head(q_h, bin_proj_h, W_h):
            # Project to binary space and threshold
            binary_query = jnp.sign(bin_proj_h @ q_h)

            # Hopfield iterations
            state = binary_query
            for _ in range(self.num_iterations):
                state = jnp.sign(W_h @ state)

            # Project back to real space
            return bin_proj_h.T @ state / self.binary_dim

        # Apply to each head
        out = jax.vmap(hopfield_head)(q, self.bin_proj, self.weight_matrix)
        """


class HopfieldHCL(eqx.Module):
    """Binary Hopfield version of HCL for inference.

    Each channel independently runs Hopfield dynamics at each spatial position.
    """

    in_channels: int
    out_channels: int
    binary_dim: int
    num_iterations: int
    kernel_size: int
    use_activation: bool
    query_conv: eqx.nn.Conv2d
    layer_norm: eqx.nn.GroupNorm
    bin_proj: jax.Array  # (out_channels, binary_dim, 1)
    weight_matrix: jax.Array  # (out_channels, binary_dim, binary_dim)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with Hopfield dynamics.

        Args:
            x: Input tensor of shape (C_in, H, W)

        Returns:
            Output tensor of shape (C_out, H, W)
        """
        _, H, W = x.shape

        # Project to queries via conv: (out_channels, H, W)
        q = self.query_conv(x)

        # For each channel and spatial position, run Hopfield
        def hopfield_channel(q_c, bin_proj_c, W_c):
            # q_c: (H, W), bin_proj_c: (binary_dim, 1), W_c: (binary_dim, binary_dim)
            def hopfield_position(q_hw):
                # Project scalar to binary space
                binary_query = jnp.sign(bin_proj_c[:, 0] * q_hw)

                # Hopfield iterations
                state = binary_query
                for _ in range(self.num_iterations):
                    state = jnp.sign(W_c @ state)

                # Project back to scalar
                return jnp.sum(bin_proj_c[:, 0] * state)

            # Apply to all spatial positions
            return jax.vmap(jax.vmap(hopfield_position))(q_c)

        # Apply to each channel
        out = jax.vmap(hopfield_channel)(q, self.bin_proj, self.weight_matrix)

        # Residual + norm
        out = self.layer_norm(out + q)
        if self.use_activation:
            out = jax.nn.gelu(out)

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


class HopfieldHCM(eqx.Module):
    """Hopfield Convolutional Model - container for Hopfield layers."""

    conv_layers: tuple[HopfieldHCL, ...]
    fc_layers: tuple[HopfieldHNL, ...]
    pool: eqx.nn.MaxPool2d

    def __init__(
        self,
        conv_layers: list[HopfieldHCL],
        fc_layers: list[HopfieldHNL],
        pool: eqx.nn.MaxPool2d | None = None,
    ):
        self.conv_layers = tuple(conv_layers)
        self.fc_layers = tuple(fc_layers)
        self.pool = (
            pool if pool is not None else eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        for conv in self.conv_layers:
            x = self.pool(conv(x))
        x = x.reshape(-1)
        for fc in self.fc_layers:
            x = fc(x)
        return x
