import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

"""
N = 10
D = 32
Dp = 2048

X = np.random.normal(size=(N, D))

print(np.linalg.norm(X, axis=-1))
rand_proj = np.random.normal(size=(D, Dp)) / np.sqrt(Dp)
proj = X @ rand_proj

print(np.linalg.norm(proj, axis=-1))
proj_t = proj @ rand_proj.T
proj_inv = proj @ np.linalg.pinv(rand_proj)

t_dists = np.linalg.norm(X - proj_t, axis=-1)
inv_dists = np.linalg.norm(X - proj_inv, axis=-1)

print(np.mean(((t_dists) / np.linalg.norm(X, axis=-1)) ** 2))
print(np.mean(((inv_dists) / np.linalg.norm(X, axis=-1)) ** 2))

print("=======BINARY=======")

X = X / np.linalg.norm(X, axis=-1, keepdims=True)
print(np.linalg.norm(X, axis=-1))
rand_proj = np.random.normal(size=(D, Dp))
# rand_proj /= np.linalg.norm(rand_proj, axis=1, keepdims=True)
Q, _ = np.linalg.qr(rand_proj.T)

bin_proj = np.sign(X @ rand_proj)
bin_Q = np.sign(X @ Q.T)

proj_t = bin_proj @ rand_proj.T
proj_inv = bin_proj @ np.linalg.pinv(rand_proj)
proj_Q = bin_Q @ Q

proj_t /= np.linalg.norm(proj_t, axis=-1, keepdims=True)
proj_inv /= np.linalg.norm(proj_inv, axis=-1, keepdims=True)
proj_Q /= np.linalg.norm(proj_Q, axis=-1, keepdims=True)

t_dists = np.linalg.norm(X - proj_t, axis=-1)
inv_dists = np.linalg.norm(X - proj_inv, axis=-1)
q_dists = np.linalg.norm(X - proj_Q, axis=-1)

print(np.mean(((t_dists) / np.linalg.norm(X, axis=-1)) ** 2))
print(np.mean(((inv_dists) / np.linalg.norm(X, axis=-1)) ** 2))
print(np.mean(((q_dists) / np.linalg.norm(X, axis=-1)) ** 2))

print("=======SPARSE=======")

X = X / np.linalg.norm(X, axis=-1, keepdims=True)
print(np.linalg.norm(X, axis=-1))
rand_proj = np.random.normal(size=(D, Dp))
# rand_proj /= np.linalg.norm(rand_proj, axis=1, keepdims=True)
Q, _ = np.linalg.qr(rand_proj.T)

bin_proj = ((X @ rand_proj) >= 0.002).astype(int)
bin_Q = ((X @ Q.T) >= 0.002).astype(int)
print(np.mean(bin_Q))
proj_t = bin_proj @ rand_proj.T
proj_inv = bin_proj @ np.linalg.pinv(rand_proj)
proj_Q = bin_Q @ Q

proj_t /= np.linalg.norm(proj_t, axis=-1, keepdims=True)
proj_inv /= np.linalg.norm(proj_inv, axis=-1, keepdims=True)
proj_Q /= np.linalg.norm(proj_Q, axis=-1, keepdims=True)

t_dists = np.linalg.norm(X - proj_t, axis=-1)
inv_dists = np.linalg.norm(X - proj_inv, axis=-1)
q_dists = np.linalg.norm(X - proj_Q, axis=-1)

print(np.mean(((t_dists) / np.linalg.norm(X, axis=-1)) ** 2))
print(np.mean(((inv_dists) / np.linalg.norm(X, axis=-1)) ** 2))
print(np.mean(((q_dists) / np.linalg.norm(X, axis=-1)) ** 2))
"""


def compare_active_d(
    d=8, k_range=[1024, 2048, 4096, 8192], z_dims=[8, 16, 32, 64, 128, 256, 512]
):
    plt.figure(figsize=(10, 6))
    # 1. Setup original data (Normalized)

    x = np.random.randn(1000, d)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    for k in k_range:
        k = int(k)
        bin_errors = []
        for z_dim in z_dims:
            # --- BINARY METHOD (SRP with Orthogonalization) ---
            # Generate Blocked Orthogonal Matrix for better binary performance
            blocks = []
            for _ in range(int(np.ceil(k / d))):
                Q, _ = np.linalg.qr(np.random.randn(d, d))
                blocks.append(Q)
            W_bin = np.vstack(blocks)[:k, :]

            # z_dims = int(sparsity * k)
            # Project to 0/1 codes
            bits = np.einsum("kd,bd->bk", W_bin, x)
            partition = np.argpartition(-bits, z_dim, axis=-1)
            # partition = np.argpartition(-np.abs(bits), z_dim, axis=-1)
            # bits = np.sign(bits)

            bits[np.arange(bits.shape[0])[:, None], partition[:, :z_dim]] = 1
            bits[np.arange(bits.shape[0])[:, None], partition[:, z_dim:]] = 0
            bits = bits.astype(int)

            # bits = (np.argpartition(np.einsum("kd,bd->bk", W_bin, x), axis=-1) >= sparsity).astype(int)

            x_hat_bin = np.einsum("bk,kd->bd", bits, W_bin)
            x_hat_bin /= np.linalg.norm(
                x_hat_bin, axis=-1, keepdims=True
            )  # Re-normalize
            bin_errors.append(np.mean(np.linalg.norm(x - x_hat_bin, axis=-1), axis=-1))
        print(bin_errors)
        plt.plot(
            z_dims,
            bin_errors,
            "s--",
            label=f"k={k}",
            # color="crimson",
            linewidth=2,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Active Dimensions")
    plt.ylabel("L2 Reconstruction Error")
    plt.title(
        f"Scaling Comparison: Reconstruction Error vs. Active Dimenisons, varying continuous dimension (d={d})"
    )
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


def compare_scaling(d_range=[128], k_range=[512, 1024, 2048, 4096, 8192]):
    plt.figure(figsize=(10, 6))
    for d in d_range:
        # 1. Setup original data (Normalized)

        x = np.random.randn(100, d)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)

        bin_errors = []
        ks = []
        bin_errors = []
        z_dims = []
        for k in k_range:
            k = int(k)
            z_dim = 256
            # z_dim = int(math.pow(k, 1 / 3))
            z_dims.append(z_dim)
            # --- BINARY METHOD (SRP with Orthogonalization) ---
            # Generate Blocked Orthogonal Matrix for better binary performance
            blocks = []
            for _ in range(int(np.ceil(k / d))):
                Q, _ = np.linalg.qr(np.random.randn(d, d))
                blocks.append(Q)
            W_bin = np.vstack(blocks)[:k, :]
            """
            W_bin = np.random.randn(k, d)
            W_bin /= np.linalg.norm(W_bin, axis=-1, keepdims=True)
            """

            # z_dims = int(sparsity * k)
            # Project to 0/1 codes
            bits = 1 - np.einsum("kd,bd->bk", W_bin, x)
            partition = np.argpartition(bits, z_dim, axis=-1)

            bits[np.arange(bits.shape[0])[:, None], partition[:, :z_dim]] = 1
            bits[np.arange(bits.shape[0])[:, None], partition[:, z_dim:]] = 0
            bits = bits.astype(int)

            # bits = (np.argpartition(np.einsum("kd,bd->bk", W_bin, x), axis=-1) >= sparsity).astype(int)

            x_hat_bin = np.einsum("bk,kd->bd", bits, W_bin)
            x_hat_bin /= np.linalg.norm(
                x_hat_bin, axis=-1, keepdims=True
            )  # Re-normalize
            bin_errors.append(np.mean(np.linalg.norm(x - x_hat_bin, axis=-1), axis=-1))

        plt.plot(
            z_dims,
            bin_errors,
            "s--",
            # label=f"Binary ($N^1/3$) sparsity)",
            # color="crimson",
            linewidth=2,
        )

        # pow = 0.500 + 8 / d
        pow = 0.5 * (1 + 1 / math.pow(d, 1 / 3))
        print(f"pow: {pow}")
        # Plotting results
        plt.plot(
            z_dims,
            np.pow(z_dims[0], pow) * bin_errors[0] / np.pow(z_dims, pow),
            "o-",
            label=f"D = {d}",
            linewidth=2,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Active Dimensions")
    plt.ylabel("L2 Reconstruction Error")
    plt.title(
        f"Scaling Comparison: Reconstruction Error vs. Active Dimenisons, varying continuous dimension (d={d})"
    )
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


compare_scaling()
# compare_active_d()
