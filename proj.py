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


def compare_scaling(d=8, k_range=[64, 128, 256, 512, 1024, 2048, 4096]):
    # 1. Setup original data (Normalized)
    x = np.random.randn(10000, d)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    plt.figure(figsize=(10, 6))

    bin_errors = []
    for k in k_range:
        # --- BINARY METHOD (SRP with Orthogonalization) ---
        # Generate Blocked Orthogonal Matrix for better binary performance
        blocks = []
        for _ in range(int(np.ceil(k / d))):
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            blocks.append(Q)
        W_bin = np.vstack(blocks)[:k, :]
        # Project to 0/1 codes
        bits = np.sign(np.einsum("kd,bd->bk", W_bin, x))
        # Reconstruct: Map {0,1} -> {-1,1} and use Transpose
        # centered_bits = 2 * bits - 1
        x_hat_bin = np.einsum("bk,kd->bd", bits, W_bin)
        x_hat_bin /= np.linalg.norm(x_hat_bin, axis=-1, keepdims=True)  # Re-normalize
        bin_errors.append(np.mean(np.linalg.norm(x - x_hat_bin, axis=-1), axis=-1))
    """
    plt.plot(
        np.array(k_range),
        bin_errors,
        "s--",
        label="Bipolar",
        # color="crimson",
        linewidth=2,
    )
    """
    for z_dims in [8, 16, 24, 32]:
        bin_errors = []
        for k in k_range:
            # --- BINARY METHOD (SRP with Orthogonalization) ---
            # Generate Blocked Orthogonal Matrix for better binary performance
            blocks = []
            for _ in range(int(np.ceil(k / d))):
                Q, _ = np.linalg.qr(np.random.randn(d, d))
                blocks.append(Q)
            W_bin = np.vstack(blocks)[:k, :]

            # z_dims = int(sparsity * k)
            # Project to 0/1 codes
            bits = 1 - np.einsum("kd,bd->bk", W_bin, x)
            partition = np.argpartition(bits, z_dims, axis=-1)

            bits[np.arange(bits.shape[0])[:, None], partition[:, :z_dims]] = 1
            bits[np.arange(bits.shape[0])[:, None], partition[:, z_dims:]] = 0
            bits = bits.astype(int)

            # bits = (np.argpartition(np.einsum("kd,bd->bk", W_bin, x), axis=-1) >= sparsity).astype(int)

            x_hat_bin = np.einsum("bk,kd->bd", bits, W_bin)
            x_hat_bin /= np.linalg.norm(
                x_hat_bin, axis=-1, keepdims=True
            )  # Re-normalize
            bin_errors.append(np.mean(np.linalg.norm(x - x_hat_bin, axis=-1), axis=-1))

        plt.plot(
            k_range,
            bin_errors,
            "s--",
            label=f"Binary ({z_dims:0.2f} active)",
            # color="crimson",
            linewidth=2,
        )

    # Plotting results
    """
    plt.plot(
        k_range,
        0.23 * 9 / (np.log2(k_range)),
        "o-",
        label="Scaling Law (1/sqrt(k))",
        color="royalblue",
        linewidth=2,
    )
    """

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Active Dimensions(k)")
    plt.ylabel("L2 Reconstruction Error")
    plt.title(f"Scaling Comparison: Sparsity level vs. Error")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


compare_scaling()
