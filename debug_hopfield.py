"""
Diagnostic: sweep binary_dim (fixed active_dims), decompose error sources layer-by-layer.

Per-layer measurements, each in *isolation* (given the original model's output from the
previous layer as input), so errors don't compound from earlier layers:

  Retrieval layers: binary vs cosine retrieval agreement, given original layer input
  Class layer: binary classification accuracy vs ground truth and vs cosine reference

Runs N_SEEDS per binary_dim to separate systematic trends from random-projection variance.

Usage:
  uv run python debug_hopfield.py
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from data import get_fashion_mnist_data
from hopfield import convert_hnm_to_hopfield
from models import HNL, HNM, HopfieldHNL, HopfieldHNM

# ── Config ─────────────────────────────────────────────────────────────────────
# CHECKPOINT = "./checkpoints/mnist_hnm/20260219_145419_base.eqx"
CHECKPOINT = "./checkpoints/mnist_hnm/20260219_160101_annealed.eqx"
# "20260218_142617_annealed.eqx"
ACTIVE_DIMS = 256
BINARY_DIMS = [512, 1024, 2048, 4096, 8192, 16384]
N_SEEDS = 5
N_TEST = 2000
BASE_SEED = 42
OUTPUT_PATH = "outputs/hopfield_binary_dim_sweep.png"
CORR_PATH = "outputs/memory_correlations.png"
# ───────────────────────────────────────────────────────────────────────────────


def make_base_model() -> HNM:
    l1, l2, l3 = jax.random.split(jax.random.PRNGKey(0), 3)
    return HNM(
        [
            HNL(784, 512, 4, 4, key=l1),
            HNL(512, 128, 10, 1, key=l3, is_class=True),
        ]
    )


def _binary_choices(hl: HopfieldHNL, X: jax.Array) -> jax.Array:
    """(N, num_heads) – memory index via binary attention."""

    def _single(x):
        q = hl.query_proj(x).reshape(hl.num_heads, hl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        bs = jnp.einsum("hbd,hd->hb", hl.bin_proj, q)
        _, idx = jax.lax.top_k(bs, hl.active_dims)
        bq = jnp.zeros_like(bs).at[jnp.arange(hl.num_heads)[:, None], idx].set(1.0)
        attn = jnp.einsum("hb,hmb->hm", bq, hl.weight_matrix) / hl.active_dims
        return jnp.argmax(attn, axis=-1)

    return jax.vmap(_single)(X)


def _cosine_choices(hnl: HNL, X: jax.Array) -> jax.Array:
    """(N, num_heads) – memory index via cosine hard-attention."""
    mn = hnl.memories / jnp.linalg.norm(hnl.memories, axis=-1, keepdims=True)

    def _single(x):
        q = hnl.query_proj(x).reshape(hnl.num_heads, hnl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        return jnp.argmax(jnp.einsum("hd,hmd->hm", q, mn), axis=-1)

    return jax.vmap(_single)(X)


def _hnl_hard_output(hnl: HNL, X: jax.Array) -> jax.Array:
    """Run one HNL layer in hard-attention mode → (N, out_feats)."""
    mn = hnl.memories / jnp.linalg.norm(hnl.memories, axis=-1, keepdims=True)

    def _single(x):
        q = hnl.query_proj(x).reshape(hnl.num_heads, hnl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        scores = jnp.einsum("hd,hmd->hm", q, mn)
        if hnl.is_class:
            return scores.flatten() * 10
        top_m = jnp.argmax(scores, axis=-1)
        probs = jax.nn.one_hot(top_m, hnl.num_mems)
        out = jnp.einsum("hm,hmd->hd", probs, mn) * jnp.sqrt(hnl.head_dim)
        return out.reshape(hnl.out_feats)

    return jax.vmap(_single)(X)


def _binary_class_acc(hl: HopfieldHNL, hnl: HNL, X: jax.Array, y: jax.Array) -> float:
    """Classification accuracy of binary class-layer given input X (from original layer-1)."""
    cosine_preds = jnp.argmax(_hnl_hard_output(hnl, X), axis=-1)  # cosine reference

    def _single(x):
        q = hl.query_proj(x).reshape(hl.num_heads, hl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        bs = jnp.einsum("hbd,hd->hb", hl.bin_proj, q)
        _, idx = jax.lax.top_k(bs, hl.active_dims)
        bq = jnp.zeros_like(bs).at[jnp.arange(hl.num_heads)[:, None], idx].set(1.0)
        attn = jnp.einsum("hb,hmb->hm", bq, hl.weight_matrix) / hl.active_dims
        return attn.flatten() * 10  # logits

    logits = jax.vmap(_single)(X)
    bin_preds = jnp.argmax(logits, axis=-1)
    acc_vs_gt = float(jnp.mean(bin_preds == y))
    acc_vs_cos = float(jnp.mean(bin_preds == cosine_preds))
    return acc_vs_gt, acc_vs_cos


def recon_cossim(hnl: HNL, hl: HopfieldHNL, X: jax.Array) -> float:
    """Layer-0 binary reconstruction quality vs true memory (given same memory)."""
    orig = _cosine_choices(hnl, X)
    mn = hnl.memories / jnp.linalg.norm(hnl.memories, axis=-1, keepdims=True)

    def _single(ch):
        true_m = mn[jnp.arange(hnl.num_heads), ch]
        bin_pat = hl.weight_matrix[jnp.arange(hl.num_heads), ch]
        recon = jnp.einsum("hb,hbd->hd", bin_pat, hl.bin_proj)
        recon = recon / jnp.linalg.norm(recon, axis=-1, keepdims=True)
        return jnp.mean(jnp.sum(true_m * recon, axis=-1))

    return float(jnp.mean(jax.vmap(_single)(orig)))


def run_seed(model, inf_model, X, y, orig_preds, bdim, key):
    hop = convert_hnm_to_hopfield(
        inf_model, key, binary_dim=bdim, active_dims=ACTIVE_DIMS
    )

    # ── Overall accuracy ──────────────────────────────────────────────────────
    logits = jax.vmap(hop)(X)
    preds = jnp.argmax(logits, axis=-1)
    acc = float(jnp.mean(preds == y))
    agreement = float(jnp.mean(preds == orig_preds))

    # ── Build isolated inputs: each layer receives the original HNL hard-attn
    #    output from the previous layer so errors don't cascade.
    layer_inputs = [X]
    for hnl in model.layers[:-1]:
        layer_inputs.append(_hnl_hard_output(hnl, layer_inputs[-1]))

    # ── Retrieval accuracy for every non-class layer ──────────────────────────
    ret = []
    class_layer_idx = None
    for i, (hnl, hl) in enumerate(zip(model.layers, hop.layers)):
        if hnl.is_class:
            class_layer_idx = i
        else:
            ret.append(
                float(
                    jnp.mean(
                        _cosine_choices(hnl, layer_inputs[i])
                        == _binary_choices(hl, layer_inputs[i])
                    )
                )
            )

    # ── Class layer accuracy ──────────────────────────────────────────────────
    X_cls = layer_inputs[class_layer_idx]
    class_acc_gt, class_acc_cos = _binary_class_acc(
        hop.layers[class_layer_idx], model.layers[class_layer_idx], X_cls, y
    )

    # ── Reconstruction quality (first retrieval layer) ────────────────────────
    recon = recon_cossim(model.layers[0], hop.layers[0], X)

    return dict(
        error=1 - acc,
        agreement=agreement,
        ret=ret,
        class_acc_gt=class_acc_gt,
        class_acc_cos=class_acc_cos,
        recon=recon,
    )


def plot_memory_correlations(model: HNM, save_path: str) -> None:
    """
    Grid of pairwise cosine-similarity heatmaps: one cell per head per layer.
    One outer column per layer; each column contains an inner grid of (M×M) matrices,
    one per head.
    """
    n_layers = len(model.layers)
    layer_names = [
        f"Layer {i}  ({'class' if l.is_class else 'retrieval'})  "
        f"({l.num_heads} head{'s' if l.num_heads > 1 else ''}, "
        f"{l.num_mems} mems, dim {l.head_dim})"
        for i, l in enumerate(model.layers)
    ]

    # ── Compute similarity matrices ──────────────────────────────────────────
    layer_sims = []
    for layer in model.layers:
        mems = layer.memories  # (H, M, d)
        mn = mems / jnp.linalg.norm(mems, axis=-1, keepdims=True)
        sims = []
        for h in range(mems.shape[0]):
            s = np.array(mn[h] @ mn[h].T)  # (M, M)
            # np.fill_diagonal(s, np.nan)  # hide self-sim
            sims.append(s)
        layer_sims.append(sims)

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8 * n_layers, 8))
    outer_gs = gridspec.GridSpec(
        1,
        n_layers,
        figure=fig,
        wspace=0.10,
        left=0.03,
        right=0.88,
        top=0.88,
        bottom=0.05,
    )

    cmap = plt.cm.RdBu_r
    cmap.set_bad(color="#cccccc")  # NaN diagonal → grey
    vmin, vmax = -1.0, 1.0

    last_im = None
    for li, (sims, name) in enumerate(zip(layer_sims, layer_names)):
        n_heads = len(sims)
        grid_cols = int(np.ceil(np.sqrt(n_heads)))  # 4 for 16 h, 1 for 1 h
        grid_rows = int(np.ceil(n_heads / grid_cols))

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            grid_rows,
            grid_cols,
            subplot_spec=outer_gs[li],
            hspace=0.05,
            wspace=0.05,
        )

        for h, sim in enumerate(sims):
            r, c = divmod(h, grid_cols)
            ax = fig.add_subplot(inner_gs[r, c])
            last_im = ax.imshow(
                sim,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                aspect="equal",
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])

        # blank out unused cells
        for h in range(n_heads, grid_rows * grid_cols):
            r, c = divmod(h, grid_cols)
            fig.add_subplot(inner_gs[r, c]).set_visible(False)

        # layer title centred above the sub-grid
        # use the bounding box of the outer subplot spec
        spec_pos = outer_gs[li].get_position(fig)
        fig.text(
            (spec_pos.x0 + spec_pos.x1) / 2,
            0.92,
            name,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # shared colorbar
    cbar_ax = fig.add_axes([0.90, 0.10, 0.018, 0.75])
    cb = plt.colorbar(last_im, cax=cbar_ax)
    cb.set_label("Cosine similarity  (diagonal masked)", fontsize=9)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.suptitle(
        "Pairwise cosine similarity between learned memories",
        fontsize=13,
        y=0.99,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Memory correlation grid saved → {save_path}")


SCATTER_PATH = "outputs/retrieval_scatter.png"
SCATTER_BINARY_DIM = 512  # single bdim for the scatter plot
N_SCATTER = 500  # number of test samples to plot

ATTENTION_PATH = "outputs/attention_comparison.png"
ATTENTION_BINARY_DIM = 512


def _hnl_attn_probs(hnl: HNL, X: jax.Array) -> jax.Array:
    """(N, H, M) – mem_probs matching HNL.__call__: softmax for retrieval, raw scores for class."""
    mn = hnl.memories / jnp.linalg.norm(hnl.memories, axis=-1, keepdims=True)

    def _single(x):
        q = hnl.query_proj(x).reshape(hnl.num_heads, hnl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        scores = jnp.einsum("hd,hmd->hm", q, mn)
        if hnl.is_class:
            return scores * 10
        return jax.nn.softmax(scores / hnl.temperature, axis=-1)

    return jax.vmap(_single)(X)


def _hopfield_attn_probs(hl: HopfieldHNL, X: jax.Array) -> jax.Array:
    """(N, H, M) – mem_probs matching HopfieldHNL.__call__: softmax for retrieval, raw for class."""

    def _single(x):
        q = hl.query_proj(x).reshape(hl.num_heads, hl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        bs = jnp.einsum("hbd,hd->hb", hl.bin_proj, q)
        _, idx = jax.lax.top_k(bs, hl.active_dims)
        bq = jnp.zeros_like(bs).at[jnp.arange(hl.num_heads)[:, None], idx].set(1.0)
        attn = jnp.einsum("hb,hmb->hm", bq, hl.weight_matrix)
        if hl.is_class:
            return attn
        return jax.nn.softmax((attn * 2.5) / hl.active_dims, axis=-1)

    return jax.vmap(_single)(X)


def plot_attention_comparison(
    model: HNM,
    hop: HopfieldHNM,
    X: jax.Array,
    y: jax.Array,
    save_path: str,
) -> None:
    """
    Side-by-side average mem_prob heatmaps for each layer, comparing HNL vs Hopfield.

    All layers (retrieval and class):
      rows = true class, cols = memory index  (C × M heatmap)
      Each row is the mean mem_probs for samples of that true class.

    Retrieval layers: one subplot-row per head.
    Class layer: one subplot-row (H=1, M=num_classes — a soft confusion matrix).

    Inputs to each layer are isolated: the HNL hard-attn output from the
    previous layer is used, so errors do not compound.
    """
    num_classes = int(y.max()) + 1

    # Compute isolated inputs (original HNL hard-attn pipeline)
    layer_inputs = [X]
    for hnl in model.layers[:-1]:
        layer_inputs.append(_hnl_hard_output(hnl, layer_inputs[-1]))

    # Collect (N, H, M) mem_probs per layer from both models
    hnl_probs_list, hop_probs_list = [], []
    for hnl, hl, X_in in zip(model.layers, hop.layers, layer_inputs):
        hnl_probs_list.append(np.array(_hnl_attn_probs(hnl, X_in)))
        hop_probs_list.append(np.array(_hopfield_attn_probs(hl, X_in)))

    y_np = np.array(y, dtype=int)

    # Total subplot rows: one per head for retrieval layers, one for the class layer
    n_rows = sum(hnl.num_heads if not hnl.is_class else 1 for hnl in model.layers)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    row = 0
    for li, hnl in enumerate(model.layers):
        hp = hnl_probs_list[li]  # (N, H, M)
        hfp = hop_probs_list[li]  # (N, H, M)
        layer_label = f"Layer {li}  ({'class' if hnl.is_class else 'retrieval'})"

        n_heads = hnl.num_heads if not hnl.is_class else 1
        for h in range(n_heads):
            head_label = "" if hnl.is_class else f"  head {h}"
            xlabel = "Memory index (= class)" if hnl.is_class else "Memory index"

            hnl_mat = np.stack(
                [hp[y_np == c, h, :].mean(axis=0) for c in range(num_classes)]
            )  # (C, M)
            hop_mat = np.stack(
                [hfp[y_np == c, h, :].mean(axis=0) for c in range(num_classes)]
            )
            vmin = min(hnl_mat.min(), hop_mat.min())
            vmax = max(hnl_mat.max(), hop_mat.max())
            diff = hnl_mat - hop_mat

            panels = [
                (hnl_mat, "HNL (cosine)", "hot", vmin, vmax),
                (hop_mat, "Hopfield (binary)", "hot", vmin, vmax),
                (
                    diff,
                    "Difference  HNL − Hopfield",
                    "RdBu_r",
                    -np.abs(diff).max(),
                    np.abs(diff).max(),
                ),
            ]
            for col, (mat, title, cmap, lo, hi) in enumerate(panels):
                ax = axes[row, col]
                im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=lo, vmax=hi)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("True class")
                ax.set_title(f"{layer_label}{head_label}\n{title}")
                ax.set_xticks(range(hnl.num_mems))
                ax.set_yticks(range(num_classes))
            row += 1

    fig.suptitle(
        f"Average attention distributions: HNL vs Hopfield\n"
        f"(binary_dim={hop.layers[0].binary_dim}, active_dims={ACTIVE_DIMS}, N={len(X)})",
        fontsize=13,
    )
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention comparison saved → {save_path}")


def plot_retrieval_scatter(
    model: HNM,
    X: jax.Array,
    y: jax.Array,
    binary_dim: int,
    save_path: str,
) -> None:
    """
    Scatter plot for layer 0:
      x-axis : cosine similarity of each memory to the query (continuous)
      y-axis : normalised binary overlap  = (bq · bm) / active_dims

    One point per (sample, head, memory) triple.
    Colour encodes whether the memory is the cosine-argmax (correct target):
      blue  = correct target memory
      red   = all other memories
    Marker shape distinguishes whether binary retrieval agreed with cosine:
      circle (o) = binary chose the same memory as cosine (for the winner)
      cross  (x) = binary chose a different memory
    """
    hop_key = jax.random.PRNGKey(BASE_SEED + 99)
    hop = convert_hnm_to_hopfield(
        eqx.nn.inference_mode(model),
        hop_key,
        binary_dim=binary_dim,
        active_dims=ACTIVE_DIMS,
    )
    hl = hop.layers[0]
    hnl = model.layers[0]

    mn = hnl.memories / jnp.linalg.norm(hnl.memories, axis=-1, keepdims=True)
    bm = hl.weight_matrix  # (H, M, binary_dim) — binary memory patterns

    cos_other, bin_other = [], []  # non-target memories (not chosen by binary)
    cos_false, bin_false = [], []  # non-target memory chosen by binary (false positive)
    cos_hit, bin_hit = [], []  # target memory, binary agreed
    cos_miss, bin_miss = [], []  # target memory, binary chose different

    def _process_single(x):
        # Continuous query
        q = hnl.query_proj(x).reshape(hnl.num_heads, hnl.head_dim)
        q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
        cos_scores = jnp.einsum("hd,hmd->hm", q, mn)  # (H, M)
        cosine_winner = jnp.argmax(cos_scores, axis=-1)  # (H,)

        # Binary query
        bs = jnp.einsum("hbd,hd->hb", hl.bin_proj, q)  # (H, binary_dim)
        _, idx = jax.lax.top_k(bs, hl.active_dims)
        bq = jnp.zeros_like(bs).at[jnp.arange(hl.num_heads)[:, None], idx].set(1.0)
        bin_scores = jnp.einsum("hb,hmb->hm", bq, bm) / hl.active_dims  # (H, M)
        binary_winner = jnp.argmax(bin_scores, axis=-1)  # (H,)

        return cos_scores, bin_scores, cosine_winner, binary_winner

    process_batch = jax.vmap(_process_single)
    cos_s, bin_s, cos_w, bin_w = process_batch(X)
    # shapes: (N, H, M), (N, H, M), (N, H), (N, H)

    N, H, M = cos_s.shape
    cos_s_np = np.array(cos_s)
    bin_s_np = np.array(bin_s)
    cos_w_np = np.array(cos_w)
    bin_w_np = np.array(bin_w)

    for n in range(N):
        for h in range(H):
            winner = cos_w_np[n, h]
            bin_winner = bin_w_np[n, h]
            agreed = winner == bin_winner
            for m in range(M):
                cs = cos_s_np[n, h, m]
                bs = bin_s_np[n, h, m]
                if m == winner:
                    if agreed:
                        cos_hit.append(cs)
                        bin_hit.append(bs)
                    else:
                        cos_miss.append(cs)
                        bin_miss.append(bs)
                else:
                    if m == bin_winner:
                        cos_false.append(cs)
                        bin_false.append(bs)
                    else:
                        cos_other.append(cs)
                        bin_other.append(bs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # non-target memories (not chosen by binary)
    ax.scatter(
        cos_other,
        bin_other,
        c="lightcoral",
        s=3,
        alpha=0.15,
        linewidths=0,
        label="Non-target (not chosen)",
        rasterized=True,
    )
    # non-target memories that binary incorrectly chose (false positives)
    ax.scatter(
        cos_false,
        bin_false,
        c="darkorange",
        s=20,
        alpha=0.8,
        marker="^",
        linewidths=0,
        label="Non-target  (binary chose — false +)",
        rasterized=True,
    )
    # target memories where binary agreed
    ax.scatter(
        cos_hit,
        bin_hit,
        c="steelblue",
        s=14,
        alpha=0.6,
        marker="o",
        linewidths=0,
        label="Target memory  (binary ✓)",
        rasterized=True,
    )
    # target memories where binary failed
    ax.scatter(
        cos_miss,
        bin_miss,
        c="crimson",
        s=20,
        alpha=0.8,
        marker="x",
        linewidths=1.2,
        label="Target memory  (binary ✗)",
        rasterized=True,
    )

    ax.set_xlabel("Cosine similarity  (continuous)", fontsize=12)
    ax.set_ylabel(f"Normalised binary overlap  = (bq·bm) / {ACTIVE_DIMS}", fontsize=12)
    ax.set_title(
        f"Layer-0 retrieval: continuous vs binary similarity\n"
        f"(binary_dim={binary_dim}, active_dims={ACTIVE_DIMS}, "
        f"N={N} samples × {H} heads × {M} memories)",
        fontsize=11,
    )
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Retrieval scatter saved → {save_path}")


def main():
    print("Loading data …")
    _, (X_full, y_full) = get_fashion_mnist_data(flatten=True)
    X_test = X_full[:N_TEST]
    y_test = y_full[:N_TEST]

    print("Loading checkpoint …")
    model = eqx.tree_deserialise_leaves(CHECKPOINT, make_base_model())
    inf_model = eqx.nn.inference_mode(model)

    print("Plotting memory correlations …")
    plot_memory_correlations(model, CORR_PATH)

    print(f"\nPlotting retrieval scatter (binary_dim={SCATTER_BINARY_DIM}) …")
    plot_retrieval_scatter(
        model,
        X_test[:N_SCATTER],
        y_test[:N_SCATTER],
        SCATTER_BINARY_DIM,
        SCATTER_PATH,
    )

    print(f"\nPlotting attention distributions (binary_dim={ATTENTION_BINARY_DIM}) …")
    hop_attn = convert_hnm_to_hopfield(
        inf_model,
        jax.random.PRNGKey(BASE_SEED + 77),
        binary_dim=ATTENTION_BINARY_DIM,
        active_dims=ACTIVE_DIMS,
    )
    plot_attention_comparison(model, hop_attn, X_test, y_test, ATTENTION_PATH)

    dummy_keys = jax.random.split(jax.random.PRNGKey(0), N_TEST)
    orig_logits = jax.vmap(inf_model, in_axes=(0, 0, None))(X_test, dummy_keys, True)
    orig_preds = jnp.argmax(orig_logits, axis=-1)
    orig_acc = float(jnp.mean(orig_preds == y_test))
    print(f"\nOriginal HNM hard-attn acc = {orig_acc:.4f}  error = {1-orig_acc:.4f}\n")

    n_ret = sum(1 for l in model.layers if not l.is_class)
    master_key = jax.random.PRNGKey(BASE_SEED)
    all_results = {}

    for bdim in BINARY_DIMS:
        if bdim < ACTIVE_DIMS:
            print(f"Skipping binary_dim={bdim} (< active_dims)")
            continue
        all_results[bdim] = []
        for s in range(N_SEEDS):
            master_key, sk = jax.random.split(master_key)
            r = run_seed(model, inf_model, X_test, y_test, orig_preds, bdim, sk)
            all_results[bdim].append(r)
            ret_str = "  ".join(f"ret{ri}={r['ret'][ri]:.3f}" for ri in range(n_ret))
            print(
                f"  bdim={bdim:5d}  s={s}  "
                f"err={r['error']:.3f}  "
                f"{ret_str}  "
                f"cls_acc={r['class_acc_gt']:.3f}  "
                f"recon={r['recon']:.3f}"
            )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    bdims = sorted(all_results.keys())

    def ms(key):
        vals = [[r[key] for r in all_results[d]] for d in bdims]
        return np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])

    def ms_ret(ri):
        vals = [[r["ret"][ri] for r in all_results[d]] for d in bdims]
        return np.array([np.mean(v) for v in vals]), np.array([np.std(v) for v in vals])

    err_m, err_s = ms("error")
    cls_m, cls_s = ms("class_acc_gt")
    clsc_m, clsc_s = ms("class_acc_cos")
    recon_m, recon_s = ms("recon")
    ret_stats = [ms_ret(ri) for ri in range(n_ret)]

    # ── Plot ──────────────────────────────────────────────────────────────────
    xs = np.array(bdims, dtype=float)
    RET_COLORS = ["steelblue", "darkorange", "mediumpurple", "mediumseagreen"]

    # Panels: error + one per retrieval layer + class_acc + recon + summary
    n_panels = 1 + n_ret + 3
    n_cols = (n_panels + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 9))
    axes_flat = axes.flatten()
    fig.suptitle(
        f"Binary Hopfield: error sources vs binary_dim "
        f"(active_dims={ACTIVE_DIMS}, {N_SEEDS} seeds, N={N_TEST})",
        fontsize=12,
    )

    def band(ax, x, m, s, color, label, ls="-"):
        ax.plot(x, m, color=color, ls=ls, lw=2, marker="o", ms=5, label=label)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)

    pi = 0  # panel index

    # 1. Overall error
    ax = axes_flat[pi]
    pi += 1
    band(ax, xs, err_m, err_s, "tomato", "Hopfield error")
    ax.axhline(
        1 - orig_acc,
        color="steelblue",
        ls="--",
        lw=1.5,
        label=f"Original ({1-orig_acc:.3f})",
    )
    ax.set_xlabel("binary_dim")
    ax.set_ylabel("Error (1 − acc)")
    ax.set_title("Overall Test Error  (mean ± 1σ)")
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # 2. Per-retrieval-layer accuracy
    for ri, (ret_m, ret_s) in enumerate(ret_stats):
        ax = axes_flat[pi]
        pi += 1
        color = RET_COLORS[ri % len(RET_COLORS)]
        band(ax, xs, ret_m, ret_s, color, f"L{ri} retrieval")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("binary_dim")
        ax.set_ylabel("Retrieval accuracy")
        ax.set_title(f"Layer-{ri}: Binary Retrieval Accuracy")
        ax.set_xscale("log", base=2)
        ax.set_ylim([0, 1.05])
        ax.grid(alpha=0.3)

    # 3. Class layer accuracy
    ax = axes_flat[pi]
    pi += 1
    band(ax, xs, cls_m, cls_s, "mediumseagreen", "vs ground truth")
    band(ax, xs, clsc_m, clsc_s, "teal", "vs cosine-attn", ls="--")
    ax.axhline(
        orig_acc, color="steelblue", ls=":", lw=1.5, label=f"Original ({orig_acc:.3f})"
    )
    ax.set_xlabel("binary_dim")
    ax.set_ylabel("Accuracy")
    ax.set_title("Class Layer Accuracy")
    ax.set_xscale("log", base=2)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 4. Reconstruction quality
    ax = axes_flat[pi]
    pi += 1
    band(ax, xs, recon_m, recon_s, "mediumpurple", "L0 recon cos-sim")
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("binary_dim")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Layer-0: Reconstruction Quality\n(binary back-proj vs true memory)")
    ax.set_xscale("log", base=2)
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    # 5. Summary: all retrieval + class on one axes
    ax = axes_flat[pi]
    pi += 1
    for ri, (ret_m, ret_s) in enumerate(ret_stats):
        band(ax, xs, ret_m, ret_s, RET_COLORS[ri % len(RET_COLORS)], f"L{ri} retrieval")
    band(ax, xs, cls_m, cls_s, "mediumseagreen", "Class acc (vs GT)")
    ax.axhline(
        orig_acc, color="steelblue", ls=":", lw=1.5, label=f"Original ({orig_acc:.3f})"
    )
    ax.set_xlabel("binary_dim")
    ax.set_ylabel("Accuracy")
    ax.set_title("All Layers: Retrieval / Classification Accuracy\n(isolated inputs)")
    ax.set_xscale("log", base=2)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    for ax in axes_flat[pi:]:
        ax.set_visible(False)

    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved → {OUTPUT_PATH}")

    # ── Summary table ─────────────────────────────────────────────────────────
    ret_hdrs = "  ".join(f"{'ret-L'+str(ri):>13}" for ri in range(n_ret))
    print("\n── Isolated per-layer accuracy (mean ± std) ─────────────────────────────")
    print(f"{'bdim':>6}  {'error':>13}  {ret_hdrs}  {'cls-acc':>13}  {'recon':>13}")

    def fmt(m, s):
        return f"{m:.4f}±{s:.4f}"

    for i, d in enumerate(bdims):
        ret_cols = "  ".join(
            f"{fmt(ret_stats[ri][0][i], ret_stats[ri][1][i]):>13}"
            for ri in range(n_ret)
        )
        print(
            f"{d:>6}  {fmt(err_m[i], err_s[i]):>13}  "
            f"{ret_cols}  "
            f"{fmt(cls_m[i], cls_s[i]):>13}  "
            f"{fmt(recon_m[i], recon_s[i]):>13}"
        )

    print("\n── Diagnosis ────────────────────────────────────────────────────────────")
    print(f"  Original hard-attn acc/error : {orig_acc:.4f} / {1-orig_acc:.4f}")
    for ri, (ret_m, ret_s) in enumerate(ret_stats):
        print(f"  Layer-{ri} retrieval @ max bdim: {ret_m[-1]:.4f}")
        print(f"  Δ ret-L{ri}  (min→max bdim): {ret_m[-1] - ret_m[0]:+.4f}")
    print(f"  Class-layer acc    @ max bdim: {cls_m[-1]:.4f}")
    print(f"  Δ cls-acc (min→max bdim): {cls_m[-1] - cls_m[0]:+.4f}")
    print(f"  Reconstruction cos @ max bdim: {recon_m[-1]:.4f}")


if __name__ == "__main__":
    main()
