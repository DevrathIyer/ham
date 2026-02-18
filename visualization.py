from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history: dict,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot loss
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    if history.get("val_acc"):
        ax.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training plot to {save_path}")

    return fig


def plot_confusion_matrix(
    y_true: jax.Array,
    y_pred: jax.Array,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Labels
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")

    return fig


def plot_image_predictions(
    model: eqx.Module,
    images: jax.Array,
    labels: jax.Array,
    n_samples: int = 16,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 12),
) -> plt.Figure:
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Get predictions (inference mode, no dropout)
    inference_model = eqx.nn.inference_mode(model)
    dummy_key = jax.random.PRNGKey(0)
    keys = jax.random.split(dummy_key, n_samples)
    logits = jax.vmap(inference_model)(images[:n_samples], key=keys)
    preds = jnp.argmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)

    for i in range(n_samples):
        ax = axes[i]

        # Handle channel dimension
        img = np.asarray(images[i])
        if img.ndim == 3:
            if img.shape[0] in [1, 3]:  # (C, H, W)
                img = img.transpose(1, 2, 0)
            if img.shape[-1] == 1:
                img = img.squeeze(-1)

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        true_label = int(labels[i])
        pred_label = int(preds[i])
        confidence = float(probs[i, pred_label])

        # Use class names if provided
        true_name = class_names[true_label] if class_names else str(true_label)
        pred_name = class_names[pred_label] if class_names else str(pred_label)

        color = "green" if true_label == pred_label else "red"
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name} ({confidence:.1%})",
            color=color,
            fontsize=8,
        )
        ax.axis("off")

    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved predictions plot to {save_path}")

    return fig


def plot_synthetic_data_2d(
    X: jax.Array,
    y: jax.Array,
    model: eqx.Module | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[1] != 2:
        raise ValueError("Data must be 2D for this visualization")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot decision boundary if model provided
    if model is not None:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100),
        )
        grid = jnp.array(np.c_[xx.ravel(), yy.ravel()])
        inference_model = eqx.nn.inference_mode(model)
        dummy_key = jax.random.PRNGKey(0)
        keys = jax.random.split(dummy_key, grid.shape[0])
        logits = jax.vmap(inference_model)(grid, key=keys)
        Z = jnp.argmax(logits, axis=-1).reshape(xx.shape)
        ax.contourf(xx, yy, np.asarray(Z), alpha=0.3, cmap="viridis")

    # Plot data points
    scatter = ax.scatter(
        X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="black", alpha=0.7
    )
    plt.colorbar(scatter, ax=ax, label="Class")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Synthetic Data" + (" with Decision Boundary" if model else ""))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved data plot to {save_path}")

    return fig


def plot_hnm_mem_weights(
    model: eqx.Module,
    image_shape: tuple[int, int] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 10),
) -> plt.Figure:
    if not hasattr(model, "layers") or len(model.layers) == 0:
        raise ValueError("Model must have a 'layers' attribute with at least one layer")

    first_hnl = model.layers[0]
    if not hasattr(first_hnl, "mem"):
        raise ValueError("First layer must be an HNL with a 'mem' attribute")

    weights = np.asarray(first_hnl.mem.weight)
    n_memories = weights.shape[0]

    if image_shape is not None:
        # Plot as grid of images
        n_cols = int(np.ceil(np.sqrt(n_memories)))
        n_rows = int(np.ceil(n_memories / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)

        vmax = np.abs(weights).max()
        for i, ax in enumerate(axes.flatten()):
            if i < n_memories:
                mem_img = weights[i].reshape(image_shape)
                im = ax.imshow(mem_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                ax.set_title(f"Memory {i}")
            ax.axis("off")

        fig.suptitle("HNM First Layer Memory Weights", fontsize=14)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Weight Value")
    else:
        # Plot as heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(weights, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="Weight Value")
        ax.set_xlabel("Input Features")
        ax.set_ylabel("Memory Units")
        ax.set_title("HNM First Layer Memory Weights")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved HNM mem weights plot to {save_path}")

    return fig


def plot_mem_correlations(
    mem_correlations: list[dict[str, float]],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Plot average pairwise memory cosine similarity per head over training epochs."""
    if not mem_correlations:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Memory Correlations (no data)")
        return fig

    # Collect all keys across snapshots
    all_keys = sorted(mem_correlations[0].keys())
    if not all_keys:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Memory Correlations (no HNL layers)")
        return fig

    epochs = range(1, len(mem_correlations) + 1)

    # Group keys by layer
    layers = {}
    for k in all_keys:
        layer = k.split("_")[0]  # e.g. "L0"
        layers.setdefault(layer, []).append(k)

    n_layers = len(layers)
    fig, axes = plt.subplots(
        1, n_layers, figsize=(figsize[0], figsize[1]), squeeze=False
    )

    cmap = plt.cm.tab10
    for col, (layer_name, keys) in enumerate(sorted(layers.items())):
        ax = axes[0, col]
        for i, k in enumerate(sorted(keys)):
            values = [snap.get(k, 0.0) for snap in mem_correlations]
            head_label = k.split("_")[1]  # e.g. "H0"
            ax.plot(epochs, values, color=cmap(i), label=head_label, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Cosine Similarity")
        ax.set_title(f"Layer {layer_name[1:]} Memory Correlation")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(-1, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved memory correlation plot to {save_path}")

    return fig


def plot_feature_importance(
    model: eqx.Module,
    feature_names: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    # Get first layer weights
    if hasattr(model, "layers") and len(model.layers) > 0:
        weights = np.asarray(model.layers[0].weight)
    else:
        raise ValueError("Model must have a 'layers' attribute with at least one layer")

    # Compute importance as mean absolute weight per feature
    importance = np.abs(weights).mean(axis=0)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]

    # Sort by importance
    indices = np.argsort(importance)[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(importance)), importance[indices])
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Mean |Weight|")
    ax.set_title("Feature Importance (First Layer Weights)")
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance plot to {save_path}")

    return fig
