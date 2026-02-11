"""Training loop and utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from tqdm import tqdm

from data import DataLoader


@dataclass
class TrainState:
    """Training state container."""

    model: eqx.Module
    opt_state: optax.OptState
    step: int
    epoch: int


@dataclass
class TrainConfig:
    """Training configuration."""

    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 5
    log_every: int = 100


class Trainer:
    """Generic trainer for Equinox models."""

    def __init__(
        self,
        model: eqx.Module,
        loss_fn: Callable,
        config: TrainConfig | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Equinox model to train
            loss_fn: Loss function (model, x, y) -> scalar
            config: Training configuration
            optimizer: Optax optimizer (default: Adam)
        """
        self.config = config or TrainConfig()
        self.loss_fn = loss_fn

        if optimizer is None:
            optimizer = optax.adam(self.config.learning_rate)

        self.optimizer = optimizer
        self.opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        self.model = model
        self.step = 0
        self.epoch = 0

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        # Setup checkpointing
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @eqx.filter_jit
    def _train_step(
        self,
        model: eqx.Module,
        opt_state: optax.OptState,
        x: jax.Array,
        y: jax.Array,
        key: jax.Array,
    ) -> tuple[eqx.Module, optax.OptState, jax.Array]:
        """Single training step."""
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(model, x, y, key=key)
        updates, opt_state = self.optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def _eval_step(
        self,
        model: eqx.Module,
        x: jax.Array,
        y: jax.Array,
        hard=False,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluation step returning loss and accuracy (inference mode, no dropout)."""
        model = eqx.nn.inference_mode(model)
        # Use a dummy key since inference_mode disables dropout
        dummy_key = jax.random.PRNGKey(0)
        keys = jax.random.split(dummy_key, x.shape[0])
        logits = jax.vmap(model, in_axes=(0, 0, None))(x, keys, hard)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        preds = jnp.argmax(logits, axis=-1)
        acc = jnp.mean(preds == y)
        return loss, acc

    def train(
        self,
        train_data: tuple[jax.Array, jax.Array],
        val_data: tuple[jax.Array, jax.Array] | None = None,
        *,
        key: jax.Array,
    ) -> eqx.Module:
        """Train the model.

        Args:
            train_data: (X_train, y_train) tuple
            val_data: Optional (X_val, y_val) tuple
            key: JAX random key

        Returns:
            Trained model
        """
        X_train, y_train = train_data
        key, loader_key = jax.random.split(key)
        train_loader = DataLoader(
            X_train, y_train, self.config.batch_size, shuffle=True, key=loader_key
        )

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for x_batch, y_batch in pbar:
                key, step_key = jax.random.split(key)
                self.model, self.opt_state, loss = self._train_step(
                    self.model, self.opt_state, x_batch, y_batch, step_key
                )

                # Compute accuracy for display (use inference mode)
                inference_model = eqx.nn.inference_mode(self.model)
                dummy_key = jax.random.PRNGKey(0)
                batch_keys = jax.random.split(dummy_key, x_batch.shape[0])
                logits = jax.vmap(inference_model)(x_batch, key=batch_keys)
                preds = jnp.argmax(logits, axis=-1)
                acc = jnp.mean(preds == y_batch)

                epoch_loss += float(loss)
                epoch_acc += float(acc)
                n_batches += 1
                self.step += 1

                pbar.set_postfix({"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"})

            # Record epoch metrics
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(avg_acc)

            # Validation
            if val_data is not None:
                val_loss, val_acc = self.evaluate(val_data)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                print(
                    f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}: train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}"
                )

            # Checkpoint
            if (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}")

        return self.model

    def evaluate(
        self, data: tuple[jax.Array, jax.Array], **kwargs
    ) -> tuple[float, float]:
        """Evaluate model on data.

        Args:
            data: (X, y) tuple

        Returns:
            (loss, accuracy) tuple
        """
        X, y = data
        loader = DataLoader(X, y, self.config.batch_size, shuffle=False)

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for x_batch, y_batch in loader:
            loss, acc = self._eval_step(self.model, x_batch, y_batch, **kwargs)
            total_loss += float(loss)
            total_acc += float(acc)
            n_batches += 1

        return total_loss / n_batches, total_acc / n_batches

    def save_checkpoint(self, name: str) -> Path:
        """Save model checkpoint.

        Args:
            name: Checkpoint name

        Returns:
            Path to saved checkpoint
        """
        ckpt_path = self.checkpoint_dir / name
        eqx.tree_serialise_leaves(str(ckpt_path) + ".eqx", self.model)
        print(f"Saved checkpoint to {ckpt_path}.eqx")
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        self.model = eqx.tree_deserialise_leaves(str(path), self.model)
        print(f"Loaded checkpoint from {path}")


def cross_entropy_loss(
    model: eqx.Module, x: jax.Array, y: jax.Array, *, key: jax.Array
) -> jax.Array:
    """Cross-entropy loss for classification.

    Args:
        model: Equinox model
        x: Input batch (N, ...)
        y: Label batch (N,)
        key: Random key for dropout

    Returns:
        Mean cross-entropy loss
    """
    keys = jax.random.split(key, x.shape[0])
    logits = jax.vmap(model)(x, key=keys)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def mse_loss(
    model: eqx.Module, x: jax.Array, y: jax.Array, *, key: jax.Array
) -> jax.Array:
    """Mean squared error loss for regression.

    Args:
        model: Equinox model
        x: Input batch (N, features)
        y: Target batch (N,)
        key: Random key for dropout

    Returns:
        Mean squared error
    """
    keys = jax.random.split(key, x.shape[0])
    preds = jax.vmap(model)(x, key=keys).squeeze()
    return jnp.mean((preds - y) ** 2)
