from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from data import DataLoader


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    epochs: int = 10
    batch_size: int = 32
    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 5
    # Temperature annealing parameters (both must be set to enable annealing)
    temp_start: float | None = (
        None  # Starting temperature (e.g., 0.1 for soft attention)
    )
    temp_end: float | None = (
        None  # Ending temperature (e.g., 0.001 for sharp attention)
    )
    temp_anneal_steps: int | None = (
        None  # Number of steps to anneal over (if None, uses all epochs)
    )


def compute_annealed_temperature(
    step: int,
    total_steps: int,
    temp_start: float,
    temp_end: float,
) -> float:
    if step >= total_steps:
        return temp_end

    progress = step / total_steps

    decay_rate = jnp.log(temp_end / temp_start)
    return temp_start * jnp.exp(decay_rate * progress)


class Trainer:
    def __init__(
        self,
        model: eqx.Module,
        loss_fn: Callable,
        config: TrainConfig | None = None,
        optimizer: optax.GradientTransformation | None = None,
    ):
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

        # Validate temperature annealing configuration
        if (self.config.temp_start is None) != (self.config.temp_end is None):
            raise ValueError(
                "Both temp_start and temp_end must be set together to enable annealing, "
                "or both must be None to disable it."
            )
        if (
            self.config.temp_start is not None
            and self.config.temp_end is not None
            and self.config.temp_start <= self.config.temp_end
        ):
            raise ValueError(
                f"temp_start ({self.config.temp_start}) must be greater than "
                f"temp_end ({self.config.temp_end}) for annealing to work."
            )

    def _compute_logits(
        self,
        model: eqx.Module,
        x: jax.Array,
        keys: jax.Array,
        hard: bool = False,
        temperature: float | None = None,
    ) -> jax.Array:
        return jax.vmap(model, in_axes=(0, 0, None, None))(x, keys, hard, temperature)

    @eqx.filter_jit
    def _train_step(
        self,
        model: eqx.Module,
        opt_state: optax.OptState,
        x: jax.Array,
        y: jax.Array,
        key: jax.Array,
        temperature: float | None = None,
    ) -> tuple[eqx.Module, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(
            model, x, y, key=key, temperature=temperature
        )
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
        temperature: float = 1.0,
        hard: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        model = eqx.nn.inference_mode(model)
        dummy_key = jax.random.PRNGKey(0)
        keys = jax.random.split(dummy_key, x.shape[0])
        logits = self._compute_logits(
            model, x, keys, hard=hard, temperature=temperature
        )
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
        X_train, y_train = train_data
        key, loader_key = jax.random.split(key)
        train_loader = DataLoader(
            X_train, y_train, self.config.batch_size, shuffle=True, key=loader_key
        )

        # Calculate total steps for temperature annealing
        steps_per_epoch = len(X_train) // self.config.batch_size
        if self.config.temp_anneal_steps is not None:
            total_anneal_steps = self.config.temp_anneal_steps
        else:
            total_anneal_steps = self.config.epochs * steps_per_epoch

        # Determine if we should use temperature annealing
        use_annealing = (
            self.config.temp_start is not None and self.config.temp_end is not None
        )

        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.epochs}")
            for x_batch, y_batch in pbar:
                # Compute annealed temperature for this step
                if use_annealing:
                    temperature = jnp.array(
                        compute_annealed_temperature(
                            self.step,
                            total_anneal_steps,
                            self.config.temp_start,
                            self.config.temp_end,
                        )
                    )
                else:
                    temperature = jnp.array(1.0)

                key, step_key = jax.random.split(key)
                self.model, self.opt_state, loss = self._train_step(
                    self.model, self.opt_state, x_batch, y_batch, step_key, temperature
                )

                # Compute accuracy for display (use inference mode with default temperature)
                # Note: We use model's default temperature (not annealed) for consistent metrics
                inference_model = eqx.nn.inference_mode(self.model)
                dummy_key = jax.random.PRNGKey(0)
                batch_keys = jax.random.split(dummy_key, x_batch.shape[0])

                # Use helper to compute logits with consistent calling convention
                logits = self._compute_logits(
                    inference_model,
                    x_batch,
                    batch_keys,
                    hard=False,
                    temperature=temperature,
                )
                preds = jnp.argmax(logits, axis=-1)
                acc = jnp.mean(preds == y_batch)

                epoch_loss += float(loss)
                epoch_acc += float(acc)
                n_batches += 1
                self.step += 1

                # Show temperature in progress bar if using annealing
                postfix = {"loss": f"{loss:.4f}", "acc": f"{acc:.4f}"}
                if use_annealing:
                    postfix["temp"] = f"{temperature:.4e}"
                pbar.set_postfix(postfix)

            # Record epoch metrics
            avg_loss = epoch_loss / n_batches
            avg_acc = epoch_acc / n_batches
            self.history["train_loss"].append(avg_loss)
            self.history["train_acc"].append(avg_acc)

            # Validation
            if val_data is not None:
                val_loss, val_acc = self.evaluate(val_data, temperature=temperature)
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
        ckpt_path = self.checkpoint_dir / name
        eqx.tree_serialise_leaves(str(ckpt_path) + ".eqx", self.model)
        print(f"Saved checkpoint to {ckpt_path}.eqx")
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        self.model = eqx.tree_deserialise_leaves(str(path), self.model)
        print(f"Loaded checkpoint from {path}")


def cross_entropy_loss(
    model: eqx.Module, x: jax.Array, y: jax.Array, *, key: jax.Array
) -> jax.Array:
    keys = jax.random.split(key, x.shape[0])
    logits = jax.vmap(model)(x, key=keys)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def mse_loss(
    model: eqx.Module, x: jax.Array, y: jax.Array, *, key: jax.Array
) -> jax.Array:
    keys = jax.random.split(key, x.shape[0])
    preds = jax.vmap(model)(x, key=keys).squeeze()
    return jnp.mean((preds - y) ** 2)


def hnm_cross_entropy_loss(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    *,
    key: jax.Array,
    temperature: float | None = None,
) -> jax.Array:
    keys = jax.random.split(key, x.shape[0])
    # HNM models take (x, key, hard, temperature)
    logits = jax.vmap(model, in_axes=(0, 0, None, None))(x, keys, False, temperature)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
