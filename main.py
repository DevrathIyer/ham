import argparse
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from data import (DataLoader, get_cifar10_data, get_fashion_mnist_data,
                  get_mnist_data)
from hopfield import convert_hnm_to_hopfield
from models import CNN, HNL, HNM, MLP, count_parameters
from training import TrainConfig, Trainer, hnm_cross_entropy_loss, mse_loss
from visualization import (plot_confusion_matrix, plot_hnm_mem_weights,
                           plot_image_predictions, plot_training_history)


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    input_shape: tuple  # (C, H, W) for images or (features,) for tabular
    class_names: list[str] | None = None
    is_image: bool = True
    is_regression: bool = False


DATASET_CONFIGS = {
    "mnist": DatasetConfig(
        name="MNIST",
        num_classes=10,
        input_shape=(1, 28, 28),
        class_names=[str(i) for i in range(10)],
    ),
    "fashion_mnist": DatasetConfig(
        name="Fashion-MNIST",
        num_classes=10,
        input_shape=(1, 28, 28),
        class_names=[
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
    ),
    "cifar10": DatasetConfig(
        name="CIFAR-10",
        num_classes=10,
        input_shape=(3, 32, 32),
        class_names=[
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
    ),
}


def load_dataset(
    dataset_name: str,
    model_type: str,
    key: jax.Array | None = None,
) -> tuple[tuple, tuple, DatasetConfig]:
    config = DATASET_CONFIGS[dataset_name]
    flatten = model_type != "cnn" and config.is_image

    if dataset_name == "mnist":
        train_data, test_data = get_mnist_data(flatten=flatten)
    elif dataset_name == "fashion_mnist":
        train_data, test_data = get_fashion_mnist_data(flatten=flatten)
    elif dataset_name == "cifar10":
        train_data, test_data = get_cifar10_data()
        if flatten:
            X_train, y_train = train_data
            X_test, y_test = test_data
            train_data = (X_train.reshape(X_train.shape[0], -1), y_train)
            test_data = (X_test.reshape(X_test.shape[0], -1), y_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_data, test_data, config


def create_model(
    model_type: str,
    config: DatasetConfig,
    key: jax.Array,
    flatten: bool = False,
) -> tuple:
    if model_type == "mlp":
        if config.is_image:
            in_feats = 1
            for dim in config.input_shape:
                in_feats *= dim
        else:
            in_feats = config.input_shape[0]

        out_feats = config.num_classes if not config.is_regression else 1

        # Scale hidden layers based on input size
        if in_feats > 1000:
            hidden_dims = [256, 128]
        elif in_feats > 100:
            hidden_dims = [64, 8]
        else:
            hidden_dims = [32, 16]

        return MLP.create(in_feats, hidden_dims, out_feats, key=key)

    elif model_type == "hnm":
        layers = []

        l1_key, l3_key = jax.random.split(key, 2)
        layers.append(HNL(784, 64, 8, 8, key=l1_key))
        layers.append(HNL(64, 8, 10, 1, key=l3_key, is_class=True))

        return HNM(layers)

    elif model_type == "cnn":
        if not config.is_image:
            raise ValueError(
                f"CNN requires image data, but {config.name} is not an image dataset"
            )

        return CNN.create(
            in_channels=config.input_shape[0],
            num_classes=config.num_classes,
            image_size=config.input_shape[1],
            key=key,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(
    dataset: str,
    model_type: str,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    convert_to_hopfield: bool = False,
    hopfield_iterations: int = 5,
    hopfield_binary_dim: int | None = None,
    temp_start: float | None = None,
    temp_end: float | None = None,
    temp_anneal_steps: int | None = None,
    resume_checkpoint: str | None = None,
) -> None:
    config = DATASET_CONFIGS[dataset]

    print("=" * 60)
    print(f"Training {model_type.upper()} on {config.name}")
    print("=" * 60)

    key = jax.random.PRNGKey(seed)
    key, data_key, model_key, train_key, hopfield_key = jax.random.split(key, 5)

    # Load data
    print(f"Loading {config.name} data...")
    flatten = model_type == "mlp" and config.is_image
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset, model_type, data_key)[
        :2
    ]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create model
    model = create_model(model_type, config, model_key, flatten=flatten)
    print(f"Model parameters: {count_parameters(model):,}")

    # Load checkpoint if resuming
    if resume_checkpoint is not None:
        model = eqx.tree_deserialise_leaves(resume_checkpoint, model)
        print(f"Loaded checkpoint from {resume_checkpoint}")

    # Select loss function
    loss_fn = mse_loss if config.is_regression else hnm_cross_entropy_loss

    # Training config
    train_config = TrainConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=f"./checkpoints/{dataset}_{model_type}",
        checkpoint_every=max(1, epochs // 5),
        temp_start=temp_start,
        temp_end=temp_end,
        temp_anneal_steps=temp_anneal_steps,
    )

    # Handle regression separately (custom training loop)
    if config.is_regression:
        _train_regression(
            model, (X_train, y_train), (X_test, y_test), train_config, train_key
        )
        return

    # Standard classification training
    trainer = Trainer(model, loss_fn, train_config)
    trained_model = trainer.train((X_train, y_train), (X_test, y_test), key=train_key)

    # Evaluate
    test_loss, test_acc = trainer.evaluate((X_test, y_test), temperature=temp_end or 1.0)
    print(f"\nFinal Test Accuracy (Soft Attention): {test_acc:.4f}")

    # Evaluate
    test_loss, test_acc = trainer.evaluate((X_test, y_test), hard=True)
    print(f"\nFinal Test Accuracy (Hard Attention): {test_acc:.4f}")

    # Convert to Hopfield if requested
    if convert_to_hopfield and model_type == "hnm":
        print("\n" + "=" * 60)
        print("Converting to Binary Hopfield Network")
        print("=" * 60)

        hopfield_model = convert_hnm_to_hopfield(
            trained_model,
            hopfield_key,
            binary_dim=hopfield_binary_dim,
            num_iterations=hopfield_iterations,
        )

        print(f"Hopfield iterations: {hopfield_iterations}")
        print(f"Binary dimension: {hopfield_binary_dim or 'auto (num_memories)'}")

        # Evaluate Hopfield model
        hopfield_logits = jax.vmap(hopfield_model)(X_test)
        hopfield_preds = jnp.argmax(hopfield_logits, axis=-1)
        hopfield_acc = jnp.mean(hopfield_preds == y_test)
        print(f"\nFinal Test Accuracy (Binary Hopfield): {hopfield_acc:.4f}")

        # Use Hopfield model for visualizations if converted
        eval_model = hopfield_model
    else:
        eval_model = trained_model

    # Visualizations
    output_dir = Path(f"./outputs/{dataset}_{model_type}")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_training_history(
        trainer.history, save_path=output_dir / "training_history.png"
    )

    # Confusion matrix
    if convert_to_hopfield and model_type == "hnm":
        # Hopfield models don't need dropout keys
        logits = jax.vmap(eval_model)(X_test)
    else:
        # Soft attention models need inference mode
        inference_model = eqx.nn.inference_mode(eval_model)
        dummy_key = jax.random.PRNGKey(0)
        test_keys = jax.random.split(dummy_key, X_test.shape[0])
        logits = jax.vmap(inference_model)(X_test, key=test_keys)
    preds = jnp.argmax(logits, axis=-1)
    plot_confusion_matrix(
        y_test,
        preds,
        class_names=config.class_names,
        save_path=output_dir / "confusion_matrix.png",
    )

    # Image predictions (for image datasets)
    if config.is_image and model_type == "cnn":
        plot_image_predictions(
            trained_model,
            X_test,
            y_test,
            n_samples=16,
            class_names=config.class_names,
            save_path=output_dir / "predictions.png",
        )

    # HNM mem weights visualization
    if model_type == "hnm":
        # Use image shape for image datasets to visualize as 2D images
        image_shape = (
            (config.input_shape[1], config.input_shape[2]) if config.is_image else None
        )
        plot_hnm_mem_weights(
            trained_model,
            image_shape=image_shape,
            save_path=output_dir / "hnm_mem_weights.png",
        )

    plt.close("all")
    print(f"Saved visualizations to {output_dir}")


def _train_regression(model, train_data, test_data, config, key):
    from tqdm import tqdm

    X_train, y_train = train_data
    X_test, y_test = test_data

    trainer = Trainer(model, mse_loss, config)
    train_loader = DataLoader(
        X_train, y_train, config.batch_size, shuffle=True, key=key
    )

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for x_batch, y_batch in pbar:
            trainer.model, trainer.opt_state, loss = trainer._train_step(
                trainer.model, trainer.opt_state, x_batch, y_batch
            )
            epoch_loss += float(loss)
            n_batches += 1
            pbar.set_postfix({"mse": f"{loss:.4f}"})

        avg_loss = epoch_loss / n_batches
        trainer.history["train_loss"].append(avg_loss)

        val_loss = mse_loss(trainer.model, X_test, y_test)
        trainer.history["val_loss"].append(float(val_loss))

        print(f"Epoch {epoch + 1}: train_mse={avg_loss:.4f}, val_mse={val_loss:.4f}")

    final_mse = mse_loss(trainer.model, X_test, y_test)
    print(f"\nFinal Test MSE: {final_mse:.4f}")

    # Visualizations
    output_dir = Path("./outputs/regression_mlp")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    epochs_range = range(1, len(trainer.history["train_loss"]) + 1)
    ax.plot(epochs_range, trainer.history["train_loss"], "b-", label="Train MSE")
    ax.plot(epochs_range, trainer.history["val_loss"], "r-", label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Regression Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_history.png", dpi=150, bbox_inches="tight")

    inference_model = eqx.nn.inference_mode(trainer.model)
    dummy_key = jax.random.PRNGKey(0)
    test_keys = jax.random.split(dummy_key, X_test.shape[0])
    preds = jax.vmap(inference_model)(X_test, key=test_keys).squeeze()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, preds, alpha=0.5, edgecolors="black")
    lims = [
        min(float(y_test.min()), float(preds.min())),
        max(float(y_test.max()), float(preds.max())),
    ]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predictions vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "predictions.png", dpi=150, bbox_inches="tight")

    plt.close("all")
    print(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="JAX/Equinox ML Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --dataset mnist --model mlp
  python main.py --dataset cifar10 --model cnn --epochs 20
  python main.py --dataset fashion_mnist --model cnn --lr 0.001

  # Train HNM without annealing, then resume with annealing:
  python main.py -d mnist -m hnm --epochs 20
  python main.py -d mnist -m hnm --epochs 20 --resume ./checkpoints/mnist_hnm/epoch_20.eqx --temp-start 1.0 --temp-end 0.005
        """,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="mnist",
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="cnn",
        choices=["mlp", "hnm", "cnn"],
        help="Model architecture",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", "-e", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--hopfield",
        action="store_true",
        help="Convert to binary Hopfield network after training (HNM only)",
    )
    parser.add_argument(
        "--hopfield-iterations",
        type=int,
        default=100,
        help="Number of Hopfield iterations (default: 5)",
    )
    parser.add_argument(
        "--hopfield-binary-dim",
        type=int,
        default=512,
        help="Binary dimension for Hopfield projection (default: num_memories)",
    )
    parser.add_argument(
        "--temp-start",
        type=float,
        default=None,
        help="Starting temperature for annealing (e.g., 1.0 for soft attention)",
    )
    parser.add_argument(
        "--temp-end",
        type=float,
        default=None,
        help="Ending temperature for annealing (e.g., 0.005 for sharp attention)",
    )
    parser.add_argument(
        "--temp-anneal-steps",
        type=int,
        default=None,
        help="Number of steps to anneal over (default: all training steps)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .eqx checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Validate combinations
    config = DATASET_CONFIGS[args.dataset]
    if args.model == "cnn" and not config.is_image:
        parser.error(
            f"{args.model.upper()} requires image data. {config.name} is not an image dataset. Use --model mlp instead."
        )

    if args.hopfield and args.model != "hnm":
        parser.error("--hopfield only works with HNM models")

    train(
        dataset=args.dataset,
        model_type=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        convert_to_hopfield=args.hopfield,
        hopfield_iterations=args.hopfield_iterations,
        hopfield_binary_dim=args.hopfield_binary_dim,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        temp_anneal_steps=args.temp_anneal_steps,
        resume_checkpoint=args.resume,
    )


if __name__ == "__main__":
    main()
