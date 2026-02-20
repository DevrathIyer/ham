import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from data import get_cifar10_data, get_fashion_mnist_data, get_mnist_data
from hopfield import convert_hnm_to_hopfield
from models import CNN, HNL, HNM, MLP, count_parameters
from training import TrainConfig, Trainer, make_hnm_loss
from visualization import (plot_confusion_matrix, plot_hnm_mem_weights,
                           plot_image_predictions, plot_mem_correlations,
                           plot_training_history)


@dataclass
class HNLConfig:
    in_feats: int
    out_feats: int
    num_mems: int
    num_heads: int
    is_class: bool = False
    dropout_rate: float = 0.0


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    input_shape: tuple  # (C, H, W) for images or (features,) for tabular
    class_names: list[str] | None = None
    is_image: bool = True
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [1024, 1024])
    hnm_layers: list[HNLConfig] = field(default_factory=list)


DATASET_CONFIGS = {
    "mnist": DatasetConfig(
        name="MNIST",
        num_classes=10,
        input_shape=(1, 28, 28),
        class_names=[str(i) for i in range(10)],
        mlp_hidden_dims=[1024, 1024],
        hnm_layers=[
            HNLConfig(784, 512, 4, 4),
            HNLConfig(512, 128, 10, 1, is_class=True),
        ],
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
        mlp_hidden_dims=[1024, 1024],
        hnm_layers=[
            HNLConfig(784, 1024, 32, 4),
            HNLConfig(1024, 32, 10, 1, is_class=True),
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
        mlp_hidden_dims=[256, 128],
        hnm_layers=[
            HNLConfig(3072, 256, 32, 4),
            HNLConfig(256, 128, 32, 4),
            HNLConfig(128, 32, 10, 1, is_class=True),
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

        return MLP.create(in_feats, config.mlp_hidden_dims, config.num_classes, key=key)

    elif model_type == "hnm":
        if not config.hnm_layers:
            raise ValueError(f"No HNM architecture defined for dataset '{config.name}'")

        keys = jax.random.split(key, len(config.hnm_layers))
        layers = [
            HNL(
                cfg.in_feats,
                cfg.out_feats,
                cfg.num_mems,
                cfg.num_heads,
                is_class=cfg.is_class,
                dropout_rate=cfg.dropout_rate,
                key=k,
            )
            for cfg, k in zip(config.hnm_layers, keys)
        ]
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
    temp_start: float | None = None,
    temp_end: float | None = None,
    temp_anneal_steps: int | None = None,
    layer_anneal: bool = False,
    restore_checkpoint: str | None = None,
    corr_penalty: float = 0.0,
    util_penalty: float = 0.0,
) -> tuple[Trainer, eqx.Module, tuple, str]:
    config = DATASET_CONFIGS[dataset]

    print("=" * 60)
    print(f"Training {model_type.upper()} on {config.name}")
    print("=" * 60)

    key = jax.random.PRNGKey(seed)
    key, data_key, model_key, train_key = jax.random.split(key, 4)

    # Load data
    print(f"Loading {config.name} data...")
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset, model_type, data_key)[
        :2
    ]
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Create model
    flatten = model_type == "mlp" and config.is_image
    model = create_model(model_type, config, model_key, flatten=flatten)
    print(f"Model parameters: {count_parameters(model):,}")

    # Resolve and load restore checkpoint
    checkpoint_dir = f"./checkpoints/{dataset}_{model_type}"
    if restore_checkpoint is not None:
        stem = Path(restore_checkpoint).stem  # strip .eqx if provided
        restore_path = f"{checkpoint_dir}/{stem}.eqx"
        model = eqx.tree_deserialise_leaves(restore_path, model)
        print(f"Loaded checkpoint from {restore_path}")

    # Checkpoint naming: timestamp + annealed/base suffix
    annealing = temp_start is not None and temp_end is not None
    if annealing and restore_checkpoint is not None:
        # Reuse the base model's timestamp: 20260216_143000_base -> 20260216_143000_annealed
        checkpoint_name = Path(restore_checkpoint).stem.rsplit("_", 1)[0] + "_annealed"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "annealed" if annealing else "base"
        checkpoint_name = f"{timestamp}_{suffix}"

    train_config = TrainConfig(
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        temp_start=temp_start,
        temp_end=temp_end,
        temp_anneal_steps=temp_anneal_steps,
        layer_anneal=layer_anneal,
        corr_penalty=corr_penalty,
        util_penalty=util_penalty,
    )

    trainer = Trainer(model, make_hnm_loss(corr_penalty, util_penalty), train_config)
    trained_model = trainer.train((X_train, y_train), (X_test, y_test), key=train_key)

    test_loss, test_acc = trainer.evaluate((X_test, y_test))
    print(f"\nFinal Test Accuracy (Soft Attention): {test_acc:.4f}")
    test_loss, test_acc = trainer.evaluate((X_test, y_test), hard=True)
    print(f"\nFinal Test Accuracy (Hard Attention): {test_acc:.4f}")

    return trainer, trained_model, (X_test, y_test), checkpoint_name


def run_hopfield(
    model: eqx.Module,
    X_test: jax.Array,
    y_test: jax.Array,
    key: jax.Array,
    iterations: int,
    binary_dim: int,
    active_dims: int,
) -> eqx.Module:
    print("\n" + "=" * 60)
    print("Converting to Binary Hopfield Network")
    print("=" * 60)

    hopfield_model = convert_hnm_to_hopfield(
        model,
        key,
        binary_dim=binary_dim,
        active_dims=active_dims,
        num_iterations=iterations,
    )

    print(f"Hopfield iterations: {iterations}")
    print(f"Binary dimension: {binary_dim}")
    print(f"Active dims: {active_dims}")

    hopfield_logits = jax.vmap(hopfield_model)(X_test)
    hopfield_preds = jnp.argmax(hopfield_logits, axis=-1)
    hopfield_acc = jnp.mean(hopfield_preds == y_test)
    print(f"\nFinal Test Accuracy (Binary Hopfield): {hopfield_acc:.4f}")

    return hopfield_model


def visualize(
    trainer: Trainer,
    eval_model: eqx.Module,
    config: DatasetConfig,
    dataset: str,
    model_type: str,
    checkpoint_name: str,
    X_test: jax.Array,
    y_test: jax.Array,
    is_hopfield: bool = False,
) -> None:
    output_dir = Path(f"./outputs/{dataset}_{model_type}/{checkpoint_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if trainer.epoch > 0:
        plot_training_history(
            trainer.history, save_path=output_dir / "training_history.png"
        )

    if is_hopfield:
        logits = jax.vmap(eval_model)(X_test)
    else:
        inference_model = eqx.nn.inference_mode(eval_model)
        dummy_key = jax.random.PRNGKey(0)
        test_keys = jax.random.split(dummy_key, X_test.shape[0])
        logits = jax.vmap(inference_model, in_axes=(0, 0, None))(
            X_test, test_keys, False
        )

    preds = jnp.argmax(logits, axis=-1)
    plot_confusion_matrix(
        y_test,
        preds,
        class_names=config.class_names,
        save_path=output_dir / "confusion_matrix.png",
    )

    if config.is_image and model_type == "cnn":
        plot_image_predictions(
            eval_model,
            X_test,
            y_test,
            n_samples=16,
            class_names=config.class_names,
            save_path=output_dir / "predictions.png",
        )

    if model_type == "hnm" and trainer.history.get("mem_correlations"):
        plot_mem_correlations(
            trainer.history["mem_correlations"],
            save_path=output_dir / "mem_correlations.png",
        )

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

  # Train HNM without annealing, then restore with annealing:
  python main.py -d mnist -m hnm --epochs 20
  python main.py -d mnist -m hnm --epochs 20 --restore 20260216_143000_base --temp-start 1.0 --temp-end 0.005

  # Train without generating visualizations:
  python main.py -d mnist -m hnm --no-visualize
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
        "--layer-anneal",
        action="store_true",
        help="Anneal each non-class HNL layer sequentially, freezing each before moving to the next",
    )
    parser.add_argument(
        "--restore",
        type=str,
        default=None,
        help="Checkpoint name to restore from (e.g. 20260216_143000_base); dataset and model are inferred from --dataset and --model",
    )
    parser.add_argument(
        "--corr-penalty",
        type=float,
        default=0.0,
        help="Weight for memory correlation penalty (default: 0.0 = off)",
    )
    parser.add_argument(
        "--util-penalty",
        type=float,
        default=0.0,
        help="Weight for memory utilization entropy bonus (default: 0.0 = off)",
    )
    parser.add_argument(
        "--hopfield",
        action="store_true",
        help="Convert to binary Hopfield network after training (HNM only)",
    )
    parser.add_argument(
        "--hopfield-iterations",
        type=int,
        default=100,
        help="Number of Hopfield iterations",
    )
    parser.add_argument(
        "-hb",
        "--hopfield-binary-dim",
        type=int,
        default=512,
        help="Binary dimension for Hopfield projection",
    )
    parser.add_argument(
        "-ha",
        "--hopfield-active-dims",
        type=int,
        default=512,
        help="Number of active (top-k) dimensions used in binary attention",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip generating visualizations after training",
    )

    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    if args.model == "cnn" and not config.is_image:
        parser.error(
            f"{args.model.upper()} requires image data. {config.name} is not an image dataset. Use --model mlp instead."
        )
    if args.hopfield and args.model != "hnm":
        parser.error("--hopfield only works with HNM models")

    trainer, trained_model, (X_test, y_test), checkpoint_name = train(
        dataset=args.dataset,
        model_type=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        temp_anneal_steps=args.temp_anneal_steps,
        layer_anneal=args.layer_anneal,
        restore_checkpoint=args.restore,
        corr_penalty=args.corr_penalty,
        util_penalty=args.util_penalty,
    )

    eval_model = trained_model
    is_hopfield = False
    if args.hopfield:
        hopfield_key = jax.random.PRNGKey(args.seed + 1)
        eval_model = run_hopfield(
            trained_model,
            X_test,
            y_test,
            hopfield_key,
            args.hopfield_iterations,
            args.hopfield_binary_dim,
            args.hopfield_active_dims,
        )
        is_hopfield = True

    if not args.no_visualize:
        visualize(
            trainer,
            eval_model,
            config,
            args.dataset,
            args.model,
            checkpoint_name,
            X_test,
            y_test,
            is_hopfield=is_hopfield,
        )


if __name__ == "__main__":
    main()
