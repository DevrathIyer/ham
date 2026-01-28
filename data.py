"""Dataset loading utilities."""

import gzip
import hashlib
import pickle
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator

import jax
import jax.numpy as jnp
import numpy as np


MNIST_URLS = {
    "train_images": "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
}

FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def _download_file(url: str, filepath: Path) -> None:
    """Download a file if it doesn't exist."""
    if filepath.exists():
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filepath)


def _read_mnist_images(filepath: Path) -> np.ndarray:
    """Read MNIST image file."""
    with gzip.open(filepath, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images


def _read_mnist_labels(filepath: Path) -> np.ndarray:
    """Read MNIST label file."""
    with gzip.open(filepath, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def get_mnist_data(
    data_dir: str | Path = "./data/mnist",
    flatten: bool = False,
    normalize: bool = True,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Load MNIST dataset.

    Args:
        data_dir: Directory to store/load data
        flatten: If True, flatten images to vectors
        normalize: If True, normalize pixel values to [0, 1]

    Returns:
        ((train_images, train_labels), (test_images, test_labels))
    """
    data_dir = Path(data_dir)

    # Download files
    for name, url in MNIST_URLS.items():
        filepath = data_dir / Path(url).name
        _download_file(url, filepath)

    # Load data
    train_images = _read_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = _read_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = _read_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    # Convert to float and normalize
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    else:
        # Add channel dimension for CNN: (N, H, W) -> (N, C, H, W)
        train_images = train_images[:, np.newaxis, :, :]
        test_images = test_images[:, np.newaxis, :, :]

    return (
        (jnp.array(train_images), jnp.array(train_labels)),
        (jnp.array(test_images), jnp.array(test_labels)),
    )


def get_fashion_mnist_data(
    data_dir: str | Path = "./data/fashion_mnist",
    flatten: bool = False,
    normalize: bool = True,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Load Fashion-MNIST dataset (MEDIUM difficulty).

    Fashion-MNIST is a drop-in replacement for MNIST with clothing items
    instead of digits. Same format (28x28 grayscale) but harder to classify.

    Difficulty: 3/5 - More complex patterns than handwritten digits.

    Classes: T-shirt/top, Trouser, Pullover, Dress, Coat,
             Sandal, Shirt, Sneaker, Bag, Ankle boot

    Args:
        data_dir: Directory to store/load data
        flatten: If True, flatten images to vectors
        normalize: If True, normalize pixel values to [0, 1]

    Returns:
        ((train_images, train_labels), (test_images, test_labels))
    """
    data_dir = Path(data_dir)

    # Download files
    for name, url in FASHION_MNIST_URLS.items():
        filepath = data_dir / Path(url).name
        _download_file(url, filepath)

    # Load data (same format as MNIST)
    train_images = _read_mnist_images(data_dir / "train-images-idx3-ubyte.gz")
    train_labels = _read_mnist_labels(data_dir / "train-labels-idx1-ubyte.gz")
    test_images = _read_mnist_images(data_dir / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_mnist_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

    # Convert to float and normalize
    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    if flatten:
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)
    else:
        # Add channel dimension for CNN: (N, H, W) -> (N, C, H, W)
        train_images = train_images[:, np.newaxis, :, :]
        test_images = test_images[:, np.newaxis, :, :]

    return (
        (jnp.array(train_images), jnp.array(train_labels)),
        (jnp.array(test_images), jnp.array(test_labels)),
    )


def get_cifar10_data(
    data_dir: str | Path = "./data/cifar10",
    normalize: bool = True,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Load CIFAR-10 dataset (HARD difficulty).

    CIFAR-10 contains 60,000 32x32 color images in 10 classes.
    Much harder than MNIST due to color and more complex objects.

    Difficulty: 4/5 - Requires deeper networks, data augmentation helps.

    Classes: airplane, automobile, bird, cat, deer,
             dog, frog, horse, ship, truck

    Args:
        data_dir: Directory to store/load data
        normalize: If True, normalize pixel values to [0, 1]

    Returns:
        ((train_images, train_labels), (test_images, test_labels))
        Images shape: (N, 3, 32, 32) - channels first format
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "cifar-10-python.tar.gz"
    extracted_dir = data_dir / "cifar-10-batches-py"

    # Download if needed
    _download_file(CIFAR10_URL, tar_path)

    # Extract if needed
    if not extracted_dir.exists():
        print("Extracting CIFAR-10...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)

    # Load training batches
    train_images = []
    train_labels = []
    for i in range(1, 6):
        batch_path = extracted_dir / f"data_batch_{i}"
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
            train_images.append(batch[b"data"])
            train_labels.extend(batch[b"labels"])

    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.array(train_labels)

    # Load test batch
    test_path = extracted_dir / "test_batch"
    with open(test_path, "rb") as f:
        test_batch = pickle.load(f, encoding="bytes")
        test_images = test_batch[b"data"]
        test_labels = np.array(test_batch[b"labels"])

    # Reshape to (N, C, H, W) - CIFAR stores as (N, 3072) row-major RGB
    train_images = train_images.reshape(-1, 3, 32, 32).astype(np.float32)
    test_images = test_images.reshape(-1, 3, 32, 32).astype(np.float32)

    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    return (
        (jnp.array(train_images), jnp.array(train_labels)),
        (jnp.array(test_images), jnp.array(test_labels)),
    )


def get_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 3,
    noise: float = 0.1,
    *,
    key: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Generate synthetic classification dataset.

    Creates clusters of points around class centers with Gaussian noise.

    Args:
        n_samples: Total number of samples
        n_features: Number of features per sample
        n_classes: Number of classes
        noise: Standard deviation of Gaussian noise
        key: JAX random key

    Returns:
        ((train_X, train_y), (test_X, test_y))
    """
    keys = jax.random.split(key, 4)

    # Generate class centers
    centers = jax.random.normal(keys[0], (n_classes, n_features)) * 3

    # Assign labels
    labels = jnp.tile(jnp.arange(n_classes), n_samples // n_classes + 1)[:n_samples]

    # Generate samples around centers
    X = centers[labels] + jax.random.normal(keys[1], (n_samples, n_features)) * noise

    # Shuffle
    perm = jax.random.permutation(keys[2], n_samples)
    X = X[perm]
    labels = labels[perm]

    # Split into train/test (80/20)
    split_idx = int(0.8 * n_samples)
    train_X, test_X = X[:split_idx], X[split_idx:]
    train_y, test_y = labels[:split_idx], labels[split_idx:]

    return (train_X, train_y), (test_X, test_y)


def get_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,
    noise: float = 0.1,
    *,
    key: jax.Array,
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Generate synthetic regression dataset.

    Creates data following y = X @ w + noise.

    Args:
        n_samples: Total number of samples
        n_features: Number of features
        noise: Standard deviation of noise
        key: JAX random key

    Returns:
        ((train_X, train_y), (test_X, test_y))
    """
    keys = jax.random.split(key, 4)

    # True weights
    w = jax.random.normal(keys[0], (n_features,))

    # Generate X
    X = jax.random.normal(keys[1], (n_samples, n_features))

    # Generate y with noise
    y = X @ w + jax.random.normal(keys[2], (n_samples,)) * noise

    # Shuffle and split
    perm = jax.random.permutation(keys[3], n_samples)
    X, y = X[perm], y[perm]

    split_idx = int(0.8 * n_samples)
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])


class DataLoader:
    """Simple batched data loader."""

    def __init__(
        self,
        X: jax.Array,
        y: jax.Array,
        batch_size: int,
        shuffle: bool = True,
        *,
        key: jax.Array | None = None,
    ):
        """Initialize data loader.

        Args:
            X: Input features
            y: Labels/targets
            batch_size: Batch size
            shuffle: Whether to shuffle each epoch
            key: JAX random key (required if shuffle=True)
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.key = key
        self.n_samples = X.shape[0]

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[jax.Array, jax.Array]]:
        """Iterate over batches."""
        indices = jnp.arange(self.n_samples)

        if self.shuffle and self.key is not None:
            self.key, subkey = jax.random.split(self.key)
            indices = jax.random.permutation(subkey, indices)

        for i in range(0, self.n_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]
