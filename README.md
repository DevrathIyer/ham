# JAX/Equinox ML Training Skeleton

A minimal, extensible framework for training small ML models using JAX and Equinox.

## Features

- **Models**: MLP, CNN, and ResidualBlock implementations using Equinox
- **Datasets**: MNIST (auto-download) and synthetic data generators
- **Training**: Generic trainer with checkpointing via Equinox serialization
- **Visualization**: Training curves, confusion matrices, predictions, decision boundaries

## Installation

```bash
cd jax_ml_skeleton
uv sync
```

## Quick Start

```bash
# Train MLP on MNIST
uv run train --task mnist_mlp --epochs 5

# Train CNN on MNIST
uv run train --task mnist_cnn --epochs 5

# Train on synthetic 2D classification
uv run train --task synthetic --epochs 20

# Train regression model
uv run train --task regression --epochs 50

# Run all examples
uv run train --task all
```

## Project Structure

```
jax_ml_skeleton/
├── src/jax_ml_skeleton/
│   ├── __init__.py
│   ├── models.py        # MLP, CNN, ResidualBlock
│   ├── data.py          # MNIST, synthetic data loaders
│   ├── training.py      # Trainer, loss functions
│   ├── visualization.py # Plotting utilities
│   └── main.py          # CLI entry point
├── checkpoints/         # Model checkpoints
├── outputs/             # Visualizations
└── pyproject.toml
```

## Usage Examples

### Custom Model Training

```python
import jax
from jax_ml_skeleton import MLP, Trainer
from jax_ml_skeleton.training import TrainConfig, cross_entropy_loss
from jax_ml_skeleton.data import get_synthetic_data

key = jax.random.PRNGKey(42)
key, data_key, model_key, train_key = jax.random.split(key, 4)

# Generate data
(X_train, y_train), (X_test, y_test) = get_synthetic_data(
    n_samples=1000, n_features=20, n_classes=5, key=data_key
)

# Create model
model = MLP(
    in_features=20,
    hidden_dims=[64, 32],
    out_features=5,
    key=model_key,
)

# Train
config = TrainConfig(learning_rate=1e-3, epochs=10, batch_size=32)
trainer = Trainer(model, cross_entropy_loss, config)
trained_model = trainer.train((X_train, y_train), (X_test, y_test), key=train_key)
```

### Loading Checkpoints

```python
trainer.save_checkpoint("my_model")
# Later...
trainer.load_checkpoint("./checkpoints/my_model.eqx")
```

### Custom Loss Function

```python
import jax.numpy as jnp

def custom_loss(model, x, y):
    logits = jax.vmap(model)(x)
    # Your custom loss computation
    return jnp.mean((logits - y) ** 2)

trainer = Trainer(model, custom_loss, config)
```

## Outputs

After training, visualizations are saved to `./outputs/<task>/`:
- `training_history.png`: Loss and accuracy curves
- `confusion_matrix.png`: Classification confusion matrix
- `predictions.png`: Sample predictions with confidence
- `decision_boundary.png`: 2D decision boundary (synthetic data)
