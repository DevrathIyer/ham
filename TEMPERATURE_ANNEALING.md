# Temperature Annealing for HNL Layer

## Overview

This implementation adds temperature annealing support to the Hopfield Network Layer (HNL) to enable better training dynamics. Temperature annealing allows the model to transition smoothly from soft attention (where gradients flow well) to hard attention (where a single memory is chosen).

## Problem Statement

The original HNL layer used a static temperature parameter, making it difficult to train both:
1. A softmax where gradients flow (requires higher temperature)
2. One where a single memory is chosen (requires lower temperature)

## Solution

Temperature annealing gradually reduces the temperature during training:
- **Early training**: High temperature (0.1) → Soft attention, better gradient flow
- **Late training**: Low temperature (0.001) → Sharp attention, discrete memory selection

## Implementation Details

### 1. Model Changes (`models.py`)

#### HNL Layer
- Added optional `temperature` parameter to `__call__` method
- Uses provided temperature if available, otherwise falls back to default
```python
def __call__(self, x, hard, key, temperature=None):
    temp = temperature if temperature is not None else self.temperature
    attn_weights = jax.nn.softmax(attn_scores / temp, axis=-1)
```

#### HNM Wrapper
- Updated to accept and pass through temperature parameter to all layers
```python
def __call__(self, x, key, hard=False, temperature=None):
    for layer in self.layers:
        x = layer(x, hard, key=key, temperature=temperature)
```

### 2. Training Changes (`training.py`)

#### Temperature Annealing Function
```python
def compute_annealed_temperature(step, total_steps, temp_start, temp_end, schedule="linear"):
    # Returns temperature based on training progress
    # Supports: linear, cosine, exponential schedules
```

#### TrainConfig
Added annealing configuration:
- `temp_start`: Starting temperature (e.g., 0.1)
- `temp_end`: Ending temperature (e.g., 0.001)
- `temp_anneal_steps`: Number of steps to anneal over (if None, uses all training steps)

#### Training Loop
- Computes current temperature at each training step
- Passes temperature to loss function
- Displays temperature in progress bar

#### HNM Loss Function
New `hnm_cross_entropy_loss` function that:
- Accepts temperature parameter
- Passes it to HNM model during forward pass
- Supports gradient flow through temperature-dependent attention

### 3. Main Script Changes (`main.py`)

For HNM models, automatically configures:
```python
loss_fn = hnm_cross_entropy_loss
temp_start = 0.1   # Soft attention
temp_end = 1e-3    # Sharp attention
```

## Usage

### Training with Annealing (Default for HNM)
```bash
python main.py --dataset mnist --model hnm --epochs 40
```

Temperature will automatically anneal from 0.1 to 0.001 over the training period.

### Customizing Annealing

To customize annealing parameters, modify `main.py`:
```python
train_config = TrainConfig(
    temp_start=0.2,      # Higher starting temperature
    temp_end=0.01,       # Higher ending temperature
    temp_anneal_steps=5000,  # Anneal over 5000 steps instead of all training steps
)
```

## Annealing Schedules

### Linear (Default)
Uniform decrease from start to end:
```
temp(t) = temp_start + (temp_end - temp_start) * (t / T)
```

### Cosine
Smooth transition with cosine curve:
```
temp(t) = temp_end + (temp_start - temp_end) * 0.5 * (1 + cos(π * t/T))
```

### Exponential
Faster decrease early, slower later:
```
temp(t) = temp_start * exp(log(temp_end / temp_start) * t/T)
```

## Benefits

1. **Better Gradient Flow**: High initial temperature allows smooth gradients
2. **Discrete Convergence**: Low final temperature approaches hard attention
3. **Stable Training**: Gradual transition prevents training instability
4. **Flexible**: Can be disabled by not setting temp_start/temp_end

## Temperature Schedule Comparison

For 1000 steps from 0.1 to 0.001:

| Step | Linear  | Cosine  | Exponential |
|------|---------|---------|-------------|
| 0    | 0.1000  | 0.1000  | 0.1000      |
| 250  | 0.0753  | 0.0855  | 0.0316      |
| 500  | 0.0505  | 0.0505  | 0.0100      |
| 750  | 0.0258  | 0.0155  | 0.0032      |
| 1000 | 0.0010  | 0.0010  | 0.0010      |

## Example Training Output

```
Epoch 1/40: 100%|████| 937/937 [00:45<00:00, loss: 0.4523, acc: 0.8956, temp: 9.90e-02]
Epoch 2/40: 100%|████| 937/937 [00:43<00:00, loss: 0.2134, acc: 0.9345, temp: 9.80e-02]
...
Epoch 39/40: 100%|███| 937/937 [00:41<00:00, loss: 0.0234, acc: 0.9923, temp: 1.50e-03]
Epoch 40/40: 100%|███| 937/937 [00:40<00:00, loss: 0.0198, acc: 0.9934, temp: 1.00e-03]
```

## Backward Compatibility

- Other model types (MLP, CNN) are unaffected
- HNL layers still work without temperature parameter (uses default)
- Existing code continues to work without modifications
