"""Diagnostic script to compare soft attention vs Hopfield outputs."""

import equinox as eqx
import jax
import jax.numpy as jnp

from data import get_mnist_data
from hopfield import convert_hcm_to_hopfield, convert_hnl_to_hopfield
from models import HCL, HCM, HNL


def compare_hnl_outputs(
    hnl: HNL,
    x: jax.Array,
    key: jax.Array,
    binary_dim: int = 64,
    num_iterations: int = 10,
):
    """Compare soft attention vs Hopfield output for a single HNL layer."""

    # Get soft attention output
    soft_model = eqx.nn.inference_mode(hnl)
    dummy_key = jax.random.PRNGKey(0)
    soft_out = soft_model(x, key=dummy_key)

    # Convert to Hopfield
    hopfield_hnl = convert_hnl_to_hopfield(
        hnl, key, binary_dim=binary_dim, num_iterations=num_iterations
    )
    hopfield_out = hopfield_hnl(x)

    # Compare
    l2_dist = jnp.linalg.norm(soft_out - hopfield_out)
    cosine_sim = jnp.dot(soft_out, hopfield_out) / (
        jnp.linalg.norm(soft_out) * jnp.linalg.norm(hopfield_out) + 1e-8
    )

    return {
        "soft_out": soft_out,
        "hopfield_out": hopfield_out,
        "l2_distance": float(l2_dist),
        "cosine_similarity": float(cosine_sim),
        "soft_norm": float(jnp.linalg.norm(soft_out)),
        "hopfield_norm": float(jnp.linalg.norm(hopfield_out)),
    }


def trace_hopfield_convergence(
    hnl: HNL, x: jax.Array, key: jax.Array, binary_dim: int = 64, max_iters: int = 20
):
    """Trace how Hopfield state evolves over iterations."""
    from hopfield import (binarize_memories, compute_hopfield_weights,
                          create_binary_projection)

    # Setup
    q = hnl.query_proj(x)
    q = q.reshape(hnl.num_heads, hnl.head_dim)

    # Create projection and weights for first head
    bin_proj = create_binary_projection(key, hnl.head_dim, binary_dim)
    binary_memories = binarize_memories(hnl.memories[0])  # First head
    projected = binary_memories @ bin_proj.T
    binary_patterns = jnp.sign(projected)
    W = compute_hopfield_weights(binary_patterns)

    # Initial state
    q_h = q[0]  # First head
    state = jnp.sign(bin_proj @ q_h)

    states = [state]
    energies = []

    for i in range(max_iters):
        # Compute energy: E = -0.5 * s^T W s
        energy = -0.5 * state @ W @ state
        energies.append(float(energy))

        # Update
        new_state = jnp.sign(W @ state)
        states.append(new_state)

        # Check convergence
        if jnp.allclose(new_state, state):
            print(f"  Converged at iteration {i+1}")
            break
        state = new_state

    # Compute state changes
    state_changes = []
    for i in range(1, len(states)):
        change = jnp.sum(states[i] != states[i - 1])
        state_changes.append(int(change))

    return {
        "energies": energies,
        "state_changes": state_changes,
        "final_state": states[-1],
        "num_iterations": len(energies),
    }


def compare_soft_vs_hopfield_retrieval(hnl: HNL, x: jax.Array, key: jax.Array):
    """Deep dive into what soft attention retrieves vs what Hopfield retrieves."""
    from hopfield import (binarize_memories, compute_hopfield_weights,
                          create_binary_projection)

    # Get query
    q = hnl.query_proj(x)
    q = q.reshape(hnl.num_heads, hnl.head_dim)
    q_h = q[0]  # First head
    memories_h = hnl.memories[0]  # (num_memories, head_dim)

    # Soft attention
    attn_scores = jnp.einsum("d,md->m", q_h, memories_h) / (
        jnp.sqrt(hnl.head_dim) * 1e-2
    )
    attn_weights = jax.nn.softmax(attn_scores)
    soft_retrieved = jnp.einsum("m,md->d", attn_weights, memories_h)

    # Which memories does soft attention focus on?
    top_k = 5
    top_indices = jnp.argsort(attn_weights)[-top_k:][::-1]
    top_weights = attn_weights[top_indices]

    print(f"  Soft attention top {top_k} memory indices: {top_indices.tolist()}")
    print(
        f"  Soft attention top {top_k} weights: {[f'{w:.3f}' for w in top_weights.tolist()]}"
    )
    print(
        f"  Attention entropy: {-jnp.sum(attn_weights * jnp.log(attn_weights + 1e-10)):.3f}"
    )

    # Hopfield retrieval
    binary_dim = hnl.num_mems
    bin_proj = create_binary_projection(key, hnl.head_dim, binary_dim)
    # Project and binarize memories
    projected_memories = memories_h @ bin_proj.T  # (num_memories, binary_dim)
    binary_memories = jnp.sign(projected_memories)
    print(binary_memories)

    # Compute Hopfield weights
    W = compute_hopfield_weights(binary_memories)

    # Project query and iterate
    binary_query = jnp.sign(bin_proj @ q_h)
    state = binary_query
    for _ in range(10):
        state = jnp.sign(W @ state)

    # Project back
    hopfield_retrieved = bin_proj.T @ state

    # Compare retrieved vectors
    print(f"\n  Soft retrieved norm: {jnp.linalg.norm(soft_retrieved):.3f}")
    print(f"  Hopfield retrieved norm: {jnp.linalg.norm(hopfield_retrieved):.3f}")
    print(
        f"  Cosine similarity: {jnp.dot(soft_retrieved, hopfield_retrieved) / (jnp.linalg.norm(soft_retrieved) * jnp.linalg.norm(hopfield_retrieved) + 1e-8):.3f}"
    )

    # Check if Hopfield output is close to any memory
    memory_sims = (
        memories_h
        @ hopfield_retrieved
        / (
            jnp.linalg.norm(memories_h, axis=1) * jnp.linalg.norm(hopfield_retrieved)
            + 1e-8
        )
    )
    best_memory_idx = jnp.argmax(memory_sims)
    print(
        f"\n  Hopfield output most similar to memory {best_memory_idx} (sim={memory_sims[best_memory_idx]:.3f})"
    )

    return {
        "soft_retrieved": soft_retrieved,
        "hopfield_retrieved": hopfield_retrieved,
        "attn_weights": attn_weights,
    }


def main():
    print("=" * 60)
    print("Hopfield vs Soft Attention Diagnostic")
    print("=" * 60)

    # Train an HNM model first
    from models import HNM
    from training import TrainConfig, Trainer, cross_entropy_loss

    binary_d = 1

    key = jax.random.PRNGKey(42)
    k1, k2, k3, k_train = jax.random.split(key, 4)

    # Get data
    (X_train, y_train), (X_test, y_test) = get_mnist_data(flatten=True)

    # Create and train HNM
    print("\nTraining HNM model (5 epochs)...")
    print("-" * 40)
    hnm = HNM.create(
        in_features=784,
        hidden_dims=[128],
        out_features=10,
        num_memories=32,
        num_heads=4,
        key=k1,
    )

    config = TrainConfig(
        learning_rate=1e-3,
        epochs=1,
        batch_size=64,
    )
    trainer = Trainer(hnm, cross_entropy_loss, config)
    trained_hnm = trainer.train((X_train, y_train), (X_test, y_test), key=k_train)

    # Evaluate
    test_loss, test_acc = trainer.evaluate((X_test, y_test))
    print(f"\nTrained model accuracy: {test_acc:.4f}")

    # Get the first HNL layer from trained model
    hnl = trained_hnm.layers[0]
    print(f"\nAnalyzing first HNL layer:")
    print(f"  in_features={hnl.in_features}, out_features={hnl.out_features}")
    print(f"  num_memories={hnl.num_mems}, num_heads={hnl.num_heads}")

    x = X_test[0]  # Single sample

    print("\n1. Comparing outputs for a single sample:")
    print("-" * 40)
    result = compare_hnl_outputs(hnl, x, k2, binary_dim=binary_d, num_iterations=10)
    print(f"  L2 distance: {result['l2_distance']:.4f}")
    print(f"  Cosine similarity: {result['cosine_similarity']:.4f}")
    print(f"  Soft output norm: {result['soft_norm']:.4f}")
    print(f"  Hopfield output norm: {result['hopfield_norm']:.4f}")

    print("\n2. Tracing Hopfield convergence:")
    print("-" * 40)
    trace = trace_hopfield_convergence(hnl, x, k2, binary_dim=binary_d, max_iters=20)
    print(f"  Energies: {[f'{e:.2f}' for e in trace['energies'][:10]]}")
    print(f"  State changes per iteration: {trace['state_changes'][:10]}")

    print("\n3. Comparing retrieval mechanisms:")
    print("-" * 40)
    compare_soft_vs_hopfield_retrieval(hnl, x, k2)

    # Test on multiple samples
    print("\n4. Statistics over 100 samples:")
    print("-" * 40)
    l2_dists = []
    cosine_sims = []
    for i in range(100):
        key_i = jax.random.fold_in(k3, i)
        result = compare_hnl_outputs(
            hnl, X_test[i], key_i, binary_dim=binary_d, num_iterations=10
        )
        l2_dists.append(result["l2_distance"])
        cosine_sims.append(result["cosine_similarity"])

    print(
        f"  Mean L2 distance: {jnp.mean(jnp.array(l2_dists)):.4f} (std: {jnp.std(jnp.array(l2_dists)):.4f})"
    )
    print(
        f"  Mean cosine similarity: {jnp.mean(jnp.array(cosine_sims)):.4f} (std: {jnp.std(jnp.array(cosine_sims)):.4f})"
    )

    # Also test full model conversion
    print("\n5. Full model comparison (soft vs Hopfield):")
    print("-" * 40)
    from hopfield import convert_hnm_to_hopfield

    hopfield_hnm = convert_hnm_to_hopfield(
        trained_hnm, k2, binary_dim=0, num_iterations=10
    )

    # Compare predictions
    soft_model = eqx.nn.inference_mode(trained_hnm)
    dummy_key = jax.random.PRNGKey(0)

    n_test = 1000
    test_keys = jax.random.split(dummy_key, n_test)
    soft_logits = jax.vmap(soft_model)(X_test[:n_test], key=test_keys)
    hopfield_logits = jax.vmap(hopfield_hnm)(X_test[:n_test])

    soft_preds = jnp.argmax(soft_logits, axis=-1)
    hopfield_preds = jnp.argmax(hopfield_logits, axis=-1)

    soft_acc = jnp.mean(soft_preds == y_test[:n_test])
    hopfield_acc = jnp.mean(hopfield_preds == y_test[:n_test])
    agreement = jnp.mean(soft_preds == hopfield_preds)

    print(f"  Soft attention accuracy: {soft_acc:.4f}")
    print(f"  Hopfield accuracy: {hopfield_acc:.4f}")
    print(f"  Prediction agreement: {agreement:.4f}")


if __name__ == "__main__":
    main()
