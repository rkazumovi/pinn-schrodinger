"""
src/train.py

Training loop for the Schrodinger Equation PINN.

Wires together:
    - src/data.py    -> LHS-sampled collocation, boundary, initial points
    - src/model.py   -> SchrodingerPINN network (x,t) -> (u,v)
    - src/losses.py  -> PDE residual + boundary + initial condition losses

Uses full-batch gradient descent with the Adam optimizer. Per-epoch loss
components are tracked in a Pandas DataFrame and exported to CSV, per
project spec (reproducibility + structured metric tracking).

Run with:
    python -m src.train
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_all_training_data
from src.model import build_model
from src.losses import total_loss

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EPOCHS = 5000
LEARNING_RATE = 1e-3
W_PDE = 1.0
W_BC = 1.0
W_IC = 1.0
LOG_EVERY = 500

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _to_tensor_dict_collocation(df: pd.DataFrame) -> dict:
    return {
        "x": tf.constant(df["x"].values.reshape(-1, 1), dtype=tf.float32),
        "t": tf.constant(df["t"].values.reshape(-1, 1), dtype=tf.float32),
    }


def _to_tensor_dict_labeled(df: pd.DataFrame) -> dict:
    return {
        "x": tf.constant(df["x"].values.reshape(-1, 1), dtype=tf.float32),
        "t": tf.constant(df["t"].values.reshape(-1, 1), dtype=tf.float32),
        "u_target": tf.constant(df["u_target"].values.reshape(-1, 1), dtype=tf.float32),
        "v_target": tf.constant(df["v_target"].values.reshape(-1, 1), dtype=tf.float32),
    }


@tf.function
def train_step(model, optimizer, collocation_batch, boundary_batch, initial_batch,
               w_pde, w_bc, w_ic):
    """
    Single training step: forward pass, compute total loss, backprop, apply gradients.

    Wrapped in @tf.function so TensorFlow traces this into a graph once and
    reuses it every epoch, instead of re-tracing the nested GradientTape
    logic in losses.py from scratch each time (meaningful CPU speedup).
    """
    with tf.GradientTape() as tape:
        loss, _ = total_loss(
            model, collocation_batch, boundary_batch, initial_batch,
            w_pde=w_pde, w_bc=w_bc, w_ic=w_ic,
        )
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, collocation_batch, boundary_batch, initial_batch,
          epochs=EPOCHS, learning_rate=LEARNING_RATE,
          w_pde=W_PDE, w_bc=W_BC, w_ic=W_IC, log_every=LOG_EVERY):
    """
    Full training loop. Returns a Pandas DataFrame of per-epoch loss history.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    history_records = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_step(model, optimizer, collocation_batch, boundary_batch, initial_batch,
                   tf.constant(w_pde, dtype=tf.float32),
                   tf.constant(w_bc, dtype=tf.float32),
                   tf.constant(w_ic, dtype=tf.float32))

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            # Recompute loss components for logging. This call happens here,
            # in plain eager Python code (NOT inside train_step's
            # @tf.function graph), so converting tensors -> floats via
            # .numpy() is safe at this point.
            _, components_t = total_loss(model, collocation_batch, boundary_batch, initial_batch,
                                          w_pde=w_pde, w_bc=w_bc, w_ic=w_ic)
            components = {k: float(v.numpy()) for k, v in components_t.items()}
            elapsed = time.time() - start_time
            record = {"epoch": epoch, "elapsed_sec": elapsed, **components}
            history_records.append(record)
            print(f"Epoch {epoch:5d}/{epochs} | "
                  f"Total: {components['total_loss']:.6e} | "
                  f"PDE: {components['pde_loss']:.6e} | "
                  f"BC: {components['boundary_loss']:.6e} | "
                  f"IC: {components['initial_loss']:.6e} | "
                  f"Elapsed: {elapsed:.1f}s")

    history_df = pd.DataFrame(history_records)
    return history_df


def main():
    print("=" * 60)
    print("Schrodinger PINN - Training")
    print("=" * 60)

    print("Generating training data (LHS sampling)...")
    data = generate_all_training_data(save_csv=True)

    collocation_batch = _to_tensor_dict_collocation(data["collocation"])
    boundary_batch = _to_tensor_dict_labeled(data["boundary"])
    initial_batch = _to_tensor_dict_labeled(data["initial"])

    print(f"Collocation points: {collocation_batch['x'].shape[0]}")
    print(f"Boundary points   : {boundary_batch['x'].shape[0]}")
    print(f"Initial points    : {initial_batch['x'].shape[0]}")

    print("-" * 60)
    print("Building model...")
    model = build_model()
    print(f"Trainable parameters: {sum(np.prod(v.shape) for v in model.trainable_variables)}")

    print("-" * 60)
    print(f"Starting training for {EPOCHS} epochs (lr={LEARNING_RATE})...")
    print("-" * 60)

    history_df = train(model, collocation_batch, boundary_batch, initial_batch)

    print("-" * 60)
    print("Training complete. Saving artifacts...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved training history -> {history_path}")

    weights_path = os.path.join(OUTPUT_DIR, "model_weights.weights.h5")
    model.save_weights(weights_path)
    print(f"Saved model weights    -> {weights_path}")

    print("-" * 60)
    print("Sanity checks:")
    initial_loss_val = history_df.iloc[0]["total_loss"]
    final_loss_val = history_df.iloc[-1]["total_loss"]
    print(f"Initial total loss: {initial_loss_val:.6e}")
    print(f"Final total loss  : {final_loss_val:.6e}")

    assert not history_df.isnull().values.any(), "NaN detected in training history"
    assert final_loss_val < initial_loss_val, "Loss did not decrease during training"

    print(f"Loss reduced by factor of {initial_loss_val / max(final_loss_val, 1e-12):.1f}x")
    print("=" * 60)
    print("Training verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()