"""
src/visualize.py

Visualization and validation for the Schrodinger Equation PINN.

Analytical solution (infinite square well, ground state initial condition,
natural units hbar = m = 1):

    psi(x,t) = sin(pi*x) * exp(-i * pi^2 * t / 2)

Since the ground state is a stationary state, splitting into real/imaginary
parts gives:

    u_exact(x,t) = sin(pi*x) * cos(pi^2 * t / 2)
    v_exact(x,t) = -sin(pi*x) * sin(pi^2 * t / 2)

And the probability density is time-INDEPENDENT:

    |psi(x,t)|^2 = u^2 + v^2 = sin^2(pi*x)   for all t

This gives a strong, easy sanity check: if the trained network's predicted
|psi|^2 changes meaningfully over time, something is wrong.

Run with:
    python -m src.visualize
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless/script runs
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import build_model

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "model_weights.weights.h5")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history.csv")


# ---------------------------------------------------------------------------
# Analytical solution
# ---------------------------------------------------------------------------
def analytical_solution(x, t):
    """
    Exact solution for the ground-state infinite square well.

    Args:
        x, t: numpy arrays of the same shape

    Returns:
        u_exact, v_exact: numpy arrays of the same shape
    """
    u_exact = np.sin(np.pi * x) * np.cos(np.pi**2 * t / 2.0)
    v_exact = -np.sin(np.pi * x) * np.sin(np.pi**2 * t / 2.0)
    return u_exact, v_exact


def load_trained_model():
    """Build the model architecture and load the trained weights."""
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    return model


# ---------------------------------------------------------------------------
# Plot 1: Loss history
# ---------------------------------------------------------------------------
def plot_loss_history(save_path):
    history_df = pd.read_csv(HISTORY_PATH)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(history_df["epoch"], history_df["total_loss"], label="Total loss", linewidth=2)
    ax.semilogy(history_df["epoch"], history_df["pde_loss"], label="PDE residual loss", linestyle="--")
    ax.semilogy(history_df["epoch"], history_df["boundary_loss"], label="Boundary loss", linestyle="--")
    ax.semilogy(history_df["epoch"], history_df["initial_loss"], label="Initial condition loss", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss History - Schrodinger PINN")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss history plot -> {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Snapshot comparisons at fixed time slices
# ---------------------------------------------------------------------------
def plot_snapshots(model, save_path, times=(0.0, 0.25, 0.5, 0.75, 1.0), n_x=200):
    x_line = np.linspace(0, 1, n_x)

    fig, axes = plt.subplots(3, len(times), figsize=(4 * len(times), 9), sharey="row")

    for col, t_val in enumerate(times):
        t_line = np.full_like(x_line, t_val)
        u_pred, v_pred = model.predict_uv(x_line, t_line)
        u_pred = u_pred.numpy().flatten()
        v_pred = v_pred.numpy().flatten()
        u_exact, v_exact = analytical_solution(x_line, t_line)

        prob_pred = u_pred**2 + v_pred**2
        prob_exact = u_exact**2 + v_exact**2

        # Row 0: u(x,t)
        ax = axes[0, col]
        ax.plot(x_line, u_exact, "k-", label="Exact", linewidth=2)
        ax.plot(x_line, u_pred, "r--", label="PINN", linewidth=2)
        ax.set_title(f"t = {t_val:.2f}")
        if col == 0:
            ax.set_ylabel("u(x,t) [Re]")
            ax.legend(fontsize=8)

        # Row 1: v(x,t)
        ax = axes[1, col]
        ax.plot(x_line, v_exact, "k-", label="Exact", linewidth=2)
        ax.plot(x_line, v_pred, "b--", label="PINN", linewidth=2)
        if col == 0:
            ax.set_ylabel("v(x,t) [Im]")

        # Row 2: |psi|^2
        ax = axes[2, col]
        ax.plot(x_line, prob_exact, "k-", label="Exact", linewidth=2)
        ax.plot(x_line, prob_pred, "g--", label="PINN", linewidth=2)
        ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("|psi(x,t)|^2")

    fig.suptitle("PINN vs Analytical Solution - Snapshots in Time", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved snapshot comparison plot -> {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Full domain heatmaps
# ---------------------------------------------------------------------------
def plot_heatmaps(model, save_path, n_x=100, n_t=100):
    x_vals = np.linspace(0, 1, n_x)
    t_vals = np.linspace(0, 1, n_t)
    X, T = np.meshgrid(x_vals, t_vals)  # shape (n_t, n_x)

    x_flat = X.flatten()
    t_flat = T.flatten()

    u_pred, v_pred = model.predict_uv(x_flat, t_flat)
    u_pred = u_pred.numpy().flatten().reshape(n_t, n_x)
    v_pred = v_pred.numpy().flatten().reshape(n_t, n_x)
    prob_pred = u_pred**2 + v_pred**2

    u_exact, v_exact = analytical_solution(X, T)
    prob_exact = u_exact**2 + v_exact**2

    abs_error = np.abs(prob_pred - prob_exact)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im0 = axes[0].pcolormesh(X, T, prob_pred, shading="auto", cmap="viridis")
    axes[0].set_title("PINN predicted |psi(x,t)|^2")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, T, prob_exact, shading="auto", cmap="viridis")
    axes[1].set_title("Analytical |psi(x,t)|^2")
    axes[1].set_xlabel("x")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, T, abs_error, shading="auto", cmap="inferno")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    fig.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved heatmap comparison plot -> {save_path}")

    return prob_pred, prob_exact, abs_error


# ---------------------------------------------------------------------------
# Quantitative validation
# ---------------------------------------------------------------------------
def compute_validation_metrics(model, n_x=100, n_t=100):
    x_vals = np.linspace(0, 1, n_x)
    t_vals = np.linspace(0, 1, n_t)
    X, T = np.meshgrid(x_vals, t_vals)
    x_flat = X.flatten()
    t_flat = T.flatten()

    u_pred, v_pred = model.predict_uv(x_flat, t_flat)
    u_pred = u_pred.numpy().flatten()
    v_pred = v_pred.numpy().flatten()
    u_exact, v_exact = analytical_solution(x_flat, t_flat)

    # Relative L2 error, treating (u,v) as a single stacked real vector
    pred_vec = np.concatenate([u_pred, v_pred])
    exact_vec = np.concatenate([u_exact, v_exact])

    l2_error = np.linalg.norm(pred_vec - exact_vec) / np.linalg.norm(exact_vec)

    prob_pred = u_pred**2 + v_pred**2
    prob_exact = u_exact**2 + v_exact**2
    prob_l2_error = np.linalg.norm(prob_pred - prob_exact) / np.linalg.norm(prob_exact)

    metrics = {
        "relative_l2_error_psi": float(l2_error),
        "relative_l2_error_prob_density": float(prob_l2_error),
        "max_abs_error_u": float(np.max(np.abs(u_pred - u_exact))),
        "max_abs_error_v": float(np.max(np.abs(v_pred - v_exact))),
    }
    return metrics


def main():
    print("=" * 60)
    print("Schrodinger PINN - Visualization & Validation")
    print("=" * 60)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No trained weights found at {WEIGHTS_PATH}. Run 'python -m src.train' first."
        )
    if not os.path.exists(HISTORY_PATH):
        raise FileNotFoundError(
            f"No training history found at {HISTORY_PATH}. Run 'python -m src.train' first."
        )

    print("Loading trained model...")
    model = load_trained_model()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("-" * 60)
    print("Generating plots...")
    plot_loss_history(os.path.join(OUTPUT_DIR, "loss_history.png"))
    plot_snapshots(model, os.path.join(OUTPUT_DIR, "snapshots_comparison.png"))
    prob_pred, prob_exact, abs_error = plot_heatmaps(model, os.path.join(OUTPUT_DIR, "heatmap_comparison.png"))

    print("-" * 60)
    print("Computing quantitative validation metrics...")
    metrics = compute_validation_metrics(model)
    for k, v in metrics.items():
        print(f"  {k:32s}: {v:.6e}")

    metrics_path = os.path.join(OUTPUT_DIR, "validation_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved validation metrics -> {metrics_path}")

    print("-" * 60)
    print("Sanity checks:")
    assert not np.isnan(metrics["relative_l2_error_psi"]), "NaN in relative L2 error"
    assert metrics["relative_l2_error_psi"] < 0.10, (
        f"Relative L2 error too high: {metrics['relative_l2_error_psi']:.4f} "
        "(expected < 0.10 given final training loss)"
    )
    print(f"Relative L2 error on psi        : {metrics['relative_l2_error_psi']*100:.3f}%")
    print(f"Relative L2 error on |psi|^2     : {metrics['relative_l2_error_prob_density']*100:.3f}%")

    print("=" * 60)
    print("Validation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()