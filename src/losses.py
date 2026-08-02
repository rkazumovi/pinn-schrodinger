"""
src/losses.py

Loss functions for the Schrodinger Equation PINN.

Physics (natural units: hbar = m = 1, V(x) = 0 inside the infinite well):

    i * psi_t = -1/2 * psi_xx

Writing psi(x,t) = u(x,t) + i*v(x,t) and separating real/imaginary parts
gives two coupled real PDE residuals that the network must drive to zero:

    f_u = u_t + 1/2 * v_xx      (from the imaginary part of the equation)
    f_v = v_t - 1/2 * u_xx      (from the real part of the equation)

Derivatives are computed with nested tf.GradientTape:
    - outer tape watches x to get the 2nd spatial derivative (u_xx, v_xx)
    - inner tape watches (x, t) to get the 1st derivatives (u_x, v_x, u_t, v_t)

Three loss terms are combined:
    1. PDE residual loss   - on collocation points (interior domain)
    2. Boundary loss       - on boundary points (x=0, x=1): u=v=0
    3. Initial loss        - on initial points (t=0): u=sin(pi*x), v=0
"""

import numpy as np
import tensorflow as tf


def compute_pde_residual(model, x, t, V=None):
    """
    Compute the real and imaginary PDE residuals f_u, f_v at given (x, t)
    collocation points, using nested automatic differentiation.

    Args:
        model: SchrodingerPINN instance (callable on (batch, 2) -> (batch, 2))
        x: tensor of shape (N, 1), requires float32
        t: tensor of shape (N, 1), requires float32
        V: optional potential function V(x) -> tensor of shape (N, 1).
           Defaults to zero everywhere (free particle / infinite well interior).

    Returns:
        f_u, f_v: tensors of shape (N, 1), the PDE residuals (should be ~0)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    t = tf.convert_to_tensor(t, dtype=tf.float32)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        tape2.watch(t)

        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(x)
            tape1.watch(t)

            inputs = tf.concat([x, t], axis=1)
            uv = model(inputs, training=True)
            u = uv[:, 0:1]
            v = uv[:, 1:2]

        u_x = tape1.gradient(u, x)
        v_x = tape1.gradient(v, x)
        u_t = tape1.gradient(u, t)
        v_t = tape1.gradient(v, t)

        del tape1

    u_xx = tape2.gradient(u_x, x)
    v_xx = tape2.gradient(v_x, x)

    del tape2

    if V is None:
        V_x = tf.zeros_like(x)
    else:
        V_x = V(x)

    # f_u = u_t + 1/2 v_xx - V*v   (V=0 term included for generality)
    # f_v = v_t - 1/2 u_xx + V*u
    f_u = u_t + 0.5 * v_xx - V_x * v
    f_v = v_t - 0.5 * u_xx + V_x * u

    return f_u, f_v


def pde_loss(model, x_collocation, t_collocation, V=None):
    """Mean-squared PDE residual loss over collocation points."""
    f_u, f_v = compute_pde_residual(model, x_collocation, t_collocation, V=V)
    return tf.reduce_mean(tf.square(f_u)) + tf.reduce_mean(tf.square(f_v))


def boundary_loss(model, x_boundary, t_boundary, u_target, v_target):
    """
    MSE loss enforcing psi(0,t) = psi(1,t) = 0, i.e. u=0 and v=0
    at the boundary points.
    """
    x_boundary = tf.convert_to_tensor(x_boundary, dtype=tf.float32)
    t_boundary = tf.convert_to_tensor(t_boundary, dtype=tf.float32)
    u_target = tf.convert_to_tensor(u_target, dtype=tf.float32)
    v_target = tf.convert_to_tensor(v_target, dtype=tf.float32)

    inputs = tf.concat([x_boundary, t_boundary], axis=1)
    uv = model(inputs, training=True)
    u_pred = uv[:, 0:1]
    v_pred = uv[:, 1:2]

    return tf.reduce_mean(tf.square(u_pred - u_target)) + tf.reduce_mean(tf.square(v_pred - v_target))


def initial_loss(model, x_initial, t_initial, u_target, v_target):
    """
    MSE loss enforcing psi(x,0) = sin(pi*x), i.e. u=sin(pi*x), v=0
    at the initial condition points.
    """
    x_initial = tf.convert_to_tensor(x_initial, dtype=tf.float32)
    t_initial = tf.convert_to_tensor(t_initial, dtype=tf.float32)
    u_target = tf.convert_to_tensor(u_target, dtype=tf.float32)
    v_target = tf.convert_to_tensor(v_target, dtype=tf.float32)

    inputs = tf.concat([x_initial, t_initial], axis=1)
    uv = model(inputs, training=True)
    u_pred = uv[:, 0:1]
    v_pred = uv[:, 1:2]

    return tf.reduce_mean(tf.square(u_pred - u_target)) + tf.reduce_mean(tf.square(v_pred - v_target))


def total_loss(model, collocation_batch, boundary_batch, initial_batch,
                w_pde=1.0, w_bc=1.0, w_ic=1.0, V=None):
    """
    Combine all three loss terms into a single weighted total loss.

    Args:
        collocation_batch: dict with keys "x", "t" (arrays/tensors of shape (N,1))
        boundary_batch: dict with keys "x", "t", "u_target", "v_target"
        initial_batch: dict with keys "x", "t", "u_target", "v_target"
        w_pde, w_bc, w_ic: scalar weights for each loss term
        V: optional potential function

    Returns:
        total: scalar total loss
        components: dict with individual loss values (for logging)
    """
    loss_pde = pde_loss(model, collocation_batch["x"], collocation_batch["t"], V=V)
    loss_bc = boundary_loss(model, boundary_batch["x"], boundary_batch["t"],
                             boundary_batch["u_target"], boundary_batch["v_target"])
    loss_ic = initial_loss(model, initial_batch["x"], initial_batch["t"],
                            initial_batch["u_target"], initial_batch["v_target"])

    total = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic

    # NOTE: components are returned as raw tensors, NOT converted with
    # float(...) / .numpy() here. This function must remain graph-safe so
    # it can be called from inside a @tf.function (e.g. train_step in
    # train.py). Calling .numpy() during graph tracing raises:
    #   AttributeError: 'SymbolicTensor' object has no attribute 'numpy'
    # Conversion to Python floats for logging/printing must happen at the
    # call site, in eager mode, outside any @tf.function.
    components = {
        "pde_loss": loss_pde,
        "boundary_loss": loss_bc,
        "initial_loss": loss_ic,
        "total_loss": total,
    }
    return total, components


def main():
    # Local imports to avoid circular dependency when run as a script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.model import build_model

    print("=" * 60)
    print("Schrodinger PINN - Loss Functions Sanity Check")
    print("=" * 60)

    model = build_model()

    # Small dummy batches (not yet the real LHS data from data.py - just
    # enough to verify shapes, gradients, and finiteness)
    n = 20
    rng = np.random.default_rng(0)

    x_col = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    t_col = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)

    x_bc = np.concatenate([np.zeros((n // 2, 1)), np.ones((n // 2, 1))]).astype(np.float32)
    t_bc = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    u_bc_target = np.zeros((n, 1), dtype=np.float32)
    v_bc_target = np.zeros((n, 1), dtype=np.float32)

    x_ic = rng.uniform(0, 1, size=(n, 1)).astype(np.float32)
    t_ic = np.zeros((n, 1), dtype=np.float32)
    u_ic_target = np.sin(np.pi * x_ic).astype(np.float32)
    v_ic_target = np.zeros((n, 1), dtype=np.float32)

    print("-" * 60)
    print("Testing PDE residual computation...")
    f_u, f_v = compute_pde_residual(model, x_col, t_col)
    print(f"f_u shape: {f_u.shape}, f_v shape: {f_v.shape}")
    assert f_u.shape == (n, 1), "f_u shape mismatch"
    assert f_v.shape == (n, 1), "f_v shape mismatch"
    assert not np.any(np.isnan(f_u.numpy())), "NaN in f_u"
    assert not np.any(np.isnan(f_v.numpy())), "NaN in f_v"
    print(f"Sample f_u[:3]: {f_u.numpy()[:3].flatten()}")
    print(f"Sample f_v[:3]: {f_v.numpy()[:3].flatten()}")

    print("-" * 60)
    print("Testing individual loss terms...")
    l_pde = pde_loss(model, x_col, t_col)
    l_bc = boundary_loss(model, x_bc, t_bc, u_bc_target, v_bc_target)
    l_ic = initial_loss(model, x_ic, t_ic, u_ic_target, v_ic_target)
    print(f"PDE loss     : {l_pde.numpy():.6f}")
    print(f"Boundary loss: {l_bc.numpy():.6f}")
    print(f"Initial loss : {l_ic.numpy():.6f}")

    for name, val in [("pde", l_pde), ("boundary", l_bc), ("initial", l_ic)]:
        assert not np.isnan(val.numpy()), f"{name} loss is NaN"
        assert val.numpy() >= 0.0, f"{name} loss is negative"

    print("-" * 60)
    print("Testing combined total_loss()...")
    collocation_batch = {"x": x_col, "t": t_col}
    boundary_batch = {"x": x_bc, "t": t_bc, "u_target": u_bc_target, "v_target": v_bc_target}
    initial_batch = {"x": x_ic, "t": t_ic, "u_target": u_ic_target, "v_target": v_ic_target}

    total, components = total_loss(model, collocation_batch, boundary_batch, initial_batch)
    # Convert tensors to Python floats here (eager mode, outside any tf.function)
    components = {k: float(v.numpy()) for k, v in components.items()}
    print("Loss components:")
    for k, val in components.items():
        print(f"  {k:15s}: {val:.6f}")

    assert abs(components["total_loss"] -
               (components["pde_loss"] + components["boundary_loss"] + components["initial_loss"])) < 1e-4, \
        "Total loss does not match sum of components (with unit weights)"

    print("-" * 60)
    print("Testing gradient flow (loss must be differentiable w.r.t. model weights)...")
    with tf.GradientTape() as tape:
        total, _ = total_loss(model, collocation_batch, boundary_batch, initial_batch)
    grads = tape.gradient(total, model.trainable_variables)
    n_none = sum(1 for g in grads if g is None)
    print(f"Trainable variables: {len(model.trainable_variables)}, gradients with None: {n_none}")
    assert n_none == 0, "Some gradients are None - graph is disconnected somewhere"

    print("=" * 60)
    print("All loss function checks passed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()