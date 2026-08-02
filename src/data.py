"""
src/data.py

Training data generation for the 1D Schrodinger Equation PINN
(Infinite Square Well / Particle in a Box).

Generates three sets of training points using Latin Hypercube Sampling (LHS)
for better space-filling coverage than plain random uniform sampling:

    1. Collocation points  -> enforce PDE residual (real + imaginary)
    2. Boundary points     -> enforce psi(0,t) = psi(1,t) = 0
    3. Initial points      -> enforce psi(x,0) = sin(pi*x) (ground state)

All point sets are stored as Pandas DataFrames and exported to CSV under
outputs/ for reproducibility, per project spec.

Physical setup:
    Domain:      x in [0, 1],  t in [0, 1]
    BC:          u(0,t) = v(0,t) = 0
                 u(1,t) = v(1,t) = 0
    IC:          u(x,0) = sin(pi*x),  v(x,0) = 0
    Potential:   V(x) = 0  (free particle / infinite well interior)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_COLLOCATION = 10000   # interior domain points for PDE residual
N_BOUNDARY = 400         # total boundary points (split across x=0 and x=1)
N_INITIAL = 400          # initial condition points at t=0

X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def _lhs_samples(n_samples: int, dims: int, bounds: list, seed: int = 42) -> np.ndarray:
    """
    Generate Latin Hypercube samples scaled to given bounds.

    Args:
        n_samples: number of sample points
        dims: number of dimensions
        bounds: list of (low, high) tuples, one per dimension
        seed: random seed for reproducibility

    Returns:
        ndarray of shape (n_samples, dims)
    """
    sampler = qmc.LatinHypercube(d=dims, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    return qmc.scale(unit_samples, lows, highs)


def generate_collocation_points(n_points: int = N_COLLOCATION, seed: int = 42) -> pd.DataFrame:
    """
    Generate interior domain points (x, t) via 2D Latin Hypercube Sampling.
    Used to enforce the PDE residual (real and imaginary parts of the
    Schrodinger equation) during training.
    """
    samples = _lhs_samples(n_points, dims=2, bounds=[(X_MIN, X_MAX), (T_MIN, T_MAX)], seed=seed)
    df = pd.DataFrame(samples, columns=["x", "t"])
    df["point_type"] = "collocation"
    return df


def generate_boundary_points(n_points: int = N_BOUNDARY, seed: int = 43) -> pd.DataFrame:
    """
    Generate boundary points at x=0 and x=1, split evenly, with t sampled
    via 1D Latin Hypercube Sampling. Enforces psi(0,t) = psi(1,t) = 0,
    i.e. u = 0 and v = 0 (target values included for convenience).
    """
    n_half = n_points // 2

    t_left = _lhs_samples(n_half, dims=1, bounds=[(T_MIN, T_MAX)], seed=seed).flatten()
    t_right = _lhs_samples(n_points - n_half, dims=1, bounds=[(T_MIN, T_MAX)], seed=seed + 1).flatten()

    df_left = pd.DataFrame({
        "x": np.zeros(n_half),
        "t": t_left,
        "u_target": np.zeros(n_half),
        "v_target": np.zeros(n_half),
    })
    df_right = pd.DataFrame({
        "x": np.ones(n_points - n_half),
        "t": t_right,
        "u_target": np.zeros(n_points - n_half),
        "v_target": np.zeros(n_points - n_half),
    })

    df = pd.concat([df_left, df_right], ignore_index=True)
    df["point_type"] = "boundary"
    return df


def generate_initial_points(n_points: int = N_INITIAL, seed: int = 44) -> pd.DataFrame:
    """
    Generate initial condition points at t=0, with x sampled via 1D
    Latin Hypercube Sampling. Enforces:
        u(x,0) = sin(pi*x)   (ground state, real part)
        v(x,0) = 0           (imaginary part starts at zero)
    """
    x_samples = _lhs_samples(n_points, dims=1, bounds=[(X_MIN, X_MAX)], seed=seed).flatten()

    df = pd.DataFrame({
        "x": x_samples,
        "t": np.zeros(n_points),
        "u_target": np.sin(np.pi * x_samples),
        "v_target": np.zeros(n_points),
    })
    df["point_type"] = "initial"
    return df


def generate_all_training_data(save_csv: bool = True) -> dict:
    """
    Generate all three training point sets and optionally save to CSV.

    Returns:
        dict with keys "collocation", "boundary", "initial" mapping to DataFrames
    """
    collocation_df = generate_collocation_points()
    boundary_df = generate_boundary_points()
    initial_df = generate_initial_points()

    data = {
        "collocation": collocation_df,
        "boundary": boundary_df,
        "initial": initial_df,
    }

    if save_csv:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for name, df in data.items():
            path = os.path.join(OUTPUT_DIR, f"{name}_points.csv")
            df.to_csv(path, index=False)
            print(f"Saved {len(df):>6} {name} points -> {path}")

    return data


def main():
    print("=" * 60)
    print("Generating training data for Schrodinger Equation PINN")
    print("=" * 60)

    data = generate_all_training_data(save_csv=True)

    print("-" * 60)
    for name, df in data.items():
        print(f"\n{name.upper()} points: {len(df)} rows")
        print(df.head(3).to_string(index=False))

    # Sanity checks
    print("-" * 60)
    print("Sanity checks:")
    coll = data["collocation"]
    bnd = data["boundary"]
    init = data["initial"]

    assert len(coll) == N_COLLOCATION, "Collocation point count mismatch"
    assert len(bnd) == N_BOUNDARY, "Boundary point count mismatch"
    assert len(init) == N_INITIAL, "Initial point count mismatch"
    assert coll["x"].between(X_MIN, X_MAX).all(), "Collocation x out of bounds"
    assert coll["t"].between(T_MIN, T_MAX).all(), "Collocation t out of bounds"
    assert set(bnd["x"].unique()) == {0.0, 1.0}, "Boundary x values incorrect"
    assert (init["t"] == 0.0).all(), "Initial points must have t=0"
    assert np.allclose(init["u_target"], np.sin(np.pi * init["x"])), "Initial u_target mismatch"

    print("All assertions passed.")
    print("=" * 60)
    print("Data generation verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()