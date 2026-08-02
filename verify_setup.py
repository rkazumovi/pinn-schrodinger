"""
verify_setup.py
Checks TensorFlow installation, version, and CPU device availability.
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import matplotlib

def main():
    print("=" * 50)
    print("Environment Verification")
    print("=" * 50)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Pandas version      : {pd.__version__}")
    print(f"NumPy version       : {np.__version__}")
    print(f"SciPy version       : {scipy.__version__}")
    print(f"Matplotlib version  : {matplotlib.__version__}")
    print("-" * 50)

    devices = tf.config.list_physical_devices()
    print("Available devices:")
    for d in devices:
        print(f"  {d}")

    cpu_devices = tf.config.list_physical_devices('CPU')
    print(f"\nCPU available: {len(cpu_devices) > 0}")

    # Quick sanity computation to confirm TF actually runs
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print("\nSanity check matmul result:")
    print(c.numpy())

    print("=" * 50)
    print("Setup verified successfully." if cpu_devices else "WARNING: No CPU device detected.")
    print("=" * 50)

if __name__ == "__main__":
    main()