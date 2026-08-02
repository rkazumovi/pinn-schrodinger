"""
src/model.py

TensorFlow neural network architecture for the Schrodinger Equation PINN.

The network approximates the complex wavefunction psi(x,t) = u(x,t) + i*v(x,t)
by taking (x, t) as input and producing (u, v) simultaneously as two output
neurons of a shared fully-connected trunk.

Architecture:
    Input:   (x, t)                  -> 2 features
    Hidden:  6 layers x 50 neurons   -> tanh activation, Glorot normal init
    Output:  (u, v)                  -> 2 features, linear activation

Tanh is used throughout because the PDE residual requires computing
d^2u/dx^2, d^2v/dx^2, du/dt, dv/dt via automatic differentiation (in
losses.py / train.py using tf.GradientTape) - tanh is smooth and has
well-behaved higher-order derivatives, unlike ReLU.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SchrodingerPINN(keras.Model):
    """
    Fully-connected neural network mapping (x, t) -> (u, v).

    u(x,t): real part of the wavefunction
    v(x,t): imaginary part of the wavefunction
    """

    def __init__(self, hidden_layers: int = 6, neurons_per_layer: int = 50, name: str = "schrodinger_pinn", **kwargs):
        super().__init__(name=name, **kwargs)

        self.hidden_layers_list = []
        for i in range(hidden_layers):
            self.hidden_layers_list.append(
                layers.Dense(
                    neurons_per_layer,
                    activation="tanh",
                    kernel_initializer="glorot_normal",
                    name=f"hidden_{i+1}",
                )
            )

        # Output layer: 2 units (u, v), no activation (linear)
        self.output_layer = layers.Dense(
            2,
            activation=None,
            kernel_initializer="glorot_normal",
            name="output_uv",
        )

    def call(self, inputs, training=False):
        """
        Forward pass.

        Args:
            inputs: tensor of shape (batch_size, 2) with columns [x, t]

        Returns:
            tensor of shape (batch_size, 2) with columns [u, v]
        """
        x = inputs
        for layer in self.hidden_layers_list:
            x = layer(x)
        return self.output_layer(x)

    def predict_uv(self, x, t):
        """
        Convenience method: given separate x and t arrays/tensors,
        returns (u, v) as separate tensors.

        Args:
            x: tensor/array of shape (N,) or (N,1)
            t: tensor/array of shape (N,) or (N,1)

        Returns:
            u, v: tensors of shape (N,1)
        """
        x = tf.reshape(tf.cast(x, tf.float32), [-1, 1])
        t = tf.reshape(tf.cast(t, tf.float32), [-1, 1])
        inputs = tf.concat([x, t], axis=1)
        uv = self(inputs, training=False)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v


def build_model(hidden_layers: int = 6, neurons_per_layer: int = 50) -> SchrodingerPINN:
    """
    Factory function to build and initialize the PINN model.

    Note: for subclassed Keras models, calling model.build(input_shape=...)
    does NOT reliably build nested layers in Keras 3 (they can be left
    marked "unbuilt" with 0 params, even though the model runs fine).
    The robust fix is to run one real forward pass on dummy data, which
    forces every Dense layer to build its weights based on the actual
    input shape it receives.
    """
    model = SchrodingerPINN(hidden_layers=hidden_layers, neurons_per_layer=neurons_per_layer)
    dummy_input = tf.zeros((1, 2), dtype=tf.float32)
    _ = model(dummy_input, training=False)  # forces all sublayers to build
    return model


def main():
    print("=" * 60)
    print("Schrodinger PINN - Model Architecture Check")
    print("=" * 60)

    model = build_model()
    model.summary()

    print("-" * 60)
    print("Forward pass sanity check")
    print("-" * 60)

    # Small batch of dummy (x, t) points
    batch_size = 5
    x_dummy = np.linspace(0.0, 1.0, batch_size).reshape(-1, 1)
    t_dummy = np.linspace(0.0, 1.0, batch_size).reshape(-1, 1)
    inputs = np.concatenate([x_dummy, t_dummy], axis=1).astype(np.float32)

    outputs = model(inputs, training=False)
    print(f"Input shape : {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print("\nSample input (x, t) -> output (u, v):")
    for i in range(batch_size):
        print(f"  ({inputs[i,0]:.3f}, {inputs[i,1]:.3f}) -> "
              f"(u={outputs[i,0]:.6f}, v={outputs[i,1]:.6f})")

    # Sanity checks
    assert outputs.shape == (batch_size, 2), "Output shape mismatch"
    assert not np.any(np.isnan(outputs.numpy())), "NaN detected in output"

    # Test predict_uv convenience method
    u, v = model.predict_uv(x_dummy, t_dummy)
    assert u.shape == (batch_size, 1), "u shape mismatch"
    assert v.shape == (batch_size, 1), "v shape mismatch"

    print("-" * 60)
    print("All assertions passed.")
    print("Model architecture verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()