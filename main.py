"""
main.py

Full pipeline entry point for the Schrodinger Equation PINN project.

Runs, in order:
    1. Data generation (LHS-sampled collocation, boundary, initial points)
    2. Model architecture build + sanity check
    3. Loss function sanity check (gradient flow verification)
    4. Training (5000 epochs, Adam optimizer)
    5. Visualization + validation against the analytical solution

Run with:
    python main.py
"""

from src.data import generate_all_training_data
from src.model import build_model
from src import train as train_module
from src import visualize as visualize_module


def main():
    print("#" * 60)
    print("# Schrodinger Equation PINN - Full Pipeline")
    print("#" * 60)

    print("\n[1/4] Generating training data...")
    data = generate_all_training_data(save_csv=True)

    collocation_batch = train_module._to_tensor_dict_collocation(data["collocation"])
    boundary_batch = train_module._to_tensor_dict_labeled(data["boundary"])
    initial_batch = train_module._to_tensor_dict_labeled(data["initial"])

    print("\n[2/4] Building model...")
    model = build_model()

    print("\n[3/4] Training...")
    history_df = train_module.train(model, collocation_batch, boundary_batch, initial_batch)

    import os
    os.makedirs(train_module.OUTPUT_DIR, exist_ok=True)
    history_df.to_csv(os.path.join(train_module.OUTPUT_DIR, "training_history.csv"), index=False)
    model.save_weights(os.path.join(train_module.OUTPUT_DIR, "model_weights.weights.h5"))

    print("\n[4/4] Generating validation plots and metrics...")
    visualize_module.plot_loss_history(os.path.join(visualize_module.OUTPUT_DIR, "loss_history.png"))
    visualize_module.plot_snapshots(model, os.path.join(visualize_module.OUTPUT_DIR, "snapshots_comparison.png"))
    visualize_module.plot_heatmaps(model, os.path.join(visualize_module.OUTPUT_DIR, "heatmap_comparison.png"))
    metrics = visualize_module.compute_validation_metrics(model)

    print("\n" + "#" * 60)
    print("Pipeline complete. Final validation metrics:")
    for k, v in metrics.items():
        print(f"  {k:32s}: {v:.6e}")
    print("#" * 60)


if __name__ == "__main__":
    main()