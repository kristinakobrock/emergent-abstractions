import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(metric_values):
    """
    Computes mean and standard deviation.
    """
    return np.mean(metric_values, axis=1)[-1], np.std(metric_values, axis=1)[-1]


def plot_and_compute_statistics(path, start_run=0, end_run=5):
    """
    Opens metrics and loss pickle files for given dataset. Parses data into
    train_losses, test_losses, train_accs, test_accs lists and uses these
    to plot metrics and calculate mean and standard deviation statistics.
    """
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    for run in range(start_run, end_run):
        data_path = path / "standard" / f"{run}" / "loss_and_metrics.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        epochs = list(range(1, len(data["metrics_train0"]) + 1))
        train_losses.append([data["loss_train"][epoch] for epoch in epochs])
        test_losses.append([data["loss_test"][epoch] for epoch in epochs])
        train_accs.append([data["metrics_train0"][epoch] for epoch in epochs])
        test_accs.append([data["metrics_test0"][epoch] for epoch in epochs])

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    for i, run in enumerate(range(start_run, end_run)):
        axes[0].plot(
            epochs,
            train_losses[i],
            label=f"Train Loss Run {i+1}",
            linestyle="-",
            color=f"C{i}",
        )
        axes[0].plot(
            epochs,
            test_losses[i],
            label=f"Test Loss Run {i+1}",
            linestyle="--",
            color=f"C{i}",
        )
        axes[1].plot(
            epochs,
            train_accs[i],
            label=f"Train Acc Run {i+1}",
            linestyle="-",
            color=f"C{i}",
        )
        axes[1].plot(
            epochs,
            test_accs[i],
            label=f"Test Acc Run {i+1}",
            linestyle="--",
            color=f"C{i}",
        )

    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Loss over Epochs")

    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy over Epochs")

    plt.tight_layout()
    plt.show()

    # Compute and print statistics
    print("Final Epoch Statistics (Mean, Std Dev):")
    train_loss_mean, train_loss_std = compute_stats(train_losses)
    test_loss_mean, test_loss_std = compute_stats(test_losses)
    train_acc_mean, train_acc_std = compute_stats(train_accs)
    test_acc_mean, test_acc_std = compute_stats(test_accs)

    print(f"Train Loss: Mean = {train_loss_mean:.3f}, Std Dev = {train_loss_std:.3f}")
    print(f"Test Loss: Mean = {test_loss_mean:.3f}, Std Dev = {test_loss_std:.3f}")
    print(f"Train Accuracy: Mean = {train_acc_mean:.3f}, Std Dev = {train_acc_std:.3f}")
    print(f"Test Accuracy: Mean = {test_acc_mean:.3f}, Std Dev = {test_acc_std:.3f}")


if __name__ == "__main__":
    dataset_path = r"results\(3,4)_game_size_10_vsf_3"  # Input your desired path
    dataset_path = Path(dataset_path)
    plot_and_compute_statistics(dataset_path)
