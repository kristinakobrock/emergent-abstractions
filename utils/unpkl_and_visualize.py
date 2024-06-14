# import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt


def plot_metrics(pickle_path):
    """
    This function parses data directly from a given loss_and_metrics.pkl file
    by unpickling the file. Then it uses the data to plot a visualization of
    training accuracy, test accuracy, training loss, and test loss.
    """
    # dont forget, when you call use pickle_path= r"relative_path"
    pickle_path = Path(pickle_path)
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # print("Keys under 'loss_train':", list(data['loss_train'].keys())[:10])  # print first 10 keys
    # print(json.dumps(data, indent=4))
    # print(data.keys())

    # Extracting data
    epochs = list(range(1, len(data["metrics_train0"]) + 1))
    train_loss = [data["loss_train"][(epoch)] for epoch in epochs]
    test_loss = [data["loss_test"][(epoch)] for epoch in epochs]
    train_accuracy = [data["metrics_train0"][(epoch)] for epoch in epochs]
    test_accuracy = [data["metrics_test0"][(epoch)] for epoch in epochs]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o")
    ax1.plot(epochs, test_loss, label="Test Loss", marker="o")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot accuracy curves
    ax2.plot(epochs, train_accuracy, label="Train Accuracy", marker="o")
    ax2.plot(epochs, test_accuracy, label="Test Accuracy", marker="o")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_ylim(0, 1)

    # Display the plot
    plt.tight_layout()
    plt.show()


# When calling the function, use r... pickle_path= r"relative_path"
plot_metrics(
    pickle_path=r"results\(3,4)_game_size_10_vsf_3\standard\5\loss_and_metrics.pkl"
)
