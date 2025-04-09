import matplotlib.pyplot as plt
import numpy as np


def show_images():
    pass


def plot_activations():
    pass


def plot_confusion_map():
    pass


def plot_training_metrics(metrics_dict):
    """
    Plots training metrics from a dictionary with subplots.

    Args:
        metrics_dict (dict): A dictionary where keys are metric names (e.g., 'train_loss', 'val_loss', 'rmse')
                             and values are lists of metric values per epoch.
    """

    num_metrics = len(metrics_dict)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(8, 4 * num_metrics))  # Adjust figsize as needed

    if num_metrics == 1:
        axes = [axes] #make sure axes is iterable in the single plot case.

    epochs = range(1, len(next(iter(metrics_dict.values()))) + 1)  # Assuming all lists have the same length

    for i, (metric_name, metric_values) in enumerate(metrics_dict.items()):
        axes[i].plot(epochs, metric_values)
        axes[i].set_title(metric_name)
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric_name)
        axes[i].grid(True)

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


def plot_sequence_predictions(input_seq, real_output_seq, predicted_output_seq, feature_names):
    """
    Plots input, real output, and predicted output sequences for each feature with correctly aligned time steps.

    Args:
        input_seq (np.array): Input sequence (time steps, features).
        real_output_seq (np.array): Real output sequence (time steps, features).
        predicted_output_seq (np.array): Predicted output sequence (time steps, features).
        feature_names (list): List of feature names.
    """
    pass
