import json
import matplotlib.pyplot as plt


def load_history(json_path):
    """Load training history from a JSON file."""
    with open(json_path, "r") as f:
        history = json.load(f)
    return history


def get_xaxis(history, use_epochs=True):
    """
    Returns the x-axis.
    If use_epochs=True, uses history['epoch'].
    Otherwise uses history['amount_samples_trained'].
    """
    if use_epochs:
        return history["epoch"], "Epoch"
    else:
        return history["amount_samples_trained"], "Samples Trained"


def plot_metric(
    history,
    metric,
    use_epochs=True,
    ax=None,
    title=None,
    ylabel=None,
):
    """
    Plot a single metric.
    """
    x, xlabel = get_xaxis(history, use_epochs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x, history[metric], linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or metric)
    ax.grid(True)

    return ax


def plot_losses(history, use_epochs=True):
    """Plot discriminator and generator losses."""
    x, xlabel = get_xaxis(history, use_epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(x, history["d_loss_real"], label="D Loss (Real)")
    plt.plot(x, history["d_loss_fake"], label="D Loss (Fake)")
    plt.plot(x, history["g_loss"], label="Generator Loss")

    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_validation(history, use_epochs=True):
    """Plot validation metrics."""
    x, xlabel = get_xaxis(history, use_epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(x, history["val_mae"], label="Validation MAE")

    plt.xlabel(xlabel)
    plt.ylabel("MAE")
    plt.title("Validation MAE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_training_overview(history, use_epochs=True):
    """
    Creates a 2-panel overview:
        - Losses
        - Validation MAE
    """
    x, xlabel = get_xaxis(history, use_epochs)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Losses
    axes[0].plot(x, history["d_loss_real"], label="D Real")
    axes[0].plot(x, history["d_loss_fake"], label="D Fake")
    axes[0].plot(x, history["g_loss"], label="Generator")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Losses")
    axes[0].legend()
    axes[0].grid(True)

    # Validation
    axes[1].plot(x, history["val_mae"], label="Validation MAE")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation Performance")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()