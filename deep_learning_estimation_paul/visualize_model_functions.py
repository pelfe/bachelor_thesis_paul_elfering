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
    """Plot a single metric."""
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

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        history["val_mae"],
        label="Validation MAE"
    )
    ax.plot(
        x,
        history["val_discriminator_fake_accuracy"],
        label="Validation D Fake Accuracy"
    )
    ax.plot(
        x,
        history["val_discriminator_real_accuracy"],
        label="Validation D Real Accuracy"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Metric Value")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_learning_rates(history):
    """
    Plot learning rates.
    Learning rates are usually constant, so this displays them as text.
    """
    plt.figure(figsize=(8, 3))

    plt.axis("off")

    text = (
        f"Discriminator learning rate: "
        f"{history['discriminator_learning_rate']}\n\n"
        f"Generator learning rate: "
        f"{history['generator_learning_rate']}"
    )

    plt.text(
        0.5,
        0.5,
        text,
        fontsize=14,
        ha="center",
        va="center"
    )

    plt.title("Learning Rates")
    plt.show()


def plot_training_overview(history, use_epochs=True):
    """
    Creates a complete training overview:
        - Losses
        - Validation MAE
        - Validation discriminator accuracies
        - Learning rates
    """
    x, xlabel = get_xaxis(history, use_epochs)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 12),
        sharex=True
    )

    # Losses
    axes[0].plot(
        x,
        history["d_loss_real"],
        label="D Real Loss"
    )
    axes[0].plot(
        x,
        history["d_loss_fake"],
        label="D Fake Loss"
    )
    axes[0].plot(
        x,
        history["g_loss"],
        label="Generator Loss"
    )

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Losses")
    axes[0].legend()
    axes[0].grid(True)


    # Validation MAE
    axes[1].plot(
        x,
        history["val_mae"],
        label="Validation MAE"
    )

    axes[1].set_ylabel("MAE")
    axes[1].set_title("Validation MAE")
    axes[1].legend()
    axes[1].grid(True)


    # Validation discriminator metrics
    axes[2].plot(
        x,
        history["val_discriminator_fake_accuracy"],
        label="Fake Accuracy"
    )
    axes[2].plot(
        x,
        history["val_discriminator_real_accuracy"],
        label="Real Accuracy"
    )

    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Validation Discriminator Accuracy")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Show learning rates separately
    plot_learning_rates(history)