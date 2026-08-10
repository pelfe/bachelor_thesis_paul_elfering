import os

# =====================================================
# TF SETTINGS
# =====================================================

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_USE_CUDNN_FRONTEND"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.optimizer.set_jit(False)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(
        gpu,
        True
    )

# =====================================================
# IMPORTS
# =====================================================

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from scipy.interpolate import make_interp_spline

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = (
    "data/models/pauls_models/"
    "mixed_15stacks_gan/discriminator.keras"
)

X_PATH = (
    "data/post_interpolation/healthy/"
    "npy_stacked_15/size64_subjectct1_x.npy"
)

BATCH_SIZE = 16

# image dimensions
HEIGHT = 64
WIDTH = 64

# number of fake images per variance
N_TEST = 200

# variance values to test
TEST_VARIANCES = np.linspace(
    0.001,
    10,
    25
)

# =====================================================
# LOAD DISCRIMINATOR
# =====================================================

print("Loading discriminator...")

discriminator = load_model(
    MODEL_PATH
)

discriminator.summary()

# =====================================================
# LOAD CONDITIONING IMAGE X
# =====================================================

print("\nLoading x...")

x = np.load(
    X_PATH
)

print(
    "x shape:",
    x.shape
)

print(
    "x range:",
    x.min(),
    x.max()
)


# =====================================================
# CREATE CONTROLLED VARIANCE IMAGES
# =====================================================

def create_images_with_variance(
        variance,
        n
):
    """
    Creates fake images:

    shape:
        (n,64,64,1)

    values:
        0-1

    variance:
        approximately controlled
    """

    CHANNELS = 15

    images = np.random.randn(
        n,
        HEIGHT,
        WIDTH,
        CHANNELS
    )

    # normalize variance to 1
    images = (
            images /
            images.std()
    )

    # scale variance
    images = (
            images *
            np.sqrt(variance)
    )

    # center intensity
    images = images + 0.5

    # keep valid image range
    images = np.clip(
        images,
        0,
        1
    )

    return images.astype(
        np.float32
    )


# =====================================================
# RUN EXPERIMENT
# =====================================================

print("\nRunning variance experiment...")

measured_variances = []
mean_predictions = []

for variance in TEST_VARIANCES:
    print(
        "\nTesting variance:",
        variance
    )

    # ---------------------------------
    # generate fake y
    # ---------------------------------

    fake_y = create_images_with_variance(
        variance,
        N_TEST
    )

    # ---------------------------------
    # select x samples
    # ---------------------------------

    x_batch = x[:N_TEST]

    # ---------------------------------
    # discriminator prediction
    # ---------------------------------

    prediction = discriminator.predict(
        [
            x_batch,
            fake_y
        ],
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # ---------------------------------
    # calculate actual variance
    # ---------------------------------

    image_variance = np.var(
        fake_y,
        axis=(1, 2, 3)
    )

    avg_variance = (
        image_variance.mean()
    )

    avg_prediction = (
        prediction.mean()
    )

    measured_variances.append(
        avg_variance
    )

    mean_predictions.append(
        avg_prediction
    )

    print(
        "Actual variance:",
        avg_variance
    )

    print(
        "Prediction:",
        avg_prediction
    )

# =====================================================
# INTERPOLATED PLOT
# =====================================================


measured_variances = np.array(
    measured_variances
)

mean_predictions = np.array(
    mean_predictions
)

# sort values

idx = np.argsort(
    measured_variances
)

measured_variances = (
    measured_variances[idx]
)

mean_predictions = (
    mean_predictions[idx]
)

# smooth curve

x_smooth = np.linspace(
    measured_variances.min(),
    measured_variances.max(),
    300
)

interpolator = make_interp_spline(
    measured_variances,
    mean_predictions,
    k=3
)

y_smooth = interpolator(
    x_smooth
)

# =====================================================
# PLOT
# =====================================================


plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    measured_variances,
    mean_predictions,
    label="Measured"
)

plt.plot(
    x_smooth,
    y_smooth,
    label="Interpolated"
)

plt.xlabel(
    "Image variance"
)

plt.ylabel(
    "Average discriminator prediction"
)

plt.title(
    "Discriminator prediction vs image variance"
)

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()

print("\nDONE")