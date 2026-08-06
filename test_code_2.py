import os

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_USE_CUDNN_FRONTEND"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.config.optimizer.set_jit(False)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import BatchNormalization


# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = (
    "data/models/external models/epoch-9"
)

X_PATH = (
    "data/post_interpolation/healthy/"
    "npy_stacked_11/size64_subjectct1_x.npy"
)

Y_PATH = (
    "data/post_interpolation/healthy/"
    "npy_stacked_11/size64_subjectct1_y.npy"
)


BATCH_SIZE = 16


# =====================================================
# LOAD SAVEDMODEL (KERAS 2)
# =====================================================

print("Loading SavedModel...")

model = load_model(
    MODEL_PATH,
    compile=False
)

model.summary()


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading data...")

x = np.load(X_PATH)
y = np.load(Y_PATH)


print("x shape:", x.shape)
print("y shape:", y.shape)

print(
    "x:",
    x.min(),
    x.max(),
    x.mean(),
    x.std()
)

print(
    "y:",
    y.min(),
    y.max(),
    y.mean(),
    y.std()
)


# =====================================================
# 1. LAYER WEIGHT ANALYSIS
# =====================================================

print("\n==============================")
print(" WEIGHT ANALYSIS")
print("==============================")


for layer in model.layers:

    weights = layer.get_weights()

    if len(weights) == 0:
        continue

    kernel = weights[0]

    print("\nLayer:", layer.name)

    print(
        "shape:",
        kernel.shape
    )

    print(
        "mean:",
        kernel.mean()
    )

    print(
        "std:",
        kernel.std()
    )

    print(
        "near zero:",
        np.mean(np.abs(kernel)<1e-7)*100,
        "%"
    )


# =====================================================
# 2. CONV ACTIVATION ANALYSIS
# =====================================================

print("\n==============================")
print(" CONV ACTIVATIONS")
print("==============================")


conv_layers = [
    l for l in model.layers
    if isinstance(l, tf.keras.layers.Conv2D)
]


for layer in conv_layers:

    intermediate = Model(
        inputs=model.inputs,
        outputs=layer.output
    )

    out = intermediate.predict(
        x,
        batch_size=BATCH_SIZE,
        verbose=0
    )


    print("\nLayer:", layer.name)

    print(
        "mean:",
        out.mean()
    )

    print(
        "std:",
        out.std()
    )

    print(
        "min:",
        out.min()
    )

    print(
        "max:",
        out.max()
    )


    # dead feature channels

    if len(out.shape) == 4:

        channel_activity = np.mean(
            np.abs(out),
            axis=(0,1,2)
        )

        dead_channels = np.sum(
            channel_activity < 1e-7
        )

        print(
            "dead channels:",
            dead_channels,
            "/",
            len(channel_activity)
        )


# =====================================================
# 3. DISCRIMINATOR OUTPUT
# =====================================================

print("\n==============================")
print(" DISCRIMINATOR OUTPUT")
print("==============================")


output = model.predict(
    x,
    batch_size=BATCH_SIZE
)


print(
    "percentiles:",
    np.percentile(
        output,
        [0,1,25,50,75,99,100]
    )
)


print(
    "mean:",
    output.mean()
)


print(
    "std:",
    output.std()
)


# =====================================================
# 4. BATCH NORMALIZATION CHECK
# =====================================================

print("\n==============================")
print(" BATCH NORMALIZATION")
print("==============================")


for layer in model.layers:

    if isinstance(layer, BatchNormalization):

        gamma, beta, moving_mean, moving_var = (
            layer.get_weights()
        )

        print("\nLayer:", layer.name)

        print(
            "gamma mean:",
            gamma.mean()
        )

        print(
            "moving mean:",
            moving_mean.mean()
        )

        print(
            "moving variance:",
            moving_var.mean()
        )


# =====================================================
# 5. TRAINING VS INFERENCE DIFFERENCE
# =====================================================

print("\n==============================")
print(" TRAINING VS INFERENCE")
print("==============================")


out_training = model(
    x,
    training=True
)

out_inference = model(
    x,
    training=False
)


print(
    "training mean:",
    float(tf.reduce_mean(out_training))
)


print(
    "inference mean:",
    float(tf.reduce_mean(out_inference))
)


print("\n==============================")
print(" DONE")
print("==============================")