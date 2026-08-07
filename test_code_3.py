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
    "data/models/pauls_models/"
    "mixed_15stacks_gan/discriminator.keras"
)

X_PATH = (
    "data/post_interpolation/healthy/"
    "npy_stacked_15/size64_subjectct1_x.npy"
)

Y_PATH = (
    "data/post_interpolation/healthy/"
    "npy_stacked_15/size64_subjectct1_y.npy"
)


BATCH_SIZE = 16


# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = load_model(MODEL_PATH)

model.summary()


# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading data...")

x = np.load(X_PATH)
y = np.load(Y_PATH)

print(x.min(), x.max(), x.mean(), x.std())
print(y.min(), y.max(), y.mean(), y.std())
print("x shape:", x.shape)
print("y shape:", y.shape)

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
        [x,y],
        batch_size=16,
        verbose=0
    )

    print(
        layer.name,
        "mean:",
        out.mean(),
        "std:",
        out.std(),
        "min:",
        out.min(),
        "max:",
        out.max()
    )

# =====================================================
# LOAD GENERATOR AND CREATE FAKE SAMPLES
# =====================================================

GENERATOR_PATH = (
    "data/models/pauls_models/"
    "mixed_15stacks_gan/generator.keras"
)

generator = load_model(GENERATOR_PATH)

fake = generator.predict(
    x,
    batch_size=BATCH_SIZE,
    verbose=0
)

print("\nFake shape:", fake.shape)
print(
    "Fake range:",
    fake.min(),
    fake.max(),
    fake.mean(),
    fake.std()
)



# ====================================================
# 1. NORMAL OUTPUT DISTRIBUTION
# =====================================================

print("\n==============================")
print(" DISCRIMINATOR OUTPUT")
print("==============================")


output = model.predict(
    [x,y],
    batch_size=BATCH_SIZE
)


print(
    "Output percentiles:",
    np.percentile(
        output,
        [0,1,25,50,75,99,100]
    )
)

print(
    "Mean output:",
    output.mean()
)


# =====================================================
# 2. GET LOGITS BEFORE SIGMOID
# =====================================================

print("\n==============================")
print(" LOGITS BEFORE SIGMOID")
print("==============================")


# assumes:
# Conv2D -> sigmoid Activation
# so layer before sigmoid is the logits

sigmoid_index = None

for i, layer in enumerate(model.layers):

    if layer.__class__.__name__ == "Activation":
        sigmoid_index = i


if sigmoid_index is not None:

    logit_layer = model.layers[sigmoid_index-1]

    logit_model = Model(
        inputs=model.inputs,
        outputs=logit_layer.output
    )

    logits = logit_model.predict(
        [x,y],
        batch_size=BATCH_SIZE
    )


    print(
        "Logit percentiles:",
        np.percentile(
            logits,
            [0,1,25,50,75,99,100]
        )
    )


    print(
        "Mean logit:",
        logits.mean()
    )

else:
    print(
        "No sigmoid Activation layer found"
    )


# =====================================================
# LOGITS TRAINING VS INFERENCE
# =====================================================

print("\n==============================")
print(" LOGITS (TRAINING VS INFERENCE)")
print("==============================")

logit_model = Model(
    inputs=model.inputs,
    outputs=logit_layer.output
)

# Real
real_logits_train = logit_model([x, y], training=True).numpy()
real_logits_test  = logit_model([x, y], training=False).numpy()

print("\nREAL LOGITS")
print("Training mean :", real_logits_train.mean())
print("Inference mean:", real_logits_test.mean())

# Fake
fake_logits_train = logit_model([x, fake], training=True).numpy()
fake_logits_test  = logit_model([x, fake], training=False).numpy()

print("\nFAKE LOGITS")
print("Training mean :", fake_logits_train.mean())
print("Inference mean:", fake_logits_test.mean())

# =====================================================
# 3. FINAL CONV WEIGHTS AND BIAS
# =====================================================

print("\n==============================")
print(" FINAL CONV PARAMETERS")
print("==============================")


conv_layers = [
    layer for layer in model.layers
    if layer.__class__.__name__ == "Conv2D"
]


final_conv = conv_layers[-1]


weights, bias = final_conv.get_weights()


print("Final Conv layer:", final_conv.name)

print(
    "Kernel mean:",
    weights.mean()
)

print(
    "Kernel std:",
    weights.std()
)

print(
    "Bias:",
    bias
)


# =====================================================
# 4. FEATURE COLLAPSE CHECK
# =====================================================

print("\n==============================")
print(" FEATURE COLLAPSE")
print("==============================")


feature_layer = conv_layers[-2]


feature_model = Model(
    inputs=model.inputs,
    outputs=feature_layer.output
)


features = feature_model.predict(
    [x,y],
    batch_size=BATCH_SIZE
)


print(
    "Feature mean:",
    features.mean()
)

print(
    "Feature std:",
    features.std()
)

print(
    "Percentage near zero:",
    np.mean(np.abs(features)<1e-7)*100,
    "%"
)



# =====================================================
# 5. BATCH NORMALIZATION CHECK
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
            "moving mean:",
            moving_mean.mean()
        )

        print(
            "moving variance:",
            moving_var.mean()
        )





# =====================================================
# 6. TRAINING VS INFERENCE (REAL + FAKE)
# =====================================================

print("\n==============================")
print(" TRAINING VS INFERENCE")
print("==============================")

# -----------------------------------------------------
# Load generator
# -----------------------------------------------------

GENERATOR_PATH = (
    "data/models/pauls_models/"
    "mixed_15stacks_gan/generator.keras"
)

generator = load_model(GENERATOR_PATH)

fake = generator.predict(
    x,
    batch_size=BATCH_SIZE,
    verbose=0
)

# -----------------------------------------------------
# REAL PAIRS
# -----------------------------------------------------

real_training = model(
    [x, y],
    training=True
).numpy()

real_inference = model(
    [x, y],
    training=False
).numpy()

print("\nREAL PAIRS")
print("Training mean :", real_training.mean())
print("Inference mean:", real_inference.mean())

print("Training percentiles:",
      np.percentile(real_training, [0,1,25,50,75,99,100]))

print("Inference percentiles:",
      np.percentile(real_inference, [0,1,25,50,75,99,100]))


# -----------------------------------------------------
# FAKE PAIRS
# -----------------------------------------------------

fake_training = model(
    [x, fake],
    training=True
).numpy()

fake_inference = model(
    [x, fake],
    training=False
).numpy()

print("\nFAKE PAIRS")
print("Training mean :", fake_training.mean())
print("Inference mean:", fake_inference.mean())

print("Training percentiles:",
      np.percentile(fake_training, [0,1,25,50,75,99,100]))

print("Inference percentiles:",
      np.percentile(fake_inference, [0,1,25,50,75,99,100]))

# =====================================================
# 7. ACTIVATION SATURATION
# =====================================================

print("\n==============================")
print(" ACTIVATION ANALYSIS")
print("==============================")


activation_layers = []

for layer in model.layers:

    if layer.__class__.__name__ in [
        "ReLU",
        "LeakyReLU",
        "Activation"
    ]:
        activation_layers.append(layer)


activation_model = Model(
    inputs=model.inputs,
    outputs=[
        layer.output
        for layer in activation_layers
    ]
)


acts = activation_model.predict(
    [x,y],
    batch_size=BATCH_SIZE
)



for layer, act in zip(
    activation_layers,
    acts
):

    print("\nLayer:", layer.name)

    print(
        "mean:",
        act.mean()
    )

    print(
        "std:",
        act.std()
    )

    print(
        "min/max:",
        act.min(),
        act.max()
    )

    if len(act.shape)==4:

        channel_activity = np.mean(
            np.abs(act),
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


print("\n==============================")
print(" DONE")
print("==============================")