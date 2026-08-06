import os

os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_USE_CUDNN_FRONTEND"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf
import numpy as np

import deep_learning_based_estimation_of_heart_surface_potentials_main.main_bayesian_cross as main_bayesian_cross



# ======================================================
# Parameters
# ======================================================

batch_size = 16
learning_rate = 0.001


train_subjects = [
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "L79")
]


# ======================================================
# Dataset
# ======================================================

def load_subject(base_path, subject):

    x = np.load(
        base_path + subject + "_x.npy",
        mmap_mode="r"
    )

    y = np.load(
        base_path + subject + "_y.npy",
        mmap_mode="r"
    )

    return x, y



class SimpleGenerator(tf.keras.utils.Sequence):

    def __init__(self, subjects, batch_size):

        xs = []
        ys = []

        for path, subject in subjects:

            x, y = load_subject(path, subject)

            xs.append(x)
            ys.append(y)


        self.x = np.concatenate(xs)
        self.y = np.concatenate(ys)

        self.batch_size = batch_size


    def __len__(self):

        return int(
            np.ceil(
                len(self.x)/self.batch_size
            )
        )


    def __getitem__(self, index):

        s = index*self.batch_size
        e = (index+1)*self.batch_size

        return (
            np.asarray(self.x[s:e]).astype(np.float32),
            np.asarray(self.y[s:e]).astype(np.float32)
        )



# ======================================================
# Diagnostics
# ======================================================

def summarize(name, x):

    print(
        f"{name:20s}",
        "shape:",
        x.shape,
        "min:",
        np.min(x),
        "max:",
        np.max(x),
        "mean:",
        np.mean(x),
        "std:",
        np.std(x)
    )



def bce(pred, target):

    eps = 1e-7

    return -np.mean(
        target*np.log(pred+eps)
        +
        (1-target)*np.log(1-pred+eps)
    )



# ======================================================
# Main test
# ======================================================

import tensorflow as tf
import numpy as np


# ======================================================
# Discriminator health test
# ======================================================

def test_trained_discriminator(d_model, g_model):

    dataset = SimpleGenerator(
        train_subjects,
        batch_size
    )


    X_realA, X_realB = dataset[0]


    # Generate fake samples
    X_fakeB = g_model.predict(
        X_realA,
        verbose=0
    )


    # Create wrong pairs
    X_wrongB = np.roll(
        X_realB,
        shift=1,
        axis=0
    )


    batch = X_realA.shape[0]

    patch = d_model.output_shape[1]


    y_real = np.ones(
        (batch, patch, patch, 1)
    )

    y_fake = np.zeros(
        (batch, patch, patch, 1)
    )


    print("\n==============================")
    print("DISCRIMINATOR HEALTH TEST")
    print("==============================")


    # --------------------------------------------------
    # 1. Predictions
    # --------------------------------------------------

    real_score = d_model.predict(
        [X_realA, X_realB],
        verbose=0
    )

    fake_score = d_model.predict(
        [X_realA, X_fakeB],
        verbose=0
    )

    wrong_score = d_model.predict(
        [X_realA, X_wrongB],
        verbose=0
    )


    print("\nSCORES")
    print("------------------------------")

    print(
        "Real pairs:",
        real_score.mean()
    )

    print(
        "Generated pairs:",
        fake_score.mean()
    )

    print(
        "Wrong real pairs:",
        wrong_score.mean()
    )



    # --------------------------------------------------
    # 2. Classification accuracy
    # --------------------------------------------------

    real_acc = np.mean(
        real_score > 0.5
    )

    fake_acc = np.mean(
        fake_score < 0.5
    )

    wrong_acc = np.mean(
        wrong_score < 0.5
    )


    print("\nACCURACY")
    print("------------------------------")

    print(
        "Real classified correctly:",
        real_acc
    )

    print(
        "Fake classified correctly:",
        fake_acc
    )

    print(
        "Wrong pairs rejected:",
        wrong_acc
    )



    # --------------------------------------------------
    # 3. BCE losses
    # --------------------------------------------------

    bce = tf.keras.losses.binary_crossentropy


    print("\nLOSSES")
    print("------------------------------")


    print(
        "Real BCE:",
        tf.reduce_mean(
            bce(y_real, real_score)
        ).numpy()
    )


    print(
        "Fake BCE:",
        tf.reduce_mean(
            bce(y_fake, fake_score)
        ).numpy()
    )


    # --------------------------------------------------
    # 4. Saturation check
    # --------------------------------------------------

    print("\nSATURATION")
    print("------------------------------")


    if real_score.mean() > 0.95:
        print(
            "WARNING: D(real) saturated"
        )

    if fake_score.mean() < 0.05:
        print(
            "WARNING: D(fake) saturated"
        )

    if (
        real_score.mean() > 0.95
        and fake_score.mean() < 0.05
    ):
        print(
            "D is extremely confident"
        )



    # --------------------------------------------------
    # 5. BatchNorm consistency
    # --------------------------------------------------

    print("\nBATCH NORMALIZATION")
    print("------------------------------")


    train_mode = d_model(
        [X_realA,X_realB],
        training=True
    ).numpy()


    eval_mode = d_model(
        [X_realA,X_realB],
        training=False
    ).numpy()


    difference = np.abs(
        train_mode - eval_mode
    ).mean()


    print(
        "training=True:",
        train_mode.mean()
    )

    print(
        "training=False:",
        eval_mode.mean()
    )

    print(
        "Difference:",
        difference
    )


    if difference > 0.1:
        print(
            "WARNING: BatchNorm statistics unstable"
        )

    # --------------------------------------------------
    # 6. Gradient test
    # --------------------------------------------------

    print("\nGRADIENT TEST")
    print("------------------------------")

    X_realA_tensor = tf.convert_to_tensor(
        X_realA,
        dtype=tf.float32
    )

    X_fake_tensor = tf.convert_to_tensor(
        X_fakeB,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(X_fake_tensor)

        pred = d_model(
            [
                X_realA_tensor,
                X_fake_tensor
            ],
            training=True
        )

        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                tf.ones_like(pred),
                pred
            )
        )

    gradients = tape.gradient(
        loss,
        X_fake_tensor
    )

    gradient_value = tf.reduce_mean(
        tf.abs(gradients)
    ).numpy()

    print(
        "Mean gradient magnitude:",
        gradient_value
    )

    if gradient_value < 1e-7:
        print(
            "WARNING: discriminator gives almost no generator gradient"
        )
    else:
        print(
            "Generator receives usable discriminator gradients"
        )



    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\nFINAL SUMMARY")
    print("------------------------------")


    if (
        real_score.mean() > 0.9
        and fake_score.mean() > 0.9
    ):
        print(
            "CRITICAL: D calls everything real"
        )

    elif (
        real_score.mean() > 0.9
        and fake_score.mean() < 0.1
    ):
        print(
            "D is very strong - check generator gradients"
        )

    elif (
        real_score.mean() > fake_score.mean()
        and wrong_score.mean() < real_score.mean()
    ):
        print(
            "D appears to learn useful conditional features"
        )

    else:
        print(
            "D behaviour is suspicious"
        )



if __name__ == "__main__":

    g_model_path = (
        "data/models/pauls_models/"
        "lbbb_15stacks_gan/generator.keras"
    )
    d_model_path = (
        "data/models/pauls_models/"
        "lbbb_15stacks_gan/discriminator.keras"
    )
    g_model = tf.keras.models.load_model(
        g_model_path
    )
    d_model = tf.keras.models.load_model(
        d_model_path
    )

    test_trained_discriminator(d_model, g_model)