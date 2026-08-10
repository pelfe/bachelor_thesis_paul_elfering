"""
train_generator_only.py

Standalone generator-only training for the heart-surface-potential
image-to-image problem.

Training:

    X  ->  Generator  ->  Y_pred
                         |
                         +--> L1 loss
                         +--> Cosine loss

No discriminator / GAN is required.

Example usage from another file:

    from train_generator_only import train_generator_only

    history = train_generator_only(
        generator=g_model,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
        train_labels=None,
        epochs=50,
        batch_size=16,
        learning_rate=0.002,
    )
"""

import gc
import random
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# ============================================================
# Reproducibility
# ============================================================

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load subject
# ============================================================

def load_subject(base_path, subject):
    """
    Load all data belonging to one subject.

    Expected files:

        <base_path><subject>_x.npy
        <base_path><subject>_x_entropy.npy
        <base_path><subject>_y.npy
        <base_path><subject>_y_entropy.npy
    """

    x_path = base_path + subject + "_x.npy"
    x_entropy_path = base_path + subject + "_x_entropy.npy"
    y_path = base_path + subject + "_y.npy"
    y_entropy_path = base_path + subject + "_y_entropy.npy"

    x = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    x_entropy = np.load(x_entropy_path, mmap_mode="r")
    y_entropy = np.load(y_entropy_path, mmap_mode="r")

    return x, y, x_entropy, y_entropy


# ============================================================
# Subject Generator
# ============================================================

class SubjectGenerator(tf.keras.utils.Sequence):

    def __init__(
        self,
        subjects,
        subject_labels=None,
        batch_size=16,
        shuffle=True,
        augment=False,
        aug_v=0.5,
        augment_factor=2,
    ):

        self.subjects = subjects
        self.subject_labels = subject_labels

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augment_factor = augment_factor

        # ----------------------------------------------------
        # Data augmentation
        # ----------------------------------------------------

        if augment:

            self.data_gen = ImageDataGenerator(
                zca_epsilon=1e-5,
                rotation_range=180 * aug_v,
                width_shift_range=aug_v,
                height_shift_range=aug_v,
                shear_range=aug_v,
                zoom_range=aug_v,
                horizontal_flip=True,
                vertical_flip=True,
            )

        # ----------------------------------------------------
        # Subject information
        # ----------------------------------------------------

        self.subject_info = []

        total_samples = 0

        for idx, (path, subject) in enumerate(self.subjects):

            x, y, x_entropy, y_entropy = load_subject(
                path,
                subject
            )

            label = None

            if self.subject_labels is not None:
                label = self.subject_labels[idx]

            self.subject_info.append(
                {
                    "path": path,
                    "subject": subject,
                    "label": label,
                    "samples": x.shape[0],
                    "start": total_samples,
                    "end": total_samples + x.shape[0],
                }
            )

            total_samples += x.shape[0]

        self.total_samples = total_samples

        # ----------------------------------------------------
        # Direct sample lookup
        # ----------------------------------------------------

        self.sample_lookup = []

        for subject in self.subject_info:

            for local_index in range(subject["samples"]):

                self.sample_lookup.append(
                    {
                        "path": subject["path"],
                        "subject": subject["subject"],
                        "local_index": local_index,
                        "label": subject["label"],
                    }
                )

        # ----------------------------------------------------
        # Indices
        # ----------------------------------------------------

        if augment:

            self.indices = np.arange(
                self.total_samples * augment_factor
            )

        else:

            self.indices = np.arange(
                self.total_samples
            )

        # ----------------------------------------------------
        # Current subject cache
        # ----------------------------------------------------

        self.current_subject = None

        self.current_x = None
        self.current_y = None

        self.current_x_entropy = None
        self.current_y_entropy = None

        self.on_epoch_end()

    # --------------------------------------------------------
    # Number of batches
    # --------------------------------------------------------

    def __len__(self):

        return int(
            np.ceil(
                len(self.indices) / self.batch_size
            )
        )

    # --------------------------------------------------------
    # Get batch
    # --------------------------------------------------------

    def __getitem__(self, batch_index):

        batch_indices = self.indices[
            batch_index * self.batch_size:
            (batch_index + 1) * self.batch_size
        ]

        x_batch = []
        y_batch = []

        x_entropy_batch = []
        y_entropy_batch = []

        augment_flags = []

        # ----------------------------------------------------
        # Load samples
        # ----------------------------------------------------

        for index in batch_indices:

            is_augmented = False

            if self.augment and index >= self.total_samples:

                index -= self.total_samples

                is_augmented = True

            augment_flags.append(is_augmented)

            sample = self.sample_lookup[index]

            subject_id = (
                sample["path"],
                sample["subject"]
            )

            # ------------------------------------------------
            # Load subject only when subject changes
            # ------------------------------------------------

            if self.current_subject != subject_id:

                (
                    self.current_x,
                    self.current_y,
                    self.current_x_entropy,
                    self.current_y_entropy,
                ) = load_subject(
                    sample["path"],
                    sample["subject"]
                )

                self.current_subject = subject_id

            local_index = sample["local_index"]

            x_batch.append(
                self.current_x[local_index]
            )

            y_batch.append(
                self.current_y[local_index]
            )

            x_entropy_batch.append(
                self.current_x_entropy[local_index]
            )

            y_entropy_batch.append(
                self.current_y_entropy[local_index]
            )

        # ----------------------------------------------------
        # Convert to numpy
        # ----------------------------------------------------

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        x_entropy_batch = np.asarray(
            x_entropy_batch
        )

        y_entropy_batch = np.asarray(
            y_entropy_batch
        )

        # ----------------------------------------------------
        # Augmentation
        #
        # We concatenate X and Y so exactly the same spatial
        # transformation is applied to both.
        # ----------------------------------------------------

        if self.augment and all(augment_flags):

            augmented_x = []
            augmented_y = []

            for x_img, y_img in zip(
                x_batch,
                y_batch
            ):

                combined = np.concatenate(
                    [
                        x_img,
                        y_img
                    ],
                    axis=-1
                )

                aug = self.data_gen.random_transform(
                    combined
                )

                augmented_x.append(
                    aug[..., :x_img.shape[-1]]
                )

                augmented_y.append(
                    aug[..., x_img.shape[-1]:]
                )

            x_batch = np.asarray(
                augmented_x
            )

            y_batch = np.asarray(
                augmented_y
            )

        return (
            x_batch,
            y_batch,
            x_entropy_batch,
            y_entropy_batch,
        )

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    def on_epoch_end(self):

        if self.shuffle:

            np.random.shuffle(
                self.indices
            )


# ============================================================
# Cosine loss
# ============================================================

def cosine_loss(y_true, y_pred):

    y_true_flat = tf.reshape(
        y_true,
        [tf.shape(y_true)[0], -1]
    )

    y_pred_flat = tf.reshape(
        y_pred,
        [tf.shape(y_pred)[0], -1]
    )

    cosine_similarity = (
        tf.reduce_sum(
            y_true_flat * y_pred_flat,
            axis=1
        )
        /
        (
            tf.norm(
                y_true_flat,
                axis=1
            )
            *
            tf.norm(
                y_pred_flat,
                axis=1
            )
            + 1e-8
        )
    )

    return tf.reduce_mean(
        1.0 - cosine_similarity
    )


# ============================================================
# Generator-only loss
# ============================================================

def generator_loss(
    y_true,
    y_pred,
    alpha=0.2,
):
    """
    Combined L1 + cosine loss.

    alpha = 0.0
        only L1

    alpha = 1.0
        only cosine

    alpha = 0.2
        80% L1 + 20% cosine
    """

    l1_loss = tf.reduce_mean(
        tf.abs(
            y_true - y_pred
        )
    )

    cos_loss = cosine_loss(
        y_true,
        y_pred
    )

    loss = (
        (1.0 - alpha) * l1_loss
        +
        alpha * cos_loss
    )

    return loss


# ============================================================
# Validation MAE
# ============================================================

def calculate_mae(
    generator,
    val_generator,
):
    """
    Calculate mean absolute error over the validation set.
    """

    maes = []

    for i in range(
        len(val_generator)
    ):

        (
            x_batch,
            y_batch,
            _,
            _,
        ) = val_generator[i]

        prediction = generator(
            x_batch,
            training=False
        )

        prediction = prediction.numpy()

        mae = np.mean(
            np.abs(
                prediction - y_batch
            )
        )

        maes.append(
            mae
        )

    return float(
        np.mean(maes)
    )


# ============================================================
# Generator-only training
# ============================================================

def train_generator_only(
    generator,
    train_subjects,
    val_subjects,
    train_labels=None,
    val_labels=None,
    epochs=10,
    batch_size=16,
    learning_rate=0.01,
    alpha=0.2,
    augment=True,
    aug_v=0.5,
    augment_factor=2,
    patience=10,
    min_delta=0.0,
    save_path=None,
):
    """
    Train the generator without a discriminator.

    Parameters
    ----------
    generator:
        A compiled or uncompiled tf.keras generator model.

    train_subjects:
        List like:

            [
                ("/path/to/data/", "subject01"),
                ("/path/to/data/", "subject02"),
            ]

    val_subjects:
        Validation subjects in the same format.

    train_labels:
        Optional subject labels. Not used by the generator,
        but accepted for compatibility with your existing code.

    val_labels:
        Optional validation labels.

    epochs:
        Number of training epochs.

    batch_size:
        Batch size.

    learning_rate:
        Adam learning rate.

    alpha:
        Weight of cosine loss.

            loss =
                (1-alpha) * L1
                + alpha * cosine

    augment:
        Enable augmentation for training.

    aug_v:
        Strength of augmentation.

    augment_factor:
        Number of times the original dataset is repeated
        when augmentation is enabled.

    patience:
        Early stopping patience.

    min_delta:
        Minimum improvement required for validation MAE.

    save_path:
        Optional path for saving the best generator.

        Example:

            "best_generator.keras"

    Returns
    -------
    generator:
        Trained generator.

    history:
        Dictionary containing training statistics.
    """

    print()
    print("=" * 60)
    print("GENERATOR-ONLY TRAINING")
    print("=" * 60)

    # ========================================================
    # Create training generator
    # ========================================================

    train_generator = SubjectGenerator(
        subjects=train_subjects,
        subject_labels=train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
        aug_v=aug_v,
        augment_factor=augment_factor,
    )

    # ========================================================
    # Create validation generator
    # ========================================================

    val_generator = SubjectGenerator(
        subjects=val_subjects,
        subject_labels=val_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )

    print(
        "Training samples:",
        train_generator.total_samples
    )

    print(
        "Validation samples:",
        val_generator.total_samples
    )

    print(
        "Training batches:",
        len(train_generator)
    )

    print(
        "Validation batches:",
        len(val_generator)
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.7
    )

    # ========================================================
    # History
    # ========================================================

    history = {
        "epoch": [],
        "train_loss": [],
        "val_mae": [],
    }

    # ========================================================
    # Best model tracking
    # ========================================================

    best_val_mae = np.inf
    epochs_without_improvement = 0

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(epochs):

        print()
        print("=" * 60)
        print(
            f"Epoch {epoch + 1}/{epochs}"
        )
        print("=" * 60)

        epoch_losses = []

        progress = tqdm(
            range(len(train_generator)),
            desc="Training",
            unit="batch"
        )

        # ====================================================
        # Training
        # ====================================================

        for step in progress:

            (
                x_batch,
                y_batch,
                x_entropy,
                y_entropy,
            ) = train_generator[step]

            # ------------------------------------------------
            # Convert to tensors
            # ------------------------------------------------

            x_batch = tf.convert_to_tensor(
                x_batch,
                dtype=tf.float32
            )

            y_batch = tf.convert_to_tensor(
                y_batch,
                dtype=tf.float32
            )

            # ------------------------------------------------
            # Forward + backward pass
            # ------------------------------------------------

            with tf.GradientTape() as tape:

                prediction = generator(
                    x_batch,
                    training=True
                )

                loss = generator_loss(
                    y_batch,
                    prediction,
                    alpha=alpha
                )

            # ------------------------------------------------
            # Calculate gradients
            # ------------------------------------------------

            gradients = tape.gradient(
                loss,
                generator.trainable_variables
            )

            # ------------------------------------------------
            # Update generator
            # ------------------------------------------------

            optimizer.apply_gradients(
                zip(
                    gradients,
                    generator.trainable_variables
                )
            )

            loss_value = float(
                loss.numpy()
            )

            epoch_losses.append(
                loss_value
            )

            progress.set_postfix(
                loss=f"{np.mean(epoch_losses):.6f}"
            )

            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            del (
                x_batch,
                y_batch,
                x_entropy,
                y_entropy,
                prediction,
                loss,
                gradients,
            )

        # ====================================================
        # Epoch training loss
        # ====================================================

        train_loss = float(
            np.mean(epoch_losses)
        )

        # ====================================================
        # Validation
        # ====================================================

        val_mae = calculate_mae(
            generator,
            val_generator
        )

        print()
        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Validation MAE: {val_mae:.6f}"
        )

        # ====================================================
        # Store history
        # ====================================================

        history["epoch"].append(
            epoch + 1
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_mae"].append(
            val_mae
        )

        # ====================================================
        # Save best model
        # ====================================================

        if val_mae < (
            best_val_mae - min_delta
        ):

            best_val_mae = val_mae
            epochs_without_improvement = 0

            print(
                f"New best validation MAE: "
                f"{best_val_mae:.6f}"
            )

            if save_path is not None:

                generator.save(
                    save_path
                )

                print(
                    "Saved best generator to:",
                    save_path
                )

        else:

            epochs_without_improvement += 1

            print(
                "No improvement for",
                epochs_without_improvement,
                "epoch(s)"
            )

        # ====================================================
        # Early stopping
        # ====================================================

        if (
            epochs_without_improvement
            >= patience
        ):

            print()
            print(
                "Early stopping."
            )

            break

        # ====================================================
        # Cleanup
        # ====================================================

        gc.collect()

    print()
    print("=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    print(
        f"Best validation MAE: "
        f"{best_val_mae:.6f}"
    )

    return generator, history


# ============================================================
# Example prediction helper
# ============================================================

def predict_generator(
    generator,
    x,
    batch_size=16,
):
    """
    Predict data using the trained generator.
    """

    predictions = []

    for start in range(
        0,
        len(x),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(x)
        )

        batch = x[start:end]

        prediction = generator.predict(
            batch,
            verbose=0
        )

        predictions.append(
            prediction
        )

    return np.concatenate(
        predictions,
        axis=0
    )
