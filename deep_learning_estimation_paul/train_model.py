# Set seed values to remove randomness from that
seed_value = 0
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.random.set_seed(seed_value)

from tensorflow.keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization
import gc
import logging
from numpy import zeros
from numpy import ones
from numpy.random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Layer, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, \
    BatchNormalization, AveragePooling2D, MaxPooling2D, UpSampling2D, ReLU, Dense, Input, GlobalAveragePooling2D
from deep_learning_based_estimation_of_heart_surface_potentials_main.show_data import *
from deep_learning_based_estimation_of_heart_surface_potentials_main.UNet import get_unet_adapted
from tqdm import tqdm


def cosine_loss(y_true, y_pred):

    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    cosine_similarity = tf.reduce_sum(
        y_true_flat * y_pred_flat,
        axis=1) / (
        tf.norm(y_true_flat, axis=1) *
        tf.norm(y_pred_flat, axis=1)
        + 1e-8)

    return tf.reduce_mean(1 - cosine_similarity)





# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape, regularization_w=0.01, ssim_l1_r=50, ssim_w=0.5, lambda1=100, alpha=0.2, learning_rate =0.002, use_acgan=False):

    # Discriminator weights should be updated separately
    d_model.trainable = False


    # connect source input and generator output to discriminator input
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)

    if isinstance(gen_out, list):
        gen_out = gen_out[0]

    dis_out = d_model([in_src, gen_out])


    def generator_image_loss(y_true, y_pred):
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))
        cos = cosine_loss(y_true, y_pred)
        return (1 - alpha) * l1 + alpha * cos


    # -------------------------------------------------
    # Normal GAN
    # -------------------------------------------------
    if use_acgan == False:

        model = Model(in_src, [dis_out, gen_out])
        losses = ['binary_crossentropy', generator_image_loss]
        loss_weights = [1,lambda1]

    # -------------------------------------------------
    # AC-GAN
    # -------------------------------------------------
    else:
        source_out, class_out = dis_out
        model = Model(in_src, [source_out, class_out, gen_out])
        losses = ['binary_crossentropy','sparse_categorical_crossentropy',generator_image_loss]
        loss_weights = [1,1,lambda1]

    # Add L1 regularization to the generator's Conv2D layers
    l1_strength = regularization_w

    for layer in g_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = l1(l1_strength)


    # initial learning rate
    initial_learning_rate = 0.002

    # decay step and decay rate
    decay_steps = 1000
    decay_rate = 0.96

    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False
    )

    opt = Adam(learning_rate=learning_rate_decay,beta_1=0.7)

    model.compile(
        loss=losses,
        optimizer=opt,
        loss_weights=loss_weights
    )

    return model


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape, weights, name='train_'):
    trainA, trainB = dataset
    # choose random instances
    if name == "train_":
        ix = randint(0, trainA.shape[0], n_samples)
    else:
        ix = list(range(len(trainA)))
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    if not np.any(weights):
        w = weights
    else:
        w = weights[ix]

    return [X1, X2], y, w


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y

def calc_mae(g_model, dataset, n_batch=16):
    x, y = dataset

    num_samples = x.shape[0]
    num_batches = int(np.ceil(num_samples / n_batch))

    total_mae = 0.0

    for i in range(num_batches):
        start_idx = i * n_batch
        end_idx = (i + 1) * n_batch
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]

        X_fakeB, _ = generate_fake_samples(g_model, x_batch, 1)
        X_fakeB = X_fakeB.squeeze()

        batch_mae = np.abs(np.subtract(X_fakeB, y_batch)).mean()
        total_mae += batch_mae

        # Clear the variables to release memory
        del X_fakeB, x_batch, y_batch
        gc.collect()

    mae = total_mae / num_batches
    mae = round(mae, 4)
    return mae

def load_subject(base_path, subject):

    x_path = base_path + subject + "_x.npy"
    x_entropy_ath = base_path + subject + "_x_entropy.npy"
    y_path = base_path + subject + "_y.npy"
    y_entropy_path = base_path + subject + "_y_entropy.npy"

    x = np.load(x_path,mmap_mode="r")
    y = np.load(y_path,mmap_mode="r")
    x_entropy = np.load(x_entropy_ath,mmap_mode="r")
    y_entropy = np.load(y_entropy_path,mmap_mode="r")

    return x, y, x_entropy, y_entropy


# -------------------------------------------------------
# Subject generator
# -------------------------------------------------------

class SubjectGenerator(tf.keras.utils.Sequence):

    def __init__(
            self,
            subjects,
            subject_labels=None,
            batch_size=16,
            shuffle=True,
            augment=False,
            aug_v=0.5,
            augment_factor=2
    ):

        self.subjects = subjects
        self.subject_labels = subject_labels

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.augment_factor = augment_factor

        if augment:
            self.data_gen = ImageDataGenerator(
                zca_epsilon=1e-5,
                rotation_range=180*aug_v,
                width_shift_range=aug_v,
                height_shift_range=aug_v,
                shear_range=aug_v,
                zoom_range=aug_v,
                horizontal_flip=True,
                vertical_flip=True,
            )

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
                    "end": total_samples+x.shape[0]
                }
            )

            total_samples += x.shape[0]

        self.total_samples = total_samples

        # -------------------------------------------------------
        # Create direct sample lookup table
        # -------------------------------------------------------
        self.sample_lookup = []

        for subject in self.subject_info:

            for local_index in range(subject["samples"]):
                self.sample_lookup.append(
                    {
                        "path": subject["path"],
                        "subject": subject["subject"],
                        "local_index": local_index,
                        "label": subject["label"]
                    }
                )

        if augment:
            self.indices = np.arange(
                self.total_samples * augment_factor
            )
        else:
            self.indices = np.arange(
                self.total_samples
            )


        self.current_subject=None
        self.current_x=None
        self.current_y=None
        self.current_x_entropy=None
        self.current_y_entropy=None

        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, batch_index):

        batch_indices = self.indices[
            batch_index * self.batch_size:
            (batch_index + 1) * self.batch_size
        ]

        x_batch = []
        y_batch = []
        class_batch = []
        x_entropy_batch = []
        y_entropy_batch = []

        augment_flags = []

        # -------------------------------------------------------
        # Load batch using direct lookup
        # -------------------------------------------------------
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

            # Load subject only if changed
            if self.current_subject != subject_id:
                self.current_x, self.current_y, \
                    self.current_x_entropy, self.current_y_entropy = load_subject(
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

            class_batch.append(
                sample["label"]
            )

            x_entropy_batch.append(
                self.current_x_entropy[local_index]
            )

            y_entropy_batch.append(
                self.current_y_entropy[local_index]
            )

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        x_entropy_batch = np.asarray(x_entropy_batch)
        y_entropy_batch = np.asarray(y_entropy_batch)

        # only augment duplicated half
        if self.augment and all(augment_flags):

            augmented_x = []
            augmented_y = []

            for x_img, y_img in zip(x_batch, y_batch):
                combined = np.concatenate(
                    [
                        x_img,
                        y_img
                    ],
                    axis=-1
                )

                aug = self.data_gen.random_transform(combined)

                augmented_x.append(
                    aug[..., :x_img.shape[-1]]
                )

                augmented_y.append(
                    aug[..., x_img.shape[-1]:]
                )

            x_batch = np.asarray(augmented_x)
            y_batch = np.asarray(augmented_y)

        class_batch = np.asarray(class_batch)

        return (
            x_batch,
            y_batch,
            x_entropy_batch,
            y_entropy_batch,
            class_batch
        )

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(
                self.indices
            )

# ======================================================
# TRAIN FUNCTION (REFERENCE STYLE + YOUR STRUCTURE)
# ======================================================
def train(
        d_model,
        g_model,
        gan_model,
        train_subjects,
        val_subjects,
        train_labels=None,
        val_labels=None,
        epochs=10,
        batch_size=16,
        discriminator_lr = 0.01,
        generator_lr = 0.01,
        label_smoothing = True,
        use_acgan=False):

    # -----------------------------
    # Enable discriminator training
    # -----------------------------

    # discriminator
    d_model.trainable = True
    d_model.compile(
        optimizer=d_model.optimizer,
        loss=d_model.loss
    )

    history = {
        "discriminator_learning_rate": discriminator_lr,
        "generator_learning_rate": generator_lr,
        "epoch": [],
        "d_loss_real": [],
        "d_loss_fake": [],
        "g_loss": [],
        "val_mae": [],
        "val_discriminator_fake_accuracy": [],
        "val_discriminator_real_accuracy": [],
        "amount_samples_trained": []
    }

    # -----------------------------
    # Create generators
    # -----------------------------
    train_generator = SubjectGenerator(
        train_subjects,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )

    val_generator = SubjectGenerator(
        val_subjects,
        batch_size=batch_size,
        shuffle=False
    )

    if use_acgan:
        n_patch = d_model.output_shape[0][1]
    else:
        n_patch = d_model.output_shape[1]

    print("Training samples:", train_generator.total_samples)
    print("Validation samples:", val_generator.total_samples)

    # ==================================================
    # Epoch loop
    # ==================================================
    for epoch in range(epochs):
        print("\n==============================")
        print("Epoch:", epoch+1)
        print("==============================")

        epoch_d_real = []
        epoch_d_fake = []
        epoch_g = []
        progress = tqdm(range(len(train_generator)),desc="Training",unit="batch")

        for step in progress:
            # --------------------------
            # Load batch
            # --------------------------
            (X_realA, X_realB, input_entropy, output_entropy, class_labels) = train_generator[step]

            batch = X_realA.shape[0]
            y_real = tf.ones((batch,n_patch,n_patch,1))
            y_fake = tf.zeros((batch,n_patch,n_patch,1))

            input_entropy_patch = tf.image.resize(input_entropy[..., None],(n_patch, n_patch), method="area")
            # --------------------------
            # Generate fake images
            # --------------------------
            X_fakeB = g_model(X_realA, training=False)

            # --------------------------
            # Train discriminator real
            # --------------------------

            if use_acgan:
                d_loss_real = d_model.train_on_batch([X_realA, X_realB], [y_real, class_labels])

            else:
                d_loss_real = d_model.train_on_batch([X_realA, X_realB], y_real)

            # --------------------------
            # Train discriminator fake
            # --------------------------
            if use_acgan:
                d_loss_fake = d_model.train_on_batch([X_realA, X_fakeB], [y_fake, class_labels])

            else:
                d_loss_fake = d_model.train_on_batch([X_realA, X_fakeB], y_fake)

            # --------------------------
            # Train generator
            # --------------------------

            if use_acgan:
                g_loss = gan_model.train_on_batch(X_realA, [y_real, class_labels, X_realB], sample_weight=[input_entropy_patch, input_entropy_patch, input_entropy])

            else:
                g_loss = gan_model.train_on_batch(X_realA, [y_real, X_realB],sample_weight=[input_entropy_patch, input_entropy])


            # --------------------------
            # Store losses
            # --------------------------
            epoch_d_real.append(float(
                    d_loss_real[0]
                    if isinstance(d_loss_real,list)
                    else d_loss_real))

            epoch_d_fake.append(float(
                    d_loss_fake[0]
                    if isinstance(d_loss_fake,list)
                    else d_loss_fake))

            epoch_g.append(float(
                g_loss[0]
                if isinstance(g_loss,list)
                else g_loss))

            progress.set_postfix({
                    "D_real":
                    f"{np.mean(epoch_d_real):.4f}",

                    "D_fake":
                    f"{np.mean(epoch_d_fake):.4f}",

                    "G":
                    f"{np.mean(epoch_g):.4f}"
                })

            # cleanup
            del (X_realA,X_realB,X_fakeB,y_real,y_fake)

        # ==================================================
        # Validation after epoch
        # ==================================================
        val_mae = []
        val_d_real_acc = []
        val_d_fake_acc = []

        for i in range(len(val_generator)):
            val_x, val_y, _, _, val_class_labels = val_generator[i]

            # -------------------------
            # Generator validation MAE
            # -------------------------
            pred = g_model.predict_on_batch(val_x)
            mae = np.mean(np.abs(pred - val_y))
            val_mae.append(mae)

            # -------------------------
            # Discriminator validation
            # -------------------------
            batch_val = val_x.shape[0]
            y_real = np.ones((batch_val, n_patch, n_patch, 1))
            y_fake = np.zeros((batch_val, n_patch, n_patch, 1))

            # -------------------------
            # Real samples accuracy
            # -------------------------
            if use_acgan:
                real_output = d_model.predict_on_batch([val_x, val_y])
                fake_output = d_model.predict_on_batch([val_x, pred])

                # AC-GAN returns:
                # [source_output, class_output]
                real_source = real_output[0]
                fake_source = fake_output[0]

            else:
                real_source = d_model.predict_on_batch([val_x, val_y])
                fake_source = d_model.predict_on_batch([val_x, pred])

            # -------------------------
            # PatchGAN accuracy
            # -------------------------
            real_acc = np.mean((real_source > 0.5) == (y_real > 0.5))
            fake_acc = np.mean((fake_source < 0.5) == (y_fake < 0.5))

            val_d_real_acc.append(float(real_acc))
            val_d_fake_acc.append(float(fake_acc))

        val_mae = np.mean(val_mae)
        val_d_real_acc = np.mean(val_d_real_acc)
        val_d_fake_acc = np.mean(val_d_fake_acc)

        print("Epoch", epoch + 1,
            "validation MAE:", val_mae,
            "D real acc:", val_d_real_acc,
            "D fake acc:", val_d_fake_acc)

        history["epoch"].append(epoch+1)
        history["d_loss_real"].append(np.mean(epoch_d_real))
        history["d_loss_fake"].append(np.mean(epoch_d_fake))
        history["g_loss"].append(np.mean(epoch_g))
        history["val_mae"].append(float(val_mae))
        history["val_discriminator_real_accuracy"].append(float(val_d_real_acc))
        history["val_discriminator_fake_accuracy"].append(float(val_d_fake_acc))
        history["amount_samples_trained"].append(len(train_generator.indices))
        # clear TF memory between epochs
        gc.collect()

    return history

