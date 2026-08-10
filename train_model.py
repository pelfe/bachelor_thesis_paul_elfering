import os
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_USE_CUDNN_FRONTEND"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

import json
import tensorflow.keras
import os
import deep_learning_estimation_paul.train_model as train_model
import deep_learning_estimation_paul.discriminator_structures as discriminator_structures

# ======================================================
#define subjects:
train_subjects_mixed = [
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10")
]

test_subjects_mixed = [
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("bachelor_thesis_paul_elfering/data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]



#define params:
T = 15                  # time window depth
lambda1 = 100           # generator loss weight
lambda2 = 0.01          # regularization weight
alpha = 0.2             # loss balancing parameter
learning_rate_discriminator = 0.01
learning_rate_generator = 0.01
dropout_ratio = 0.2     # dropout probability
label_smoothing = False
use_acgan = False
# start training:

model_name = "original_gan_tiantian_mixed"
batch_size = 16

train_generator = train_model.SubjectGenerator(
    train_subjects_mixed,
    batch_size=batch_size,
    shuffle=True,
    augment=True,
    aug_v=0.5,
    augment_factor=2
)

# -------------------------------------------------------
# Create model
# -------------------------------------------------------
input_shape = train_generator[0][0].shape[1:]

input_img = tensorflow.keras.Input(
    shape=input_shape
)


g_model = train_model.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = discriminator_structures.define_1x1_patch_relu_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate_discriminator,
    decay_rate=0.96,
)
gan_model = train_model.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
    learning_rate=learning_rate_generator)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = train_model.train(
    d_model,
    g_model,
    gan_model,
    train_subjects_mixed,
    test_subjects_mixed,
    discriminator_lr = learning_rate_discriminator,
    generator_lr = learning_rate_generator,
    epochs=15,
    batch_size=batch_size,
    label_smoothing=label_smoothing,
    use_acgan=use_acgan
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("bachelor_thesis_paul_elfering/data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)






#define params:
T = 15                  # time window depth
lambda1 = 100           # generator loss weight
lambda2 = 0.01          # regularization weight
alpha = 0.2             # loss balancing parameter
learning_rate_discriminator = 0.01
learning_rate_generator = 0.01
dropout_ratio = 0.2     # dropout probability
label_smoothing = False
use_acgan = False
# start training:

model_name = "leaky_relu_mixed"
batch_size = 16

train_generator = train_model.SubjectGenerator(
    train_subjects_mixed,
    batch_size=batch_size,
    shuffle=True,
    augment=True,
    aug_v=0.5,
    augment_factor=2
)

# -------------------------------------------------------
# Create model
# -------------------------------------------------------
input_shape = train_generator[0][0].shape[1:]

input_img = tensorflow.keras.Input(
    shape=input_shape
)


g_model = train_model.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = discriminator_structures.define_1x1_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate_discriminator,
    decay_rate=0.96,
)
gan_model = train_model.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
    learning_rate=learning_rate_generator)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = train_model.train(
    d_model,
    g_model,
    gan_model,
    train_subjects_mixed,
    test_subjects_mixed,
    discriminator_lr = learning_rate_discriminator,
    generator_lr = learning_rate_generator,
    epochs=15,
    batch_size=batch_size,
    label_smoothing=label_smoothing,
    use_acgan=use_acgan
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("bachelor_thesis_paul_elfering/data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)



#define params:
T = 15                  # time window depth
lambda1 = 100           # generator loss weight
lambda2 = 0.01          # regularization weight
alpha = 0.2             # loss balancing parameter
learning_rate_discriminator = 0.01
learning_rate_generator = 0.01
dropout_ratio = 0.2     # dropout probability
label_smoothing = True
use_acgan = False
# start training:

model_name = "label_smoothing_mixed"
batch_size = 16

train_generator = train_model.SubjectGenerator(
    train_subjects_mixed,
    batch_size=batch_size,
    shuffle=True,
    augment=True,
    aug_v=0.5,
    augment_factor=2
)

# -------------------------------------------------------
# Create model
# -------------------------------------------------------
input_shape = train_generator[0][0].shape[1:]

input_img = tensorflow.keras.Input(
    shape=input_shape
)


g_model = train_model.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = discriminator_structures.define_1x1_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate_discriminator,
    decay_rate=0.96,
)
gan_model = train_model.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
    learning_rate=learning_rate_generator)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = train_model.train(
    d_model,
    g_model,
    gan_model,
    train_subjects_mixed,
    test_subjects_mixed,
    discriminator_lr = learning_rate_discriminator,
    generator_lr = learning_rate_generator,
    epochs=15,
    batch_size=batch_size,
    label_smoothing=label_smoothing,
    use_acgan=use_acgan
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("bachelor_thesis_paul_elfering/data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)




















import deep_learning_estimation_paul.train_generator_only as train_generator_only
#define params:
T = 15                  # time window depth
lambda1 = 100           # generator loss weight
lambda2 = 0.01          # regularization weight
alpha = 0.2             # loss balancing parameter
learning_rate_generator = 0.01
dropout_ratio = 0.2     # dropout probability
# start training:

model_name = "no_discriminator_model"
batch_size = 16

train_generator = train_model.SubjectGenerator(
    train_subjects_mixed,
    batch_size=batch_size,
    shuffle=True,
    augment=True,
    aug_v=0.5,
    augment_factor=2
)

# -------------------------------------------------------
# Create model
# -------------------------------------------------------
input_shape = train_generator[0][0].shape[1:]

input_img = tensorflow.keras.Input(
    shape=input_shape
)


g_model = train_model.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------
history = train_generator_only(
        generator=g_model,
        train_subjects=train_subjects_mixed,
        val_subjects=test_subjects_mixed,
        train_labels=None,
        epochs=15,
        batch_size=16,
        learning_rate=learning_rate_generator,
    )


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("bachelor_thesis_paul_elfering/data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)