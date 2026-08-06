model_name = "mixed_15stacks_gan"
batch_size = 16
train_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10")
]

test_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]

train_generator = main_bayesian_cross.SubjectGenerator(
    train_subjects,
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


g_model = main_bayesian_cross.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = main_bayesian_cross.define_1x1_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate,
    decay_rate=0.96,
)
gan_model = main_bayesian_cross.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = main_bayesian_cross.train(
    d_model,
    g_model,
    gan_model,
    train_subjects,
    test_subjects,
    epochs=15,
    batch_size=batch_size,
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)







train_subjects = [
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10")
]

test_subjects = [
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]

train_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),

    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
]


train_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
]

test_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
]




model_name = "mixed_3x3_15stacks_ac_gan"
batch_size = 16
train_subjects = [
    #("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10")
]

test_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]

train_labels = [
    0,0,0,0,0,0,0,       # healthy ct2-ct8
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, # idiopathic
    1,1,1,1,1                # lbbb
]

test_labels = [
    0,0,0,       # healthy
    0,0,0,0,0,   # idiopathic
    1,1          # lbbb
]

train_generator = main_bayesian_cross.SubjectGenerator(
    train_subjects,
    subject_labels=train_labels,
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


g_model = main_bayesian_cross.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = main_bayesian_cross.define_2x2_ac_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate,
    decay_rate=0.96,
    n_classes= 2
)
gan_model = main_bayesian_cross.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
    use_acgan = True
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = main_bayesian_cross.train(
    d_model,
    g_model,
    gan_model,
    train_subjects,
    test_subjects,
    train_labels=train_labels,
    val_labels=test_labels,
    epochs=15,
    batch_size=batch_size,
    use_acgan=True
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)





# -------------------------------------------------------
# Dataset setup
# -------------------------------------------------------
model_name = "mixed_15stacks_gan"
batch_size = 16
train_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10")
]

test_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]

train_generator = main_bayesian_cross.SubjectGenerator(
    train_subjects,
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


g_model = main_bayesian_cross.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = main_bayesian_cross.define_1x1_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate,
    decay_rate=0.96,
)
gan_model = main_bayesian_cross.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = main_bayesian_cross.train(
    d_model,
    g_model,
    gan_model,
    train_subjects,
    test_subjects,
    epochs=15,
    batch_size=batch_size,
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)




model_name = "mixed_2x2_15stacks_ac_gan"
batch_size = 16
train_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "B6"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT2"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT3"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT4"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject", "ECGiCRT10")
]

test_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]

train_labels = [
    0,0,0,0,0,0,0,0,       # healthy ct2-ct8
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, # idiopathic
    1,1,1,1,1                # lbbb
]

test_labels = [
    0,0,0,       # healthy
    0,0,0,0,0,   # idiopathic
    1,1          # lbbb
]

train_generator = main_bayesian_cross.SubjectGenerator(
    train_subjects,
    subject_labels=train_labels,
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


g_model = main_bayesian_cross.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = main_bayesian_cross.define_2x2_ac_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate,
    decay_rate=0.96,
    n_classes= 2
)
gan_model = main_bayesian_cross.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
    use_acgan = True
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = main_bayesian_cross.train(
    d_model,
    g_model,
    gan_model,
    train_subjects,
    test_subjects,
    train_labels=train_labels,
    val_labels=test_labels,
    epochs=15,
    batch_size=batch_size,
    use_acgan=True
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)




# ======================================================
# RUN TRAINING
# ======================================================
model_name = "idiohealthy_15stacks_gan"
batch_size = 16
train_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct1"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct2"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct3"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct4"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct5"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct6"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct7"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A91"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "A95"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "B62"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F4"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "F5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "G39"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "H19"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "I60"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "J74"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "K72"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "L8"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "M58"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "P12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Q55"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "R9"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "S7"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "T12"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W2"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W5"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W18"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "W93"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X2")
]

test_subjects = [
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct9"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct10"),
    ("data/post_interpolation/healthy/npy_stacked_15/size64_subject", "ct11"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "X35"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y46"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Y82"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z3"),
    ("data/post_interpolation/idiopathic/npy_stacked_15/size64_subject", "Z34"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","ECGiCRT17"),
    ("data/post_interpolation/lbbb/npy_stacked_15/size64_subject","L79")
]

train_generator = main_bayesian_cross.SubjectGenerator(
    train_subjects,
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


g_model = main_bayesian_cross.get_unet_adapted(
    input_img,
    n_filters=32,
    activation="relu",
    dropout=dropout_ratio,
    batchnorm=True,
    dropout_skip=0,
    stack_count=input_shape[-1],
    conc_layers=[True,True,True,True]
)
d_model = main_bayesian_cross.define_1x1_patch_discriminator(
    input_shape,
    n_filters=32,
    activation="relu",
    initial_lr=learning_rate,
    decay_rate=0.96,
)
gan_model = main_bayesian_cross.define_gan(
    g_model,
    d_model,
    input_shape,
    regularization_w=lambda2,
    lambda1=lambda1,
    alpha=alpha,
)

# -------------------------------------------------------
# Train
# -------------------------------------------------------

history = main_bayesian_cross.train(
    d_model,
    g_model,
    gan_model,
    train_subjects,
    test_subjects,
    epochs=15,
    batch_size=batch_size,
)


# ======================================================
# SAVE MODELS
# ======================================================
save_dir = ("data/models/pauls_models/"+ model_name)
os.makedirs(save_dir,exist_ok=True)

g_model.save(save_dir+"/generator.keras")
d_model.save(save_dir+"/discriminator.keras")
gan_model.save(save_dir+"/gan.keras")
with open(save_dir+"/history.json","w") as f:
    json.dump(history,f,indent=4)