# Set seed values to remove randomness from that
seed_value = 0
import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Layer, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, \
    BatchNormalization, AveragePooling2D, MaxPooling2D, UpSampling2D, ReLU, Dense, Input, GlobalAveragePooling2D
from deep_learning_based_estimation_of_heart_surface_potentials_main.show_data import *









def define_test_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate=0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 16, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (2, 2), padding='valid', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(model.summary())

    # initial learning rate
    initial_learning_rate = initial_lr

    # decay step and decay rate
    decay_steps = 1000

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    # compile model
    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model






# define the discriminator model
def define_1x1_patch_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate=0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 16, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (2, 2), padding='valid', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(model.summary())

    # initial learning rate
    initial_learning_rate = initial_lr

    # decay step and decay rate
    decay_steps = 1000

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    # compile model
    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_2x2_patch_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate=0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 8, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (3, 3), padding='valid', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(model.summary())

    # initial learning rate
    initial_learning_rate = initial_lr

    # decay step and decay rate
    decay_steps = 1000

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    # compile model
    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_4x4_patch_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate=0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (5, 5), padding='valid', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(model.summary())

    # initial learning rate
    initial_learning_rate = initial_lr

    # decay step and decay rate
    decay_steps = 1000

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    # compile model
    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_8x8_patch_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate=0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (1, 1), padding='valid', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(model.summary())

    # initial learning rate
    initial_learning_rate = initial_lr

    # decay step and decay rate
    decay_steps = 1000

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    # compile model
    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

def define_16x16_patch_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate=0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(negative_slope=0.2)(d)

    d = Conv2D(n_filters * 2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(negative_slope=0.2)(d)

    # patch output
    d = Conv2D(1, (1, 1), padding='valid', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)
    # define model
    model = Model([in_src_image, in_target_image], patch_out)
    print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
    print(model.summary())

    # initial learning rate
    initial_learning_rate = initial_lr

    # decay step and decay rate
    decay_steps = 1000

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    # compile model
    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model



def define_2x2_ac_patch_discriminator(
        image_shape,
        n_classes,
        n_filters=32,
        activation='relu',
        initial_lr=0.001,
        decay_rate=0.96):

    init = RandomNormal(stddev=0.02)

    in_src = Input(shape=image_shape)
    in_target = Input(shape=image_shape)

    merged = Concatenate()([in_src, in_target])

    d = Conv2D(n_filters,3,strides=2,padding='same',
               kernel_initializer=init)(merged)
    d = Activation(activation)(d)

    d = Conv2D(n_filters*2,3,strides=2,padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)

    d = Conv2D(n_filters*4,3,strides=2,padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)

    d = Conv2D(n_filters*8,3,strides=2,padding='same',
               kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)

    # ----------------------------
    # Source output (PatchGAN)
    # ----------------------------
    source = Conv2D(
        1,
        3,
        padding="valid",
        kernel_initializer=init)(d)

    source_out = Activation(
        "sigmoid",
        name="source")(source)

    # ----------------------------
    # Class output
    # ----------------------------
    c = GlobalAveragePooling2D()(d)

    class_out = Dense(
        n_classes,
        activation="softmax",
        name="class")(c)

    model = Model(
        [in_src,in_target],
        [source_out,class_out]
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=1000,
        decay_rate=decay_rate
    )

    opt = Adam(lr_schedule,beta_1=0.5)

    model.compile(
        optimizer=opt,
        loss=[
            "binary_crossentropy",
            "sparse_categorical_crossentropy"
        ],
        loss_weights=[1.0,1.0]
    )

    return model