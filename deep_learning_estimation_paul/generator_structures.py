import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"


import tensorflow as tf
tf.config.optimizer.set_jit(False)
from tensorflow.keras.layers import Conv2D, \
    MaxPool2D, Conv2DTranspose, Input, Activation, \
    Concatenate, CenterCrop, BatchNormalization, MaxPooling2D,\
    Dropout, concatenate, UpSampling2D, AveragePooling2D, Conv3D, Add, Multiply, Permute, Softmax
from tensorflow.keras import Model

img_dim = 32
epochs = 10
filters = 32
dropout = 0


def self_attention_block(x):
    feature_size = x.shape[3]
    f = Conv2D(filters=feature_size // 2, kernel_size=1, strides=1, padding='same')(x)
    g = Conv2D(filters=feature_size // 2, kernel_size=1, strides=1, padding='same')(x)
    h = Conv2D(filters=feature_size // 2, kernel_size=1, strides=1, padding='same')(x)

    gt = Permute((2, 1, 3))(g)

    gf = Multiply()([gt, f])
    sm = Softmax(axis=2)(gf)

    v = Multiply()([h, sm])
    output = Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same')(v)
    return output


def attention_block(gating, x):
    feature_size = x.shape[3]
    # Make filter amount of x and gating equal
    phi_g = Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same')(gating)

    concat_xg = Add()([phi_g, x])

    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(act_xg)

    result = Multiply()([psi, x])
    result_bn = BatchNormalization()(result)

    return result_bn


def decoder_block_adapted(n_filters, batchnorm, input_layer, skip_layer, has_skip_connection):
    u7 = UpSampling2D((2, 2), interpolation="bilinear")(input_layer)
    if has_skip_connection:
        # skip_layer = single_conv2d(skip_layer, n_filters, kernel_size=3, batchnorm=batchnorm)
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c7


def decoder_block_base_unet(n_filters, batchnorm, input_layer, skip_layer, has_skip_connection, dropout=0):
    u7 = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(input_layer)

    if has_skip_connection:
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c7


def conv2d_block(input_tensor, n_filters, activation='relu', kernel_size=3, batchnorm=True, stride=1):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               strides=(1, 1), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)  # Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               strides=(stride, stride), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)  # Activation('relu')(x)

    return x


def single_conv2d(input_tensor, n_filters, activation='relu', kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)  # Activation('relu')(x)
    return x


def decoder_block(n_filters, batchnorm, input_layer, skip_layer, has_skip_connection, activation='relu', dropout=0,
                  convSkip=False):
    u7 = UpSampling2D((2, 2), interpolation="nearest")(input_layer)  # nearest, bilinear

    if has_skip_connection:
        if convSkip:
            skip_layer = single_conv2d(skip_layer, n_filters, activation=activation)
            skip_layer = single_conv2d(skip_layer, n_filters, activation=activation)
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)

    c7 = single_conv2d(u7, n_filters, activation=activation, kernel_size=3, batchnorm=batchnorm)
    return c7


def get_unet_adapted(input_img, n_filters=64, dropout=0.2, activation='relu', batchnorm=True, dropout_skip=0,
                     stack_count=15, conc_layers=[True, True, True, True]):
    scaler = 2
    scaler_2 = 1
    # input_img = input_img + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)
    # Encoder model
    c1 = single_conv2d(input_img, n_filters, activation=activation, kernel_size=3, batchnorm=batchnorm)  # single_conv2d
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = single_conv2d(p1, n_filters * scaler ** 1 * scaler_2, activation=activation, kernel_size=3,
                       batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = single_conv2d(p2, n_filters * scaler ** 2 * scaler_2, activation=activation, kernel_size=3,
                       batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = single_conv2d(p3, n_filters * scaler ** 3 * scaler_2, activation=activation, kernel_size=3,
                       batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * scaler ** 4 * scaler_2, activation=activation, kernel_size=3,
                      batchnorm=batchnorm)
    # Decoder model
    c6 = decoder_block(n_filters * scaler ** 3 * scaler_2, batchnorm, input_layer=c5, skip_layer=c4,
                       has_skip_connection=conc_layers[0], activation=activation, dropout=dropout_skip, convSkip=True)
    c7 = decoder_block(n_filters * scaler ** 2 * scaler_2, batchnorm, input_layer=c6, skip_layer=c3,
                       has_skip_connection=conc_layers[1], activation=activation, dropout=dropout_skip, convSkip=True)
    c8 = decoder_block(n_filters * scaler ** 1 * scaler_2, batchnorm, input_layer=c7, skip_layer=c2,
                       has_skip_connection=conc_layers[2], activation=activation, dropout=dropout_skip, convSkip=True)
    c9 = decoder_block(n_filters, batchnorm, input_layer=c8, skip_layer=c1, has_skip_connection=conc_layers[3],
                       activation=activation, dropout=dropout_skip, convSkip=True)
    outputs = Conv2D(stack_count, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary()
    return model


def get_unet(input_img, n_filters=32, dropout=0, batchnorm=True, conc_layers=[True, True, True, True]):
    # Encoder model
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # Decoder model
    c6 = decoder_block_base_unet(n_filters * 8, batchnorm, input_layer=c5, skip_layer=c4,
                                 has_skip_connection=conc_layers[0])
    c7 = decoder_block_base_unet(n_filters * 4, batchnorm, input_layer=c6, skip_layer=c3,
                                 has_skip_connection=conc_layers[1])
    c8 = decoder_block_base_unet(n_filters * 2, batchnorm, input_layer=c7, skip_layer=c2,
                                 has_skip_connection=conc_layers[2])
    c9 = decoder_block_base_unet(n_filters, batchnorm, input_layer=c8, skip_layer=c1,
                                 has_skip_connection=conc_layers[3])

    # Output
    outputs = Conv2D(15, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary()
    return model

