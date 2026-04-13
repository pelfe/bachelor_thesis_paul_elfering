import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, \
    MaxPool2D, Conv2DTranspose, Input, Activation, \
    Concatenate, CenterCrop, BatchNormalization, MaxPooling2D,\
    Dropout, concatenate, UpSampling2D, AveragePooling2D
from tensorflow.keras import Model


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def inception_block(input_tensor, n_filters, batchnorm):
    direct = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    if batchnorm:
        direct = BatchNormalization()(direct)

    tower_1 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    tower_1 = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(tower_1)
    if batchnorm:
        tower_1 = BatchNormalization()(tower_1)

    tower_2 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(input_tensor)
    tower_2 = Conv2D(n_filters, (5, 5), padding='same', activation='relu')(tower_2)
    if batchnorm:
        tower_2 = BatchNormalization()(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
    tower_3 = Conv2D(n_filters, (1, 1), padding='same', activation='relu')(tower_3)

    if batchnorm:
        tower_3 = BatchNormalization()(tower_3)

    output = concatenate([direct, tower_1, tower_2, tower_3], axis=3)

    return output


def build_unet_inception(input_img, n_filters=32, dropout=0.2, batchnorm=True, conc_layers=[True, True, True, True]):
    # Encoder model
    c1 = inception_block(input_img, n_filters * 1, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = inception_block(p1, n_filters * 2, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = inception_block(p2, n_filters * 4, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = inception_block(p3, n_filters * 8, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = inception_block(p4, n_filters=n_filters * 8, batchnorm=batchnorm)

    # Decoder model
    c7 = decoder_block_transconv(n_filters * 8, batchnorm, input_layer=c5, skip_layer=c4, has_skip_connection=conc_layers[0], dropout=dropout)
    c7 = decoder_block_transconv(n_filters * 4, batchnorm, input_layer=c7, skip_layer=c3, has_skip_connection=conc_layers[1], dropout=dropout)
    c8 = decoder_block_transconv(n_filters * 2, batchnorm, input_layer=c7, skip_layer=c2, has_skip_connection=conc_layers[2], dropout=dropout)
    c9 = decoder_block_transconv(n_filters, batchnorm, input_layer=c8, skip_layer=c1, has_skip_connection=conc_layers[3], dropout=dropout)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


def decoder_block_transconv(n_filters, batchnorm, input_layer, skip_layer, has_skip_connection, dropout):
    u7 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(input_layer)
    if has_skip_connection:
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c7


def decoder_block(n_filters, batchnorm, input_layer, skip_layer, has_skip_connection, dropout):
    u7 = UpSampling2D((2, 2), interpolation="bilinear")(input_layer)
    if has_skip_connection:
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c7


def build_unet_inception_2(input_img, n_filters=32, dropout=0.2, batchnorm=True, conc_layers=[True, True, True, True]):
    # Encoder model
    c1 = inception_block(input_img, n_filters * 1, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = inception_block(p1, n_filters * 2, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = inception_block(p2, n_filters * 4, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = inception_block(p3, n_filters * 8, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    # Bottleneck
    c5 = inception_block(p4, n_filters=n_filters * 16, batchnorm=batchnorm)

    # Decoder model
    c7 = decoder_block(n_filters * 8, batchnorm, input_layer=c5, skip_layer=c4, has_skip_connection=conc_layers[0], dropout=dropout)
    c7 = decoder_block(n_filters * 4, batchnorm, input_layer=c7, skip_layer=c3, has_skip_connection=conc_layers[1], dropout=dropout)
    c8 = decoder_block(n_filters * 2, batchnorm, input_layer=c7, skip_layer=c2, has_skip_connection=conc_layers[2], dropout=dropout)
    c9 = decoder_block(n_filters, batchnorm, input_layer=c8, skip_layer=c1, has_skip_connection=conc_layers[3], dropout=dropout)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    print(model.summary())
    return model
