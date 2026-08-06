from tensorflow.keras.layers import Conv2D, \
    MaxPool2D, Conv2DTranspose, Input, Activation, \
    Concatenate, CenterCrop, BatchNormalization, MaxPooling2D,\
    Dropout, concatenate, UpSampling2D, AveragePooling2D, Conv3D, Add, Multiply, Permute, Softmax
from tensorflow.keras import Model

from deep_learning_based_estimation_of_heart_surface_potentials_main.test_model import *
from deep_learning_based_estimation_of_heart_surface_potentials_main.show_data import *

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
    u7 = Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding='same', kernel_initializer='he_normal')(input_layer)

    if has_skip_connection:
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)

    c7 = conv2d_block(u7, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c7


def conv2d_block(input_tensor, n_filters, activation='relu', kernel_size=3, batchnorm=True, stride=1):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               strides=(1,1), kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)#Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               strides=(stride,stride), kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)#Activation('relu')(x)

    return x

def single_conv2d(input_tensor, n_filters, activation='relu', kernel_size=3, batchnorm=True):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), \
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation(activation)(x)#Activation('relu')(x)
    return x

def decoder_block(n_filters, batchnorm, input_layer, skip_layer, has_skip_connection, activation='relu', dropout=0, convSkip=False):
    # u7 = Conv2DTranspose(n_filters, (2,2), strides=(2,2), padding='same', kernel_initializer='he_normal')(input_layer)
    u7 = UpSampling2D((2, 2), interpolation="nearest")(input_layer)# nearest, bilinear
    # u7 = single_conv2d(u7, n_filters, kernel_size=4, batchnorm=True)

    if has_skip_connection:
        if convSkip:
            skip_layer = single_conv2d(skip_layer, n_filters, activation=activation)
            skip_layer = single_conv2d(skip_layer, n_filters, activation=activation)
        u7 = concatenate([u7, skip_layer])
    u7 = Dropout(dropout)(u7)

    c7 = single_conv2d(u7, n_filters, activation=activation, kernel_size=3, batchnorm=batchnorm)
    #c7 = conv2d_block(u7, n_filters, kernel_size=3, batchnorm=batchnorm)
    return c7


def get_unet_adapted(input_img, n_filters=32, dropout=0.2, activation='relu', batchnorm=True, dropout_skip=0, stack_count=6, conc_layers=[False, False, False, False]):
    scaler = 2
    scaler_2 = 1 
    # input_img = input_img + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)
    # Encoder model
    c1 = single_conv2d(input_img, n_filters, activation=activation, kernel_size=3, batchnorm=batchnorm)# single_conv2d
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = single_conv2d(p1, n_filters * scaler ** 1 * scaler_2, activation=activation, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = single_conv2d(p2, n_filters * scaler ** 2 * scaler_2, activation=activation, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = single_conv2d(p3, n_filters * scaler ** 3 * scaler_2, activation=activation, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)


    # # 32x32, 2x2 latent representation
    # # Bottleneck
    # c4 = conv2d_block(p3, n_filters=n_filters * scaler ** 4 * scaler_2, activation=activation, kernel_size=3, batchnorm=batchnorm)
    # # Decoder model
    # c5 = decoder_block(n_filters * scaler ** 3 * scaler_2, batchnorm, input_layer=c4, skip_layer=c3, has_skip_connection=conc_layers[0], activation=activation, dropout=dropout_skip, convSkip=True)
    # c6 = decoder_block(n_filters * scaler ** 2 * scaler_2, batchnorm, input_layer=c5, skip_layer=c2, has_skip_connection=conc_layers[1], activation=activation, dropout=dropout_skip, convSkip=True)
    # c7 = decoder_block(n_filters, batchnorm, input_layer=c6, skip_layer=c1, has_skip_connection=conc_layers[2], activation=activation, dropout=dropout_skip, convSkip=True)
    # outputs = Conv2D(stack_count, (1, 1), activation='sigmoid')(c7)

    # 32x32, 2x2 latent representation
    # Bottleneck
    c5 = conv2d_block(p4, n_filters=n_filters * scaler ** 4 * scaler_2, activation=activation, kernel_size=3, batchnorm=batchnorm)
    # Decoder model
    c6 = decoder_block(n_filters * scaler ** 3 * scaler_2, batchnorm, input_layer=c5, skip_layer=c4, has_skip_connection=conc_layers[0], activation=activation, dropout=dropout_skip, convSkip=True)
    c7 = decoder_block(n_filters * scaler ** 2 * scaler_2, batchnorm, input_layer=c6, skip_layer=c3, has_skip_connection=conc_layers[1], activation=activation, dropout=dropout_skip, convSkip=True)
    c8 = decoder_block(n_filters * scaler ** 1 * scaler_2, batchnorm, input_layer=c7, skip_layer=c2, has_skip_connection=conc_layers[2], activation=activation, dropout=dropout_skip, convSkip=True)
    c9 = decoder_block(n_filters, batchnorm, input_layer=c8, skip_layer=c1, has_skip_connection=conc_layers[3], activation=activation, dropout=dropout_skip, convSkip=True)
    outputs = Conv2D(stack_count, (1, 1), activation='sigmoid')(c9)
    
    # # 32x32, 2x2 latent representation
    # # Bottleneck
    # c5 = conv2d_block(p4, n_filters=n_filters * scaler ** 4 * scaler_2, kernel_size=3, batchnorm=batchnorm)
    # # Decoder model
    # c6 = decoder_block(n_filters * scaler ** 3 * scaler_2, batchnorm, input_layer=c5, skip_layer=c4, has_skip_connection=conc_layers[0], dropout=dropout_skip, convSkip=False)
    # c7 = decoder_block(n_filters * scaler ** 2 * scaler_2, batchnorm, input_layer=c6, skip_layer=c3, has_skip_connection=conc_layers[1], dropout=dropout_skip, convSkip=False)
    # c8 = decoder_block(n_filters * scaler ** 1 * scaler_2, batchnorm, input_layer=c7, skip_layer=c2, has_skip_connection=conc_layers[2], dropout=dropout_skip, convSkip=False)
    # c9 = decoder_block(n_filters, batchnorm, input_layer=c8, skip_layer=c1, has_skip_connection=conc_layers[3], dropout=dropout_skip, convSkip=False)
    # outputs = Conv2D(stack_count, (1, 1), activation='sigmoid')(c9)
    
    # # 32x32, 1x1 latent representation
    # c5  = single_conv2d(p4, n_filters * scaler ** 4 * scaler_2, kernel_size=3, batchnorm=batchnorm)
    # p5 = MaxPooling2D((2, 2))(c5)
    # p5 = Dropout(dropout)(p5)    
    # # Bottleneck
    # c6 = conv2d_block(p5, n_filters=n_filters * scaler ** 5 * scaler_2, kernel_size=3, batchnorm=batchnorm)
    # # Decoder model
    # c7 = decoder_block(n_filters * scaler ** 4 * scaler_2, batchnorm, input_layer=c6, skip_layer=c5, has_skip_connection=conc_layers[0], dropout=dropout_skip, convSkip=True)
    # c8 = decoder_block(n_filters * scaler ** 3 * scaler_2, batchnorm, input_layer=c7, skip_layer=c4, has_skip_connection=conc_layers[1], dropout=dropout_skip, convSkip=True)
    # c9 = decoder_block(n_filters * scaler ** 2 * scaler_2, batchnorm, input_layer=c8, skip_layer=c3, has_skip_connection=conc_layers[2], dropout=dropout_skip, convSkip=True)
    # c10 = decoder_block(n_filters * scaler ** 1 * scaler_2, batchnorm, input_layer=c9, skip_layer=c2, has_skip_connection=conc_layers[3], dropout=dropout_skip, convSkip=True)
    # c11 = decoder_block(n_filters, batchnorm, input_layer=c10, skip_layer=c1, has_skip_connection=conc_layers[4], dropout=dropout_skip, convSkip=True)
    # outputs = Conv2D(stack_count, (1, 1), activation='sigmoid')(c11)


    # # # 64x64, 1x1 latent representation
    # c5  = single_conv2d(p4, n_filters * scaler ** 4 * scaler_2, kernel_size=3, batchnorm=batchnorm)
    # p5 = MaxPooling2D((2, 2))(c5)
    # p5 = Dropout(dropout)(p5)

    # c6  = single_conv2d(p5, n_filters * scaler ** 5 * scaler_2, kernel_size=3, batchnorm=batchnorm)
    # p6 = MaxPooling2D((2, 2))(c6)
    # p6 = Dropout(dropout)(p6)
    # # Bottleneck
    # c7 = conv2d_block(p6, n_filters=n_filters * scaler ** 6 * scaler_2, kernel_size=3, batchnorm=batchnorm)
    # # Decoder model
    # c8  = decoder_block(n_filters * scaler ** 5 * scaler_2, batchnorm, input_layer=c7, skip_layer=c6, has_skip_connection=conc_layers[0], dropout=dropout_skip, convSkip=True)
    # c9  = decoder_block(n_filters * scaler ** 4 * scaler_2, batchnorm, input_layer=c8, skip_layer=c5, has_skip_connection=conc_layers[1], dropout=dropout_skip, convSkip=True)
    # c10 = decoder_block(n_filters * scaler ** 3 * scaler_2, batchnorm, input_layer=c9, skip_layer=c4, has_skip_connection=conc_layers[2], dropout=dropout_skip, convSkip=True)
    # c11 = decoder_block(n_filters * scaler ** 2 * scaler_2, batchnorm, input_layer=c10, skip_layer=c3, has_skip_connection=conc_layers[3], dropout=dropout_skip, convSkip=True)
    # c12 = decoder_block(n_filters * scaler ** 1 * scaler_2, batchnorm, input_layer=c11, skip_layer=c2, has_skip_connection=conc_layers[4], dropout=dropout_skip, convSkip=True)
    # c13 = decoder_block(n_filters, batchnorm, input_layer=c12, skip_layer=c1, has_skip_connection=conc_layers[5], dropout=dropout_skip, convSkip=True)
    # outputs = Conv2D(stack_count, (1, 1), activation='sigmoid')(c13)


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
    c6 = decoder_block_base_unet(n_filters * 8, batchnorm, input_layer=c5, skip_layer=c4, has_skip_connection=conc_layers[0])
    c7 = decoder_block_base_unet(n_filters * 4, batchnorm, input_layer=c6, skip_layer=c3, has_skip_connection=conc_layers[1])
    c8 = decoder_block_base_unet(n_filters * 2, batchnorm, input_layer=c7, skip_layer=c2, has_skip_connection=conc_layers[2])
    c9 = decoder_block_base_unet(n_filters, batchnorm, input_layer=c8, skip_layer=c1, has_skip_connection=conc_layers[3])

    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    model.summary()
    return model


def train_unet(model_prefix ="unet"):
    input_img = Input((img_dim, img_dim, 1), name='img')
    model = get_unet(input_img, n_filters=filters, dropout=dropout, batchnorm=False)
    model.summary()
    for i in range(3, 4):
        model = get_unet(input_img, n_filters=filters, dropout=dropout, batchnorm=False)
        history = train_model(model, False, img_dim, img_dim, epochs=epochs, dataset_nr=i)
        save_result(model, history, i, get_model_name(model_prefix))
        show_model_examples(model, i)


def train_unet_adapted(model_name = "unet-v3-allskip-conv"):
    input_img = Input((img_dim, img_dim, 1), name='img')
    model = get_unet_adapted(input_img, n_filters=filters, dropout=dropout, batchnorm=True)
    model.summary()
    for i in range(3, 4):
        model = get_unet_adapted(input_img, n_filters=filters, dropout=dropout, batchnorm=True)
        history = train_model(model, False, img_dim, img_dim, epochs=epochs, dataset_nr=i)
        save_result(model, history, i, get_model_name(model_name))
        show_model_examples(model, i)


def show_model_history(block=True, model_prename = "unet"):
    historys = []
    names = []
    for i in range(4):
        historys.append(retrieve_history(get_model_name(model_prename), i, img_dim, epochs))
        names.append(get_model_name(model_prename) + "-data_" + str(i))
        print(historys[i]['val_mean_squared_error'][7])
    visualize_data(historys, names, block=block)


def get_model_name(prename = "unet", filters=filters, dropout=dropout):
    adaptations = "n-data-"
    return adaptations + "_" + prename + "_f" + str(filters) + "_d" + str(dropout)


def show_model_examples(model, val_nr):
    (train_x, train_y), (test_X, test_y) = get_data(img_dim, datasets[val_nr], False)
    train_x = train_x[np.arange(0, train_x.shape[0], 100), :, :]
    train_y = train_y[np.arange(0, train_y.shape[0], 100), :, :]
    test_X = test_X[np.arange(0, test_X.shape[0], 30), :, :]
    test_y = test_y[np.arange(0, test_y.shape[0], 30), :, :]

    #show_intermetdiate_images(model, test_X[0])
    show_model_predictions(model, train_x, train_y, test_X, test_y, img_dim)