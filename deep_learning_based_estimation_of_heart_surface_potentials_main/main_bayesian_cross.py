# Set seed values to remove randomness from that
seed_value= 0
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
from keras.layers import Layer, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, AveragePooling2D, MaxPooling2D, UpSampling2D, ReLU, Dense, Input
from deep_learning_based_estimation_of_heart_surface_potentials_main.show_data import *
from deep_learning_based_estimation_of_heart_surface_potentials_main.UNet import get_unet_adapted
import gc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def ssim_loss(y_true, y_pred):
    # SSIM loss calculation
    ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0) + 0.0
    return ssim_loss

class L1RegularizationLayer(Layer):
    def __init__(self, strength=0.01, **kwargs):
        super(L1RegularizationLayer, self).__init__(**kwargs)
        self.strength = strength

    def call(self, inputs):
        self.add_loss(self.strength * tf.reduce_sum(tf.abs(inputs)+0.0))
        return inputs
# def generator_l1_regularization(y_true, y_pred):
#     return tf.reduce_mean(tf.abs(y_pred - y_true))


def create_feedforward_model(hidden_units_1, hidden_units_2):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(252, 252,)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(hidden_units_1, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(hidden_units_2, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(252 * 252, activation='sigmoid'))
    model.add(tf.keras.layers.Reshape((252, 252)))

    model.summary()
    return model


# define the discriminator model
def define_discriminator(image_shape, n_filters=32, activation='relu', initial_lr=0.001, decay_rate = 0.96):
    # weights init
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    # concatenate images channel-wise
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = Activation(activation)(d)#ReLU()(d)

    d = Conv2D(n_filters * 2, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)#ReLU()(d)

    d = Conv2D(n_filters * 4, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)#ReLU()(d)

    d = Conv2D(n_filters * 8, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)#ReLU()(d)

    d = Conv2D(n_filters * 16, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = Activation(activation)(d)#ReLU()(d)

    # patch output
    d = Conv2D(1, (2,2), padding='valid', kernel_initializer=init)(d)
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


# define an encoder block
def encoder_block(layer_in, n_filters, kernel_size=3, stride_size=2, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), strides=(stride_size, stride_size), \
            kernel_initializer=init, padding='same')(layer_in)
    if batchnorm:
        x = BatchNormalization()(x, training=True)
    x = Activation('relu')(x)

    return x


def decoder_block(layer_in, skip_in, n_filters, dropout=0.0):
    # weight init
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    # g = UpSampling2D((2, 2), interpolation="bilinear")(layer_in)
    g = BatchNormalization()(g, training=True)

    if dropout > 0.0:
        g = Dropout(dropout)(g, training=True)
    # merge with skip connection
    if skip_in is not None:
        g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)

    # g = single_conv2d(g, n_filters, kernel_size=3, batchnorm=False)

    return g

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape, regularization_w=0.01, ssim_l1_r=50, ssim_w=0.5):
    # Discriminator weights should be updated separately
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False

    in_src = Input(shape=image_shape)
    entropy_input = Input(shape=(1,))
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    
    # connect the source input and generator output to the discriminator input
    # print(in_src, gen_out)

    in_src = Input(shape=image_shape)

    gen_out = g_model(in_src)

    if isinstance(gen_out, list):
        gen_out = gen_out[0]


    dis_out = d_model([in_src, gen_out])

    # src image as input, discriminator classification and generated image as output
    model = Model([in_src, entropy_input],[dis_out, gen_out])

    # Add L1 regularization to the generator's Conv2D layers
    l1_strength = regularization_w  # You can adjust the regularization strength
    for layer in g_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Apply L1 regularization to kernel weights
            layer.kernel_regularizer = l1(l1_strength)

    # initial learning rate
    initial_learning_rate = 0.002

    # decay step and decay rate
    decay_steps = 1000
    decay_rate = 0.96

    # 创建学习率衰减策略
    learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=False  # become smooth when setting to True
    )

    opt = Adam(learning_rate=learning_rate_decay, beta_1=0.7)
    model.compile(loss=['binary_crossentropy', 'mae', ssim_loss], optimizer=opt, loss_weights=[1, ssim_l1_r*(1-ssim_w), ssim_l1_r*ssim_w])
    
    return model


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape, weights, name='train_'):
    trainA, trainB = dataset
    # choose random instances
    if name=="train_":
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


# def calc_mae(g_model, dataset):
#     x, y = dataset

#     # generate a batch of fake samples
#     X_fakeB, _ = generate_fake_samples(g_model, x, 1)
#     X_fakeB = X_fakeB.squeeze()
#     mae = np.abs(np.subtract(X_fakeB, y)).mean()
#     mae = round(mae, 4)
#     return mae
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



# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=20, name="train_"):
    # select a sample of input images
    [X_realA, X_realB], _, _ = generate_real_samples(dataset, n_samples, 1, [], name)

    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    X_fakeB = X_fakeB.squeeze()
    mse = np.square(np.subtract(X_fakeB, X_realB)).mean()
    mse = round(mse, 4)
    # print(mse)

    if name=="train_":
        fig = plt.figure(figsize=(64, 64))
        colums = 3
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(n_samples, colums),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        share_all=True
                        )
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        for i, ax in enumerate(grid):
            if i % colums == 0:
                ax.imshow(X_realA[i // colums][:,:,1], cmap='gray', vmin=0, vmax=1, interpolation='none')
            elif i % colums == 1:
                ax.imshow(X_realB[i // colums][:,:,1], cmap='gray', vmin=0, vmax=1, interpolation='none')
                # [:,:,1]
            elif i % colums == 2:
                ax.imshow(X_fakeB[i // colums][:,:,1], cmap='gray', vmin=0, vmax=1, interpolation='none')
                # print(str(i // colums) + " : " + str(get_pearson_correlation_coefficient(X_fakeB[i // colums], X_realB[i // colums])))

        filename1 = "/home/P70066803/pythonProject/DL_paper/plots/train"  + str(step) + "_e" + str(mse) + ".png"
        plt.savefig(filename1)
        plt.close()
    else:
        path = "/home/P70066803/pythonProject/DL_paper/plots/test" + str(step)
        if not os.path.exists(path):
            os.makedirs(path)
        fig = plt.figure(figsize=(2,2))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.1, share_all=True)
        grid[0].get_yaxis().set_ticks([])
        grid[0].get_xaxis().set_ticks([])
        for ind in range(0, X_realB.shape[0], 10):
            for i, ax in enumerate(grid):
                if i % 2 == 0:
                    ax.imshow(X_realB[ind][:,:,1], cmap='gray', vmin=0, vmax=1, interpolation='none')
                elif i % 2 == 1:
                    ax.imshow(X_fakeB[ind][:,:,1], cmap='gray', vmin=0, vmax=1, interpolation='none')          
                filename1 = "/home/P70066803/pythonProject/DL_paper/plots/test" + str(step) + '/' + str(ind) + ".png"
                plt.savefig(filename1, bbox_inches='tight', pad_inches=0)
        # Release memory by deleting unnecessary variables
        del X_realA, X_realB, X_fakeB
        gc.collect()
        plt.close()


def train(d_model, g_model, gan_model, all_vali_beats, img_dim=64, stack_count=10, interval=5, aug_v=0.5, n_epochs=8, n_batch=8, use_entropy=False):
    training_data_dir = '/home/P70066803/data/images/'
    types=['B6','F4','F5','L8','M7','R9','S7','W2','W5','X2','Z3',
     'A95','G39','H19','I60','J74','K72','M58','T12','W18',
     'W93','X35','Y46','Y82','Z34','Q55','A91','H63','P12',
     'ct1','ct2','ct3','ct4','ct5','ct6','ct7','ct8','ct9','ct10','ct11']
    n_patch = d_model.output_shape[1] # Patch is always 1 in final model
    mae_scores = []

    for i in range(0, len(all_vali_beats), 7):
        cv_val_beats = all_vali_beats[i:i + 5]
        cv_train_beats =  [beat for beat in all_vali_beats if beat not in cv_val_beats]      
        train_beats = cv_train_beats.copy()
        val_beats = cv_val_beats.copy()

        print("Retrieving data...")
        train_x, train_y, train_x_entropy, train_y_entropy = get_data_specific_heartbeats_tian(train_beats, img_size=img_dim, types=types, body_data_dir=training_data_dir, stack_count=stack_count, interval=interval, use_entropy=use_entropy)
        val_x, val_y, _, _ = get_data_specific_heartbeats_tian(val_beats, img_size=img_dim, types=types, body_data_dir=training_data_dir, stack_count=stack_count, interval=interval, use_entropy=use_entropy)

        train_x_shuffle = np.arange(len(train_x))
        # Shuffle the array randomly
        np.random.shuffle(train_x_shuffle)

        # Define the data augmentation transformations you want to apply
        data_gen = ImageDataGenerator(
            zca_epsilon=1e-05,
            rotation_range=180*aug_v,     # Randomly rotate the images within ±10 degrees
            width_shift_range=aug_v, # Randomly shift the images horizontally by up to 10% of the image width
            height_shift_range=aug_v,# Randomly shift the images vertically by up to 10% of the image height
            shear_range=aug_v,       # Randomly apply shear transformations
            zoom_range=aug_v,        # Randomly zoom in/out of the images
            horizontal_flip=True,  # Randomly flip the images horizontally
            vertical_flip=True,    # Randomly flip the images vertically
        )
        # Create a generator for training data with augmentation
        train_data_gen = data_gen.flow(train_x, train_y, batch_size=n_batch, shuffle=False)
        train_x_aug = []
        train_y_aug = []
        for i in range(len(train_data_gen)):
            batch_x, batch_y = train_data_gen[i]
            train_x_aug.append(batch_x)
            train_y_aug.append(batch_y)
        # Concatenate the list of arrays into a single array
        train_x_aug = np.concatenate(train_x_aug, axis=0)
        train_y_aug = np.concatenate(train_y_aug, axis=0)
        train_x_aug = train_x_aug[train_x_shuffle].tolist()
        train_y_aug = train_y_aug[train_x_shuffle].tolist()
        train_x_entropy_aug = train_x_entropy[train_x_shuffle].tolist()

        for i in train_x_shuffle:
            train_x_aug.append(train_x[i])
            train_y_aug.append(train_y[i])
            train_x_entropy_aug.append(train_x_entropy[i])
        train_x_aug = np.array(train_x_aug)
        train_y_aug = np.array(train_y_aug)
        train_x_entropy_aug = np.array(train_x_entropy_aug)
        print("Data augmentation completed")  
            
        # calculate the number of batches per training epoch
        bat_per_epo = int(len(train_x_aug) / n_batch)
        print('batch number: ', bat_per_epo)  
        
        train_data_split = (train_x_aug, train_y_aug)
        val_data_split = (val_x, val_y)     
        # enumerate epochs
        for epoch in range(n_epochs):
            print('epoch: ', epoch)
            for batch_index in range(bat_per_epo):
                # select a batch of real samples
                [X_realA, X_realB], y_real, w = generate_real_samples(train_data_split, n_batch, n_patch, train_x_entropy_aug, name="train_")
                # generate a batch of fake samples
                X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

                # update discriminator for real sample classification
                d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
                # update discriminator for generated sample classification
                d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
                # update the generator
                g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB], sample_weight=w)

                del X_realA, X_realB, y_real, w, X_fakeB, y_fake
                gc.collect()  # Trigger the garbage collector to release memory
                # release the memory
                tf.keras.backend.clear_session()

        mae = calc_mae(g_model, val_data_split, n_batch)
        mae_scores.append(mae)
        print("MAE value is: " + str(mae) + "after each cross validation.")   

    mae_performance = np.mean(mae_scores)
    print("MAE value is: " + str(mae_performance) + "after the whole cross validation.")                   
    return mae_performance


def target_function(ssim_l1_r):
    # regularization_w,dropout, ssim_w, lr, dropout, stack_count, ssim_l1_r
    # test beats
    all_test_beats = [0,3,5,6,9,12,15,16,19,20,21,23,25,28,29,31,33,35,37,39]
    all_beats = list(range(40))
    all_trainable_vali_beats = [beat for beat in all_beats if beat not in all_test_beats]
    all_vali_beats = all_trainable_vali_beats

    regularization_w = 0.01
    ssim_w = 0.8
    lr = 0.01
    dropout = 0.2
    stack_count = 11
    # ssim_l1_r = 50


    img_dim=64
    interval=5
    aug_v=0.5
    n_epochs=15
    batch_size=16
    activation='relu'
    use_entropy = True
    image_shape = (img_dim, img_dim, stack_count)
    input_img = Input((img_dim, img_dim, stack_count), name='img')

    decay_rate = 0.96  
    aug_v = 0.5

    # define the models
    d_model = define_discriminator(image_shape, n_filters=16, activation=activation, initial_lr=lr, decay_rate=decay_rate)
    g_model = get_unet_adapted(input_img, n_filters=32, activation=activation, dropout=dropout, batchnorm=True, dropout_skip=0, stack_count=stack_count, conc_layers=[False, False, False, False])
    # define the composite model
    gan_model = define_gan(g_model, d_model, image_shape, regularization_w=regularization_w, ssim_l1_r=ssim_l1_r, ssim_w=ssim_w)

    try:
        # train model
        error = train(d_model, g_model, gan_model, all_vali_beats, img_dim=img_dim, stack_count=stack_count, interval=interval, aug_v=aug_v, n_epochs=n_epochs, n_batch=batch_size, use_entropy=use_entropy)
        acc = -error
        if np.isnan(acc):
            logging.warning("NaN value encountered in accuracy. Returning a default value.")
            return -1
        else:
            return acc
    except (ValueError, RuntimeError) as e:
        logging.error(f"Error in training simulation: {e}")
        return -1 
    



# regularization_w = [0.001, 0.01, 0.05]
# ssim_w = [0.2, 0.4, 0.6, 0.8]
# lr = [0.001, 0.01, 0.05, 0.1] 
# dropout = [0.1, 0.2, 0.3, 0.4] 
# stack_count = [7, 9, 11, 13, 15, 17, 19]
ssim_l1_r = [0, 50, 100, 150, 200]

# 定义参数值的映射函数，将索引映射回实际值
# def map_param1(index):
#     # return (regularization_w[1]-regularization_w[0])*index + regularization_w[0]
#     return regularization_w[round(index)]
# def map_param2(index):
#     # return (ssim_w[1]-ssim_w[0])*index + ssim_w[0]
#     return ssim_w[round(index)]
# def map_param3(index):
#     # return (lr[1]-lr[0])*index + lr[0]
#     return lr[round(index)]
# def map_param4(index):
#     # return (dropout[1]-dropout[0])*index + dropout[0]
#     return dropout[round(index)]
# def map_param5(index):
#     return stack_count[round(index)]
def map_param7(index):
    return ssim_l1_r[round(index)]

# 优化过程中使用映射函数
def optimized_function(param7_index):#param1_index, param4_index, param2_index, param3_index, param5_index, 
    # param1 = map_param1(param1_index)
    # param2 = map_param2(param2_index)
    # param3 = map_param3(param3_index)
    # param4 = map_param4(param4_index)
    # param5 = map_param5(param5_index)
    param7 = map_param7(param7_index)
    return target_function(param7) # param1, param4, param2, param3, param5, 



if __name__ == "__main__":
    # Limit GPU memory growth to a fraction of total GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e) 
     

    optimizer = BayesianOptimization(
        f=optimized_function,
        pbounds={'param7_index': (0, len(ssim_l1_r)-1)},
        # 'param1_index': (0, len(regularization_w)-1), 'param4_index': (0, len(dropout)-1), 'param2_index': (0, len(ssim_w)-1), 'param3_index': (0, len(lr)-1),
                #  'param4_index': (0, len(dropout)-1), 'param5_index': (0, len(stack_count)-1), 'param7_index': (0, len(ssim_l1_r)-1)
        random_state=42,
        verbose=2
    )

    optimizer.maximize(init_points=3, n_iter=8) 

    for i, res in enumerate(optimizer.res):
        print(f"Iteration {i + 1}:")
        print(f"  Parameters: {res['params']}")
        print(f"  Target Value: {res['target']}")

    print("Optimal result", optimizer.max)


    # print("Initial parameters:", optimizer.space.target_to_args(optimizer.space.target))
    # for i in range(3 + 10):
    #     optimizer.maximize(init_points=3, n_iter=1)
    #     print(f"Iteration {i + 1}:")
    #     print(f"  Parameters: {optimizer.max['params']}")
    #     print(f"  Target: {optimizer.max['target']}")
    
    # best_hyperparameters = optimizer.max['params']
    # best_performance = optimizer.max['target']
    # print("Best Hyperparameters:", best_hyperparameters)
    # print("Best Performance:", best_performance)