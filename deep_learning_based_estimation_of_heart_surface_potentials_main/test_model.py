import matplotlib.pyplot as plt
from .create_data import *
import tensorflow as tf


def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=6))


def MS_SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0, filter_size=6, power_factors=(0.0448, 0.2856, 0.3001)))


def train_model(model, flattened, chest_img_size, heart_img_size, epochs=10, dataset_nr=0, single_person=False):
    print('Starting to extract data. This could take a minute...')
    if single_person:
        (train_x, train_y), (test_X, test_y) = get_data(chest_img_size, datasets[dataset_nr], flattened)
    else:
        (train_x, train_y), (test_X, test_y) = get_data(chest_img_size, datasets[dataset_nr], flattened)
    # model.summary()
    print('Data has been extracted...')
    model.compile(optimizer='adam',
                  loss='mse'
                  #loss=SSIMLoss
                  )
                  #metrics=['mean_absolute_error'])

    print('Start training...')
    history = model.fit(train_x,
                        train_y,
                        epochs=epochs,
                        batch_size=16,
                        validation_data=(test_X, test_y))

    # print('Get final MSE of validation data')
    # test_mse = get_mse_data(model, test_X, test_y, heart_img_size)
    # train_mse = get_mse_data(model, train_x, train_y, heart_img_size)
    return history


def get_mse_data(model, x, y, heart_img_size=32):
    predicted = model.predict(x)

    predicted = predicted.reshape((predicted.shape[0], heart_img_size, heart_img_size))
    y = y.reshape((y.shape[0], heart_img_size, heart_img_size))

    mse = np.square(np.subtract(predicted, y)).mean()
    mse = round(mse, 4)
    return mse


def test_model(model, test_X, test_y, chest_img_size=32, heart_img_size=32):
    predicted = model.predict(test_X)

    predicted = predicted.reshape((predicted.shape[0], heart_img_size, heart_img_size))
    test_X = test_X.reshape((test_X.shape[0], chest_img_size, chest_img_size))
    test_y = test_y.reshape((test_y.shape[0], heart_img_size, heart_img_size))

    plot_test(test_X, test_y, predicted, chest_img_size)


def show_history(history):
    plt.plot(history.history['mean_squared_error'], label='MSE')
    plt.plot(history.history['val_mean_squared_error'], label='val_MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MeanSquaredError')
    plt.ylim([0, 0.015])
    plt.legend(loc='upper right')
    plt.show()


def plot_test(x, y, pred, img_size, am=5):
    plt.figure(figsize=(img_size, img_size))
    for j in range(1, am + 1):
        for i in range((j-1) * 135, j *135):
            plt.subplot(15, 9, i + 1 - (j-1) * 135)
            plt.xticks([])
            plt.title = "Plot " + str(j) + "/" + str(am)
            plt.yticks([])
            plt.grid(False)
            if i % 3 == 0:
                plt.imshow(x[i], cmap='gray')
            elif i % 3 == 1:
                plt.imshow(y[i], cmap='gray')
            elif i % 3 == 2:
                plt.imshow(pred[i], cmap='gray')
        plt.show()