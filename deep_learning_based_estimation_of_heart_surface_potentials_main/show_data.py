import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import math
from create_data import *
import cv2
from test_model import SSIMLoss
from skimage import metrics
from scipy.ndimage import median_filter


def save_error():
    path = '/home/P70066803/pythonProject/DL_paper/inception/ivf_ct' + '/epoch_errors.npy'
    content = np.load(path)

    plt.figure(figsize=(9,9))
    new_ticks = np.arange(0, len(content[0]) // 2, step=0.5)
    plt.plot(new_ticks, content[0], label="Train" , color='red')
    plt.plot(new_ticks, content[1], label="Test", color='blue', linestyle="--")

    plt.xlabel('Epoch')
    plt.ylabel('mae')
    plt.xticks(np.arange(1, len(content[0]) // 2 + 1, step=1))
    # plt.ylim([0, 0.155])
    # plt.xlim(0, 60)
    plt.legend(loc='upper right')
    plt.savefig("/home/P70066803/pythonProject/DL_paper/plots/error.png", bbox_inches='tight')
    plt.show()

def plot_correlation_coefficients(models, min_value=-0.25):
    dogs = ('Dog 1', 'Dog 2', 'Dog 3', 'Dog 4')
    all_bars = dogs * 5
    beat_types = ["SR", "LVlat", "LVant", "LVpost", "RVlat", 'Other']
    plot_icon = ['v', '.', 's', 'd', '*', 'P']
    color = ['blue', 'green', 'yellow', 'purple', 'red', 'gray']

    for i, model in enumerate(models):
        for dog in dogs:
            for i, beat_type in enumerate(beat_types):
                cc_array = models[model][dog][beat_type]
                plt.scatter([dog] * len(cc_array), cc_array, c=color[i], marker=plot_icon[i])

    plt.xticks(np.arange(len(all_bars)), all_bars)
    plt.ylim([min_value, 1])
    plt.show()


def get_pearson_correlation_coefficient(predicted, actual):
    x_sum = np.sum(predicted)
    y_sum = np.sum(actual)
    x_y = np.sum(np.multiply(predicted, actual))
    x_x = np.sum(np.square(predicted))
    y_y = np.sum(np.square(actual))

    size = actual.size

    try:
        pce = (size * x_y - x_sum * y_sum) / (math.sqrt((size * x_x - x_sum * x_sum) * (size * y_y - y_sum * y_sum))+1e-6)
        return pce
    except:
        return 0
    # pce = np.corrcoef(predicted.flatten(), actual.flatten())[0][1]


def visualize_data(historys, names, show_train=True, show_val=True, block=True):
    for i in range(len(names)):
        if show_train:
            plt.plot(historys[i]['mean_squared_error'], label=names[i] + "_MSE")
        if show_val:
            plt.plot(historys[i]['val_mean_squared_error'], label=names[i] + "_val_MSE")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 0.05])
    plt.legend(loc='upper right')
    if not block:
        plt.ion()
    plt.show()


def retrieve_gan_history(model_name):
    path = 'gan_results/' + model_name + '/results.npy'
    error_results = np.load(path)
    return error_results


def show_history_gan(model_names):
    colors = ['red', 'blue', 'green', 'yellow']
    for i, model in enumerate(model_names):
        error_results = retrieve_gan_history(model)
        if i == 1:
            model = "UNET Base"
        else:
            model = "UNET Adapted"
        plt.plot(error_results[0][:25], label="Train " + model, color=colors[i])
        plt.plot(error_results[1][:25], label="Test " + model, color=colors[i], linestyle="--")

    plt.xlabel('Epoch')
    plt.ylabel('mae')
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
    plt.ylim([0, 0.15])
    plt.xlim(0, 24)
    plt.legend(loc='upper right')
    plt.savefig('out.png', bbox_inches='tight')
    plt.show()
    

def retrieve_history(model_name, val_nr, img_size, epochs):
    path = "results/" + model_name + "/valdata_" + str(val_nr) + "/size_" + str(img_size) + "-epoch_" + str(epochs)
    return np.load(path + '/train_history.npy',allow_pickle='TRUE').item()


def retrieve_model(model_name, val_nr, img_size, epochs):
    path = "results/" + model_name + "/valdata_" + str(val_nr) + "/size_" + str(img_size) + "-epoch_" + str(epochs) + "/model"
    model = keras.models.load_model(path, custom_objects = {'SSIMLoss': SSIMLoss})

    return model


def get_visual_model(model):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    visual_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    return visual_model


def show_attention_images(visual_model, x):
    x = x.reshape((1,) + x.shape)

    feature_maps = visual_model.predict(x)
    x = feature_maps[87][0, :, :, 0] # 87
    return x


def order_link_array(arr):
    nrs = []
    for i in range(len(arr)):
        start_index = arr[i].rfind('/') + 1
        end_index = len(arr[i]) - 4
        nr = int(arr[i][start_index:end_index])
        nrs.append(nr)

    ordered = []
    order = sorted(range(len(nrs)), key=lambda k: nrs[k])
    for index in order:
        ordered.append(arr[index])
    return ordered


def get_data_specific_heartbeats(heartbeat_nrs = [], img_size=128, types=[], body_data_dir=data_dir_qrs):
    x, y = get_image_paths_all(body_data_dir + "body", data_dir_full + "heart", am=1, types=types, body_nrs=heartbeat_nrs)
    y = order_link_array(y)
    x = order_link_array(x)

    x = paths_to_np_images(x, False, (img_size, img_size)) / 255
    y = paths_to_np_images(y, False, (img_size, img_size)) / 255
    return x, y

# Tiantian
def get_data_specific_heartbeats_tian(heartbeat_nrs = [], img_size=32, types=[], body_data_dir=data_dir_qrs, stack_count=6, interval=2, use_entropy=False, smooth=False):
    x, y = get_image_paths_qrs(body_data_dir + "body", data_dir_full + "heart", am=1, types=types, body_nrs=heartbeat_nrs, stack_count=stack_count)

    x, x_entropy = paths_to_np_images_time(x, (img_size, img_size), interval=interval, stack_count=stack_count, use_entropy=use_entropy, smooth=smooth)
    y, y_entropy = paths_to_np_images_time(y, (img_size, img_size), interval=interval, stack_count=stack_count, use_entropy=use_entropy, smooth=smooth)
    x = x.squeeze()
    y = y.squeeze()
    return x, y, x_entropy, y_entropy

def show_heartbeat_ecg_tian(models, patient_nr, del_nr=4, img_size=32, ecg_amount=3, types=[], stack_count=20, interval=2, use_entropy=False, smooth=True):
    print('Retrieving image data for patient ' + str(patient_nr))
    # activation_map = calculate_activation_map_image(y_pred, ecg_y_r, ecg_y_r.shape[0])
    # qrs-s-t
    x, y = get_image_paths_qrs(data_dir_qrs + "body-qrs-s-t", data_dir_full + "heart", am=1, types=types, body_nrs=[patient_nr], stack_count=stack_count, all=True)
    x, x_entropy = paths_to_np_images_time(x, (img_size, img_size), interval=interval, stack_count=stack_count, use_entropy=use_entropy, smooth=smooth)
    y, y_entropy = paths_to_np_images_time(y, (img_size, img_size), interval=interval, stack_count=stack_count, use_entropy=use_entropy, smooth=smooth)
    # x = x.squeeze()
    # y = y.squeeze()
    print('Predicting patient ...')
    y_pred = []
    model = models
    predicteds = model.predict(x).squeeze()
    if len(x.shape)==4 and x.shape[-1]>1:
        predicted = predicteds[:,:,:,del_nr]
        y_d = y[:,:,:,del_nr]
    else:
        predicted = predicteds.squeeze()
        y_d = y.squeeze()

    mae = abs(np.subtract(predicted, y_d)).mean()
    mae = round(mae, 4)

    ecg_y_pred = predicted.transpose([1, 2, 0])
    y_pred = ecg_y_pred[::ecg_amount, ::ecg_amount, :-1]
    ecg_y = y_d.transpose([1, 2, 0])
    ecg_y_r = ecg_y[::ecg_amount, ::ecg_amount, :-1]

    # smooth
    window_size = 25
    for column in range(img_size):
        for row in range(img_size):
            padded_ecg_y = np.pad(ecg_y_r[column][row], (window_size // 2, window_size // 2), mode='edge')
            ecg_y_r[column][row] = np.convolve(padded_ecg_y, np.ones(window_size)/window_size, mode='valid')

    pcc = []
    mae = []
    for i in range(img_size // ecg_amount):
        for j in range(img_size // ecg_amount):
            pcc.append(get_pearson_correlation_coefficient(y_pred[i][j], ecg_y_r[i][j]))
            mae.append(np.subtract(y_pred[i][j], ecg_y_r[i][j]).mean())

    activation_map = calculate_activation_map(model=model, heartbeat_nr=patient_nr, types=types, del_nr=del_nr, stack_count=stack_count, img_size=img_size)
    recovery_map = calculate_recovery_map(model=model, heartbeat_nr=patient_nr, types=types, del_nr=del_nr, stack_count=stack_count, img_size=img_size)

    return {
        "actu":ecg_y_r,
        "pred":y_pred,
        "pcc":pcc,
        "mae":mae,
        "AT":activation_map,
        "RT":recovery_map
    }

def calculate_recovery_map(model, heartbeat_nr, types, del_nr=4, stack_count=20, img_size=32):
    # Get data in numpy form
    x_qrs, y_qrs = get_image_paths_all(data_dir_qrs + "body-qrs-s", data_dir_full + "heart", am=1, types=types, body_nrs=[heartbeat_nr])
    x, y = get_image_paths_all(data_dir_qrst + "body-qrs-s-t", data_dir_full + "heart", am=1, types=types, body_nrs=[heartbeat_nr])
    x = order_link_array(x)
    y = order_link_array(y)

    x = paths_to_np_images(x, False, (img_size, img_size)) / 255
    y = paths_to_np_images(y, False, (img_size, img_size)) / 255
    if stack_count > 1:
        x = x[:-(stack_count - del_nr),:,:]
        y = y[:-(stack_count - del_nr),:,:]
    st_size = len(y) - len(y_qrs)
    print(len(y), len(y_qrs))

    x_expanded = np.expand_dims(x, axis=-1)  
    x_repreated = np.repeat(x_expanded, stack_count, axis=-1)
    y_expanded = np.expand_dims(y, axis=-1)  
    y_repreated = np.repeat(y_expanded, stack_count, axis=-1)

    # Predict heartbeats
    y_repreated_pred = model.predict(x_repreated)
    y_pred = y_repreated_pred[:,:,:,0].squeeze()
    x = x_repreated[:,:,:,0].squeeze()
    y = y_repreated[:,:,:,0].squeeze()

    actual_gradient_map = np.zeros((st_size, img_size, img_size))
    predicted_gradient_map = np.zeros((st_size, img_size, img_size))

    # smooth
    window_size = 25
    padding_size = window_size-1
    y = y.transpose([1, 2, 0])
    for column in range(img_size):
        for row in range(img_size):
            padded_ecg_y = np.pad(y[column][row], (padding_size // 2, padding_size // 2), mode='edge')        
            y[column][row] = np.convolve(padded_ecg_y, np.ones(window_size)/window_size, mode='valid')
    y = y.transpose([2, 0, 1])

    # Calculate individual gradients for all pixels
    for i in range(st_size - 1):
        for column in range(img_size):
            for row in range(img_size):
                actual_gradient_map[i][column][row] = y[len(y_qrs) + i+1][column][row] - y[len(y_qrs) + i][column][row]
                predicted_gradient_map[i][column][row] = y_pred[len(y_qrs) + i+1][column][row] - y_pred[len(y_qrs) + i][column][row]

    # Get index of smallest gradient for each pixel
    actual_activation_map = (np.argmax(actual_gradient_map, axis=0) + len(y_qrs)-del_nr) 
    predicted_activation_map = (np.argmax(predicted_gradient_map, axis=0) + len(y_qrs)-del_nr) 
    actual_activation_map_normal = actual_activation_map / np.max(np.array(actual_activation_map))
    predicted_activation_map_normal = predicted_activation_map / np.max(np.array(predicted_activation_map))

    return {
        "actual":actual_activation_map,
        "predicted":predicted_activation_map,
        "cc":metrics.structural_similarity(predicted_activation_map_normal, actual_activation_map_normal)
        # get_pearson_correlation_coefficient(predicted_activation_map, actual_activation_map)
    }

def calculate_activation_map(model, heartbeat_nr, types, del_nr=4, stack_count=20, img_size=32):
    # Get data in numpy form
    x, y = get_image_paths_all(data_dir_qrs + "body-qrs", data_dir_full + "heart", am=1, types=types, body_nrs=[heartbeat_nr])
    x = order_link_array(x)
    y = order_link_array(y)

    x = paths_to_np_images(x, False, (img_size, img_size)) / 255
    y = paths_to_np_images(y, False, (img_size, img_size)) / 255

    x_expanded = np.expand_dims(x, axis=-1)  
    x_repreated = np.repeat(x_expanded, stack_count, axis=-1)
    y_expanded = np.expand_dims(y, axis=-1)  
    y_repreated = np.repeat(y_expanded, stack_count, axis=-1)

    # Predict heartbeats
    y_repreated_pred = model.predict(x_repreated)
    y_pred = y_repreated_pred[:,:,:,0].squeeze()
    x = x_repreated[:,:,:,0].squeeze()
    y = y_repreated[:,:,:,0].squeeze()
    x = x[del_nr:,:,:]
    y = y[del_nr:,:,:]

    actual_gradient_map = np.zeros((len(y), img_size, img_size))
    predicted_gradient_map = np.zeros((len(y), img_size, img_size))

    # smooth
    window_size = 25
    y = y.transpose([1, 2, 0])
    for column in range(img_size):
        for row in range(img_size):
            padded_ecg_y = np.pad(y[column][row], (window_size // 2, window_size // 2), mode='edge')
            y[column][row] = np.convolve(padded_ecg_y, np.ones(window_size)/window_size, mode='valid')
    y = y.transpose([2, 0, 1])

    # # (temporal) Calculate individual gradients for all pixels 
    # for i in range(len(y) - 1):
    #     for column in range(img_size):
    #         for row in range(img_size):
    #             actual_gradient_map[i][column][row] = y[i+1][column][row] - y[i][column][row]
    #             predicted_gradient_map[i][column][row] = y_pred[i+1][column][row] - y_pred[i][column][row]
    # # Get index of smallest gradient for each pixel
    # actual_activation_map = np.argmin(actual_gradient_map, axis=0) # 1000/2048 is frequency
    # predicted_activation_map = np.argmin(predicted_gradient_map, axis=0) # 1000/2048 is frequency
    # actual_activation_map_normal = actual_activation_map / np.max(np.array(actual_activation_map))
    # predicted_activation_map_normal = predicted_activation_map / np.max(np.array(predicted_activation_map))


    # (spatial-temporal) 
    dVdt = np.diff(y, axis=0)
    dVdx, dVdy = np.gradient(y, axis=(1,2))
    normSpatialGradient = np.sqrt(dVdx**2 + dVdy**2)
    if dVdt.shape[0] > normSpatialGradient.shape[0]:
        dVdt = dVdt[:-1, :, :]
    elif dVdt.shape[0] < normSpatialGradient.shape[0]:
        normSpatialGradient = normSpatialGradient[1:, :, :]
    weightedMinimum = dVdt * normSpatialGradient
    actual_activation_map = np.argmin(weightedMinimum, axis=0)

    dVdt = np.diff(y_pred, axis=0)
    dVdx, dVdy = np.gradient(y_pred, axis=(1,2))
    normSpatialGradient = np.sqrt(dVdx**2 + dVdy**2)
    if dVdt.shape[0] > normSpatialGradient.shape[0]:
        dVdt = dVdt[:-1, :, :]
    elif dVdt.shape[0] < normSpatialGradient.shape[0]:
        normSpatialGradient = normSpatialGradient[1:, :, :]
    weightedMinimum = dVdt * normSpatialGradient
    predicted_activation_map = np.argmin(weightedMinimum, axis=0)

    # # smooth
    # actual_activation_map = median_filter(actual_activation_map, size=3)
    # predicted_activation_map = median_filter(predicted_activation_map, size=3)

    # Get index of smallest gradient for each pixel
    actual_activation_map_normal = actual_activation_map / np.max(np.array(actual_activation_map))
    predicted_activation_map_normal = predicted_activation_map / np.max(np.array(predicted_activation_map))
    

    return {
        "actual":actual_activation_map,
        "predicted":predicted_activation_map,
        "cc":metrics.structural_similarity(predicted_activation_map_normal, actual_activation_map_normal)
        # get_pearson_correlation_coefficient(predicted_activation_map, actual_activation_map)
    }




def show_heartbeat_ecg(models, heartbeat_nr, img_size=128, ecg_amount=3, types=[], allowed_image_nrs=np.arange(0, 2500)):
    print('Retrieving image data for heartbeat ' + str(heartbeat_nr))
    x, y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", am=1, types=types, body_nrs=[heartbeat_nr], allowed_img_nrs=allowed_image_nrs)
    y = order_link_array(y)
    x = order_link_array(x)

    x = paths_to_np_images(x, False, (img_size, img_size)) / 255
    y = paths_to_np_images(y, False, (img_size, img_size)) / 255

    print('Predicting heartbeats...')
    y_pred = []
    for model in models:
        predicted = model.predict(x).squeeze()
        mae = abs(np.subtract(predicted, y)).mean()
        mae = round(mae, 4)
        print(mae)

        ecg_y_pred = predicted.transpose([1, 2, 0]) * 200 - 100
        ecg_y_pred_r = cv2.resize(ecg_y_pred, dsize=(img_size // ecg_amount, img_size // ecg_amount),
                                  interpolation=cv2.INTER_CUBIC)
        y_pred.append(ecg_y_pred_r)

    ecg_y = y.transpose([1, 2, 0]) * 200 - 100
    ecg_y_r = cv2.resize(ecg_y, dsize=(img_size // ecg_amount, img_size // ecg_amount), interpolation=cv2.INTER_CUBIC)

    activation_map = calculate_activation_map_image(y_pred[0], ecg_y_r, ecg_y_r.shape[0])
    fig, axs = plt.subplots(img_size // ecg_amount, img_size // ecg_amount)
    for i in range(img_size // ecg_amount):
        for j in range(img_size // ecg_amount):
            axs[i, j].plot(ecg_y_r[i][j], color='blue')
            axs[i, j].axvline(activation_map['actual'][i][j], linestyle="--", color='blue')
            for z in range(len(models)):
                if z == 0:
                    axs[i, j].plot(y_pred[z][i][j], color='red')
                    axs[i, j].axvline(activation_map['predicted'][i][j], linestyle="--", color='red')
                else:
                    axs[i, j].plot(y_pred[z][i][j])

            axs[i, j].get_xaxis().set_ticks([])
            axs[i, j].get_yaxis().set_ticks([])

    plt.setp(axs, ylim=(-100, 100))
    plt.show()


def show_model_predictions(model, x, y, t_x, t_y, img_size=32):
    pred = model.predict(x)
    t_pred = model.predict(t_x)
    pred = pred.reshape((pred.shape[0], pred.shape[1], pred.shape[2]))
    t_pred = t_pred.reshape((t_pred.shape[0], t_pred.shape[1], t_pred.shape[2]))

    mse = np.square(np.subtract(t_pred, t_y)).mean()
    mse = round(mse, 4)
    print("mse = " + str(mse))
    fig = plt.figure(figsize=(img_size, img_size))

    colums = 7
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(15, colums),  # creates 2x2 grid of axes
                     axes_pad=0.03,  # pad between axes in inch.
                     share_all=True
                     )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    visual_model = get_visual_model(model)

    for i, ax in enumerate(grid):
        # Iterating over the grid returns the Axes.
        if i % colums == 0:
            ax.imshow(x[i // colums], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        elif i % colums == 1:
            ax.imshow(y[i // colums], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        elif i % colums == 2:
            ax.imshow(pred[i // colums], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        elif i % colums == 4:
            ax.imshow(t_x[i // colums], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        elif i % colums == 5:
            ax.imshow(t_y[i // colums], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        elif i % colums == 6:
            ax.imshow(t_pred[i // colums], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
            print(str(i // colums) + " : " + str(get_pearson_correlation_coefficient(t_pred[i // colums], t_y[i // colums])))
    plt.show()


def show_intermetdiate_images(model, x):
    # output intermediate representations for all layers except the first layer.
    layer_outputs = [layer.output for layer in model.layers[1:]]
    visual_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

    x = x.reshape((1,) + x.shape)

    feature_maps = visual_model.predict(x)

    # Collect the names of each layer except the first one for plotting
    layer_names = [layer.name for layer in model.layers[1:]]
    model.summary()
    # list_to_show = ["softmax_7"]
    list_to_show = ["conv2d_56", "activation_53",
                    'conv2d_78', 'conv2d_80', 'conv2d_82', 'conv2d_83']
    # Plotting intermediate representation images layer by layer
    for layer_name, feature_map in zip(layer_names, feature_maps):
        if layer_name in list_to_show: # skip fully connected layers

            n_features = feature_map.shape[-1]

            size = feature_map.shape[1]
            # Tile our feature images in matrix `display_grid
            display_grid = np.zeros((size, size * n_features))
            # Fill out the matrix by looping over all the feature images of your image
            for i in range(n_features):
                x = feature_map[0, :, :, i]

                display_grid[:, i * size : (i + 1) * size] = x

            scale = 20. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            if layer_name == "conv2d_116":
                plt.imshow(display_grid, cmap='gray', vmin = 0, vmax = 1,interpolation='none')
            else:
                plt.imshow(display_grid, aspect='auto', cmap='viridis')

    plt.show()


def get_model_name(prename = "unet", filters=16, dropout=0):
    adaptations = "n-data-"
    return adaptations + "_" + prename + "_f" + str(filters) + "_d" + str(dropout)


def show_multiple_model_examples_heartbeat(models, img_size, types, heartbeat, min_frame=0):
    #x, y = get_image_paths_all(data_dir_full + "body", data_dir_full + "heart", am=6, body_nrs=[heartbeat])
    x, y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", am=1, types=types, body_nrs=[heartbeat])
    y = order_link_array(y)
    x = order_link_array(x)

    x = paths_to_np_images(x, False, (img_size, img_size)) / 255
    y = paths_to_np_images(y, False, (img_size, img_size)) / 255
    y_pred = []
    for model in models:
        print('Predicting heartbeats...')
        y_pred.append(model.predict(x).squeeze())

    fig = plt.figure(figsize=(img_size, img_size))
    colums = 2 + len(models)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(18, colums),  # creates 2x2 grid of axes
                     axes_pad=0.03,  # pad between axes in inch.
                     share_all=True
                     )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    for i, ax in enumerate(grid):
        if i % colums == 0:
            ax.imshow(x[i // colums + min_frame], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        elif i % colums == 1:
            ax.imshow(y[i // colums + min_frame], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
        else:
            ax.imshow(y_pred[i % colums - 2][i // colums + min_frame], cmap='gray', vmin = 0, vmax = 1,interpolation='none')
    plt.show()


def calculate_activation_map_image(predicted, actual, img_size=32):
    pic_am = actual.shape[2]
    actual_gradient_map = np.zeros((pic_am, img_size, img_size))
    predicted_gradient_map = np.zeros((pic_am, img_size, img_size))
    # Calculate individual gradients for all pixels
    for i in range(pic_am - 2):
        for column in range(img_size):
            for row in range(img_size):
                actual_gradient_map[i][column][row] = actual[column][row][i] - actual[column][row][i + 1]
                predicted_gradient_map[i][column][row] = predicted[column][row][i] - predicted[column][row][i + 1]

    # Get index of smallest gradient for each pixel
    actual_activation_map = np.argmax(actual_gradient_map, axis=0)
    predicted_activation_map = np.argmax(predicted_gradient_map, axis=0)

    return {
        "actual":actual_activation_map,
        "predicted":predicted_activation_map
    }


from matplotlib import cm, colors
def image_to_color(bspm, hspm):
    fig, axes = plt.subplots(nrows=1 * len(hspm), ncols=2)
    for i in range(len(bspm)):
        axes[i][0].imshow(bspm[i], vmin=0, vmax=255, cmap="jet_r")
        axes[i, 0].axis('off')
    for i in range(len(hspm)):
        axes[i][1].imshow(hspm[i], vmin=0, vmax=255, cmap="jet_r")
        axes[i, 1].axis('off')

    axes[0, 0].set_title("BSPM")
    axes[0, 1].set_title("HSPM")
    cmap = cm.get_cmap("jet_r")
    norm = colors.Normalize(-100, 100)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.ravel().tolist())

    plt.show()
    fig.savefig('out.png', bbox_inches='tight', pad_inches=0.2)


if __name__ == "__main__":
    q = ["SR", "LVlat", "LVant", "LVpost", "RVlat", "Dog1", "Dog2", "Dog3", "Dog4"]
    for i in range(3, 9):
        test_class_name = q[i]
        show_history_gan(["qrs-gan8-" + test_class_name])
    # plot_correlation_coefficients([1, 2])
    show_history_gan(["qrs-gan5-LVpost"])
    show_history_gan(["qrs-15-epochs-unet", "qrs-15-epochs-unet-f48"])
    show_history_gan(["qrs-15-epochs-unet", "qrst-15-epochs-unet"])
    show_history_gan(['qrs-30-epochs', "qrs-15-epochs-unet"])
    #show_history_gan(['qrs-lr_0002-b1_0.7', 'qrs-lr_0008-b1_0.7'])
    # dataset_nr = 3
    # img_nr = 35
    # body_nr = 19
    # image_bspm = Image.open('E:\\thesis_data\\all\\body\\body' + str(body_nr) + '\\' + str(img_nr) + '.jpg')
    # bspm_1 = np.asarray(image_bspm)
    # image_bspm = Image.open('E:\\thesis_data\\all\\body\\body' + str(67) + '\\' + str(51) + '.jpg')
    # bspm_2 = np.asarray(image_bspm)
    #
    # image_hspm = Image.open('E:\\thesis_data\\all\\heart\\heart' + str(body_nr) + '\\' + str(img_nr) + '.jpg')
    # hspm_1 = np.asarray(image_hspm)
    # image_hspm = Image.open('E:\\thesis_data\\all\\heart\\heart' + str(67) + '\\' + str(51) + '.jpg')
    # hspm_2 = np.asarray(image_hspm)
    # image_to_color([bspm_1, bspm_2], [hspm_1, hspm_2])
    # model_gan_base = keras.models.load_model(("gan_results\\batch16\\dataset" + str(dataset_nr)))
    # activation_map = calculate_activation_map(model_gan_base, 2)

    # model_unet_a = retrieve_model(model_name=get_model_name("unet-v3-allskip-conv-alldata", filters=16, dropout=0),
    #                               val_nr=dataset_nr,
    #                               img_size=32,
    #                               epochs=10)
    # model_unet_af = retrieve_model(model_name=get_model_name("unet-allskip3", filters=32, dropout=0),
    #                               val_nr=dataset_nr,
    #                               img_size=32,
    #                               epochs=10)
    #
    # model_gan2 = keras.models.load_model("gan_results4\\dataset" + str(dataset_nr))
    # model_gan_a = keras.models.load_model("gan_results-new\\dataset" + str(dataset_nr))
    # model_gan2.summary()
    # q = [4, 8, 19, 20, 39, 40, 41, 42, 43, 44, 72, 73, 74, 75, 76, 77, 79]
    # # show_heartbeat_ecg(model_unet_af, 39, 32, 4)
    # # show_model_examples_heartbeat(model_unet_af, 32, 39)
    # avg = 0
    # for heartbeat in q:
    #     # show_heartbeat_ecg(model_gan2, heartbeat, 32, 4)
    #     # show_heartbeat_ecg(model_gan_a, heartbeat, 32, 4)
    #     print('----')
    #     # show_model_examples_heartbeat(model_unet_a, 32, heartbeat)
    #     show_multiple_model_examples_heartbeat([model_gan2, model_gan_a], 32, heartbeat)
