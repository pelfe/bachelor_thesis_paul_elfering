import keras

from show_data import *
import matplotlib.pyplot as plt
import pandas as pd
from skimage import metrics
from sklearn.metrics.pairwise import cosine_similarity

def calculate_correlations(img_dim, val_data_nr):
    (train_x, train_y), (test_x, test_y) = get_data(img_dim, datasets[val_data_nr], flattened=False)

    correlation = np.zeros(test_x.shape[0])
    for i in range(test_x.shape[0]):
        correlation_i = np.zeros(train_x.shape[0])
        for j in range(train_x.shape[0]):
            subtracted = np.absolute(test_x[i] - train_x[j])
            correlation_i[j] = np.mean(subtracted)
        correlation_i = np.sort(correlation_i)
        correlation[i] = np.mean(correlation_i[:2])
        print(str(i) + " --- " + str(test_x.shape[0]))

    return correlation


def save_all_correlations():
    img_dim = 32
    for i in range(6, 7):
        correlation = calculate_correlations(img_dim, i)
        with open('processed_data/size' + str(img_dim) + '_dataset-nr' + str(i) + 'l2.npy', 'wb') as f:
            np.save(f, correlation)


def load_correlations(dataset_nr):
    with open('processed_data/size32_dataset-nr' + str(dataset_nr) + 'l2.npy', 'rb') as f:
        correlations = np.load(f)
    return correlations


def correlation_vs_ME():
    img_dim = 32
    dataset_nr = 6

    (train_x, train_y), (test_x, test_y) = get_data(img_dim, datasets[dataset_nr], flattened=False)

    correlations = load_correlations(dataset_nr)

    path = "gan_results/dataset" + str(dataset_nr)
    model = keras.models.load_model(path)
    t_pred = model.predict(test_x)

    mse_array = np.zeros(t_pred.shape[0])
    for i in range(len(mse_array)):
        mse = np.square(np.subtract(t_pred[i], test_y[i])).mean()
        mse = round(mse, 4)
        mse_array[i] = mse

    plt.scatter(mse_array, correlations, c ="blue")
    # 但是根据函数calculate_correlations看出，这个cc是计算的x，而不是y，或许需要改进？
    plt.xlim([0, 0.10])
    plt.ylim([0, 0.12])
    plt.show()


def get_pearson_correlation_coefficient(predicted, actual):
    x_sum = np.sum(predicted)
    y_sum = np.sum(actual)
    x_y = np.sum(np.multiply(predicted, actual))
    x_x = np.sum(np.square(predicted))
    y_y = np.sum(np.square(actual))

    size = actual.size
    pce = (size * x_y - x_sum * y_sum) / (math.sqrt((size * x_x - x_sum * x_sum) * (size * y_y - y_sum * y_sum)) + 1e-10)

    return pce


def calculate_pce_all(dataset_array, model, flattened=False, img_dim = 32, dog_nr=-1):
    (train_x, train_y), (test_x, test_y) = get_data(img_dim, dataset_array, flattened=flattened, dog_nr=dog_nr)

    pce_array = calculate_pcc_data(test_x, test_y, model)
    return pce_array


def calculate_pcc_data(test_x, test_y, model, stack_count=6):
    t_pred = model.predict(test_x)
    t_pred = np.squeeze(t_pred)
    test_y = np.squeeze(test_y)

    pce_array = np.zeros(t_pred.shape[0])   
    for i in range(len(pce_array)):
        if stack_count > 1:
            pce_stack = np.zeros(stack_count)
            for j in range(stack_count):
                pce_stack[j] = get_pearson_correlation_coefficient(t_pred[i,:,:,j], test_y[i,:,:,j])
            pce_array[i] = np.mean(pce_stack)
        else:
            pce_array[i] = get_pearson_correlation_coefficient(t_pred[i,:,:], test_y[i,:,:])

    return pce_array

def calculate_ssim_data(test_x, test_y, model, stack_count=6):
    t_pred = model.predict(test_x)
    t_pred = np.squeeze(t_pred)
    test_y = np.squeeze(test_y)

    pce_array = np.zeros(t_pred.shape[0])   
    for i in range(len(pce_array)):
        if stack_count > 1:
            pce_stack = np.zeros(stack_count)
            for j in range(stack_count):
                pce_stack[j] = metrics.structural_similarity(t_pred[i,:,:,j], test_y[i,:,:,j])
            pce_array[i] = np.mean(pce_stack)
        else:
            pce_array[i] = metrics.structural_similarity(t_pred[i,:,:], test_y[i,:,:])
            # pce_array[i] = tf.image.ssim(tf.image.convert_image_dtype(tf.expand_dims(t_pred[i,:,:,j], axis=0), tf.float32),
            #                              tf.image.convert_image_dtype(tf.expand_dims(test_y[i,:,:,j], axis=0), tf.float32), max_val=1.0)

    return pce_array

def calculate_cos_data(test_x, test_y, model, stack_count=6):
    t_pred = model.predict(test_x)
    t_pred = np.squeeze(t_pred)
    test_y = np.squeeze(test_y)

    pce_array = np.zeros(t_pred.shape[0])   
    for i in range(len(pce_array)):
        if stack_count > 1:
            pce_stack = np.zeros(stack_count)
            for j in range(stack_count):
                pce_stack[j] = cosine_similarity(t_pred[i,:,:,j].reshape(1, -1), test_y[i,:,:,j].reshape(1, -1))
            pce_array[i] = np.mean(pce_stack)
        else:
            pce_array[i] = cosine_similarity(t_pred[i,:,:].reshape(1, -1), test_y[i,:,:].reshape(1, -1))
    return pce_array


def calculate_mse_dataset(x, y, model):
    t_pred = model.predict(x)
    t_pred = np.squeeze(t_pred)

    me_array = np.zeros(t_pred.shape[0])
    for i in range(len(me_array)):
        me_array[i] = np.mean((np.abs(y[i] - t_pred[i])) ** 2)

    return me_array


def calculate_mae_dataset(x, y, model):
    t_pred = model.predict(x)
    t_pred = np.squeeze(t_pred)

    me_array = np.zeros(t_pred.shape[0])
    for i in range(len(me_array)):
        me_array[i] = np.mean(np.abs(y[i] - t_pred[i]))

    return me_array


def calculate_mean_difference_all(dataset_array, model, flattened=False, dog_nr=-1):
    img_dim = 32
    (train_x, train_y), (test_x, test_y) = get_data(img_dim, dataset_array, flattened=flattened, dog_nr=dog_nr)

    return calculate_mae_dataset(test_x, test_y, model)


def visualize_boxplot(pces, names, limit=[0,1], path=''):
    plt.figure(figsize=(9, 9))
    bp = plt.boxplot(pces, showmeans=True)
    plt.ylim(limit)
    print(list(range(0, len(names))))
    plt.xticks(list(range(1, len(names) + 1)), names)
    
    if path:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    plt.show()
    return bp


def get_box_plot_data(labels, bp):
    rows_list = []

    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)


def get_model(path):
    print(path)
    return keras.models.load_model(path, custom_objects={'SSIMLoss': SSIMLoss}, compile=False)


def visualize_cc_over_time(cc_data):
    x_total = np.array([])
    for i in range(100):
        t = []
        for j in range(len(cc_data)):
            if len(cc_data[j]) > i:
                t.append(cc_data[j][i])

        if(len(t)) < 25:
            break
        x_total = np.append(x_total, np.mean(t))
    cc_p, = plt.plot(np.arange(0, len(x_total) * 1000/2048, 1000/2048), x_total)
    plt.ylim([0.49, 1.01])
    qrs_data = loadmat('/home/P70066803/data/resources/beatstian.mat')
    beats = qrs_data['beatstian']
    qrs = beats['QRSind'][0]
    x_scatter = []
    y_scatter = []
    max_x = math.ceil(len(x_total) * 1000/2048)
    print(max_x)
    print('---')
    for i in range(60, len(qrs)):
        if len(qrs[i][0]) == 0:
            continue
        x, y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", am=1, body_nrs=[i + 1],allowed_img_nrs=[qrs[i][0][0]])
        y = paths_to_np_images(y, False, (32, 32)).flatten() / 255

        start = np.average(y)

        x, y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", am=1, body_nrs=[i + 1],allowed_img_nrs=[qrs[i][0][len(qrs[i][0]) // 2]])
        y = paths_to_np_images(y, False, (32, 32)).flatten() / 255
        mid = np.average(y)
        if start > mid:
            continue
        print(i)
        for nr, j in enumerate(qrs[i][0]):
            x, y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", am=1, body_nrs=[i + 1], allowed_img_nrs=[j])
            y = paths_to_np_images(y, False, (32, 32)).flatten() / 255

            x_scatter.append(int(nr / len(qrs[i][0]) * max_x))
            y_scatter.append(np.average(y) - start)

    # plt.scatter(x_scatter, y_scatter)
    # plt.show()

    def average(x, y, max):
        avg = []
        for i in range(max):
            x_i = [j for j in range(len(x)) if x[j] == i]
            y_i = [y[j] for j in range(len(y)) if j in x_i]
            avg.append(np.average(y_i))
        return avg

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(interval, window, 'same')

    # plot(x_scatter, y_scatter, "k.")
    avg = average(x_scatter, y_scatter, max_x)

    y_av = movingaverage(avg, 3)
    y_av = (y_av - np.min(y_av)) / np.ptp(y_av)
    print(np.max(y_av))
    print(np.min(y_av))
    y_av = y_av * 0.5 + 0.5
    print(np.max(y_av))
    print(np.min(y_av))

    qrs_p, = plt.plot(list(range(0, max_x)), y_av, "r")
    plt.legend([cc_p, qrs_p], ['Mean correlation', 'Mean potential'])
    # plt.xlim(0, max_x)
    print('-------------')
    print('finished')
    plt.show()


def visualize_act_rec_scatter(activation, recovery):
    x_total = np.array([])
    for i in range(len(activation)):
        x = abs(activation[i]['predicted'].flatten() - activation[i]['actual'].flatten())
        x_total = np.concatenate((x_total, x))
    print(x_total.mean())
    plt.axvline(x_total.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.hist(x_total, bins=[*range(-150, 150, 2)])
    plt.xticks(range(-150, 150, 10))
    plt.show()


def visualize_activation_map(actual, predictedArray):
    # Get the largest qrs size
    data = loadmat('/home/P70066803/data/resources/beatstian.mat')
    beats = data['beatstian']
    qrs = beats['QRSind'][0]
    largest_qrs_window = 0
    for i in range(len(qrs)):
        largest_qrs_window = max(len(qrs[i][0]), largest_qrs_window)
    largest_qrs_window = 175
    print(actual.shape)

    fig, axes = plt.subplots(nrows=1 + len(predictedArray), ncols=1)
    im = axes[0].imshow(actual, vmin=0, vmax=largest_qrs_window)
    for i in range(len(predictedArray)):
        axes[1 + i].imshow(predictedArray[i], vmin=0, vmax=largest_qrs_window)

    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()


def visualize_violinplot(data, names, limit=(0, 1), widths=0.4):
    plt.figure(figsize=(9, 9))
    vp = plt.violinplot(data, points=100, widths=widths,
                             showmeans=False, showextrema=False, showmedians=False)
    medians = []
    percentile25 = []
    percentile75 = []
    for i in range(len(data)):
        medians.append(np.percentile(data[i], [50]))
        percentile25.append(np.percentile(data[i], [25]))
        percentile75.append(np.percentile(data[i], [75]))
    # quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    means = []
    for i in range(len(data)):
        means.append(np.mean(data[i]))

    inds = np.arange(1, len(means) + 1)
    size = 300
    medians_scatter = plt.scatter(inds, medians, marker='_', color='orange', s=size, zorder=3)
    percentile25_scatter = plt.scatter(inds, percentile25, marker='_', color='black', s=size // 2, zorder=3)
    percentile75_scatter = plt.scatter(inds, percentile75, marker='_', color='black', s=size // 2, zorder=3)
    means_scatter = plt.scatter(inds, means, marker='^', color='green', s=40, zorder=3)

    plt.xticks(list(range(1, len(names) + 1)), names)
    plt.legend([medians_scatter, percentile25_scatter, means_scatter], ['median', 'IQR', 'mean'])
    plt.ylim(limit[0], limit[1])
    plt.ylabel('Correlation Coefficient')
    plt.show()
