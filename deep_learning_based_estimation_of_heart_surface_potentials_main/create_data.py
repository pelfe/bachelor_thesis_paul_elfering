import os
from PIL import Image
import numpy as np
import re
import math
import random
import scipy
from scipy.stats import entropy

################# THIS IS OLD EXPERIMENTS CODE. PLEASE FORGET
datasets = [[1, 6, 26, 55, 56, 57], # Sinus beats
            [3, 7, 15, 16, 17, 18, 29, 36, 37, 38, 67, 68, 69, 70, 71], # LVLat
            [2, 10, 11, 12, 13 ,14, 28, 31, 32, 33, 34, 35, 63, 64, 65, 66], # LVant
            [4, 8, 19, 20, 39, 40, 41, 42, 43, 44, 72, 73, 74, 75, 76, 77, 79], # LVpost
            [5, 9, 30, 45, 46, 47, 48, 49, 80, 81, 82, 83, 84, 85, 86, 87, 88], # RVlat
            [15, 36, 68, 10, 33, 64, 41, 73, 45, 80], # NEW
            [15, 36, 67, 10, 31, 63, 39, 72, 45, 80],# OLD
            [67, 68, 69, 70, 71, 63, 64, 65, 66, 72, 73, 74, 75, 76, 77, 79, 89, 90, 91, 80, 81, 82, 83, 84, 85, 86, 87, 88, 58, 59, 60, 61, 62, 78, 92, 93] # Train dog 123
            ]
################# end of unused code





# Directorys with images you want to use of heartbeats. Eg, data_dir_qrs contains only image pairs during the QRS complex
# Only BSPM is needed
# Dir structure :
#       path
#           body
#               body1
#                   1.jpg, 2.jpg, 3.jpg.....
#               body2
#                   1.jpg, 2.jpg, 3.jpg.....

# OPTIONAL
# data_dir_qrs = ".\\resources\\data\\qrs\\"
# data_dir_qrst = ".\\resources\\data\\qrs+t\\"
data_dir_qrs = "/home/P70066803/data/images/"
data_dir_qrst = "/home/P70066803/data/images/"

# Dir structure :
#       path
#           body
#               body1
#                   1.jpg, 2.jpg, 3.jpg.....
#               body2
#                   1.jpg, 2.jpg, 3.jpg.....
#           heart
#               heart1
#                   1.jpg, 2.jpg, 3.jpg.....
# REQUIRED : Data directory which contains ALL BSPM and HSPM images
data_dir_full = "/home/P70066803/data/images/"
# data_dir_full = ".\\resources\\data\\all\\"


def get_image_paths(input_path, output_path, skip_am=2):
    dirs = next(os.walk(input_path))[1]
    input_data = []
    output_data = []
    for dir_name in dirs:
        for dir_ in os.walk(input_path + "/" + dir_name):
            dir_path = dir_[0]
            body_nr = re.findall(r'\d+', dir_path)[0]
            img_am = len(dir_[2])
            for img_nr in range(1, img_am, skip_am):
                input_data.append(dir_path + "/" + str(img_nr) + ".png")
                output_data.append(output_path + "/heart" + body_nr + "/" + str(img_nr) + ".png")

    return input_data, output_data


def get_image_paths_all(input_path, output_path, am=1, types=[], body_nrs=[], allowed_img_nrs=np.arange(2499)):
    dirs = next(os.walk(input_path))[1]
    input_data = []
    output_data = []
    for dir_name in dirs:
        body_nr = types.index(dir_name)
        for dir_ in os.walk(input_path + "/" + dir_name):
            dir_path = dir_[0]

            if int(body_nr) in body_nrs:
                for img in os.listdir(dir_path):
                    img_nr = int(re.findall(r'\d+', img)[0])
                    if img_nr % am == 0 and img_nr in allowed_img_nrs:
                        input_data.append(data_dir_full + "body/" + types[body_nr] + "/" + img)
                        output_data.append(output_path + "/"  + types[body_nr] + "/" + img)
    return input_data, output_data

def get_image_paths_qrs(input_path, output_path, am=1, types=[], body_nrs=[], stack_count=6, all=False):
    dirs = next(os.walk(input_path))[1]
    input_data = []
    output_data = []
    num = 0
    for dir_name in dirs:
        body_nr = types.index(dir_name)
        if dir_name[:2] == "ct":
            path = "/home/P70066803/data/mat data/ct/" + dir_name + "_beats.mat"
        else:
            path = "/home/P70066803/data/mat data/ivf/" + dir_name + "_beats.mat"
        beats = scipy.io.loadmat(path)
        QRSind = np.array(beats['beats1']['QRSind'][0, 0]) - 1
        STind = np.array(beats['beats1']['STind'][0, 0]) - 1
        for dir_ in os.walk(input_path + "/" + dir_name):
            dir_path = dir_[0]          

            if int(body_nr) in body_nrs:
                num += STind[0,-1] - QRSind[0,0] + 1 - stack_count + 1
                for img in os.listdir(dir_path):
                    img_nr = int(re.findall(r'\d+', img)[0])
                    if all==False:
                        if img_nr % am == 0 and img_nr in np.arange(QRSind[0,0], STind[0,-1]+1):
                            input_data.append(data_dir_full + "body/" + types[body_nr] + "/" + img)
                            output_data.append(output_path + "/"  + types[body_nr] + "/" + img)
                    else:
                        input_data.append(data_dir_full + "body/" + types[body_nr] + "/" + img)
                        output_data.append(output_path + "/"  + types[body_nr] + "/" + img)

    return input_data, output_data


def rescale_image(image, new_img_size):
    input_size = image.shape[0]
    output_size = new_img_size[0]
    if input_size == output_size:
        return image
    bin_size = input_size // output_size
    small_image = image.reshape((1, output_size, bin_size,
                                       output_size, bin_size)).max(4).max(2)

    return small_image


def paths_to_np_images(paths, flattened=True, img_size=(32, 32)):
    if flattened:
        data = np.empty((len(paths), img_size[0] * img_size[1]))
    else:
        data = np.empty((len(paths), img_size[0], img_size[1]))
    for i, path in enumerate(paths):
        image = Image.open(path).convert("L")
        image_re = image.resize(img_size)

        img_array = np.asarray(image_re)
        # img_array = rescale_image(img_array, img_size)        
        if flattened:
            img_array = img_array.flatten()

        data[i] = img_array
    return data

def paths_to_np_images_time(paths, img_size=(32, 32), interval=2, stack_count=6, use_entropy=False, smooth=False):
    stacked_images_array = []
    stacked_entropy_array = []
    image_paths_grouped = group_paths_by_folder(paths)  # sort paths based on file

    for image_paths in image_paths_grouped.values():
        stacked_images, stacked_entropy = stack_images(image_paths, img_size=img_size, interval=interval, stack_count=stack_count, smooth=smooth)
        stacked_images_array.append(stacked_images)
        stacked_entropy_array.append(stacked_entropy)
    
    stacked_images_data = np.concatenate(stacked_images_array, axis=0)
    
    stacked_entropy_data = np.concatenate(stacked_entropy_array, axis=0) #(160, 1, 1, 2)
    if stacked_entropy_data.shape[-1] >1:
        stacked_entropy_data = stacked_entropy_data.squeeze()
        stacked_entropy_data = np.mean(stacked_entropy_data, axis=-1)
    else:
        stacked_entropy_data = stacked_entropy_data.squeeze() 

    stacked_entropy_data = stacked_entropy_data / np.max(stacked_entropy_data)
    stacked_entropy_data = np.expand_dims(stacked_entropy_data, axis=1)
    stacked_entropy_data = np.expand_dims(stacked_entropy_data, axis=1)
    stacked_entropy_data = np.repeat(stacked_entropy_data, img_size[0], axis=1)
    stacked_entropy_data = np.repeat(stacked_entropy_data, img_size[0], axis=2)
    if use_entropy:
        stacked_entropy_data = stacked_entropy_data
    else:
        stacked_entropy_data = np.ones_like(stacked_entropy_data)

    return stacked_images_data, stacked_entropy_data

    
def group_paths_by_folder(paths):
    path_groups = {}
    for path in paths:
        folder_path = os.path.dirname(path)
        if folder_path in path_groups:
            path_groups[folder_path].append(path)
        else:
            path_groups[folder_path] = [path]
        path_groups[folder_path]=order_link_array(path_groups[folder_path])
    return path_groups
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
def stack_images(image_paths, img_size, interval=2, stack_count=6, smooth=False):
    if not image_paths or len(image_paths) < stack_count:
        raise ValueError("Invalid image_paths. The number of image_paths must be greater than or equal to stack_count.")
    
    stacked_images_data = np.empty(((len(image_paths)-stack_count)//interval+1, img_size[0], img_size[1], stack_count))
    stacked_entropy_data = np.empty(((len(image_paths)-stack_count)//interval+1, 1, 1, stack_count))

    for k, i in enumerate(range(0, len(image_paths)-stack_count, interval)):
        images = []
        entropy_data = []
        for j in range(stack_count):
            image_path = image_paths[i + j]
            try:
                image = Image.open(image_path).convert("L")
                image_re = image.resize(img_size)
            except IOError:
                # Handle image loading errors
                raise IOError(f"Failed to load image: {image_path}")
            
            img_array = np.asarray(image_re)

            entropy_data.append(entropy(np.histogram(img_array.flatten(), bins=np.arange(np.min(img_array), np.max(img_array) + 2))[0]))

            img_array = img_array / 255
            images.append(img_array)

        stacked_images = np.stack(images, axis=-1)  # stack images
        # smooth
        if smooth:
            window_size=15
            window_function = np.ones(window_size) / window_size
            padded_images = np.pad(stacked_images, ((0, 0), (0, 0), (window_size // 2, window_size // 2)), mode='edge')
            stacked_images = np.apply_along_axis(lambda x: np.convolve(x, window_function, mode='valid'), axis=2, arr=padded_images)

        stacked_images_data[k] = stacked_images
        
        stacked_entropy = np.stack(entropy_data, axis=-1)  # stack entropy values
        stacked_entropy_data[k] = stacked_entropy
    
    # stacked_entropy_data = np.mean(stacked_entropy_data / np.sum(stacked_entropy_data) * stack_count)
    stacked_entropy_data = stacked_entropy_data / np.sum(stacked_entropy_data) * stack_count
    # stacked_entropy_data / np.max(stacked_entropy_data)
    # stacked_entropy_data = 1 - stacked_entropy_data

    return stacked_images_data, stacked_entropy_data



def get_data_first_part(normalize_factor=1., flattened=True, chest_img_size=(128, 128), heart_img_size=(128, 128), val_data=[1,2,3,4,5], train_data=None):
    if train_data is None:
        nrs = np.arange(start=0, stop=94, step=1)
        indices = np.argwhere(np.isin(nrs, val_data))
        train_data = np.delete(nrs, indices)
        train_data = np.delete(train_data, datasets[0])
        print(train_data)

    #train_x_spec, train_y_spec = get_image_paths_all(data_dir_full + "body", data_dir_full + "heart", 12, train_data)
    train_x, train_y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", 1, train_data)
    # train_x = np.concatenate((train_x, train_x_spec), axis=0)
    # train_y = np.concatenate((train_y, train_y_spec), axis=0)

    val_x, val_y = get_image_paths_all(data_dir_qrs + "body", data_dir_full + "heart", 1, val_data)
    print(len(train_x), len(train_y))
    print(len(val_x), len(val_y))
    train_x = paths_to_np_images(train_x, flattened, chest_img_size) / normalize_factor
    train_y = paths_to_np_images(train_y, flattened, heart_img_size) / normalize_factor
    val_x = paths_to_np_images(val_x, flattened, chest_img_size) / normalize_factor
    val_y = paths_to_np_images(val_y, flattened, heart_img_size) / normalize_factor
    return (train_x, train_y), (val_x, val_y)


def save_data(val_nr):
    val_data_str = ""
    for nr in datasets[val_nr]:
        val_data_str += "_" + str(nr)
    size = 32
    (train_x, train_y), (test_x, test_y) = get_data_first_part(normalize_factor=255.,
                                                               flattened=False,
                                                               chest_img_size=(size, size),
                                                               heart_img_size=(size, size),
                                                               val_data=datasets[val_nr])

    with open('processed_data/size'+ str(size) + '_val' + val_data_str + '.npy', 'wb') as f:
        np.save(f, train_x)
        np.save(f, train_y)

        np.save(f, test_x)
        np.save(f, test_y)


def save_data_person(person_nr, dataset_array):
    size = 32
    (train_x, train_y), (test_x, test_y) = get_data_first_part(normalize_factor=255.,
                                                               flattened=False,
                                                               chest_img_size=(size, size),
                                                               heart_img_size=(size, size),
                                                               val_data=dataset_array[1],
                                                               train_data=dataset_array[0])

    with open('processed_data/qrs-size'+ str(size) + '_person' + str(person_nr) + '.npy', 'wb') as f:
        np.save(f, train_x)
        np.save(f, train_y)

        np.save(f, test_x)
        np.save(f, test_y)


def rotate_image(img, rotation_am):
    rotated_img = np.rot90(img, k=rotation_am, axes=(0, 1))

    return rotated_img


def shift_image(img, x, y):
    shifted_img = np.roll(img, (x, y), axis=(1, 0))

    return shifted_img


def get_data(size, val_data, flattened, dog_nr=-1, augment=False):
    if dog_nr >= 0:
        path = "processed_data/qrs-size" + str(size) + '_person' + str(dog_nr)
    else:
        path = "processed_data/size" + str(size) + "_val"
        #path = "processed_data/size" + str(size) + "_val"
        for nr in val_data:
            path += "_" + str(nr)

    with open(path + '.npy', 'rb') as f:
        train_x = np.load(f)
        train_y = np.load(f)
        test_x = np.load(f)
        test_y = np.load(f)

    if flattened:
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
        train_y = train_y.reshape(train_y.shape[0], train_y.shape[1] * train_y.shape[2])
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
        test_y = test_y.reshape(test_y.shape[0], test_y.shape[1] * test_y.shape[2])

    if augment:
        rotations = 1
        shiftings = 0
        augmentations_x = np.zeros((train_x.shape[0] * (rotations + shiftings), size, size))
        augmentations_y = np.zeros((train_x.shape[0] * (rotations + shiftings), size, size))

        for i in range(len(train_x)):
            rots = [1, 2, 3] # Amount of times to rotate 90 degree
            rots.remove(random.choice(rots))

            for j in range(min(rotations, len(rots))):
                new_x = rotate_image(train_x[i], rots[j])
                new_y = rotate_image(train_y[i], rots[j])
                augmentations_x[i * (rotations + shiftings) + j] = new_x
                augmentations_y[i * (rotations + shiftings) + j] = new_y

            for j in range(shiftings):
                x_shift = random.randint(-6, 6)
                y_shift = random.randint(-6, 6)
                if x_shift == 0 and y_shift == 0:
                    x_shift += random.randint(1, 6)
                    y_shift += random.randint(1, 6)

                augmentations_x[i * (rotations + shiftings) + rotations + j] = shift_image(train_x[i], x_shift, y_shift)
                augmentations_y[i * (rotations + shiftings) + rotations + j] = shift_image(train_y[i], x_shift, y_shift)
        train_x = np.concatenate((train_x, augmentations_x), axis=0)
        train_y = np.concatenate((train_y, augmentations_y), axis=0)

    return (train_x, train_y), (test_x, test_y)


def save_result(model, history, val_data, model_name, epochs=10):
    input_shape = model.get_config()["layers"][0]["config"]["batch_input_shape"]
    input_size = input_shape[1]
    if len(input_shape) < 3:
        input_size = int(math.sqrt(input_shape[1]))

    base_path = "results/" + model_name + "/valdata_" + str(val_data)
    dir_name = base_path + "/size_" + str(input_size) + "-epoch_" + str(epochs)
    model.save(dir_name + "/model/")
    np.save(dir_name + '/train_history.npy', history.history)

from scipy.io import loadmat
if __name__ == "__main__":
    save_data(7)
    # save_data_person(4, dataset_dog_4)
    # save_data(3)
    # save_data(1)
    # save_data(2)
    # save_data(3)
    # save_data(4)

    # (train_x, train_y), (test_X, test_y) = get_data(32, datasets[1], False, augment=True)
