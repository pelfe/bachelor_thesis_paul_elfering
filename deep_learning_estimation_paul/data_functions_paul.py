import os

import numpy as np
from PIL import Image
from scipy.stats import entropy
import re


#def sort_files(files):
#    return sorted(
#        files,
#        key=lambda p: int(re.search(r'_t(\d+)', p).group(1))
#    )


def sort_files(files):
    return sorted(
        files,
        key=lambda p: int(re.search(r'(\d+)\.png$', p).group(1))
    )


def stack_images(image_paths, img_size, interval=2, stack_count=6, smooth=False):
    if not image_paths or len(image_paths) < stack_count:
        raise ValueError("Invalid image_paths. The number of image_paths must be greater than or equal to stack_count.")

    stacked_images_data = np.empty(
        ((len(image_paths) - stack_count) // interval + 1, img_size[0], img_size[1], stack_count))
    stacked_entropy_data = np.empty(((len(image_paths) - stack_count) // interval + 1, 1, 1, stack_count))

    for k, i in enumerate(range(0, len(image_paths) - stack_count, interval)):
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

            entropy_data.append(
                entropy(np.histogram(img_array.flatten(), bins=np.arange(np.min(img_array), np.max(img_array) + 2))[0]))

            img_array = img_array / 255
            images.append(img_array)

        stacked_images = np.stack(images, axis=-1)  # stack images
        # smooth
        if smooth:
            window_size = 15
            window_function = np.ones(window_size) / window_size
            padded_images = np.pad(stacked_images, ((0, 0), (0, 0), (window_size // 2, window_size // 2)), mode='edge')
            stacked_images = np.apply_along_axis(lambda x: np.convolve(x, window_function, mode='valid'), axis=2,
                                                 arr=padded_images)

        stacked_images_data[k] = stacked_images

        stacked_entropy = np.stack(entropy_data, axis=-1)  # stack entropy values
        stacked_entropy_data[k] = stacked_entropy

    # stacked_entropy_data = np.mean(stacked_entropy_data / np.sum(stacked_entropy_data) * stack_count)
    stacked_entropy_data = stacked_entropy_data / np.sum(stacked_entropy_data) * stack_count
    # stacked_entropy_data / np.max(stacked_entropy_data)
    # stacked_entropy_data = 1 - stacked_entropy_data

    return stacked_images_data, stacked_entropy_data


def paths_to_np_images_time(paths, img_size=(32, 32), interval=2, stack_count=6, use_entropy=False, smooth=False):
    stacked_images_array = []
    stacked_entropy_array = []# sort paths based on file


    stacked_images, stacked_entropy = stack_images(paths, img_size=img_size, interval=interval,
                                                   stack_count=stack_count, smooth=smooth)
    stacked_images_array.append(stacked_images)
    stacked_entropy_array.append(stacked_entropy)

    stacked_images_data = np.concatenate(stacked_images_array, axis=0)

    stacked_entropy_data = np.concatenate(stacked_entropy_array, axis=0)  # (160, 1, 1, 2)
    if stacked_entropy_data.shape[-1] > 1:
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

    return stacked_images_data
    #, stacked_entropy_data

def rescale_image(image, new_img_size):
    input_size = image.shape[0]
    output_size = new_img_size[0]
    if input_size == output_size:
        return image
    bin_size = input_size // output_size
    small_image = image.reshape((1, output_size, bin_size,
                                       output_size, bin_size)).max(4).max(2)

    return small_image

def path_to_images(paths, img_size=64):
    images = []

    for i, path in enumerate(paths):
        image = Image.open(path).convert("L")
        image_re = image.resize([img_size, img_size])

        img_array = np.asarray(image_re)
        img_array = rescale_image(img_array, [img_size, img_size])

        images.append(img_array)
    return images

def get_diretory_image_paths(dir_path):
    img_paths = []
    for img in os.listdir(dir_path):
        img_paths.append(os.path.join(dir_path, img))

    return img_paths

def images_into_npy(heart_folder, torso_folder, output_folder, subject, img_size=64):
    x_dir_path = torso_folder + "/" + subject
    y_dir_path = heart_folder + "/" + subject

    x_paths = sort_files(get_diretory_image_paths(x_dir_path))
    y_paths = sort_files(get_diretory_image_paths(y_dir_path))

    x = paths_to_np_images_time(x_paths, [img_size, img_size], 1, 11)
    y = paths_to_np_images_time(y_paths, [img_size, img_size], 1, 11)

    # 👇 ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    file_path = os.path.join(
        output_folder,
        f"size{img_size}_subject{subject}.npy"
    )

    with open(file_path, "wb") as f:
        np.save(f, x)
        np.save(f, y)

def is_data_in_order(heart_folder, torso_folder, output_folder, subject, img_size = 64):
    x_dir_path = torso_folder + "/" + subject
    y_dir_path = heart_folder + "/" + subject

    x_paths = get_diretory_image_paths(x_dir_path)
    x_paths = sort_files(x_paths)
    y_paths = get_diretory_image_paths(y_dir_path)
    y_paths = sort_files(y_paths)

    print(x_paths)
    return

def load_data(path):
    with open(path + '_x.npy', 'rb') as f:
        x = np.load(f)
    with open(path + '_y.npy', 'rb') as f:
        y = np.load(f)
    return x, y