from multiprocessing import freeze_support

import numpy as np
import scipy.io
from numba import cuda
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import scipy.interpolate
import multiprocessing as mp

def initialize_cuda():
    # Import and initialize CUDA in each child process
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def thin_plate_spline_3d(scatter_points, scatter_values, grid_points, gpu_index):
    # Convert input arrays to TensorFlow tensors
    scatter_points = tf.convert_to_tensor(scatter_points, dtype=tf.float32)
    scatter_values = tf.convert_to_tensor(scatter_values, dtype=tf.float32)
    grid_points = tf.convert_to_tensor(grid_points, dtype=tf.float32)

    # Compute pairwise distances between scatter points and grid points
    pairwise_distances = tf.norm(tf.expand_dims(scatter_points, axis=1) - tf.expand_dims(grid_points, axis=0), axis=-1)

    # Compute the thin plate spline radial basis function (RBF)
    rbf = tf.square(pairwise_distances) * tf.math.log(pairwise_distances + 1e-6)

    # Construct the thin plate spline matrix
    ones = tf.ones(shape=(tf.shape(scatter_points)[0], 1), dtype=tf.float32)
    tps_matrix = tf.concat([ones, pairwise_distances, scatter_points], axis=-1)
    print(tps_matrix.shape)
    # Solve for thin plate spline coefficients
    tps_coefficients = tf.linalg.lstsq(tps_matrix, tf.concat([scatter_values, tf.zeros_like(scatter_values)], axis=-1))
    print(tps_coefficients.shape)

    # Compute the interpolated values for grid points
    with tf.device(f"/device:GPU:{gpu_index}"):
        interpolated_values = tf.matmul(rbf, tps_coefficients)

    # Return the interpolated values
    return interpolated_values.numpy()


def define_2d_grid_points(scatter_points, num_points):
    min_x, min_y = np.min(scatter_points, axis=0)
    max_x, max_y = np.max(scatter_points, axis=0)

    x_grid = np.linspace(min_x, max_x, num_points)
    y_grid = np.linspace(min_y, max_y, num_points)
    xv, yv = np.meshgrid(x_grid, y_grid)

    grid_points = np.column_stack((xv.ravel(), yv.ravel()))

    return grid_points


@cuda.jit
def compute_barycentric_coordinates_gpu(grid_points, vertices, faces, bary_coords, face_indices):
    i, j = cuda.grid(2)

    if i < grid_points.shape[0] and j < faces.shape[0]:
        v0 = vertices[faces[j, 0]]
        v1 = vertices[faces[j, 1]]
        v2 = vertices[faces[j, 2]]

        p = grid_points[i]

        denominator = (v0[0] - v2[0]) * (v1[1] - v2[1]) - (v0[1] - v2[1]) * (v1[0] - v2[0])
        alpha = ((p[0] - v2[0]) * (v1[1] - v2[1]) - (p[1] - v2[1]) * (v1[0] - v2[0])) / denominator
        beta = ((v0[0] - v2[0]) * (p[1] - v2[1]) - (v0[1] - v2[1]) * (p[0] - v2[0])) / denominator
        gamma = 1 - alpha - beta

        if alpha >= 0 and beta >= 0 and gamma >= 0:
            bary_coords[i, 0] = alpha
            bary_coords[i, 1] = beta
            bary_coords[i, 2] = gamma
            face_indices[i] = j


def compute_barycentric_coordinates(grid_points, vertices, faces):
    bary_coords = cuda.device_array((grid_points.shape[0], 3), dtype=np.float32)
    face_indices = cuda.device_array(grid_points.shape[0], dtype=np.int32)

    threadsperblock = (16, 16)
    blockspergrid_x = (grid_points.shape[0] + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (faces.shape[0] + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    compute_barycentric_coordinates_gpu[blockspergrid, threadsperblock](grid_points, vertices, faces, bary_coords, face_indices)

    return bary_coords.copy_to_host(), face_indices.copy_to_host()


def find_nearest_vertex(grid_point, vertices, vertices3D):
    distances = np.linalg.norm(vertices - grid_point, axis=1)
    nearest_vertex_index = np.argmin(distances)
    return vertices3D[nearest_vertex_index]


def map_2d_grid_onto_3d_geometry(grid_points2D, geometry_vertices, geometry_faces, geometry_vertices3d, values, index):
    grid_points_device = cuda.to_device(grid_points2D)
    vertices_device = cuda.to_device(geometry_vertices)
    faces_device = cuda.to_device(geometry_faces)

    bary_coords, face_indices = compute_barycentric_coordinates(grid_points_device, vertices_device, faces_device)

    mapped_points_3d = np.zeros((grid_points2D.shape[0], 3))
    for i in range(grid_points2D.shape[0]):
        if np.sum(bary_coords[i]) > 0:
            face_vertices = geometry_vertices3d[geometry_faces[face_indices[i], :]]
            mapped_points_3d[i] = np.dot(bary_coords[i], face_vertices)
        else:
            mapped_points_3d[i] = find_nearest_vertex(grid_points2D[i, :], geometry_vertices, geometry_vertices3d)

    interp = scipy.interpolate.Rbf(geometry_vertices3d[:, 0], geometry_vertices3d[:, 1], geometry_vertices3d[:, 2], values, function='thin_plate', smooth=0)
    interpolated = interp(mapped_points_3d[:,0], mapped_points_3d[:,1], mapped_points_3d[:,2])
    interpolated_z_values = interpolated.reshape([-1,1])
    # interpolated_z_values = thin_plate_spline_3d(geometry_vertices3d, values, mapped_points_3d, index)
    return mapped_points_3d, interpolated_z_values


def process_chunk(chunk_index, chunk_data):
    #bullseye_faces, bullseye_vertices, bullseye_faces, bullseye_vertices3D
    gpu_index = chunk_index % num_gpus

    # Initialize CUDA in each child process
    initialize_cuda()

    mapped_points_3d, interpolated_z_values = map_2d_grid_onto_3d_geometry(chunk_data['grid_points'],
                                                                           bullseye_vertices,
                                                                           bullseye_faces,
                                                                           bullseye_vertices3D,
                                                                           chunk_data['values'],
                                                                           gpu_index)
    normalized_arr = (interpolated_z_values - np.min(interpolated_z_values)) / \
                     (np.max(interpolated_z_values) - np.min(interpolated_z_values))
    image1 = normalized_arr.reshape(chunk_data['grid_shape'])
    save_path = f'/home/P70066803/data/ct1/h_{chunk_index}.png'
    plt.imsave(save_path, image1, cmap='gray', origin='lower')

    return {'chunk_index': chunk_index,
            'mapped_points_3d': mapped_points_3d,
            'interpolated_z_values': interpolated_z_values}


def process_chunk_wrapper(args):
    return process_chunk(*args)



# Load the data
file_list = {
    'B6',
    'ECGiCRT2',
    'ECGiCRT3',
    'ECGiCRT4',
    'ECGiCRT6',
    'ECGiCRT10',
    'ECGiCRT17',
    'L79'
}
ind = 1
path = r"C:\Users\paule\Desktop\Data Science and AI\Year 3\Bachelor Thesis\Data\test_folder"
filename_beats = f'ct1_beats'
filename_bullseye = f'ct1_bullseye'
bullseye = scipy.io.loadmat(os.path.join(path, filename_bullseye))
beats = scipy.io.loadmat(os.path.join(path, filename_beats))
global bullseye_vertices
bullseye_vertices = np.array(bullseye['bullseye']['xy'][0, 0])
global bullseye_faces
bullseye_faces = np.array(bullseye['bullseye']['facesxy'][0, 0] - 1)
global bullseye_vertices3D
bullseye_vertices3D = np.array(bullseye['bullseye']['XY3D'][0, 0])
map_array = np.array(bullseye['bullseye']['map'][0, 0])
V = np.array(beats['beats1']['potsTikhonov'][0, 0])
values = V[map_array[:, 1] - 1, :]

# Set the number of GPUs available
num_gpus = 2




# Define the 2D grid points
num_points = 100
grid_points = define_2d_grid_points(bullseye_vertices, num_points)

# Split the grid points into chunks for parallel processing
chunk_size = V.shape[1] // num_gpus
chunks = []
for i in range(num_gpus - 1):
    chunk = {'grid_points': grid_points,
             'values': values[:, i * chunk_size:(i + 1) * chunk_size]}
    chunks.append(chunk)
last_chunk = {'grid_points': grid_points,
              'values': values[:, (num_gpus - 1) * chunk_size:]}
chunks.append(last_chunk)

if __name__ == '__main__':
    freeze_support()

    # Process each chunk in parallel using multiple processes
    pool = mp.Pool(processes=num_gpus)
    results = pool.map(process_chunk_wrapper, enumerate(chunks))
    pool.close()
    pool.join()