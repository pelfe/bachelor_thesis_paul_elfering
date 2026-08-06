import traceback

import numpy as np
import scipy.io
from numba import cuda
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import scipy.interpolate
import multiprocessing as mp

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
    #face_indices = cuda.device_array(grid_points.shape[0], dtype=np.int32)
    face_indices = cuda.to_device(
        np.full(grid_points.shape[0], -1, dtype=np.int32)
    )
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
            print("face_indices dtype:", face_indices.dtype)
            print("face_indices shape:", face_indices.shape)
            print("current index:", face_indices[i])
            print("min face index:", face_indices.min())
            print("max face index:", face_indices.max())
            face_vertices = geometry_vertices3d[geometry_faces[face_indices[i], :]]
            mapped_points_3d[i] = np.dot(bary_coords[i], face_vertices)
        else:
            mapped_points_3d[i] = find_nearest_vertex(grid_points2D[i, :], geometry_vertices, geometry_vertices3d)

    print(f"\n[PID {os.getpid()}] Building Rbf", flush=True)

    print(f"geometry_vertices3d.shape = {geometry_vertices3d.shape}", flush=True)
    print(f"values.shape = {np.shape(values)}", flush=True)

    print(f"len(x) = {len(geometry_vertices3d[:, 0])}", flush=True)
    print(f"len(y) = {len(geometry_vertices3d[:, 1])}", flush=True)
    print(f"len(z) = {len(geometry_vertices3d[:, 2])}", flush=True)
    print(f"len(values) = {len(values)}", flush=True)

    print(f"geometry_vertices3d.dtype = {geometry_vertices3d.dtype}", flush=True)
    print(f"values.dtype = {np.asarray(values).dtype}", flush=True)
    interp = scipy.interpolate.Rbf(geometry_vertices3d[:, 0], geometry_vertices3d[:, 1], geometry_vertices3d[:, 2], values, function='thin_plate', smooth=0)
    interpolated = interp(mapped_points_3d[:,0], mapped_points_3d[:,1], mapped_points_3d[:,2])
    interpolated_z_values = interpolated.reshape([-1,1])
    # interpolated_z_values = thin_plate_spline_3d(geometry_vertices3d, values, mapped_points_3d, index)
    return mapped_points_3d, interpolated_z_values


ind = 1
path = f"/home/UMRobotics/PycharmProjects/bachelor_thesis_paul_elfering/data/pre_interpolation/healthy"
filename_beats = f'ct{ind}_beats'
filename_bullseye = f'ct{ind}_bullseye'
bullseye = scipy.io.loadmat(os.path.join(path, filename_bullseye))
beats = scipy.io.loadmat(os.path.join(path, filename_beats))
bullseye_vertices = np.array(bullseye['bullseye']['xy'][0, 0])
bullseye_faces = np.array(bullseye['bullseye']['facesxy'][0, 0] - 1)
bullseye_vertices3D = np.array(bullseye['bullseye']['XY3D'][0, 0])
map_array = np.array(bullseye['bullseye']['map'][0, 0])     
V = np.array(beats['beats1']['potsTikhonov'][0, 0])
values = V[map_array[:, 1] - 1, :]
print(f"values.shape = {np.shape(values)}", flush=True)

num_points = 64
grid_points = define_2d_grid_points(bullseye_vertices, num_points)

for index,index_values in enumerate(values.T):
    print(index_values)
    print(index)
    mapped_points_3d, interpolated_z_values = map_2d_grid_onto_3d_geometry(grid_points,
                                                                           bullseye_vertices,
                                                                           bullseye_faces,
                                                                           bullseye_vertices3D,
                                                                           index_values,
                                                                           index)
    normalized_arr = (interpolated_z_values - np.min(interpolated_z_values)) / \
                     (np.max(interpolated_z_values) - np.min(interpolated_z_values))
    image1 = normalized_arr.reshape(num_points, num_points)
    save_path = f'/home/UMRobotics/PycharmProjects/bachelor_thesis_paul_elfering/data/test/ct1/h_{index}.png'
    plt.imsave(save_path, image1, cmap='gray', origin='lower')