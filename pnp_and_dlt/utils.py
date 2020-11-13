import os
import numpy as np


def load_data(path_to_data):
    k = np.loadtxt(os.path.join(path_to_data, 'K.txt'))
    points_world = np.loadtxt(os.path.join(path_to_data, 'p_W_corners.txt'), delimiter=',')
    points_camera = np.loadtxt(os.path.join(path_to_data, 'detected_corners.txt'))
    # Homogenous coordinates and dimensions. Scale from [cm] to [m]
    points_world = np.concatenate([points_world / 100.0, np.ones((points_world.shape[0], 1))], axis=1)
    points_camera = points_camera.reshape((-1, 12, 2))
    points_camera = np.concatenate([points_camera, np.ones((points_camera.shape[0], 12, 1))], axis=2)
    return k, points_world, points_camera
