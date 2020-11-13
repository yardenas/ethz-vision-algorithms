import os

import numpy as np
import matplotlib.pyplot as plt
import common.load_images
import common.plot_trajectory_3d
from scene_reconstruction.eight_point import compute_essential_matrix
from scene_reconstruction.utils import decompose_essential_matrix, disambiguate_relative_pose
from scene_reconstruction.linear_triangulation import linear_triangulation


def load_points(path):
    def standardize_points(path_to_points):
        points = np.loadtxt(path_to_points).ravel()
        points = points.reshape([-1, 2])
        return np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

    return standardize_points(os.path.join(path, 'matches0001.txt')), standardize_points(
        os.path.join(path, 'matches0002.txt'))


def main():
    images_generator = common.load_images.images('data')
    image_1, _ = next(images_generator)
    image_2, _ = next(images_generator)
    k = np.array([[1379.74, 0.0, 760.35],
                  [0.0, 1382.08, 503.41],
                  [0.0, 0.0, 1.0]])
    points_1, points_2 = load_points('data')
    essential_matrix = compute_essential_matrix(points_1, points_2, k, k, True)
    rots, ts = decompose_essential_matrix(essential_matrix)
    r, t = disambiguate_relative_pose(rots, ts, points_1, points_2, k, k)
    m_1 = k.dot(np.eye(3, 4))
    m_2 = k.dot(np.column_stack([r, t]))
    points_3d = linear_triangulation(points_1, points_2, m_1, m_2)
    plotter = common.plot_trajectory_3d.PoseEstimationPlotter(points_3d,
                                                              xlim=(-1.5, 1.5),
                                                              ylim=(-1.5, 1.5),
                                                              zlim=(-1.5, 1.5))
    plotter.update(m_2)
    plt.show()


if __name__ == '__main__':
    main()
