import numpy as np

import common.tools


def linear_triangulation(points_1, points_2, m_1, m_2):
    n_points = points_1.shape[0]
    p = np.zeros((n_points, 4), dtype=np.float)
    for i, (point_1, point_2) in enumerate(zip(points_1, points_2)):
        a = np.matmul(cross_matrix(point_1), m_1)
        a = np.concatenate((a, np.matmul(cross_matrix(point_2), m_2)), axis=0)
        _, _, vh = np.linalg.svd(a)
        p[i, :] = vh[-1, :]
    return common.tools.to_homogeneous(p)


def cross_matrix(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
