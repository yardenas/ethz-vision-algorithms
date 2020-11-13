import numpy as np


def linear_triangulation(points_1, points_2, m_1, m_2):
    n_points = points_1.shape[0]
    p = np.zeros((n_points, 4), dtype=np.float)
    for i, (point_1, point_2) in enumerate(zip(points_1, points_2)):
        a = np.matmul(cross_matrix(point_1), m_1)
        np.concatenate((a, np.matmul(cross_matrix(point_2), m_2)), axis=0)
        _, _, vh = np.linalg.svd(a)
        p[i, :] = vh[-1, :]
    p = p / np.expand_dims(p[:, -1], axis=1)
    return p[:, :-1]


def cross_matrix(x):
    return np.array([[x[0], -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])
