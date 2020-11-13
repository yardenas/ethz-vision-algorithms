import numpy as np

from scene_reconstruction.linear_triangulation import linear_triangulation


def decompose_essential_matrix(essential_matrix):
    u, _, vh = np.linalg.svd(essential_matrix)
    u3 = u[:, 2]
    w = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])
    r = u.dot(w).dot(vh)
    r = np.stack([r, u.dot(w.T).dot(vh)], axis=2)
    r[:, :, 0] *= np.sign(np.linalg.det(r[:, :, 0]))
    r[:, :, 1] *= np.sign(np.linalg.det(r[:, :, 1]))
    u3 /= (np.linalg.norm(u3) + 1e-6)
    return r, u3


def disambiguate_relative_pose(r, u3, points_1, points_2, k_1, k_2):
    best_r = r[:, :, 0]
    best_u = u3
    m_1 = k_1.dot(np.eye(3, 4))

    def count_frontal_points(r_test, t_test):
        projection_2 = k_2.dot(np.column_stack([r_test, t_test]))
        p_1 = linear_triangulation(points_1, points_2, m_1, projection_2)
        p_2 = np.column_stack([r_test, t_test]).dot(p_1.T).T
        return np.count_nonzero(p_1[:, -2] > 0.0) + np.count_nonzero(p_2[:, -2] > 0.0)

    total_points_best = count_frontal_points(r[:, :, 0], u3)
    num_conf_1 = count_frontal_points(r[:, :, 0], -u3)
    if num_conf_1 > total_points_best:
        total_points_best = num_conf_1
        best_u = -u3
    num_conf_2 = count_frontal_points(r[:, :, 1], u3)
    if num_conf_2 > total_points_best:
        total_points_best = num_conf_2
        best_r = r[:, :, 1]
        best_u = u3
    num_conf_3 = count_frontal_points(r[:, :, 1], -u3)
    if num_conf_3 > total_points_best:
        best_r = r[:, :, 1]
        best_u = -u3
    return best_r, best_u
