import numpy as np


def compute_essential_matrix(points_1, points_2, k_1, k_2, normalize=True):
    f = fundamental_eight_point(points_1, points_2, normalize)
    return np.linalg.inv(k_1).T.dot(f).dot(np.linalg.inv(k_2))


def fundamental_eight_point(points_1, points_2, normalize):
    if normalize:
        t_1, t_2 = normalization_transforms(points_1, points_2)
        points_1 = t_1 * points_1.T
        points_2 = t_2 * points_2.T
    n_points = points_1.shape[0]
    q = np.zeros((n_points, 9), dtype=np.float)
    for i, (point_1, point_2) in enumerate(zip(points_1, points_2)):
        q[i, :] = np.kron(point_1, point_2)
    _, _, vh = np.linalg.svd(q)
    f = vh[-1, :].reshape((3, 4))
    u, s, v_t = np.linalg.svd(f)
    s[-1, -1] = 0.0
    f = u.dot(s).dot(v_t)
    if normalize:
        f = t_2.T.dot(f).dot(t_1)
    return f


def normalization_transforms(points_1, points_2):
    mu_1 = np.average(points_1[:, :-1], axis=0)
    s_1 = np.sqrt(2) / np.std(points_1[:, :-1], axis=0)
    mu_2 = np.average(points_2[:, :-1], axis=0)
    s_2 = np.sqrt(2) / np.std(points_2[:, :-1], axis=0)
    t_1 = np.array([s_1, 0, -s_1 * mu_1[0],
                    0, s_1, -s_1 * mu_1[1],
                    [0, 0, 1]])
    t_2 = np.array([s_2, 0, -s_2 * mu_2[0],
                    0, s_2, -s_2 * mu_2[1],
                    [0, 0, 1]])
    return t_1, t_2
