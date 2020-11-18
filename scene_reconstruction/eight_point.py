import numpy as np


def compute_essential_matrix(points_1, points_2, k_1, k_2, normalize=True):
    f = fundamental_eight_point(points_1, points_2, normalize)
    return np.matmul(k_2.T, np.matmul(f, k_1))


def fundamental_eight_point(points_1, points_2, normalize):
    if normalize:
        t_1, t_2 = normalization_transforms(points_1, points_2)
        points_1_norm = np.matmul(t_1, points_1.T).T
        points_2_norm = np.matmul(t_2, points_2.T).T
    else:
        points_1_norm = points_1
        points_2_norm = points_2
    n_points = points_1_norm.shape[0]
    q = np.zeros((n_points, 9), dtype=np.float)
    for i, (point_1, point_2) in enumerate(zip(points_1_norm, points_2_norm)):
        q[i, :] = np.kron(point_1, point_2)
    _, _, vh = np.linalg.svd(q)
    f = vh[-1, :].reshape((3, 3))
    u, s, vt = np.linalg.svd(f)
    s[-1] = 0.0
    f = np.matmul(u, np.matmul(np.diag(s), vt))
    if normalize:
        f = np.matmul(t_2.T, np.matmul(f, t_1))
    return f / f[-1, -1]


def normalization_transforms(points_1, points_2):
    mu_1 = np.mean(points_1[:, :-1], axis=0)
    s_1 = np.sqrt(2) / np.std(points_1[:, :-1])
    mu_2 = np.mean(points_2[:, :-1], axis=0)
    s_2 = np.sqrt(2) / np.std(points_2[:, :-1])
    t_1 = np.array([[s_1, 0, -s_1 * mu_1[0]],
                    [0, s_1, -s_1 * mu_1[1]],
                    [0, 0, 1]])
    t_2 = np.array([[s_2, 0, -s_2 * mu_2[0]],
                    [0, s_2, -s_2 * mu_2[1]],
                    [0, 0, 1]])
    return t_1, t_2
