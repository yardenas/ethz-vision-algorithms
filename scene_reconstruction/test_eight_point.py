import numpy as np

import common.tools
import scene_reconstruction.eight_point


def cost_point_to_epipolar_line(f, p_1, p_2):
    all_points = np.row_stack([p_1, p_2]).T
    epi_lines = np.column_stack([f.T.dot(p_2.T), f.dot(p_1.T)])
    denom = epi_lines[0, :] ** 2 + epi_lines[1, :] ** 2
    return np.sqrt((((epi_lines * all_points).sum(axis=0) ** 2) / denom).sum() / p_1.shape[0])


def test_triangulation():
    np.random.seed(42)
    n = 40
    p = np.random.rand(n, 4)
    p[:, 2] = p[:, 2] * 5 + 10
    p[:, 3] = 1.0
    m_1 = np.array([[500.0, 0.0, 320.0, 0.0],
                    [0.0, 500.0, 240, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    m_2 = np.array([[500.0, 0.0, 320.0, -100.0],
                    [0.0, 500.0, 240.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    p1 = common.tools.to_homogeneous(m_1.dot(p.T).T)
    p2 = common.tools.to_homogeneous(m_2.dot(p.T).T)
    sigma = 1e-1
    noisy_p1 = p1 + sigma * np.random.rand(*p1.shape)
    noisy_p2 = p2 + sigma * np.random.rand(*p2.shape)
    f = scene_reconstruction.eight_point.fundamental_eight_point(noisy_p1, noisy_p2, True)
    cost_algebraic = np.linalg.norm((noisy_p2.T * f.dot(noisy_p1.T)).sum(axis=0)) / np.sqrt(n)
    cost_epipolar_dist = cost_point_to_epipolar_line(f, noisy_p1, noisy_p2)
    print("Algebraic is: ", cost_algebraic)
    print("Epipolar is: ", cost_epipolar_dist)


if __name__ == '__main__':
    test_triangulation()
