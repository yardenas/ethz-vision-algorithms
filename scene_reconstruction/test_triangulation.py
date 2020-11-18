import numpy as np

import common.tools
import scene_reconstruction.linear_triangulation


def test_triangulation():
    np.random.seed(42)
    n = 10
    p = np.random.rand(n, 4)
    p[:, 2] = p[:, 2] * 5 + 10
    p[:, 3] = 1.0
    m_1 = np.array([[500.0, 0.0, 320.0, 0.0],
                    [0.0, 500.0, 240, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    m_2 = np.array([[500.0, 0.0, 320.0, -100.0],
                    [0.0, 500.0, 240.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0]])
    p_1 = common.tools.to_homogeneous(m_1.dot(p.T).T)
    p_2 = common.tools.to_homogeneous(m_2.dot(p.T).T)
    p_est = scene_reconstruction.linear_triangulation.linear_triangulation(p_1, p_2, m_1, m_2)
    print("Error is: ", p_est - p)


if __name__ == '__main__':
    test_triangulation()
