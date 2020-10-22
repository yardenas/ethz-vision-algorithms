import numpy as np


def sobel_x_3():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)


def sobel_y_3():
    return sobel_x_3().T
