import numpy as np
import cv2 as cv

import point_tracker.sobel as sobel


class HarrisCornerDetector(object):
    def __init__(self, kernel_size, kappa):
        self._kernel = np.ones((kernel_size, kernel_size), np.float32)
        self._kappa = kappa
        pass

    def response(self, image):
        i_x = cv.filter2D(image, -1, sobel.sobel_x_3(), borderType=cv.BORDER_CONSTANT)
        i_y = cv.filter2D(image, -1, sobel.sobel_y_3(), borderType=cv.BORDER_CONSTANT)
        sum_i_x_2 = cv.filter2D(i_x ** 2, -1, self._kernel, borderType=cv.BORDER_CONSTANT)
        sum_i_y_2 = cv.filter2D(i_y ** 2, -1, self._kernel, borderType=cv.BORDER_CONSTANT)
        sum_i_x_y = cv.filter2D(i_x * i_y, -1, self._kernel, borderType=cv.BORDER_CONSTANT)
        temp1 = sum_i_x_2 + sum_i_y_2
        temp2 = np.sqrt(4 * (sum_i_x_y ** 2) + (sum_i_x_2 - sum_i_y_2) ** 2)
        lambda_1 = temp1 + temp2
        lambda_2 = temp1 - temp2
        result = np.minimum(lambda_1, lambda_2)
        return np.where(result < 0.0, 0.0, result)
