import numpy as np
import cv2 as cv

import point_tracker.sobel as sobel


class HarrisCornerDetector(object):
    def __init__(self, kernel_size, sigma, kappa):
        self._kernel = cv.getGaussianKernel(kernel_size, sigma)
        self._kappa = kappa

    def response(self, image):
        i_x = cv.filter2D(image, -1, sobel.sobel_x_3(), borderType=cv.BORDER_CONSTANT)
        i_y = cv.filter2D(image, -1, sobel.sobel_y_3(), borderType=cv.BORDER_CONSTANT)
        sum_i_x_2 = cv.filter2D(i_x ** 2, -1, self._kernel, borderType=cv.BORDER_CONSTANT)
        sum_i_y_2 = cv.filter2D(i_y ** 2, -1, self._kernel, borderType=cv.BORDER_CONSTANT)
        sum_i_x_y = cv.filter2D(i_x * i_y, -1, self._kernel, borderType=cv.BORDER_CONSTANT)
        result = sum_i_x_2 * sum_i_y_2 - sum_i_x_y ** 2 - self._kappa * ((sum_i_x_2 + sum_i_y_2) ** 2)
        return result
