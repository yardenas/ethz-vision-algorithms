import numpy as np

import cv2 as cv


def select_keypoints(response, k_best, suppression_kernel_size):
    suppressed = non_maximum_suppression(response, suppression_kernel_size)
    k_best_flat_indices = np.argsort(suppressed, axis=None)[-k_best:]
    return np.unravel_index(k_best_flat_indices, response.shape)


def non_maximum_suppression(response, kernel_size):
    dilation_size = 2 * kernel_size + 1
    dilated = cv.dilate(response, cv.getStructuringElement(cv.MORPH_RECT, (dilation_size, dilation_size)))
    return np.where(dilated == response, dilated, -1e-6)
