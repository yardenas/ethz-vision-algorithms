import numpy as np

import stereo_reconstruction.utils as utils


class DisparityGenerator(object):
    def __init__(self, patch_radius, min_disp, max_disp):
        self._descriptor_size = patch_radius * 2 + 1
        self._min_disp = min_disp
        self._max_disp = max_disp

    def generate(self, left_image, right_image):
        disparities_tmp = np.zeros_like(left_image)
        for row in range(left_image.shape[0]):
            disparities_tmp[row, :] = self.compute_row_disparities(left_image, right_image, row)
        return disparities_tmp

    def compute_row_disparities(self, left_image, right_image, row):
        left_patches, right_patches = self.epipolar_line_patches(left_image, right_image, row)
        distance_matrix = np.linalg.norm(left_patches[:, None, :] - right_patches[None, :, :], axis=-1)
        correspondences = np.argmin(distance_matrix, axis=0)
        disparities = np.clip(np.arange(correspondences.shape[0], dtype=np.int) - correspondences,
                              self._min_disp,
                              self._max_disp)
        return disparities

    def get_patch(self, image, center_x, center_y):
        return utils.get_descriptor(image, (center_x, center_y), self._descriptor_size)

    def epipolar_line_patches(self, left_image, right_image, row):
        cols = left_image.shape[1]
        left_patches = np.zeros((cols, self._descriptor_size ** 2), dtype=np.float32)
        right_patches = np.zeros((cols, self._descriptor_size ** 2), dtype=np.float32)
        for col in range(cols):
            left_patches[col, :] = self.get_patch(left_image, col, row)
            right_patches[col, :] = self.get_patch(right_image, col, row)
        return left_patches, right_patches
