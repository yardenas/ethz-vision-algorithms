import os
import cv2 as cv
import matplotlib.pyplot as plt

from stereo_reconstruction.disparity_generator import DisparityGenerator
import stereo_reconstruction.utils as utils


def main():
    (left_image, right_image), _ = \
        zip(next(utils.images(os.path.join('data', 'left'), 'png', cv.IMREAD_GRAYSCALE)),
            next(utils.images(os.path.join('data', 'right'), 'png', cv.IMREAD_GRAYSCALE)))
    size = (int(left_image.shape[0] / 2), int(left_image.shape[1] / 2))
    cv.resize(left_image, size, left_image, interpolation=cv.INTER_LINEAR)
    cv.resize(right_image, size, right_image, interpolation=cv.INTER_LINEAR)
    patch_radius = 5
    min_disp = 5
    max_disp = 50
    disparity = DisparityGenerator(patch_radius, min_disp, max_disp).generate(left_image, right_image)
    plt.imshow(disparity, cmap='viridis')
    plt.show()


if __name__ == '__main__':
    main()
