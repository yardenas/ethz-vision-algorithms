import os
import cv2 as cv
import matplotlib.pyplot as plt

from stereo_reconstruction.disparity_generator import DisparityGenerator
import common.load_images


def main():
    (left_image, right_image), _ = \
        zip(next(common.load_images.images(os.path.join('data', 'left'), 'png', cv.IMREAD_GRAYSCALE)),
            next(common.load_images.images(os.path.join('data', 'right'), 'png', cv.IMREAD_GRAYSCALE)))
    size = (int(left_image.shape[1] / 2), int(left_image.shape[0] / 2))
    left_image = cv.resize(left_image, size, interpolation=cv.INTER_LINEAR)
    right_image = cv.resize(right_image, size, interpolation=cv.INTER_LINEAR)
    patch_radius = 5
    min_disp = 5
    max_disp = 50
    disparity = DisparityGenerator(patch_radius, min_disp, max_disp).generate(left_image, right_image)
    plt.imshow(disparity, cmap='viridis')
    plt.show()


if __name__ == '__main__':
    main()
