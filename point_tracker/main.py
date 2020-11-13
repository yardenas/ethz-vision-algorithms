import argparse

import cv2 as cv

from point_tracker.shi_tomasi import ShiTomasiCornerDetector
from point_tracker.harris import HarrisCornerDetector
import point_tracker.utils as utils
import point_tracker.plotting as plotting

corner_patch_size = 2
sigma = 0.4
harris_kappa = 0.15
num_keypoints = 200


def part_1():
    image, image_id = next(utils.images('data', 'png', cv.IMREAD_GRAYSCALE))
    shi_tomasi_response = ShiTomasiCornerDetector(corner_patch_size).response(image)
    harris_response = HarrisCornerDetector(corner_patch_size, sigma, harris_kappa).response(image)
    plotting.plot_corner_responses(shi_tomasi_response, harris_response, image_id)


def part_2():
    image, image_id = next(utils.images('data', 'png', cv.IMREAD_GRAYSCALE))
    corners_coords = utils.select_keypoints(
        HarrisCornerDetector(corner_patch_size, sigma, harris_kappa).response(image),
        num_keypoints)
    plotting.plot_harris_corners(image, corners_coords)


def part_3():
    raise NotImplemented("This part is not implemented")


def part_4():
    raise NotImplemented("This part is not implemented")


def part_5():
    import matplotlib.pyplot as plt
    descriptor_size = 9
    match_lambda = 2
    images = [image for image, _ in utils.images('data', 'png', cv.IMREAD_GRAYSCALE)]
    frame = images[0]
    prev_corners = utils.select_keypoints(HarrisCornerDetector(corner_patch_size, sigma, harris_kappa).response(frame),
                                          num_keypoints)
    prev_descriptors = utils.generate_descriptors(prev_corners, frame, descriptor_size)
    match_plotter = plotting.MatchesPlotter()
    plt.ion()
    for i in range(1, len(images)):
        frame = images[i]
        corners = utils.select_keypoints(
            HarrisCornerDetector(corner_patch_size, sigma, harris_kappa).response(frame),
            num_keypoints)
        descriptors = utils.generate_descriptors(corners, frame, descriptor_size)
        assignments = utils.match_descriptors(descriptors, prev_descriptors, match_lambda)
        match_plotter.plot_matches(frame, corners, prev_corners, assignments)
        prev_descriptors = descriptors
        prev_corners = corners
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_1', action='store_true')
    parser.add_argument('--part_2', action='store_true')
    parser.add_argument('--part_3', action='store_true')
    parser.add_argument('--part_4', action='store_true')
    parser.add_argument('--part_5', action='store_true')
    args = parser.parse_args()

    if args.part_1:
        part_1()

    if args.part_2:
        part_2()

    if args.part_3:
        part_3()

    if args.part_4:
        part_4()

    if args.part_5:
        part_5()
