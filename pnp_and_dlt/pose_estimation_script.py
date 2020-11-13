import os
import re
import argparse

import cv2 as cv
import numpy as np

import pnp_and_dlt.utils as utils
from pnp_and_dlt.direct_linear_transform import DirectLinearTransform
import common.plot_trajectory_3d
import common.load_images


def draw_points_on_image(image, camera_points, reprojected_points):
    for reprojected_point, camera_point in zip(reprojected_points, camera_points):
        cv.circle(image, tuple(reprojected_point[:-1].astype(np.int)), 2, (0, 0, 255))
        cv.circle(image, tuple(camera_point[:-1].astype(np.int)), 3, (0, 255, 0))
    cv.imshow("Re-projected points", image)
    cv.waitKey(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    args = parser.parse_args()
    calibration_mat, points_world, points_camera = utils.load_data(args.data_dir)
    print("Re-projecting points in path: '{}' using camera calibration matrix {}"
          .format(args.data_dir, calibration_mat))
    dlt = DirectLinearTransform(calibration_mat, points_world)
    plotter = common.plot_trajectory_3d.PoseEstimationPlotter(points_world)
    for image, image_name in common.load_images.images(os.path.join(args.data_dir, 'images_undistorted'),
                                                       dtype=np.uint8):
        image_id = int(re.sub("[^0-9]", "", image_name)) - 1
        reprojected_points, projection_matrix = dlt.reproject_points(points_camera[image_id, ...])
        draw_points_on_image(image, points_camera[image_id, ...], reprojected_points)
        plotter.update(projection_matrix)


if __name__ == '__main__':
    main()
