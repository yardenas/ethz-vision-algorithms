import os
import time
import re

import pnp_and_dlt.utils as utils
from pnp_and_dlt.direct_linear_transform import DirectLinearTransform


def main():
    calibration_mat, points_world, points_camera = utils.load_data('data')
    dlt = DirectLinearTransform(calibration_mat, points_world)
    for image, image_name in utils.images(os.path.join('data', 'images_undistorted')):
        image_id = int(re.sub("[^0-9]", "", image_name)) - 1
        reprojected_points = dlt.reproject_points(points_camera[image_id, ...])

        time.sleep(33 / 1000)


if __name__ == 'main':
    main()
