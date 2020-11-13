import os
import glob
import re

import numpy as np
import cv2 as cv


def images(images_path, extension='jpg', imread_flag=cv.IMREAD_UNCHANGED, dtype=np.float32):
    images_wildcard = os.path.join(images_path, '*.' + extension)
    images_names = [name for name in glob.glob(images_wildcard)]
    images_names.sort(key=lambda x: re.sub("[^0-9]", "", x))
    for image_name in images_names:
        img = cv.imread(image_name, imread_flag).astype(dtype)
        yield img, image_name
