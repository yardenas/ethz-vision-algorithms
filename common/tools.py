import numpy as np


def to_homogeneous(p):
    return p / np.expand_dims(p[:, -1], axis=1)
