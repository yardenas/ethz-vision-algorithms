class DirectLinearTransform(object):
    def __init__(self, calibration_matrix):
        self._k = calibration_matrix

    def estimate_pose_dlt(self, points_camera, points_world):
        pass
