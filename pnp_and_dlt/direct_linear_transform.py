import numpy as np


class DirectLinearTransform(object):
    def __init__(self, calibration_matrix, points_world):
        self._k_inv = np.linalg.inv(np.array(calibration_matrix))
        self._k = np.array(calibration_matrix)
        self._points_world = points_world

    def estimate_pose_dlt(self, points_camera):
        points_camera_calibrated = self._k_inv.dot(points_camera.T).T
        assert points_camera_calibrated.shape == points_camera.shape, "Fucked up points normalization."
        assert points_camera_calibrated.shape == self._points_world.shape, "Input shape mismatch"
        x_mul = np.hstack([self._points_world,
                           np.zeros_like(self._points_world),
                           -(self._points_world.T * points_camera_calibrated[:, 0]).T])

        y_mul = np.hstack([np.zeros_like(self._points_world),
                           self._points_world,
                           -(self._points_world.T * points_camera_calibrated[:, 1]).T])
        q = np.empty((x_mul.shape[0] + y_mul.shape[0], x_mul.shape[1]))
        q[::2, :] = x_mul
        q[1::2, :] = y_mul
        _, _, vh = np.linalg.svd(q)
        # Converting to a [3 x 4] matrix, and enforcing det(R) = 1 as described statement.pdf
        alpha_m = vh[-1, :].reshape((3, 4)) * np.sign(vh[-1, -1])
        r = alpha_m[:3, :3]
        u, _, vh_r = np.linalg.svd(r)
        r_tilde = u * vh_r
        alpha = np.linalg.norm(r_tilde) / np.linalg.norm(r)
        return np.hstack([r_tilde, np.expand_dims(alpha_m[:-1, -1], axis=1) / alpha])

    def reproject_points(self, points_camera):
        projection_matrix = self.estimate_pose_dlt(points_camera)
        return (self._k.dot(projection_matrix)).dot(self._points_world.T).T
