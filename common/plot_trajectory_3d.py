import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_positions_3d(self, xs, ys, zs):
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class PoseEstimationPlotter(object):
    def __init__(self, points_world, pause=0.005):
        self._pause = pause
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_xlabel('X')
        self._ax.set_xlim(-0.3, 0.3)
        self._ax.set_ylabel('Y')
        self._ax.set_ylim(-0.3, 0.3)
        self._ax.set_zlabel('Z')
        self._ax.set_zlim(-0.6, 0.6)
        self._ax.invert_zaxis()
        arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
        self._x = Arrow3D([0, 0], [0, 0], [0, 0], **arrow_prop_dict, color='r')
        self._y = Arrow3D([0, 0], [0, 0], [0, 0], **arrow_prop_dict, color='g')
        self._z = Arrow3D([0, 0], [0, 0], [0, 0], **arrow_prop_dict, color='b')
        self._ax.add_artist(self._x)
        self._ax.add_artist(self._y)
        self._ax.add_artist(self._z)
        self._ax.scatter(points_world[:, 0], points_world[:, 1], points_world[:, 2], c='limegreen')

    def update(self, projection_matrix):
        projection_matrix_inv = np.linalg.inv(np.concatenate([
            projection_matrix, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0))
        translation = projection_matrix_inv[:-1, -1]
        rotation = projection_matrix_inv[:3, :3]
        x_start, x_end = positions_from_translation_and_rotation(translation, rotation[:, 0])
        self._x.set_positions_3d([x_start[0], x_end[0]], [x_start[1], x_end[1]], [x_start[2], x_end[2]])
        y_start, y_end = positions_from_translation_and_rotation(translation, rotation[:, 1])
        self._y.set_positions_3d([y_start[0], y_end[0]], [y_start[1], y_end[1]], [y_start[2], y_end[2]])
        z_start, z_end = positions_from_translation_and_rotation(translation, rotation[:, 2])
        self._z.set_positions_3d([z_start[0], z_end[0]], [z_start[1], z_end[1]], [z_start[2], z_end[2]])
        plt.pause(self._pause)
        plt.draw()


def positions_from_translation_and_rotation(start, direction, length=0.1):
    end = start + length * direction
    return start, end
