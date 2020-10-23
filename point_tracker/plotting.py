import numpy as np
import matplotlib.pyplot as plt


def plot_corner_responses(shi_tomasi, harris, img_id):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(shi_tomasi.astype(np.float32))
    ax1.set_title("Shi-Tomasi")
    ax2.imshow(harris.astype(np.float32))
    ax2.set_title("Harris")
    fig.suptitle("Corner detection responses" + img_id)
    plt.show()


def plot_harris_corners(image, corners_coords):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.scatter(corners_coords[1], corners_coords[0], marker='+', c='r', s=4)
    plt.title("Harris k-best corners")
    plt.show()


class MatchesPlotter(object):
    def __init__(self):
        self._fig, self._ax = plt.subplots(1, 1)
        self._points = None
        self._image = None
        self._lines = None
        self._init = False

    def plot_matches(self, image, corners, prev_corners, assignments):
        # Points with no assignment are marked with the last column
        corners_to_prev_corners = assignments[assignments < assignments.shape[0]]
        prev_corners_with_match = np.nonzero(assignments < assignments.shape[0])[0]
        x_from = prev_corners[1][prev_corners_with_match]
        y_from = prev_corners[0][prev_corners_with_match]
        x_to = corners[1][corners_to_prev_corners]
        y_to = corners[0][corners_to_prev_corners]
        if not self._init:
            self._image = self._ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            self._points, = self._ax.plot(x_from, y_from, '--rx',
                                          linestyle=' ', markersize=5, markeredgewidth=2)
            self._lines = self._ax.plot([x_to, x_from], [y_to, y_from],
                                        '--bo',
                                        fillstyle='none',
                                        linewidth=1.5,
                                        linestyle='-',
                                        markersize=5,
                                        markeredgewidth=0.5)
            self._init = True
        else:
            self._image.set_array(image)
            self._points.set_data([x_from, y_from])
            for i in range(prev_corners_with_match.shape[0]):
                if i < len(self._lines):
                    self._lines[i].set_ydata([y_to[i], y_from[i]])
                    self._lines[i].set_xdata([x_to[i], x_from[i]])
                else:
                    self._lines += self._ax.plot([x_to[i], x_from[i]], [y_to[i], y_from[i]],
                                                 '--bo',
                                                 fillstyle='none',
                                                 linewidth=2.5,
                                                 linestyle='-',
                                                 markersize=3)
            while len(self._lines) > prev_corners_with_match.shape[0]:
                self._lines[0].remove()
                del self._lines[0]

        plt.pause(0.01)
