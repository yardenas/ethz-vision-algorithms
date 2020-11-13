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

