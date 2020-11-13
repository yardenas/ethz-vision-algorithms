import numpy as np


def generate_descriptors(corners, frame, descriptor_size):
    descriptors = np.zeros((len(corners[0]), descriptor_size ** 2), dtype=np.float32)
    for i, (corner_x, corner_y) in enumerate(zip(corners[0], corners[1])):
        descriptors[i, ...] = get_descriptor(frame, (corner_x, corner_y), descriptor_size)
    return descriptors


def get_descriptor(image, center, descriptor_size):
    patch_x = np.clip(int(center[0] - descriptor_size / 2.0), 0, image.shape[0] - descriptor_size)
    patch_y = np.clip(int(center[1] - descriptor_size / 2.0), 0, image.shape[1] - descriptor_size)
    return image[patch_x:patch_x + descriptor_size, patch_y:patch_y + descriptor_size].ravel()


def match_descriptors(query, database, _lambda):
    distance_matrix = np.linalg.norm(query[:, None, :] - database[None, :, :], axis=-1)
    min_nonzero = np.min(distance_matrix[np.nonzero(distance_matrix)])
    filtered = np.where(distance_matrix < _lambda * min_nonzero, distance_matrix, np.inf)
    # If even the minimum is at has infinity cost, there is no match at all - put it in dummy column
    filtered = np.row_stack((filtered, np.ones(filtered.shape[1]) * 1e16))
    min_indices = np.argmin(filtered, axis=0)
    return min_indices
