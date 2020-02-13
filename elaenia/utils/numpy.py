import numpy as np


def unpack_objects(array: np.ndarray) -> np.ndarray:
    """
    If `array` contains objects, covert them to arrays, adding one
    dimension, so that the result is a 2D numerical numpy array.
    The objects must be iterable.
    """
    if array.dtype == np.object:
        return np.array([np.array(list(obj)) for obj in array])
    else:
        return array
