import numpy as np


def normalize(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Normalizes input values to a [0, 1] range based on the provided bounds.

    Args:
        x: Input array.
        xmin: Minimum value used for normalization.
        xmax: Maximum value used for normalization.

    Returns:
        Normalized numpy array.
    """
    return (x - xmin) / (xmax - xmin)
