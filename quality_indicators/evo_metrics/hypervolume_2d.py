import numpy as np


def hypervolume_2d(x: np.ndarray, y: np.ndarray, ref_point=(1.05, 0.95)) -> float:
    """
    Computes the 2D Hypervolume (HV) for a set of points in a minimization problem.

    The Hypervolume metric measures the area of the objective space dominated
    by the current front and bounded by a reference point. A higher HV value
    indicates a better trade-off between objectives and better convergence
    toward the Pareto front.

    Args:
        x: Values for Objective 1 (normalized, to be minimized).
        y: Values for Objective 2 (normalized, to be minimized).
        ref_point: Bounding reference point (W) in normalized space. Default
            is (1.05, 0.95) so that all Pareto points contribute.

    Returns:
        The calculated Hypervolume area.
    """
    points = np.column_stack((x, y))

    # Discard points that do not improve on the reference point
    points = points[
        (points[:, 0] < ref_point[0]) &
        (points[:, 1] < ref_point[1])
    ]

    if points.size == 0:
        return 0.0

    # Sort by the first objective and integrate with the "staircase" method
    points = points[np.argsort(points[:, 0])]

    hv = 0.0
    prev_y = ref_point[1]

    for px, py in points:
        width = ref_point[0] - px
        height = prev_y - py

        if width > 0 and height > 0:
            hv += width * height
            prev_y = py

    return hv
