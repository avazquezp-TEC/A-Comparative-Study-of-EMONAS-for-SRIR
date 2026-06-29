import numpy as np

def hypervolume_2d(x: np.ndarray, y: np.ndarray,
                   ref_point=(1.05, 0.95)) -> float:
    """
    Computes the 2D Hypervolume (HV) for a set of points in a minimization problem.

    The Hypervolume metric measures the area of the objective space that is 
    dominated by the current non-dominated front and bounded by a reference point.
    A higher HV value indicates a better trade-off between objectives and 
    better convergence toward the Pareto front.

    Args:
        x (np.ndarray): Array of values for Objective 1 (normalized, to be minimized).
        y (np.ndarray): Array of values for Objective 2 (normalized, to be minimized).
        ref_point (tuple): The bounding reference point (W) in the normalized space. 
                           Default is (1.05, 0.95) to ensure all Pareto points contribute.

    Returns:
        float: The calculated Hypervolume area.
    """
    # Combine objective arrays into a coordinate matrix
    points = np.column_stack((x, y))
    
    # Filter out points that are worse than the reference point (non-contributing)
    points = points[
        (points[:, 0] < ref_point[0]) &
        (points[:, 1] < ref_point[1])
    ]

    # If no points are within the reference bounds, HV is zero
    if points.size == 0:
        return 0.0

    # Sort points by the first objective (x) to use the "staircase" integration method
    points = points[np.argsort(points[:, 0])]

    hv = 0.0
    prev_y = ref_point[1]

    # Calculate the area by iterating through the sorted non-dominated points
    # This sums the rectangular areas defined by each point and the reference point
    for px, py in points:
        width = ref_point[0] - px
        height = prev_y - py
        
        # Only add positive areas (ensure the point actually contributes)
        if width > 0 and height > 0:
            hv += width * height
            # Update previous Y to avoid overlapping areas (staircase step)
            prev_y = py

    return hv