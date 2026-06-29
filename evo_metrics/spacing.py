import numpy as np

# Spacing Metric Implementation ------------------------------

def calculate_spacing(non_dominated_psnr, non_dominated_params, normalize_by_range=True):
    """
    Calculates the Spacing (S) metric for a set of non-dominated solutions 
    based on Schott (1995).

    The Spacing metric measures the relative distance between consecutive 
    solutions in the non-dominated front. A value of 0.0 indicates that all 
    solutions are equidistantly spaced.

    Formula:
    --------
        d_i = min_{j≠i} || p_i - p_j ||      (distance to nearest neighbor)
        S   = sqrt( 1/(N-1) * Σ (d_avg - d_i)² )

    Cross-Algorithm Comparability:
    ------------------------------
    Since objectives are normalized to [0, 1] using a global reference point 
    before calling this function, the Euclidean distances are inherently 
    comparable across different algorithms.

    If `normalize_by_range=True`, the result is divided by the diagonal of 
    the front's bounding box. This makes the metric invariant to the 
    front's extent, providing a cleaner comparison when algorithms produce 
    fronts of significantly different sizes.

        S_norm = S / diagonal(front)

    Parameters:
    -----------
    non_dominated_params : array-like
        Normalized values for the first objective.
    non_dominated_psnr   : array-like
        Normalized values for the second objective.
    normalize_by_range   : bool
        If True, normalizes S by the front's bounding box diagonal.
        Recommended for comparing different algorithms. Default: True.

    Returns:
    --------
    float or None:
        The Spacing value. Returns None if there are fewer than 2 points.
        Returns 0.0 if there are exactly 2 points.
    """
    # Create the front array by pairing the objectives
    front = np.array(list(zip(non_dominated_psnr, non_dominated_params)))

    # Sort solutions by the first objective to ensure geometric consistency
    front = front[front[:, 0].argsort()]
    N = len(front)

    # Edge cases: Spacing is not meaningful for a single point
    if N < 2:
        return None
    if N == 2:
        return 0.0

    # Calculate the distance to the nearest neighbor for each solution
    d_values = []
    for i in range(N):
        # Compute Euclidean distance to all other points
        distances = [np.linalg.norm(front[i] - front[j])
                     for j in range(N) if j != i]
        # Identify the minimum distance (nearest neighbor)
        d_values.append(min(distances))

    # Calculate the mean of nearest-neighbor distances
    mean_d = np.mean(d_values)

    # Calculate the standard deviation of these distances (the Spacing value)
    spacing_value = np.sqrt(
        sum((d - mean_d) ** 2 for d in d_values) / (N - 1)
    )

    # Apply optional range normalization
    if normalize_by_range:
        # Define the bounding box of the current front
        mins = front.min(axis=0)
        maxs = front.max(axis=0)
        # Calculate the diagonal of the bounding box
        diag = np.linalg.norm(maxs - mins)
        
        # Avoid division by zero if the front is a single point (shouldn't happen here)
        if diag > 0:
            spacing_value /= diag

    return spacing_value