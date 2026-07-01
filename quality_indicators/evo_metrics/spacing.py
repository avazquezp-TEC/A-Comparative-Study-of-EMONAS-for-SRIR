import numpy as np


def calculate_spacing(non_dominated_psnr, non_dominated_params, normalize_by_range=True):
    """
    Calculates the Spacing (S) metric for a set of non-dominated solutions,
    based on Schott (1995).

    The Spacing metric measures the relative distance between consecutive
    solutions in the non-dominated front. A value of 0.0 indicates that all
    solutions are equidistantly spaced.

        d_i = min_{j != i} || p_i - p_j ||     (distance to nearest neighbor)
        S   = sqrt( 1/(N-1) * sum (d_avg - d_i)^2 )

    Since objectives are normalized to [0, 1] using a global reference point
    before calling this function, the Euclidean distances are inherently
    comparable across different algorithms.

    If `normalize_by_range=True`, the result is additionally divided by the
    diagonal of the front's bounding box:

        S_norm = S / diagonal(front)

    This makes the metric invariant to the front's extent, which helps when
    comparing algorithms that produce fronts of very different sizes.

    Args:
        non_dominated_psnr: Normalized values for the second objective.
        non_dominated_params: Normalized values for the first objective.
        normalize_by_range: If True, normalizes S by the front's bounding
            box diagonal. Recommended when comparing different algorithms.

    Returns:
        The Spacing value, or None if fewer than 2 points are given, or 0.0
        if exactly 2 points are given.
    """
    front = np.array(list(zip(non_dominated_psnr, non_dominated_params)))
    front = front[front[:, 0].argsort()]
    N = len(front)

    if N < 2:
        return None
    if N == 2:
        return 0.0

    d_values = []
    for i in range(N):
        distances = [np.linalg.norm(front[i] - front[j]) for j in range(N) if j != i]
        d_values.append(min(distances))

    mean_d = np.mean(d_values)
    spacing_value = np.sqrt(sum((d - mean_d) ** 2 for d in d_values) / (N - 1))

    if normalize_by_range:
        mins = front.min(axis=0)
        maxs = front.max(axis=0)
        diag = np.linalg.norm(maxs - mins)
        if diag > 0:
            spacing_value /= diag

    return spacing_value
