import numpy as np


def get_non_dominated_solutions(psnr: np.ndarray, params: np.ndarray, extra: np.ndarray | None = None):
    """
    Identifies the non-dominated subset of solutions (the Pareto front) from a
    given set of points.

    Assumes a MINIMIZATION problem for both objectives (e.g. minimizing error
    and number of params). Point A dominates point B if A is not worse than B
    in any objective and strictly better in at least one.

    Args:
        psnr: Array of values for the first objective.
        params: Array of values for the second objective.
        extra: Optional array of auxiliary data aligned with psnr/params
            (e.g. decoded architecture strings). Filtered alongside the
            objectives so it always stays aligned with the returned front.

    Returns:
        (psnr, params) restricted to the non-dominated subset, or
        (psnr, params, extra) if `extra` was provided.
    """
    points = np.column_stack((psnr, params))

    # Remove duplicate solutions before the dominance comparison
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    psnr_u = np.asarray(psnr)[unique_indices]
    params_u = np.asarray(params)[unique_indices]
    points_u = points[unique_indices]
    extra_u = np.asarray(extra)[unique_indices] if extra is not None else None

    # Vectorized pairwise dominance check: dominates[i, j] is True if
    # candidate j dominates candidate i.
    a = points_u[:, None, :]
    b = points_u[None, :, :]
    not_worse = np.all(b <= a, axis=2)
    strictly_better = np.any(b < a, axis=2)
    dominates = not_worse & strictly_better
    np.fill_diagonal(dominates, False)

    is_dominated = dominates.any(axis=1)
    keep = ~is_dominated

    if extra_u is not None:
        return psnr_u[keep], params_u[keep], extra_u[keep]
    return psnr_u[keep], params_u[keep]
