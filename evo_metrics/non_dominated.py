import numpy as np

# Non-dominated solutions Logic ------------------------------

def get_non_dominated_solutions(
    psnr: np.ndarray,
    params: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies the non-dominated subset of solutions (the Pareto Front) 
    from a given set of points.

    This implementation assumes a MINIMIZATION problem for both objectives 
    (e.g., minimizing error/params). A point 'A' dominates point 'B' if 
    'A' is not worse than 'B' in any objective and strictly better in 
    at least one.

    Args:
        psnr (np.ndarray): Array containing values for the first objective (e.g., PSNR error).
        params (np.ndarray): Array containing values for the second objective (e.g., Parameters).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two arrays (psnr, params) 
                                       representing only the non-dominated solutions.
    """
    # Stack objectives into a single coordinate matrix
    points = np.column_stack((psnr, params))

    # Data Cleaning: Remove duplicate solutions to optimize the comparison loop
    points_unique, unique_indices = np.unique(points, axis=0, return_index=True)
    params_u = params[unique_indices]
    psnr_u   = psnr[unique_indices]

    N = points_unique.shape[0]
    # Initialize a mask to keep track of dominated solutions
    is_dominated = np.zeros(N, dtype=bool)

    # Perform a pairwise comparison to identify dominance
    # Note: For large datasets, consider using a more efficient algorithm 
    # like Kung's algorithm or Divide & Conquer.
    for i in range(N):
        for j in range(N):
            if i != j:
                # Check dominance conditions for minimization:
                # Point j dominates point i if it's better or equal in all 
                # objectives and strictly better in at least one.
                if (
                    (points_unique[j][0] <  points_unique[i][0] and
                     points_unique[j][1] <= points_unique[i][1]) or
                    (points_unique[j][0] <= points_unique[i][0] and
                     points_unique[j][1] <  points_unique[i][1])
                ):
                    is_dominated[i] = True
                    break # Optimization: if point i is dominated once, we can skip other checks

    # Filter out the dominated points to retrieve the Pareto front
    non_dominated = ~is_dominated
    return psnr_u[non_dominated], params_u[non_dominated]