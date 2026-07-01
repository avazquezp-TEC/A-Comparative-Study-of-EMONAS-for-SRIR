import numpy as np


def calculate_spread(non_dominated_psnr, non_dominated_params, ref_extremes=None):
    """
    Calculates the Spread (Delta) metric for a set of non-dominated solutions.

    This metric assesses both the distribution (uniformity) and the extent
    (coverage) of the Pareto front, following Deb et al. (NSGA-II, 2002):

        Delta = (d_f + d_l + sum|d_i - d_avg|) / (d_f + d_l + (N-1) * d_avg)

    Args:
        non_dominated_psnr: Values for the second objective (e.g. PSNR),
            expected in normalized [0, 1] space.
        non_dominated_params: Values for the first objective (e.g. Params),
            expected in normalized [0, 1] space.
        ref_extremes: Optional [psnr_min, param_min, psnr_max, param_max]
            reference points representing the true/desired limits. Required
            to measure global coverage across different algorithms. If None,
            only internal spacing uniformity is evaluated.

    Returns:
        A float where 0.0 represents an ideal distribution (perfectly
        uniform and reaching both extremes), or None if fewer than 3 points
        are provided.
    """
    front = np.array(list(zip(non_dominated_psnr, non_dominated_params)))
    front = front[front[:, 0].argsort()]
    N = len(front)

    if N < 3:
        return None

    distances = [np.linalg.norm(front[i] - front[i + 1]) for i in range(N - 1)]
    mean_d = np.mean(distances)

    if ref_extremes is None:
        d_f = 0.0
        d_l = 0.0
    else:
        # extreme_1: (min_psnr, max_params) | extreme_2: (max_psnr, min_params)
        extreme_1 = np.array([ref_extremes[0], ref_extremes[3]])
        extreme_2 = np.array([ref_extremes[2], ref_extremes[1]])

        d_f = np.linalg.norm(front[0] - extreme_1)
        d_l = np.linalg.norm(front[-1] - extreme_2)

    sum_deviation = sum(abs(d - mean_d) for d in distances)

    numerator = d_f + d_l + sum_deviation
    denominator = d_f + d_l + (N - 1) * mean_d

    return numerator / denominator if denominator != 0 else 0.0
