import numpy as np
import matplotlib.pyplot as plt

# Utility functions ------------------------------
def normalize(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Normalizes input values to a [0, 1] range based on provided bounds.
    
    Args:
        x: Input array.
        xmin: Minimum value for normalization.
        xmax: Maximum value for normalization.
        
    Returns:
        Normalized numpy array.
    """
    return (x - xmin) / (xmax - xmin)

def calculate_spread(non_dominated_psnr, non_dominated_params, ref_extremes=None):
    """
    Calculates the Spread (Δ) metric for a set of non-dominated solutions.
    This metric assesses both the distribution (uniformity) and the extent 
    (coverage) of the Pareto front.

    The calculation follows the formula by Deb et al. (NSGA-II, 2002):
    
        Δ = (d_f + d_l + Σ|d_i - d̄|) / (d_f + d_l + (N-1)·d̄)

    Parameters
    ----------
    non_dominated_psnr   : array-like 
        Values for the second objective (e.g., PSNR).
    non_dominated_params : array-like 
        Values for the first objective (e.g., Model Parameters).
    ref_extremes : list, optional
        Reference points representing the true/desired limits: 
        (psnr_min, param_min, psnr_max, param_max).
        Required to measure global coverage across different algorithms.
        If None, the metric only evaluates internal spacing uniformity.

    Returns
    -------
    float or None: 
        A value where 0.0 represents an ideal distribution (perfectly uniform 
        and reaching both extremes). Returns None if less than 3 points are found.
    """
    # Create front coordinates and sort by the first objective to compute distances
    front = np.array(list(zip(non_dominated_psnr, non_dominated_params)))
    front = front[front[:, 0].argsort()]
    N = len(front)

    # Spread calculation requires at least 3 points to assess uniformity
    if N < 3:
        return None
    
    # Calculate Euclidean distances between consecutive solutions in the front
    distances = [np.linalg.norm(front[i] - front[i + 1]) for i in range(N - 1)]
    mean_d = np.mean(distances)

    if ref_extremes is None: 
        # Without reference extremes, only internal uniformity is measured (coverage ignored)
        d_f = 0.0
        d_l = 0.0
    else:
        # Define the ideal extreme points in the objective space
        # extreme_1: (min_psnr, max_params) | extreme_2: (max_psnr, min_params)
        extreme_1 = np.array([ref_extremes[0], ref_extremes[3]])
        extreme_2 = np.array([ref_extremes[2], ref_extremes[1]])
        
        # d_f: Distance from the first obtained solution to the first true extreme
        d_f = np.linalg.norm(front[0] - extreme_1)
        # d_l: Distance from the last obtained solution to the second true extreme
        d_l = np.linalg.norm(front[-1] - extreme_2)

    # Sum of deviations from the average distance
    sum_deviation = sum(abs(d - mean_d) for d in distances)
    
    numerator   = d_f + d_l + sum_deviation
    denominator = d_f + d_l + (N - 1) * mean_d

    return numerator / denominator if denominator != 0 else 0.0

def main():
    """
    Example usage script to visualize and test the Spread metric.
    """
    x_min = 0.01
    x_max = 1 - x_min
    puntos = 10
    
    # Generate points following a geometric space to create non-uniform distribution
    x = np.geomspace(x_min, x_max, num=puntos)
    y = 1 * np.exp(-4 * x)
    x = np.array(x)
    y = np.array(y)
    
    # Define normalized reference extremes [min_y, min_x, max_y, max_x]
    ref_extremes = (0, 0, 1, 1)
    
    # Visualization of the tested front and extremes
    plt.scatter(x, y, label="Front Points")
    plt.scatter(ref_extremes[0], ref_extremes[3], color='red', label="Extreme 1")
    plt.scatter(ref_extremes[2], ref_extremes[1], color='red', label="Extreme 2")
    plt.legend()
    plt.title("Sample Pareto Front for Spread Calculation")
    plt.show()
    
    spread = calculate_spread(x, y, ref_extremes)
    print(f"Calculated Spread (Δ): {spread:.4f}")

if __name__ == "__main__":
    main()