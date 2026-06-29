import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from evo_metrics.file_handl import load_seeds
from evo_metrics.non_dominated import get_non_dominated_solutions

# Utility functions ------------------------------

def normalize(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    """
    Standardizes values to the [0, 1] range.
    
    Args:
        x: Array of values to normalize.
        xmin: Global minimum value.
        xmax: Global maximum value.
        
    Returns:
        Normalized numpy array.
    """
    return (x - xmin) / (xmax - xmin)

# Reference point computation ------------------------------

def get_algorithm_extrema(dfs: dict[int, pd.DataFrame]) -> tuple[float, float, float, float]:
    """
    Retrieves the absolute minimum and maximum values for PSNR and parameters 
    within a single algorithm's execution (across all seeds).
    
    Returns:
        tuple: (min_psnr, max_psnr, min_param, max_param)
    """
    max_psnr = max(df["predicted_psnr"].max() for df in dfs.values())
    min_psnr = min(df["predicted_psnr"].min() for df in dfs.values())

    max_param = max(df["params"].max() for df in dfs.values())
    min_param = min(df["params"].min() for df in dfs.values())

    return min_psnr, max_psnr, min_param, max_param


def get_global_reference_point(all_files: dict) -> list[float]:
    """
    Scans all results from all algorithms to determine a global bounding box.
    This is critical for ensuring that normalization is consistent across 
    different methods during hypervolume calculation.

    Args:
        all_files (dict): Nested dictionary {algorithm_name: {seed: file_path}}.

    Returns:
        list: [min_psnr, min_param, max_psnr, max_param] representing the 
              global limits of the search space.
    """
    min_psnr  = float("inf")
    max_psnr  = float("-inf")
    min_param = float("inf")
    max_param = float("-inf")

    for algorithm, paths in sorted(all_files.items()):
        print(f"\nProcessing algorithm [{algorithm}] with {len(paths)} seeds")
        dfs = load_seeds(paths)
        if not dfs:
            continue

        # Get local extrema for the current algorithm
        mn_psnr, mx_psnr, mn_param, mx_param = get_algorithm_extrema(dfs)

        # Update global extrema
        min_psnr  = min(min_psnr,  mn_psnr)
        max_psnr  = max(max_psnr,  mx_psnr)
        min_param = min(min_param, mn_param)
        max_param = max(max_param, mx_param)

    return [min_psnr, min_param, max_psnr, max_param]

# Global reference extremes------------------------------

def get_nadir_vector(all_files: dict[str, dict[int, Path]], ref=None) -> list[float]:
    """
    Identifies the Nadir and Ideal points based on the aggregate non-dominated 
    front of all tested algorithms. 

    This function should be called ONCE before specific algorithm analysis 
    to provide a shared reference for metrics like Spread (coverage).

    Parameters:
    -----------
    all_files : dict
        A dictionary containing paths for all algorithms and seeds.
    ref : list, optional
        Global reference points [min_psnr, min_param, max_psnr, max_param] 
        to return the vector in normalized space.

    Returns:
    --------
    list: [ideal_psnr, ideal_param, nadir_psnr, nadir_param]
    """
    all_nd_params = []
    all_nd_psnr   = []

    # Extract non-dominated solutions from the last generation of every run
    for algorithm, paths in sorted(all_files.items()):
        dfs = load_seeds(paths)
        for seed, df in dfs.items():
            last_gen = df["generation"].max()
            group    = df[df["generation"] == last_gen]

            psnr = group["predicted_psnr"].to_numpy()
            params = group["params"].to_numpy()

            # Local non-dominated solutions for this run
            nd_ps, nd_p = get_non_dominated_solutions(psnr, params)
            all_nd_params.append(nd_p)
            all_nd_psnr.append(nd_ps)

    # Combine all local fronts to find the "Best Known" Pareto front
    union_p  = np.concatenate(all_nd_params)
    union_ps = np.concatenate(all_nd_psnr)

    # Global non-dominated solutions (the collective Pareto front)
    Pareto_psnr, Pareto_params = get_non_dominated_solutions(union_ps, union_p)
    
    # Normalize the combined front if reference points are provided
    if ref != None:
        Pareto_psnr = normalize(Pareto_psnr, ref[0], ref[2])
        Pareto_params = normalize(Pareto_params, ref[1], ref[3])    

    # Extract Nadir (worst values in the front) and Ideal (best values in the front)
    nadir_param = Pareto_params.max()
    nadir_psnr  = Pareto_psnr.max()
    ideal_param = Pareto_params.min()
    ideal_psnr  = Pareto_psnr.min()

    return [ideal_psnr, ideal_param, nadir_psnr, nadir_param]