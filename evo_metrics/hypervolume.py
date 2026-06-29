import numpy as np
import pandas as pd
from evo_metrics.hypervolume_2d import hypervolume_2d
from evo_metrics.spread import calculate_spread, normalize
from evo_metrics.non_dominated import get_non_dominated_solutions
from evo_metrics.spacing import calculate_spacing
from evo_metrics.plotting import plot_HV
import matplotlib.pyplot as plt

def compute_and_plot_hv(dfs: dict[int, pd.DataFrame], 
                        ref: list[float], 
                        algorithm: str,
                        nadir_vector: list | None = None,
                        dir_save: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes performance metrics (Hypervolume, Spread, and Spacing) across multiple 
    independent runs (seeds) and visualizes the Hypervolume (HV) evolution.

    Args:
        dfs (dict): Dictionary mapping seed IDs to their respective DataFrames.
        ref (list): Normalization reference [min_psnr, min_param, max_psnr, max_param].
        algorithm (str): Name of the algorithm for labeling and titles.
        nadir_vector (list, optional): Reference extremes for Spread calculation in normalized space.
        dir_save (str, optional): Directory path to save the generated plots.

    Returns:
        tuple: (all_No_non_dominated, spread_all_seeds, spacing_all_seeds) as numpy arrays.
    """
    hv_all_seeds         = []
    all_No_non_dominated = []
    spread_all_seeds     = []
    spacing_all_seeds    = []
    all_front_range      = []

    for seed, df in dfs.items():
        hv_per_gen = []
        last_gen   = df["generation"].max()

        # Iterate through generations to calculate the Hypervolume curve
        for gen, group in df.groupby("generation"):
            # Normalize objectives to a [0, 1] range based on provided reference
            psnr   = normalize(group["predicted_psnr"].to_numpy(), ref[0], ref[2])
            params = normalize(group["params"].to_numpy(),          ref[1], ref[3])

            # Calculate 2D Hypervolume for the current generation
            hv = hypervolume_2d(params, psnr)
            hv_per_gen.append(hv)

            # Store normalized data for the final generation to compute end-of-run metrics
            if gen == last_gen:
                last_psnr   = psnr
                last_params = params

        # Extract the non-dominated front from the last generation
        nd_psnr, nd_params = get_non_dominated_solutions(last_psnr, last_params)
        
        # Sort solutions by parameters to calculate the front range
        idx     = nd_params.argsort()
        nd_p_s  = nd_params[idx]
        nd_ps_s = nd_psnr[idx]

        # Calculate the Euclidean distance between the two extreme points of the front
        front_range = np.linalg.norm(
            np.array([nd_p_s[-1], nd_ps_s[-1]]) -
            np.array([nd_p_s[0],  nd_ps_s[0]])
        )
        
        all_No_non_dominated.append(len(nd_params))

        # Diversity metrics: Spread (coverage/uniformity) and Spacing (relative distance)
        spread  = calculate_spread(nd_psnr, nd_params, ref_extremes=nadir_vector)
        spacing = calculate_spacing(nd_psnr, nd_params)

        if spread is not None:
            spread_all_seeds.append(spread)
        if spacing is not None:
            spacing_all_seeds.append(spacing)

        hv_all_seeds.append(hv_per_gen)

    # Convert results to numpy arrays for statistical analysis
    all_No_non_dominated = np.array(all_No_non_dominated)
    spread_all_seeds     = np.array(spread_all_seeds)
    spacing_all_seeds    = np.array(spacing_all_seeds)
    hv_all_seeds         = np.array(hv_all_seeds)
    all_front_range      = np.array(front_range)

    # Logging average performance across all seeds
    print(f"  Avg non-dominated solutions (last gen) : {all_No_non_dominated.mean():.2f}")
    print(f"  Avg Spread  Δ  (last gen)              : {spread_all_seeds.mean():.4f}")
    print(f"  Avg Spacing S  (last gen)              : {spacing_all_seeds.mean():.4f}")
    print(f"  Avg Front Range (last gen)             : {all_front_range.mean():.4f}")

    # Calculate mean and standard deviation of HV across seeds for the plot
    hv_mean = np.mean(hv_all_seeds, axis=0)
    hv_std  = np.std(hv_all_seeds,  axis=0)
    generations = np.arange(len(hv_mean))

    # Fit a 3rd-degree polynomial to identify the overall HV trend
    poly     = np.poly1d(np.polyfit(generations, hv_mean, 3))
    trend    = poly(generations)
    best_gen = np.argmax(trend)
    best_val = trend[best_gen]

    print(f"  Maximum tendency HV                    : {best_val:.6f}")

    # Generate and save the Hypervolume evolution plot
    plot_HV(algorithm, hv_mean, hv_std, generations, trend, best_gen, best_val, 25000, dir_save)

    return all_No_non_dominated, spread_all_seeds, spacing_all_seeds