import numpy as np
import pandas as pd

from evo_metrics.file_handl import load_seeds
from evo_metrics.non_dominated import get_non_dominated_solutions
from evo_metrics.utils import normalize

# Reference point computation ------------------------------


def get_algorithm_extrema(dfs: dict[int, pd.DataFrame]) -> tuple[float, float, float, float]:
    """
    Retrieves the absolute minimum and maximum values for PSNR and Params
    within a single algorithm's execution (across all its seeds).

    Returns:
        (min_psnr, max_psnr, min_param, max_param)
    """
    max_psnr = max(df["predicted_psnr"].max() for df in dfs.values())
    min_psnr = min(df["predicted_psnr"].min() for df in dfs.values())

    max_param = max(df["params"].max() for df in dfs.values())
    min_param = min(df["params"].min() for df in dfs.values())

    return min_psnr, max_psnr, min_param, max_param


def get_global_reference_point(all_files: dict) -> list[float]:
    """
    Scans every algorithm's results to determine a global bounding box. This
    is critical to keep normalization consistent across different methods
    when computing the Hypervolume.

    Args:
        all_files: Nested dictionary {algorithm_name: {seed: file_path}}.

    Returns:
        [min_psnr, min_param, max_psnr, max_param], the global limits of the
        search space.
    """
    min_psnr = float("inf")
    max_psnr = float("-inf")
    min_param = float("inf")
    max_param = float("-inf")

    for algorithm, paths in sorted(all_files.items()):
        print(f"\nProcessing algorithm [{algorithm}] with {len(paths)} seeds")
        dfs = load_seeds(paths)
        if not dfs:
            continue

        mn_psnr, mx_psnr, mn_param, mx_param = get_algorithm_extrema(dfs)

        min_psnr = min(min_psnr, mn_psnr)
        max_psnr = max(max_psnr, mx_psnr)
        min_param = min(min_param, mn_param)
        max_param = max(max_param, mx_param)

    return [min_psnr, min_param, max_psnr, max_param]


# Global reference extremes ------------------------------


def get_nadir_vector(all_files: dict[str, dict[int, "Path"]], ref: list[float] | None = None) -> list[float]:
    """
    Identifies the Nadir and Ideal points based on the aggregate non-dominated
    front of every algorithm. Should be called once, before per-algorithm
    analysis, to provide a shared reference for metrics like Spread.

    Args:
        all_files: Dictionary containing paths for all algorithms and seeds.
        ref: Optional global reference point [min_psnr, min_param, max_psnr,
            max_param] used to return the vector in normalized space.

    Returns:
        [ideal_psnr, ideal_param, nadir_psnr, nadir_param]
    """
    all_nd_params = []
    all_nd_psnr = []

    for algorithm, paths in sorted(all_files.items()):
        dfs = load_seeds(paths)
        for seed, df in dfs.items():
            last_gen = df["generation"].max()
            group = df[df["generation"] == last_gen]

            psnr = group["predicted_psnr"].to_numpy()
            params = group["params"].to_numpy()

            nd_psnr, nd_params = get_non_dominated_solutions(psnr, params)
            all_nd_params.append(nd_params)
            all_nd_psnr.append(nd_psnr)

    union_params = np.concatenate(all_nd_params)
    union_psnr = np.concatenate(all_nd_psnr)

    pareto_psnr, pareto_params = get_non_dominated_solutions(union_psnr, union_params)

    if ref is not None:
        pareto_psnr = normalize(pareto_psnr, ref[0], ref[2])
        pareto_params = normalize(pareto_params, ref[1], ref[3])

    nadir_param = pareto_params.max()
    nadir_psnr = pareto_psnr.max()
    ideal_param = pareto_params.min()
    ideal_psnr = pareto_psnr.min()

    return [ideal_psnr, ideal_param, nadir_psnr, nadir_param]
