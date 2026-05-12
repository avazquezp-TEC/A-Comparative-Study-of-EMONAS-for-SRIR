"""
Compute_hv.py
-------------------
This script computes the reference point and hypervolume evolution for
multi-objective optimization algorithms.

It searches CSV files with the following naming convention:
    outputs/seed_<N>/ensemble_mean_population_seed_<N>__<date>_algorithm_<prefix>.csv

For each algorithm it:
  1. Finds global min and max values (reference point).
  2. Computes and plots hypervolume evolution per generation.
  3. Computes Spread (Δ) and Spacing (S) metrics using a shared global
     reference front (union of all non-dominated solutions across all
     algorithms and seeds) so that metrics are comparable.

Fixes applied vs. original:
  - Spread was computed without ref_extremes (only measured internal
    uniformity, not coverage). Now uses nadir_vector.
  - params/psnr were taken from the last loop iteration scope instead of
    explicitly from the last generation. Now the last generation is
    extracted explicitly.
  - ref_extremes was local per algorithm, making Spread incomparable
    across algorithms. Now a single nadir_vector is built once
    from the union of all non-dominated fronts before any algorithm is
    analysed.
  - Spacing now sorts the front before computing nearest-neighbour
    distances and optionally normalizes S by the bounding-box diagonal
    of the front so it is scale-invariant and comparable across algorithms
    that may produce fronts of different extents (normalize_by_range=True).

Usage:
    python compute_hv.py --base-dir outputs --algorithm imia
"""

# Imports ------------------------------
import argparse
import warnings
from pathlib import Path

from evo_metrics.file_handl import find_csv_files, load_seeds
from evo_metrics.ref_point import get_global_reference_point, get_nadir_vector
from evo_metrics.plotting import  plot_pareto, plot_spread, plot_spacing
from evo_metrics.hypervolume import compute_and_plot_hv
warnings.filterwarnings("ignore")


# Main analysis ------------------------------

def analyze_all_algorithms(all_files: dict, ref: list[float], dir_save = None) -> None:
    """
    Runs hypervolume + quality-indicator analysis for all algorithms.
    all files is a dict with the path of all the csv files
    """
    
    nadir_vector = get_nadir_vector(all_files, ref)# get normalize

    nondominated = []
    spread_all   = []
    spacing_all  = []
    algorithms   = []

    for algorithm, paths in sorted(all_files.items()):
        print(f"\nAnalyzing hypervolume for [{algorithm}]")
        dfs = load_seeds(paths)
        if not dfs:
            continue

        non_dom, spread, spacing = compute_and_plot_hv(dfs, ref, algorithm, nadir_vector, dir_save)
        
        nondominated.append(non_dom)
        spread_all.append(spread)
        spacing_all.append(spacing)
        algorithms.append(algorithm.upper())

    plot_pareto(nondominated, algorithms, dir_save)
    plot_spread(spread_all,   algorithms, dir_save)
    plot_spacing(spacing_all, algorithms, dir_save)


# Entry point ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hypervolume analysis tool")
    parser.add_argument("--base-dir",  default="outputs",
                        help="Base directory containing seed folders")
    parser.add_argument("--algorithm", default=None,
                        help="Analyze a specific algorithm only")
    parser.add_argument("--dir_save", default=None,
                        help="Directory where plots are saved")

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    print(f"\n🔍 Searching CSV files in: {base_dir}")

    all_files = find_csv_files(base_dir, args.algorithm)
    if not all_files:
        print("⚠ No CSV files found. Check directory and naming.")
        return
    # ref = get_nadir_vector(all_files) # get nadir points
    ref = get_global_reference_point(all_files)# from maximum

    print("\nGlobal reference point:")
    print(f"  PSNR  min / max  : {ref[0]} / {ref[2]}")
    print(f"  Param min / max  : {ref[1]} / {ref[3]}")

    analyze_all_algorithms(all_files, ref, args.dir_save)


if __name__ == "__main__":
    main()