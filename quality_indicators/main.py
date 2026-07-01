"""
main.py
-------
Quality-indicator analysis for multi-objective evolutionary optimization runs.

Searches CSV files with the naming convention:
    <base-dir>/seed_<N>/ensemble_mean_population_seed_<N>_..._algorithm_<name>.csv

For each algorithm found, it:
  1. Computes and plots the Hypervolume (HV) evolution across generations.
  2. Extracts the combined non-dominated front of the last generation across
     all seeds (with architecture genes, if the CSV contains a
     'decoded_architecture' column) and saves it to genes_<algorithm>.csv.
  3. Computes Spread (Δ) and Spacing (S) per seed, using a shared global
     reference front (nadir/ideal vector) so results are comparable across
     algorithms.

Once every algorithm has been processed, it also produces:
  - A boxplot of the number of non-dominated solutions per algorithm.
  - A combined boxplot of Spread and Spacing per algorithm.
  - The individual Pareto front of each algorithm.
  - The combined Pareto front of all algorithms, together with the overall
    "known" global Pareto front.

Usage:
    python main.py --base-dir outputs --dir_save outputs/plots
    python main.py --base-dir outputs --algorithm imia --dir_save outputs/plots
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

from evo_metrics.file_handl import find_csv_files, load_seeds, save_genes_to_csv
from evo_metrics.ref_point import get_global_reference_point, get_nadir_vector
from evo_metrics.non_dominated import get_non_dominated_solutions
from evo_metrics.metrics import analyze_algorithm
from evo_metrics.plotting import (
    plot_pareto,
    plot_combined_metrics,
    plot_pareto_front,
    plot_all_pareto_fronts,
)

warnings.filterwarnings("ignore")


def analyze_all_algorithms(all_files: dict, ref: list[float], dir_save: str | None = None) -> None:
    """Runs the full quality-indicator analysis for every algorithm found."""
    nadir_vector = get_nadir_vector(all_files, ref)

    counts_all, spread_all, spacing_all, algorithms = [], [], [], []
    fronts = {}  # {algorithm: (psnr, params)} used for the combined Pareto plot
    global_psnr, global_params = [], []

    for algorithm, paths in sorted(all_files.items()):
        print(f"\nAnalyzing [{algorithm}]")
        dfs = load_seeds(paths)
        if not dfs:
            continue

        result = analyze_algorithm(dfs, ref, algorithm, nadir_vector, dir_save)

        counts_all.append(result["counts"])
        spread_all.append(result["spread"])
        spacing_all.append(result["spacing"])
        algorithms.append(algorithm.upper())

        fronts[algorithm] = (result["front_psnr"], result["front_params"])
        plot_pareto_front(algorithm,
                           result["all_psnr"], result["all_params"],
                           result["front_psnr"], result["front_params"],
                           dir_save)

        if result["front_genes"] is not None:
            save_genes_to_csv(result["front_genes"], result["front_psnr"],
                               result["front_params"], algorithm, dir_save)
        else:
            print(f"  ⚠ No 'decoded_architecture' column found — skipping genes CSV for [{algorithm}]")

        global_psnr.append(result["front_psnr"])
        global_params.append(result["front_params"])

    plot_pareto(counts_all, algorithms, dir_save)
    plot_combined_metrics(spread_all, spacing_all, algorithms, dir_save)

    global_psnr = np.concatenate(global_psnr)
    global_params = np.concatenate(global_params)
    nd_psnr, nd_params = get_non_dominated_solutions(global_psnr, global_params)
    plot_all_pareto_fronts(fronts, nd_psnr, nd_params, dir_save)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality-indicator analysis tool")
    parser.add_argument("--base-dir", default="outputs",
                         help="Base directory containing seed folders")
    parser.add_argument("--algorithm", default=None,
                         help="Analyze a specific algorithm only")
    parser.add_argument("--dir_save", default=None,
                         help="Directory where plots and CSVs are saved")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    print(f"\n🔍 Searching CSV files in: {base_dir}")

    all_files = find_csv_files(base_dir, args.algorithm)
    if not all_files:
        print("⚠ No CSV files found. Check directory and naming.")
        return

    ref = get_global_reference_point(all_files)
    print("\nGlobal reference point:")
    print(f"  PSNR  min / max : {ref[0]} / {ref[2]}")
    print(f"  Param min / max : {ref[1]} / {ref[3]}")

    analyze_all_algorithms(all_files, ref, args.dir_save)


if __name__ == "__main__":
    main()
