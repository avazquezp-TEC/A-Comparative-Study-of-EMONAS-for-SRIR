import numpy as np
import pandas as pd

from evo_metrics.hypervolume_2d import hypervolume_2d
from evo_metrics.non_dominated import get_non_dominated_solutions
from evo_metrics.spread import calculate_spread
from evo_metrics.spacing import calculate_spacing
from evo_metrics.utils import normalize
from evo_metrics.plotting import plot_HV

GENE_COLUMN = "decoded_architecture"


def _hv_curve(df: pd.DataFrame, ref: list[float]) -> list[float]:
    """Computes the Hypervolume for every generation of a single run."""
    hv_per_gen = []
    for _, group in df.groupby("generation"):
        psnr = normalize(group["predicted_psnr"].to_numpy(), ref[0], ref[2])
        params = normalize(group["params"].to_numpy(), ref[1], ref[3])
        hv_per_gen.append(hypervolume_2d(params, psnr))
    return hv_per_gen


def _summarize_hv(hv_curves: list[list[float]]):
    """Aggregates HV curves across seeds and fits a trend line."""
    hv = np.array(hv_curves)
    hv_mean = hv.mean(axis=0)
    hv_std = hv.std(axis=0)
    generations = np.arange(len(hv_mean))

    poly = np.poly1d(np.polyfit(generations, hv_mean, 3))
    trend = poly(generations)
    best_gen = int(np.argmax(trend))
    best_val = trend[best_gen]

    return hv_mean, hv_std, generations, trend, best_gen, best_val


def analyze_algorithm(dfs: dict[int, pd.DataFrame],
                       ref: list[float],
                       algorithm: str,
                       nadir_vector: list[float],
                       dir_save: str | None = None) -> dict:
    """
    Computes, for a single algorithm across all its seeds:
      - The Hypervolume evolution (plotted and saved).
      - The combined non-dominated front of the last generation, together
        with architecture genes if the CSV files contain them.
      - Spread (Delta) and Spacing (S), one value per seed, for boxplot
        comparisons across algorithms.

    Args:
        dfs: {seed_id: DataFrame} for this algorithm.
        ref: Global reference point [min_psnr, min_param, max_psnr, max_param].
        algorithm: Algorithm name (used for labels/filenames).
        nadir_vector: Shared nadir/ideal vector used by the Spread metric.
        dir_save: Directory where the HV plot is saved.

    Returns:
        dict with keys: counts, spread, spacing, hv_mean, hv_std,
        all_psnr, all_params (union of every seed's non-dominated solutions,
        before merging across seeds), front_psnr, front_params (the final
        combined non-dominated front), front_genes (None if not available).
    """
    hv_curves = []
    counts, spreads, spacings = [], [], []
    seed_fronts_psnr, seed_fronts_params, seed_fronts_genes = [], [], []

    has_genes = all(GENE_COLUMN in df.columns for df in dfs.values())

    for seed, df in dfs.items():
        hv_curves.append(_hv_curve(df, ref))

        last_gen = df["generation"].max()
        last = df[df["generation"] == last_gen]
        raw_psnr = last["predicted_psnr"].to_numpy()
        raw_params = last["params"].to_numpy()

        if has_genes:
            genes = last[GENE_COLUMN].to_numpy()
            nd_psnr, nd_params, nd_genes = get_non_dominated_solutions(raw_psnr, raw_params, genes)
            seed_fronts_genes.append(nd_genes)
        else:
            nd_psnr, nd_params = get_non_dominated_solutions(raw_psnr, raw_params)

        seed_fronts_psnr.append(nd_psnr)
        seed_fronts_params.append(nd_params)
        counts.append(len(nd_params))

        nd_psnr_n = normalize(nd_psnr, ref[0], ref[2])
        nd_params_n = normalize(nd_params, ref[1], ref[3])

        spread = calculate_spread(nd_psnr_n, nd_params_n, ref_extremes=nadir_vector)
        spacing = calculate_spacing(nd_psnr_n, nd_params_n)
        if spread is not None:
            spreads.append(spread)
        if spacing is not None:
            spacings.append(spacing)

    # Combine every seed's front into one overall non-dominated front
    all_psnr = np.concatenate(seed_fronts_psnr)
    all_params = np.concatenate(seed_fronts_params)

    if has_genes:
        all_genes = np.concatenate(seed_fronts_genes)
        front_psnr, front_params, front_genes = get_non_dominated_solutions(all_psnr, all_params, all_genes)
    else:
        front_psnr, front_params = get_non_dominated_solutions(all_psnr, all_params)
        front_genes = None

    counts = np.array(counts)
    spreads = np.array(spreads)
    spacings = np.array(spacings)

    print(f"  Avg non-dominated solutions (last gen) : {counts.mean():.2f}")
    print(f"  Avg Spread  Delta (last gen)            : {spreads.mean():.4f}")
    print(f"  Avg Spacing S     (last gen)            : {spacings.mean():.4f}")

    hv_mean, hv_std, generations, trend, best_gen, best_val = _summarize_hv(hv_curves)
    print(f"  Maximum tendency HV                     : {best_val:.6f}")

    plot_HV(algorithm, hv_mean, hv_std, generations, trend, best_gen, best_val, 25000, dir_save)

    return {
        "counts": counts,
        "spread": spreads,
        "spacing": spacings,
        "hv_mean": hv_mean,
        "hv_std": hv_std,
        "all_psnr": all_psnr,
        "all_params": all_params,
        "front_psnr": front_psnr,
        "front_params": front_params,
        "front_genes": front_genes,
    }
