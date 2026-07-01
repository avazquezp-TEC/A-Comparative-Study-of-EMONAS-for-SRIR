import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- Global Configuration ---
TITLE_SIZE = 32
AXIS_LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 24

# Maps internal algorithm identifiers to their academic notation
ALGORITHM_LABELS = {
    "sms_emoa": "SMS-EMOA",
    "nsga3": "NSGA-III",
    "cmopso": "CMOPSO",
    "imia": "IMIA",
}

plt.rcParams.update({
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': AXIS_LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE,
})


def get_display_name(alg_name: str) -> str:
    """Returns the formatted display name of an algorithm for plots."""
    return ALGORITHM_LABELS.get(alg_name.lower(), alg_name.upper())


def _save_or_show(save_path: str | None, filename: str) -> None:
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Hypervolume evolution ------------------------------


def plot_HV(algorithm, hv_mean, hv_std, generations, trend, best_gen, best_val, No_eval, save_path=None):
    """Plots the Hypervolume evolution over the course of the optimization."""
    plt.figure(figsize=(10, 8))

    total_generations = len(generations)
    eval_per_gen = No_eval / total_generations if total_generations > 1 else No_eval

    evaluations = np.array(generations) * eval_per_gen
    best_eval = best_gen * eval_per_gen

    plt.errorbar(evaluations, hv_mean, yerr=hv_std, fmt="o", color='blue',
                 ecolor='lightcoral', elinewidth=2, capsize=4, alpha=0.7,
                 label="Mean HV ± Std", zorder=2)

    plt.plot(evaluations, trend, "k--", linewidth=2.5, label="Tendency", zorder=3)

    plt.scatter(best_eval, best_val, c="red", marker="x", s=120,
                linewidths=2.5, label="Maximum Tendency", zorder=4)

    ymin, _ = plt.ylim()
    xmin, _ = plt.xlim()
    plt.vlines(best_eval, ymin=ymin, ymax=best_val, colors='black',
               linestyles='dashed', linewidth=1.2, zorder=4)
    plt.hlines(best_val, xmin=xmin, xmax=best_eval, colors='black',
               linestyles='dashed', linewidth=1.2, zorder=4)

    plt.xlabel("Evaluations")
    plt.ylabel("Hypervolume")
    plt.grid(True)
    plt.legend(loc='lower right', framealpha=1.0)
    plt.xlim(0, No_eval * 1.02)
    plt.tight_layout()

    _save_or_show(save_path, f"hv_{algorithm}.png")


# Non-dominated count comparison ------------------------------


def plot_pareto(nondominated, algorithms, save_path=None):
    """Boxplot comparing the number of non-dominated solutions per algorithm."""
    plt.figure(figsize=(12, 6))
    display_algs = [get_display_name(alg) for alg in algorithms]

    plt.boxplot(nondominated, labels=display_algs)
    plt.ylabel("No. of Non-Dominated Solutions")
    plt.grid(True)
    plt.tight_layout()

    _save_or_show(save_path, "pareto.png")


# Spread & Spacing comparison ------------------------------


def plot_combined_metrics(spread_all, spacing_all, algorithms, save_path=None):
    """
    Single boxplot chart where each algorithm has two side-by-side boxes
    representing Spread and Spacing.
    """
    data_list = []
    display_algs = [get_display_name(alg) for alg in algorithms]

    for i, alg_name in enumerate(display_algs):
        for val in spread_all[i]:
            data_list.append({"Algorithm": alg_name, "Value": val, "Metric": "Spread (Δ)"})
        for val in spacing_all[i]:
            data_list.append({"Algorithm": alg_name, "Value": val, "Metric": "Spacing"})

    df = pd.DataFrame(data_list)

    plt.figure(figsize=(14, 7))
    sns.boxplot(data=df, x="Algorithm", y="Value", hue="Metric", palette="Set2")
    plt.xlabel("Algorithms")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Metrics", loc="upper right")
    plt.tight_layout()

    _save_or_show(save_path, "combined_metrics.png")


# Pareto fronts ------------------------------


def plot_pareto_front(algorithm, all_psnr, all_params, front_psnr, front_params, save_path=None):
    """
    Plots a single algorithm's non-dominated solutions.

    Args:
        all_psnr, all_params: Union of every seed's own non-dominated front
            (i.e. before merging across seeds), plotted as blue dots.
        front_psnr, front_params: Final combined non-dominated front across
            all seeds, highlighted with black X markers.
    """
    display_name = get_display_name(algorithm)

    plt.figure(figsize=(8, 5))
    plt.scatter(all_psnr, all_params, color="royalblue", edgecolors="black",
                s=50, alpha=0.7, zorder=2, label="Non-dominated per seed")
    plt.scatter(front_psnr, front_params, color="black", marker="x", s=110,
                linewidths=2, zorder=3, label="Combined non-dominated front")
    plt.title(f"Non-dominated front - {display_name}")
    plt.xlabel("Predicted PSNR (dB)")
    plt.ylabel("Params")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6, zorder=1)
    plt.tight_layout()

    _save_or_show(save_path, f"pareto_front_{algorithm.lower()}.png")


def plot_all_pareto_fronts(fronts: dict, global_psnr, global_params, save_path=None):
    """
    Plots every algorithm's non-dominated front together with the overall
    (cross-algorithm) known Pareto front.

    Args:
        fronts: {algorithm: (psnr, params)}.
        global_psnr, global_params: The overall non-dominated front computed
            from the union of every algorithm's front.
    """
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']

    plt.figure(figsize=(10, 6))
    for i, (algorithm, (psnr, params)) in enumerate(fronts.items()):
        plt.scatter(psnr, params, label=get_display_name(algorithm),
                    marker=markers[i % len(markers)], alpha=0.8, s=100)

    plt.scatter(global_psnr, global_params, label="Known Global Pareto Front",
                marker='x', color='black', s=80, zorder=5)

    plt.xlabel("PSNR (dB)")
    plt.ylabel("Params")
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    _save_or_show(save_path, "pareto_front_all.png")
