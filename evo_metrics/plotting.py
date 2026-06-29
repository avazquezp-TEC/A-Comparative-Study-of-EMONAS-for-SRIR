import matplotlib.pyplot as plt
import numpy as np
import os

# --- Global Configuration Variables ---
# Standardized sizes for high-quality figures (DPI 300)
TITLE_SIZE = 32
AXIS_LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 24

# --- Algorithm Name Mapping ---
# Dictionary to convert internal identifiers to academic notation
ALGORITHM_LABELS = {
    "sms_emoa": "SMS-EMOA",
    "nsga3": "NSGA-III"
}

def get_display_name(alg_name):
    """
    Returns the formatted name of an algorithm for plots. 
    If not found in ALGORITHM_LABELS, returns the name in uppercase.
    """
    return ALGORITHM_LABELS.get(alg_name.lower(), alg_name.upper())

# Apply global Matplotlib styling for consistent aesthetics across all plots
plt.rcParams.update({
    'axes.titlesize': TITLE_SIZE,
    'axes.labelsize': AXIS_LABEL_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'legend.fontsize': LEGEND_SIZE
})

# Plotting Functions ------------------------------

def plot_HV(algorithm, hv_mean, hv_std, generations, trend,
            best_gen, best_val, No_eval, save_path=None):
    """
    Plots the Hypervolume evolution over the course of the optimization.
    Converts generations to objective evaluations for a fairer comparison.
    """
    plt.figure(figsize=(10, 8))

    # --- Scale Conversion: Generations → Evaluations ---
    total_generations = len(generations)
    
    # Calculate evaluations per generation to align the X-axis
    if total_generations > 1:
        eval_per_gen = No_eval / total_generations
    else:
        eval_per_gen = No_eval
    
    # Transform generation indices into evaluation counts
    evaluations = np.array(generations) * eval_per_gen
    best_eval = best_gen * eval_per_gen

    # Plot Mean HV with Standard Deviation as error bars
    plt.errorbar(evaluations, hv_mean, yerr=hv_std, fmt="o", color='blue',
                 ecolor='lightcoral', elinewidth=2, capsize=4, alpha=0.7,
                 label="Mean HV ± Std", zorder=2)

    # Plot the calculated tendency (3rd degree polynomial)
    plt.plot(evaluations, trend, "k--", linewidth=2.5, label="Tendency", zorder=3)

    # Highlight the maximum point in the tendency curve
    plt.scatter(best_eval, best_val, c="red", marker="x", s=120,
                linewidths=2.5, label="Maximum Tendency", zorder=4)

    # Add visual reference lines for the best tendency point
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
    
    # Adjust X-limit to match total evaluations with a 2% margin
    plt.xlim(0, No_eval * 1.02)
    
    plt.tight_layout()

    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"hv_{algorithm}.png") 
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pareto(nondominated, algorithms, save_path=None):
    """
    Generates a boxplot comparing the number of non-dominated solutions 
    produced by different algorithms in their last generation.
    """
    plt.figure(figsize=(12, 6))
    
    # Convert internal names to display names for labels
    display_algs = [get_display_name(alg) for alg in algorithms]
    
    plt.boxplot(nondominated, labels=display_algs)
    plt.ylabel("No. of Non-Dominated Solutions")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, "pareto.png") 
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spread(spread_all, algorithms, save_path=None):
    """
    Generates a boxplot comparison of the Spread (Δ) metric across algorithms.
    Lower values indicate better coverage and distribution.
    """
    plt.figure(figsize=(15, 6))
    
    display_algs = [get_display_name(alg) for alg in algorithms]
    
    plt.boxplot(spread_all, labels=display_algs)
    plt.ylabel("Spread (Δ)")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, "spread.png") 
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_spacing(spacing_all, algorithms, save_path=None):
    """
    Generates a boxplot comparison of the Spacing metric across algorithms.
    Lower values indicate a more uniform distribution of points.
    """
    plt.figure(figsize=(15, 6))
    
    display_algs = [get_display_name(alg) for alg in algorithms]
    
    plt.boxplot(spacing_all, labels=display_algs)
    plt.ylabel("Spacing")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, "spacing.png") 
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()