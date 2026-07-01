# Quality Indicators — Multi-Objective Evolutionary Optimization

This file computes the hypervolume, number of non dominated solutions, spread and spacing of the 
algorithms across all the seeds and extract the nondominated solutions for training.

## Structure

```
main.py                        # single entry point
evo_metrics/
    file_handl.py               # find/load CSV files, save genes CSV
    non_dominated.py            # vectorized non-dominated (Pareto) sorting
    hypervolume_2d.py           # 2D hypervolume computation
    spread.py                   # Spread (Δ) metric
    spacing.py                  # Spacing (S) metric
    ref_point.py                # global reference point + nadir/ideal vector
    metrics.py                  # per-algorithm analysis (HV, front, spread, spacing)
    plotting.py                 # all plots
    utils.py                    # normalize()
```

## Expected input

```
outputs/
  seed_<N>/
    ensemble_mean_population_seed_<N>_..._algorithm_<name>.csv
```

Each CSV must contain the columns `generation`, `predicted_psnr`, `params`,
and optionally `decoded_architecture` (used to export the non-dominated
architecture genes).

## What it produces (in `--dir_save`)

- `hv_<algorithm>.png` — Hypervolume evolution per algorithm.
- `genes_<algorithm>.csv` — Non-dominated genes (architecture, PSNR, params)
  across all seeds, for every algorithm that has a `decoded_architecture`
  column.
- `pareto.png` — Boxplot of the number of non-dominated solutions per
  algorithm.
- `combined_metrics.png` — Boxplot of Spread and Spacing per algorithm.
- `pareto_front_<algorithm>.png` — Individual non-dominated front per
  algorithm.
- `pareto_front_all.png` — Combined non-dominated front of all algorithms,
  plus the overall "known" global Pareto front.

## Usage

```bash
python quality_indicators/main.py --base-dir outputs --dir_save Plots
python quality_indicators/main.py --base-dir outputs --algorithm imia --dir_save Plots
```
