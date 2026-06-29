# Evolutionary Multi-Objective NAS for Super-Resolution Image Restoration

This repository contains the implementation and comparative analysis of Evolutionary Multi-Objective Neural Architecture Search (EMONAS) algorithms applied to Super-Resolution Image Restoration (SRIR).

## Abstract

This work presents a comparative analysis of EMONAS algorithms for SRIR. While deep learning models for this task have advanced significantly, they are often over-parameterized due to manual design, leaving a vast solution space unexplored. We evaluate three underexplored alternatives against the state-of-the-art NSGA-III:

- **CMOPSO**: A Competitive Mechanism-based Multi-Objective Particle Swarm Optimizer designed for many-objective problems.
- **IMIA**: An Indicator-based Multi-objective Immune Algorithm that guides the search through quality metrics.
- **SMS-EMOA**: A steady-state evolutionary strategy that uses hypervolume contribution to maintain population diversity.

Performance is measured using Hypervolume (HV), Spread (Δ), Spacing (S), and Pareto Front Rank.

## Getting Started

### 1. Neural Architecture Search (Optimization)

The `main.py` script is the primary entry point for executing the NAS process. It uses surrogate models to evaluate architectures efficiently without full training.

**Usage example:**

```bash
python main.py --search imia --seed 1 --pop-size 20 --n_gen 300 --verbose
````
### Key Arguments

- `--search`: Algorithm selection (`nsga3`, `cmopso`, `imia`, `sms_emoa`)
- `--pop_size`: Population size (default: 20)
- `--n_gen`: Number of generations (default: 100)
- `--models_dir`: Directory containing the surrogate `.pkl` models
- `--output_file`: (Optional) Custom path for the results CSV

## Quality Indicator Analysis (Post-Processing)

After running the experiments, use `quality_indicator_evo.py` to process the results, normalize objectives, and calculate quality metrics.

**Usage example:**

```bash
python quality_indicator_evo.py --base-dir ./outputs --dir_save ./results
````
**Features:**

- `Global normalization`: Computes a shared reference point across all algorithms to ensure fair metric comparison
- `Metric suite`: Calculates evolution of HV, final Spread, and Spacing
- `Visualizations`: Generates Pareto front plots and boxplots for comparative analysis

# Evaluation Workflow

- **Validation**:  
  Run the following command to verify algorithm convergence on a standard benchmark:
  ```bash
  python ZDT1_algorithm_test.py
  ````
- **Execution**:
Run main.py with multiple seeds for each algorithm. Results are stored in outputs/seed_N/.

- **Efficiency**:
Extract mean computation times:
  ```bash
  python avg_runtime.py --dir ./logs
  ````

- **Indicators**:
Generate the scientific plots used in the paper:
  ```bash
  python quality_indicator_evo.py
  ````
## 🎓 Citation
If you use this code or our findings in your research, please cite this work as follows (currently under review):
# License
This project is licensed under the MIT License - see the LICENSE file for details.
