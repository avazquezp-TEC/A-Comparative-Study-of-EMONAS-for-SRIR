import os
import re
from pathlib import Path

import pandas as pd

# Regex to extract the seed number and algorithm name from standardized filenames.
# Example: ensemble_mean_population_seed_42_10_50_algorithm_nsga3.csv
CSV_PATTERN = re.compile(
    r'ensemble_mean_population_seed_(\d+)_\d+_\d+_algorithm_([a-zA-Z0-9_]+?).csv',
    re.IGNORECASE
)

GENE_COLUMN = "decoded_architecture"

# File discovery ------------------------------


def find_csv_files(base_dir: Path, algorithm_filter: str | None) -> dict[str, dict[int, Path]]:
    """
    Recursively searches through seed directories to identify and group CSV result files.

    Expects a directory structure where each run is stored in a 'seed_*' subfolder,
    and uses a regular expression to parse metadata directly from the filenames.

    Args:
        base_dir: Root directory containing the experimental results.
        algorithm_filter: If provided, only files matching this algorithm name
            will be indexed.

    Returns:
        Nested dictionary {algorithm_name: {seed_id: file_path}}.
    """
    files: dict[str, dict[int, Path]] = {}

    for seed_dir in sorted(base_dir.glob("seed_*")):
        for csv_file in seed_dir.glob("ensemble_mean*.csv"):
            match = CSV_PATTERN.match(csv_file.name)
            if not match:
                continue

            seed = int(match.group(1))
            algorithm = match.group(2)

            if algorithm_filter and algorithm != algorithm_filter:
                continue

            files.setdefault(algorithm, {})[seed] = csv_file

    return files


def load_seeds(paths: dict[int, Path]) -> dict[int, pd.DataFrame]:
    """
    Reads multiple CSV files into a dictionary of DataFrames.

    Args:
        paths: Dictionary mapping seed IDs to their respective file paths.

    Returns:
        Dictionary {seed_id: pd.DataFrame} with the loaded data.
    """
    dataframes = {}

    for seed, path in sorted(paths.items()):
        try:
            dataframes[seed] = pd.read_csv(path)
        except Exception as exc:
            print(f"  ⚠ Error reading seed {seed} from {path}: {exc}")

    return dataframes


# CSV export ------------------------------


def save_genes_to_csv(genes, psnr, params, algorithm: str, dir_save: str | None) -> None:
    """
    Saves the non-dominated architecture genes of an algorithm to a CSV file,
    together with their PSNR and Params values.

    Args:
        genes: Array of decoded architecture strings.
        psnr: Array of PSNR values aligned with genes.
        params: Array of Params values aligned with genes.
        algorithm: Algorithm name, used to build the output filename.
        dir_save: Directory where the CSV file is written. If None, saving
            is skipped.
    """
    if dir_save is None:
        print(f"  ⚠ No dir_save specified — skipping genes CSV for [{algorithm}]")
        return

    genes_clean = pd.Series(genes).str.strip().str.replace(" ", ",", regex=False)
    df = pd.DataFrame({
        "algorithm": algorithm,
        "psnr": psnr,
        "params": params,
        "genes": genes_clean,
    })

    os.makedirs(dir_save, exist_ok=True)
    filepath = os.path.join(dir_save, f"genes_{algorithm.lower()}.csv")
    df.to_csv(filepath, index=False)
    print(f"  Saved genes CSV: {filepath}")
