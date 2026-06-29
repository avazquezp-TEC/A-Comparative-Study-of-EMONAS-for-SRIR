from pathlib import Path
import pandas as pd
import re

# Regex to extract the seed number and algorithm name from standardized filenames
# Example: ensemble_mean_population_seed_42_10_50_algorithm_nsga3.csv
CSV_PATTERN = re.compile(
    r'ensemble_mean_population_seed_(\d+)_\d+_\d+_algorithm_([a-zA-Z0-9_]+?).csv',
    re.IGNORECASE
)

# File handling ------------------------------

def find_csv_files(base_dir: Path, algorithm_filter: str | None) -> dict[str, dict[int, Path]]:
    """
    Recursively searches through seed directories to identify and group CSV result files.

    This function expects a directory structure where each run is stored in a 'seed_*' 
    subfolder. It uses regular expressions to parse metadata directly from the filenames.

    Args:
        base_dir (Path): The root directory containing the experimental results.
        algorithm_filter (str | None): If provided, only files matching this algorithm 
                                       name will be indexed.

    Returns:
        dict: A nested dictionary structure {algorithm_name: {seed_id: file_path}}.
    """
    files = {}

    # Iterate through all directories matching the 'seed_*' pattern
    for seed_dir in sorted(base_dir.glob("seed_*")):
        # Look for CSV files starting with the expected prefix
        for csv_file in seed_dir.glob("ensemble_mean*.csv"):
            match = CSV_PATTERN.match(csv_file.name)
            if not match:
                continue

            # Extract seed and algorithm name from the filename groups
            seed = int(match.group(1))
            algorithm = match.group(2)

            # Apply algorithm filter if specified
            if algorithm_filter and algorithm != algorithm_filter:
                continue

            # Organize the file path in the nested dictionary
            files.setdefault(algorithm, {})[seed] = csv_file

    return files


def load_seeds(paths: dict[int, Path]) -> dict[int, pd.DataFrame]:
    """
    Reads multiple CSV files and loads them into a dictionary of DataFrames.

    Args:
        paths (dict): A dictionary mapping seed IDs to their respective file paths.

    Returns:
        dict: A dictionary {seed_id: pd.DataFrame} containing the loaded data.
    """
    dataframes = {}

    # Sort items by seed ID to ensure consistent processing order
    for seed, path in sorted(paths.items()):
        try:
            # Load the experimental results using pandas
            dataframes[seed] = pd.read_csv(path)
        except Exception as exc:
            # Log an error message if the file is corrupted or missing
            print(f"  ⚠ Error reading seed {seed} from {path}: {exc}")

    return dataframes