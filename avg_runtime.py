"""
avg_runtime.py
--------------
Calculates the average execution time (in minutes) from logs generated
by the following algorithms: sms_emoa, imia, cmopso, nsga3.

Expected naming convention:
    <prefix>_seed<N>.log
    Examples: imia_seed1.log, nsga3_seed15.log

Usage:
    # Process all algorithms in the current directory
    python avg_runtime.py

    # Process only a specific algorithm
    python avg_runtime.py --prefix imia

    # Custom directory and specific algorithm
    python avg_runtime.py --dir /path/to/logs --prefix cmopso
"""

import re
import sys
import argparse
from pathlib import Path

# Supported algorithm prefixes
PREFIXES = ['sms_emoa', 'imia', 'cmopso', 'nsga3']
# Regex to find the runtime pattern in the log files
PATTERN = re.compile(r'\[INFO\] Total runtime:\s+([\d.]+)\s+minutes', re.IGNORECASE)


def extract_runtime(log_path: Path) -> float | None:
    """
    Reads a log file and extracts the runtime in minutes.
    Returns None if the pattern is not found.
    """
    try:
        with open(log_path, 'r', errors='replace') as f:
            content = f.read()
        match = PATTERN.search(content)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"  [ERROR] Could not read {log_path.name}: {e}")
    return None


def process_prefix(prefix: str, log_dir: Path) -> dict:
    """
    Processes all logs matching a specific prefix and calculates statistics.
    """
    runtimes = []
    missing_data = []
    files_processed = []

    # Search for files starting with the prefix and ending in .log
    for log_file in log_dir.glob(f"{prefix}_seed*.log"):
        t = extract_runtime(log_file)
        if t is not None:
            runtimes.append(t)
            files_processed.append((log_file.name, t))
        else:
            missing_data.append(log_file.name)

    if not runtimes:
        return {}

    return {
        'avg': sum(runtimes) / len(runtimes),
        'min': min(runtimes),
        'max': max(runtimes),
        'count': len(runtimes),
        'missing': missing_data,
        'files': sorted(files_processed)
    }


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Average runtime calculator for algorithm logs.")
    parser.add_argument('--dir', type=str, default='.', 
                        help='Directory containing the .log files. Default: current directory.')
    parser.add_argument('--prefix', type=str, choices=PREFIXES, default=None,
                        help=f'Specific algorithm prefix to process. Options: {PREFIXES}.')
    parser.add_argument('--verbose', action='store_true', 
                        help='Show detailed runtime for each individual file.')
    args = parser.parse_args()

    log_dir = Path(args.dir).resolve()
    if not log_dir.is_dir():
        print(f"[ERROR] Directory not found: {log_dir}")
        sys.exit(1)

    prefixes = [args.prefix] if args.prefix else PREFIXES

    print(f"\n{'='*55}")
    print(f"  Log Directory : {log_dir}")
    print(f"  Algorithms     : {', '.join(prefixes)}")
    print(f"{'='*55}\n")

    any_found = False
    for prefix in prefixes:
        stats = process_prefix(prefix, log_dir)
        if not stats:
            continue

        any_found = True
        print(f"  [{prefix.upper()}]  ({stats['count']} seeds found)")
        print(f"    Average : {stats['avg']:.4f} min")
        print(f"    Minimum : {stats['min']:.4f} min")
        print(f"    Maximum : {stats['max']:.4f} min")

        if stats['missing']:
            print(f"    ⚠ Missing runtime info in: {', '.join(stats['missing'])}")

        if args.verbose:
            print("    Detailed Breakdown:")
            for name, t in stats['files']:
                print(f"      - {name}: {t:.4f} min")
        print("-" * 30)

    if not any_found:
        print("❌ No matching log files were found in the specified directory.")

if __name__ == "__main__":
    main()