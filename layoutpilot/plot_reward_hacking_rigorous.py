"""Regenerate Figure 1 from saved rigorous Best-of-N CSV results.

Usage:
    python -m layoutpilot.plot_reward_hacking_rigorous \
      --run-dir experiment_outputs/rigorous_reward_hacking
"""
import argparse
import csv
import json
from pathlib import Path
from layoutpilot.evaluate_reward_hacking_rigorous import plot_curves


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, default=None)
    a = p.parse_args(argv)
    with (a.run_dir/'curves.csv').open(newline='', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    numeric = [key for key in data[0] if key not in ('subset',)]
    for row in data:
        for key in numeric:
            row[key] = int(row[key]) if key in ('budget','best_of_n','seed_pairs','templates') else float(row[key])
    metadata = json.loads((a.run_dir/'summary.json').read_text(encoding='utf-8'))
    destination = a.output or a.run_dir/'proxy_gold_vs_sqrt_kl.png'
    plot_curves(data, destination, status=metadata['status'])
    print(destination)


if __name__ == '__main__':
    main()
