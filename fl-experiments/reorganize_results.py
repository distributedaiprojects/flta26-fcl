"""
Reorganize flat result files in the results directory into folders.

results_info_2026-04-16_10-54-35_exp271_federated_results_automated_tmp.csv
  -> results_info_2026-04-16_10-54-35_exp271/federated_results_automated_tmp.csv

results_info_2026-04-16_10-54-35_exp271.txt
  -> results_info_2026-04-16_10-54-35_exp271/results_info.txt
"""

import os
import re
import shutil
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results/scenario_8_fmnist"

# Match: results_info_<datetime>_exp<num>_<rest>
PATTERN = re.compile(r"^(results_info_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_exp\d+)_(.+)$")
# Match: results_info_<datetime>_exp<num>.txt (no extra suffix)
TXT_PATTERN = re.compile(r"^(results_info_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_exp\d+)(\.txt)$")

moved = 0
for f in sorted(RESULTS_DIR.iterdir()):
    if not f.is_file():
        continue

    m = PATTERN.match(f.name)
    if m:
        folder_name, rest = m.group(1), m.group(2)
    else:
        m2 = TXT_PATTERN.match(f.name)
        if m2:
            folder_name = m2.group(1)
            rest = "results_info" + m2.group(2)
        else:
            continue

    dest_dir = RESULTS_DIR / folder_name
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / rest
    shutil.move(str(f), str(dest))
    print(f"  {f.name}  ->  {folder_name}/{rest}")
    moved += 1

print(f"\nMoved {moved} files.")
