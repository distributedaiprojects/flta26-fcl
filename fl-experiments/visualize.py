"""
Overlay per-client accuracy from two experiments on the same plot.

Usage:
    python visualize_compare.py <exp_id_1> <exp_id_2>
    python visualize_compare.py 261 262
"""

import sys
import re
import ast
import glob
import matplotlib.pyplot as plt
from pathlib import Path
RESULTS_DIR = "/Users/milenaangelova/git-repo/hints/fl-experiments/results/scenario_8_fmnist"

def parse_log(log_path):
    import csv as csvmod
    clients_data = {}
    with open(log_path, newline="") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            round_num = int(row["round"])
            raw = row.get("client_label_metrics", "")
            if not raw:
                continue
            try:
                metrics = ast.literal_eval(raw)
            except Exception:
                continue
            for client_id, entries in metrics.items():
                cid = int(client_id)
                clients_data.setdefault(cid, {})
                for entry in entries:
                    key = tuple(entry["labels"])
                    clients_data[cid].setdefault(key, {"rounds": [], "accuracy": []})
                    clients_data[cid][key]["rounds"].append(round_num)
                    clients_data[cid][key]["accuracy"].append(entry["accuracy"])
    return clients_data


def classify_label_sets(clients_data):
    result = {}
    for cid, sets in clients_data.items():
        ordered = sorted(sets.items(), key=lambda kv: kv[1]["rounds"][0])
        classified = []
        for i, (labels, data) in enumerate(ordered):
            classified.append({
                "labels": list(labels),
                "kind": "old" if i == 0 else "new",
                "data": data,
            })
        result[cid] = classified
    return result


def find_latest_result(exp_id):
    pattern = RESULTS_DIR+"/results_info_*_exp"+str(exp_id)
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(f"No results found for exp{exp_id}")
    folder = Path(matches[-1])
    csv = folder / "federated_results_automated_tmp.csv"
    if not csv.exists():
        sys.exit(f"CSV not found: {csv}")
    return csv


def plot_compare(clients1, clients2, exp1, exp2, out_path):
    all_cids = sorted(set(clients1) | set(clients2))
    ncols = 2
    nrows = (len(all_cids) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5), squeeze=False)
    axes = axes.flatten()

    for idx, cid in enumerate(all_cids):
        ax = axes[idx]
        for clients, exp_id, color_old, color_new, ls in [
            (clients1, exp1, "#E53935", "#FF8A80", "-"),
            (clients2, exp2, "#1E88E5", "#82B1FF", "--"),
        ]:
            if cid not in clients:
                continue
            for entry in clients[cid]:
                d = entry["data"]
                tag = entry["kind"]
                color = color_old if tag == "old" else color_new
                ax.plot(d["rounds"], d["accuracy"], color=color, linewidth=1.5,
                        linestyle=ls, marker="o", markersize=2,
                        label=f"exp{exp_id} {tag} {entry['labels']}")

        ax.set_title(f"Client {cid}", fontsize=9)
        ax.set_xlabel("Round", fontsize=8)
        ax.set_ylabel("Accuracy (%)", fontsize=8)
        ax.set_ylim(0, 105)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=5, loc="lower right")

    for idx in range(len(all_cids), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Client Accuracy: exp{exp1} vs exp{exp2}", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# 282 - KD Fashion-MNIST
# 284 - FedAvg Fashion-MNIST

# 287 - KD CIFAR10
# 289 - FedAvg CIFAR10

exp1, exp2 = "322","324"

csv1 = find_latest_result(exp1)
csv2 = find_latest_result(exp2)

clients1 = classify_label_sets(parse_log(csv1))
clients2 = classify_label_sets(parse_log(csv2))

out_dir = Path(RESULTS_DIR+"/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / f"exp_{exp1}_vs_{exp2}_client_accuracy.png"
plot_compare(clients1, clients2, exp1, exp2, out_path)


