"""
Visualize PR-AUC on each client's initial (original) target label set across FL rounds.

Reads client_label_metrics from the CSV — the first entry per client per round
is always the original label set, so this directly measures forgetting over time.

Usage:
    python visualize_initial_label_prauc.py <csv_path> [experiment_id [experiment_id ...]]

If no experiment_id is given, all experiments are plotted.

Example Usage: 
python visualize_initial_label_prauc.py /proj/er-ailml-lcm/code/hints/fl/fl-experiments/results/results_info_2026-03-29_00-47-24_exp228_federated_results_automated_tmp.csv

"""


import sys
import ast
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 0 drift, fmnist, 212, 214
# 0 drift, cifar, 216, 218

# 20 drift, fmnist, 220, 222
# 20 drift, cifar, 224, 226

# 50 drift, fmnist, 228, 230
# 50 drift, cifar, 232, 234

# 80 drift, fmnist, 236, 238
# 80 drift, cifar, 240, 242

# 100 drift, fmnist, 244, 246
# 100 drift, cifar, 248, 250



def parse_metrics(raw):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def load(csv_path, exp_ids=None):
    df = pd.read_csv(csv_path)
    if "client_label_metrics" not in df.columns:
        raise ValueError("Column 'client_label_metrics' not found.")
    df = df.dropna(subset=["client_label_metrics"])
    if exp_ids:
        df = df[df["experiment_id"].isin([int(e) for e in exp_ids])]
    return df

def extract(df):
    """
    Returns:
      { experiment_id -> { client_id -> { 'label_sets':      [[...], ...],  # one per distinct snapshot
                                          'rounds_per_set':  [[...], ...],
                                          'pr_auc_per_set':  [[...], ...] } } }
    """
    # Intermediate: exp -> client -> label_set_tuple -> {rounds, pr_aucs}
    raw = {}
    for _, row in df.sort_values("round").iterrows():
        exp_id = row["experiment_id"]
        round_num = int(row["round"])
        metrics = parse_metrics(row["client_label_metrics"])
        if not metrics:
            continue
        raw.setdefault(exp_id, {})
        for client_id, entries in metrics.items():
            cid = int(client_id)
            if not isinstance(entries, list) or len(entries) == 0:
                continue
            raw[exp_id].setdefault(cid, {})
            for entry in entries:
                key = tuple(entry["labels"])
                raw[exp_id][cid].setdefault(key, {"rounds": [], "pr_auc": []})
                raw[exp_id][cid][key]["rounds"].append(round_num)
                raw[exp_id][cid][key]["pr_auc"].append(entry["precision_recall_auc"])

    # Reorder so index-0 is always the initial (first-seen) label set
    result = {}
    for exp_id, clients in raw.items():
        result[exp_id] = {}
        for cid, sets in clients.items():
            # Sort by first round each label set appeared
            ordered = sorted(sets.items(), key=lambda kv: kv[1]["rounds"][0])
            result[exp_id][cid] = {
                "label_sets":     [list(k) for k, _ in ordered],
                "rounds_per_set": [v["rounds"] for _, v in ordered],
                "pr_auc_per_set": [v["pr_auc"] for _, v in ordered],
            }
    return result


# Fixed colors per label-set index: index 0 = initial (red), rest cycle through these
_COLORS = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
           "#00ACC1", "#F4511E", "#6D4C41", "#546E7A", "#FFB300"]


def plot_experiment(exp_id, clients_data, out_dir):
    num_clients = len(clients_data)
    ncols = 2
    nrows = (num_clients + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows * 3.2), sharex=False)
    axes = axes.flatten()

    for idx, (cid, data) in enumerate(sorted(clients_data.items())):
        ax = axes[idx]
        has_multiple = len(data["label_sets"]) > 1

        for i, (lbl_set, rounds, pr_auc) in enumerate(
            zip(data["label_sets"], data["rounds_per_set"], data["pr_auc_per_set"])
        ):
            # Only plot the most-recent (last) label set in addition to the initial one
            if i > 0 and i != len(data["label_sets"]) - 1:
                continue
            color = _COLORS[i % len(_COLORS)]
            label = f"labels {lbl_set}" + (" (initial)" if i == 0 else " (current)")
            ax.plot(rounds, pr_auc, color=color, linewidth=1.8,
                    linestyle="-" if i == 0 else "--", label=label)

        ax.set_title(f"Client {cid}  |  initial: {data['label_sets'][0]}", fontsize=9)
        ax.set_xlabel("Round", fontsize=8)
        ax.set_ylabel("PR-AUC", fontsize=8)
        ax.set_ylim(0.4, 1.05)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if has_multiple:
            ax.legend(fontsize=7, loc="lower right")

    for idx in range(num_clients, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        f"Experiment {exp_id} — PR-AUC on initial & current target labels per client",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    out_path = out_dir / f"exp{exp_id}_initial_label_prauc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    exp_ids = sys.argv[2:] if len(sys.argv) > 2 else None

    df = load(csv_path, exp_ids)
    all_data = extract(df)

    out_dir = csv_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp_id, clients_data in all_data.items():
        plot_experiment(exp_id, clients_data, out_dir)


if __name__ == "__main__":
    main()
