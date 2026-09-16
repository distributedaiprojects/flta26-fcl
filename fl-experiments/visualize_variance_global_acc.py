from pathlib import Path
import glob
import ast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Better visibility for papers
# -----------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
})


# -----------------------------
# Labels
# -----------------------------
label_dictionary = {
    "312": "KD-FCL",
    "314": "FedAvg",
    "315": "FedProx",
    "331": "CFeD",

    "317": "KD-FCL",
    "319": "FedAvg",
    "320": "FedProx",
    "332": "CFeD",

    "322": "KD-FCL",
    "324": "FedAvg",
    "325": "FedProx",
    "333": "CFeD",

    "327": "KD-FCL",
    "329": "FedAvg",
    "330": "FedProx",
    "334": "CFeD",
}


# -----------------------------
# Fixed colors
# -----------------------------
method_colors = {
    "KD-FCL": "#ff7f0e",   # orange
    "FedAvg": "#817e7e",  # gray
    "FedProx": "#F02525", # red
    "CFeD": "#9467bd",    # purple
}


# -----------------------------
# Load all runs
# -----------------------------
def load_results(
    list_filenames,
    base_path,
):
    all_dfs = []

    for exp_id in list_filenames:

        files = sorted(
            glob.glob(
                f"{base_path}/*_exp{exp_id}_federated_results_automated_tmp.csv"
            )
        )

        print(f"Experiment {exp_id}: {len(files)} runs found")

        for run_id, file in enumerate(files):

            df = pd.read_csv(file)

            df.columns = df.columns.str.strip()

            df["experiment_id"] = str(exp_id)
            df["run_id"] = run_id

            all_dfs.append(df)

    if not all_dfs:
        raise ValueError(
            "No CSV files found. Check base_path and experiment IDs."
        )

    return pd.concat(
        all_dfs,
        ignore_index=True,
    )


# -----------------------------
# Average client accuracy per run
# -----------------------------

def parse_client_accuracies(value):
    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            sanitized = value.strip().strip("[]")
            return [float(x) for x in sanitized.split(",") if x.strip()]

    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)

    return [float(value)]


def get_average_run_accuracies(df):
    df = df.copy()
    df["client_accuracies_parsed"] = (
        df["client_accuracies"]
        .apply(parse_client_accuracies)
    )
    df["average_client_accuracy"] = (
        df["client_accuracies_parsed"]
        .apply(lambda xs: np.mean(xs) if xs else np.nan)
    )

    avg_df = (
        df.groupby(
            ["experiment_id", "run_id"]
        )["average_client_accuracy"]
        .mean()
        .reset_index()
    )

    return avg_df


# -----------------------------
# Density plot
# -----------------------------
def plot_density(avg_df):

    plt.figure(
        figsize=(12, 8)
    )

    for exp_id, exp_df in avg_df.groupby(
        "experiment_id"
    ):

        label = label_dictionary.get(
            exp_id,
            exp_id,
        )

        sns.kdeplot(
            data=exp_df,
            x="average_client_accuracy",
            fill=True,
            linewidth=3,
            alpha=0.35,
            label=label,
            color=method_colors[label],
        )

    plt.xlabel(
        "Average Client Accuracy Across Rounds",
        fontweight="bold",
    )

    plt.ylabel(
        "Density",
        fontweight="bold",
    )

    plt.tick_params(
        width=2,
        length=6,
    )

    plt.legend(
        frameon=True,
        fancybox=True,
    )

    plt.grid(
        alpha=0.2,
    )

    plt.tight_layout()

    plt.savefig(
        "density_plot_average_client_accuracy_5_runs_case_2_cifar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    list_filenames = [
        "327",
        "329",
        "330",
        "334"
    ]

    df = load_results(
        list_filenames=list_filenames,
        base_path="fl-experiments/results/scenario_2_cifar_360/",
    )

    avg_df = get_average_run_accuracies(df)

    print("\nAverage client accuracy per run:\n")

    print(
        avg_df[
            [
                "experiment_id",
                "run_id",
                "average_client_accuracy",
            ]
        ]
    )

    print("\nOverall statistics:\n")

    print(
        avg_df.groupby(
            "experiment_id"
        )["average_client_accuracy"]
        .agg(
            ["mean", "std", "var"]
        )
        .rename(
            index=label_dictionary
        )
    )

    plot_density(
        avg_df
    )