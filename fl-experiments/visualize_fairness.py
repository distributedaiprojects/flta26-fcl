from pathlib import Path
import glob
import ast

import numpy as np
import pandas as pd


CONVERGENCE_WINDOW = 11
ROUND_COL = "round"

RESULTS_PATH_NAME = "scenario_2_cifar_360"
BASE_PATH = f"fl-experiments/results/{RESULTS_PATH_NAME}"

LIST_FILENAMES = [
    "312",  # KD-FCL
    "314",  # FedAvg
    "315",  # FedProx
    "331",  # CFeD

    "317",  # KD-FCL
    "319",  # FedAvg
    "320",  # FedProx
    "332",  # CFeD

    "322",  # KD-FCL
    "324",  # FedAvg
    "325",  # FedProx
    "333",  # CFeD 

    "327",  # KD-FCL
    "329",  # FedAvg
    "330",  # FedProx
    "334",  # CFeD
]


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


def load_results(list_filenames, base_path):
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
            df["method"] = label_dictionary.get(str(exp_id), str(exp_id))
            df["run_id"] = run_id
            df["source_file"] = Path(file).name

            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No CSV files found.")

    return pd.concat(all_dfs, ignore_index=True)


def parse_client_accuracies(value):
    if pd.isna(value):
        return []

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return [float(x) for x in parsed]
        except Exception:
            value = value.strip().strip("[]")
            return [float(x) for x in value.split(",") if x.strip()]

    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]

    return [float(value)]


def normalize_accuracies(xs):
    if len(xs) == 0:
        return xs

    # Convert percent scale 0–100 to probability scale 0–1
    if np.nanmax(xs) > 1.0:
        return [x / 100.0 for x in xs]

    return xs


def prepare_dataframe(df):
    df = df.copy()

    if "client_accuracies" not in df.columns:
        raise ValueError("Column 'client_accuracies' not found.")

    if ROUND_COL not in df.columns:
        raise ValueError(
            f"Round column '{ROUND_COL}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    df["client_accuracies_parsed"] = (
        df["client_accuracies"]
        .apply(parse_client_accuracies)
        .apply(normalize_accuracies)
    )

    # Mean client accuracy per round
    df["mean_client_accuracy"] = (
        df["client_accuracies_parsed"]
        .apply(lambda xs: np.mean(xs) if len(xs) > 0 else np.nan)
    )

    # Fairness metric:
    # variance of accuracy scores across clients per round
    df["client_accuracy_variance"] = (
        df["client_accuracies_parsed"]
        .apply(lambda xs: np.var(xs, ddof=1) if len(xs) > 1 else np.nan)
    )

    return df


def calculate_fairness_variance_convergence_window(df):
    per_run_rows = []
    per_round_rows = []

    for (exp_id, method, run_id), run_df in df.groupby(
        ["experiment_id", "method", "run_id"]
    ):
        max_round = run_df[ROUND_COL].max()
        window_start = max_round - CONVERGENCE_WINDOW + 1

        conv_df = run_df[
            run_df[ROUND_COL].between(window_start, max_round)
        ].copy()

        for _, row in conv_df.iterrows():
            per_round_rows.append({
                "experiment_id": exp_id,
                "method": method,
                "run_id": run_id,
                "round": row[ROUND_COL],
                "client_accuracy_variance": row["client_accuracy_variance"],
                "mean_client_accuracy": row["mean_client_accuracy"],
            })

        per_run_rows.append({
            "experiment_id": exp_id,
            "method": method,
            "run_id": run_id,
            "window_start_round": window_start,
            "window_end_round": max_round,
            "mean_fairness_variance": conv_df["client_accuracy_variance"].mean(),
            "mean_accuracy_convergence_window": conv_df["mean_client_accuracy"].mean(),
        })

    per_round_fairness = pd.DataFrame(per_round_rows)
    per_run_fairness = pd.DataFrame(per_run_rows)

    summary_fairness = (
        per_run_fairness
        .groupby(["experiment_id", "method"])
        .agg(
            runs=("run_id", "count"),
            mean_fairness_variance=("mean_fairness_variance", "mean"),
            std_fairness_variance=("mean_fairness_variance", "std"),
            min_fairness_variance=("mean_fairness_variance", "min"),
            max_fairness_variance=("mean_fairness_variance", "max"),
            mean_accuracy_convergence_window=(
                "mean_accuracy_convergence_window",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_fairness["95ci_low"] = (
        summary_fairness["mean_fairness_variance"]
        - 1.96
        * summary_fairness["std_fairness_variance"]
        / np.sqrt(summary_fairness["runs"])
    )

    summary_fairness["95ci_high"] = (
        summary_fairness["mean_fairness_variance"]
        + 1.96
        * summary_fairness["std_fairness_variance"]
        / np.sqrt(summary_fairness["runs"])
    )

    return per_round_fairness, per_run_fairness, summary_fairness


if __name__ == "__main__":
    df = load_results(
        list_filenames=LIST_FILENAMES,
        base_path=BASE_PATH,
    )

    df = prepare_dataframe(df)

    per_round_fairness, per_run_fairness, summary_fairness = (
        calculate_fairness_variance_convergence_window(df)
    )

    print("\nFairness variance per run over convergence window:\n")
    print(per_run_fairness)

    print("\nSummary fairness variance over convergence window:\n")
    print(summary_fairness)

    per_round_fairness.to_csv(
        f"fairness_variance_per_round_convergence_window_{RESULTS_PATH_NAME}.csv",
        index=False,
    )

    per_run_fairness.to_csv(
        f"fairness_variance_per_run_convergence_window_{RESULTS_PATH_NAME}.csv",
        index=False,
    )

    summary_fairness.to_csv(
        f"summary_fairness_variance_convergence_window_{RESULTS_PATH_NAME}.csv",
        index=False,
    )