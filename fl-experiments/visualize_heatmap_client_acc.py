from pathlib import Path
import glob

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


def visualize(list_experiments: list[str]) -> None:
    all_exp_dfs = []

    for experiment_id in list_experiments:
        files = glob.glob(
            f"fl-experiments/results/scenario_2_cifar_360/*_exp{experiment_id}_federated_results_automated_tmp.csv"
        )

        if not files:
            print(f"Warning: no files found for experiment {experiment_id}")
            continue

        exp_dfs = []

        for file in files:
            df_temp = pd.read_csv(file)
            df_temp.columns = df_temp.columns.str.strip()

            df_temp["experiment_id"] = str(experiment_id)

            num_clients = int(df_temp["num_selected_clients"].iat[0])

            def parse_client_accuracy(a: any, clientid: int) -> float:
                if pd.isna(a):
                    return float("nan")
                if not isinstance(a, str):
                    a = str(a)
                parts = [p.strip() for p in a.strip().strip("[]").split(",") if p.strip() != ""]
                if clientid < 0 or clientid >= len(parts):
                    return float("nan")
                try:
                    return float(parts[clientid])
                except ValueError:
                    return float("nan")

            for clientid in range(num_clients):
                df_temp[f"client_{clientid}_accuracy"] = (
                    df_temp["client_accuracies"].apply(
                        lambda a: parse_client_accuracy(a, clientid)
                    )
                )

            exp_dfs.append(df_temp)

        all_exp_dfs.append(
            pd.concat(exp_dfs, ignore_index=True)
        )

    if not all_exp_dfs:
        raise ValueError(
            "No experiment data loaded. Check the glob path and experiment IDs."
        )

    df = pd.concat(all_exp_dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    experiment_label_map = {
        "311": "SS-FCL",
        "312": "KD-FCL",
        "313": "SS-KD-FCL",
        "314": "FedAvg",
        "315": "FedProx",
        "316": "SS-FCL",
        "317": "KD-FCL",
        "318": "SS-KD-FCL",
        "319": "FedAvg",
        "320": "FedProx",
        "321": "SS-FCL",
        "322": "KD-FCL",
        "323": "SS-KD-FCL",
        "324": "FedAvg",
        "325": "FedProx",
        "326": "SS-FCL",
        "327": "KD-FCL",
        "328": "SS-KD-FCL",
        "329": "FedAvg",
        "330": "FedProx",
        "331": "CFeD",
        "332": "CFeD",
        "333": "CFeD",
        "334": "CFeD",
    }

    tmp_df = df[df["experiment_id"].isin(list_experiments)]

    client_labels = [
        f"client_{client_id}_accuracy"
        for client_id in range(
            int(df["num_selected_clients"].iat[0])
        )
    ]

    grouped_mean_df = (
        tmp_df.groupby("experiment_id")[client_labels]
        .mean()
        .reindex(list_experiments)
    )

    grouped_std_df = (
        tmp_df.groupby("experiment_id")[client_labels]
        .std()
        .reindex(list_experiments)
    )

    grouped_mean_df.index = [
        experiment_label_map.get(exp, exp)
        for exp in grouped_mean_df.index
    ]

    grouped_std_df.index = [
        experiment_label_map.get(exp, exp)
        for exp in grouped_std_df.index
    ]

    grouped_mean_df.columns = list(
        range(len(grouped_mean_df.columns))
    )

    grouped_std_df.columns = list(
        range(len(grouped_std_df.columns))
    )

    output_dir = Path("fl-experiments/results/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_combined_heatmap(
        data_mean: pd.DataFrame,
        data_std: pd.DataFrame,
        title: str,
        filename: str,
        cmap: str = "coolwarm",
        transpose: bool = False,
    ) -> None:

        if transpose:
            data_mean = data_mean.T
            data_std = data_std.T

        annot_labels = data_mean.copy().astype(str)

        for j in range(data_mean.shape[1]):
            col_max = data_mean.iloc[:, j].max()

            for i in range(data_mean.shape[0]):
                mean_val = data_mean.iloc[i, j]
                std_val = data_std.iloc[i, j]

                if mean_val == col_max:
                    annot_labels.iloc[i, j] = (
                        rf"$\bf{{{mean_val:.2f}}}$"
                        + f"\n±{std_val:.2f}"
                    )
                else:
                    annot_labels.iloc[i, j] = (
                        f"{mean_val:.2f}\n±{std_val:.2f}"
                    )

        plt.figure(
            figsize=(
                max(14, data_mean.shape[1] * 1.0),
                max(5, data_mean.shape[0] * 1.1),
            )
        )

        ax = sns.heatmap(
            data_mean,
            annot=annot_labels,
            fmt="",
            cmap=cmap,
            square=False,
            linewidths=1.2,
            linecolor="white",
            annot_kws={
                "fontsize": 15,   
                "color": "black",
            },
            cbar_kws={
                "label": "Mean Accuracy",
                "shrink": 0.9,
            },
        )

        ax.set_aspect("auto")
        ax.collections[0].set_alpha(0.70)

        plt.xlabel(
            "Clients" if not transpose else "Models",
            fontsize=20,
            fontweight="bold",
        )

        plt.ylabel(
            "Models" if not transpose else "Clients",
            fontsize=20,
            fontweight="bold",
        )

        plt.xticks(
            fontsize=18,
            fontweight="bold",
            rotation=0 if not transpose else 45,
        )

        plt.yticks(
            fontsize=18,
            fontweight="bold",
            rotation=0,
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(
            labelsize=16,
            width=2,
            length=6,
        )

        cbar.set_label(
            "Mean Accuracy",
            fontsize=18,
            fontweight="bold",
        )

        # Optional title
        # plt.title(
        #     title,
        #     fontsize=22,
        #     fontweight="bold",
        #     pad=10,
        # )

        plt.tight_layout()

        plt.savefig(
            output_dir / filename,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        plt.show()
        plt.close()

    plot_combined_heatmap(
        grouped_mean_df,
        grouped_std_df,
        "Client Accuracy Mean ± Std",
        "client_accuracy_mean_std_heatmap_scenario_2_cifar_360.png",
        transpose=False,
    )


if __name__ == "__main__":
    list_filenames = [
        "326",
        "327",
        "328",
        "329",
        "330",
        "334"
    ]

    visualize(list_filenames)