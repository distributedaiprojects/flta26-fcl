from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np


def visualize(list_experiments: list[str]) -> None:
    all_exp_dfs = []

    for experiment_id in list_experiments:
        files = glob.glob(
            f"fl-experiments/results/scenario_6_fmnist_360/*_exp{experiment_id}_federated_results_automated_tmp.csv"
        )

        if not files:
            print(f"Warning: no files found for experiment {experiment_id}")
            continue

        exp_dfs = []

        for file in files:
            df_temp = pd.read_csv(file)
            df_temp["experiment_id"] = str(experiment_id)

            num_clients = int(df_temp["num_selected_clients"].iat[0])

            for clientid in range(num_clients):
                df_temp[f"client_{clientid}_accuracy"] = df_temp["client_accuracies"].apply(
                    lambda a: float(
                        a.split(",", -1)[clientid]
                        .replace("[", "")
                        .replace("]", "")
                        .strip()
                    )
                )

            exp_dfs.append(df_temp)

        all_exp_dfs.append(pd.concat(exp_dfs, ignore_index=True))

    if not all_exp_dfs:
        raise ValueError("No experiment data loaded. Check the glob path and experiment IDs.")

    df = pd.concat(all_exp_dfs, ignore_index=True)
    df.columns = df.columns.str.strip()

    experiment_label_map = {
        "301": "SS-FCL",
        "302": "KD-FCL",
        "303": "SS-KD-FCL",
        "304": "FedAvg",
        "305": "FedProx",
    }

    tmp_df = df[df["experiment_id"].isin(list_experiments)]

    client_labels = [
        f"client_{client_id}_accuracy"
        for client_id in range(int(df["num_selected_clients"].iat[0]))
    ]

    grouped_mean_df = tmp_df.groupby("experiment_id")[client_labels].mean()
    grouped_std_df = tmp_df.groupby("experiment_id")[client_labels].std()

    grouped_mean_df = grouped_mean_df.reindex(list_experiments)
    grouped_std_df = grouped_std_df.reindex(list_experiments)

    grouped_mean_df.index = [
        experiment_label_map.get(exp, exp) for exp in grouped_mean_df.index
    ]
    grouped_std_df.index = [
        experiment_label_map.get(exp, exp) for exp in grouped_std_df.index
    ]

    grouped_mean_df.columns = list(range(len(grouped_mean_df.columns)))
    grouped_std_df.columns = list(range(len(grouped_std_df.columns)))

    output_dir = Path("fl-experiments/results/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    def plot_combined_bubble(
        data_mean: pd.DataFrame,
        data_std: pd.DataFrame,
        title: str,
        cmap: str = "Set2",
    ) -> None:
        x_labels = data_mean.columns
        y_labels = data_mean.index

        mean_values = data_mean.values
        std_values = data_std.values

        mean_range = mean_values.max() - mean_values.min()

        if mean_range == 0:
            sizes = np.full_like(mean_values, 300)
        else:
            sizes = (mean_values - mean_values.min()) / mean_range * 500 + 100

        x_pos, y_pos = np.meshgrid(
            range(len(x_labels)),
            range(len(y_labels))
        )

        x_flat = x_pos.flatten()
        y_flat = y_pos.flatten()
        sizes_flat = sizes.flatten()
        colors_flat = std_values.flatten()

        plt.figure(
            figsize=(
                max(5, len(x_labels) * 0.75),
                max(3, len(y_labels) * 0.6),
            )
        )

        scatter = plt.scatter(
            x_flat,
            y_flat,
            s=sizes_flat,
            c=colors_flat,
            cmap=cmap,
            alpha=0.85,
            edgecolors="none",
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label("Std")

        for idx in range(len(x_flat)):
            mean_val = mean_values.flatten()[idx]
            std_val = std_values.flatten()[idx]

            plt.annotate(
                f"{mean_val:.2f}\n±{std_val:.2f}",
                (x_flat[idx], y_flat[idx]),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

        plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
        plt.yticks(range(len(y_labels)), y_labels)

        plt.xlim(-0.2, len(x_labels) - 0.8)
        plt.ylim(len(y_labels) - 0.8, -0.2)

        plt.xlabel("Clients")
        plt.ylabel("Models")
        # plt.title(title)

        ax = plt.gca()
        ax.margins(x=0.01, y=0.01)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.tight_layout(pad=0.4)

    plot_combined_bubble(
        grouped_mean_df,
        grouped_std_df,
        "Client Accuracy Mean ± Std",
    )

    combined_filename = output_dir / "client_accuracy_mean_std_together_scenario_6_fmnist.png"

    plt.savefig(combined_filename, dpi=200)
    plt.show()
    plt.close()


if __name__ == "__main__":
    list_filenames = ["301", "302", "303", "304", "305"]

    visualize(list_filenames)