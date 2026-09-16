from pathlib import Path
import glob
import ast

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


label_dictionary = {
    "311": "311 - SS-FCL",
    "312": "312 - KD-FCL",
    "313": "313 - SS-KD-FCL",
    "314": "314 - FedAvg",
    "315": "315 - FedProx",
    "331": "331 - CFeD",

    "316": "316 - SS-FCL",
    "317": "317 - KD-FCL",
    "318": "318 - SS-KD-FCL",
    "319": "319 - FedAvg",
    "320": "320 - FedProx",
    "332": "332 - CFeD",

    "321": "321 - SS-FCL",
    "322": "322 - KD-FCL",
    "323": "323 - SS-KD-FCL",
    "324": "324 - FedAvg",
    "325": "325 - FedProx",
    "333": "333 - CFeD",

    "326": "326 - SS-FCL",
    "327": "327 - KD-FCL",
    "328": "328 - SS-KD-FCL",
    "329": "329 - FedAvg",
    "330": "330 - FedProx",
    "334": "334 - CFeD",

}


PALETTE = {
    "SS-FCL": "#1f77b4",
    "KD-FCL": "#ff7f0e",
    "SS-KD-FCL": "#2ca02c",
    "FedAvg": "#000000",
    "FedProx": "#d62728",
    "CFeD": "#9467bd"
}


METHOD_ORDER = [
    "SS-FCL",
    "KD-FCL",
    "SS-KD-FCL",
    "FedAvg",
    "FedProx",
    "CFeD"
]


def parse_client_accuracies(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [float(v) for v in parsed]
            return [float(parsed)]
        except Exception:
            try:
                return [
                    float(v.strip())
                    for v in value.replace("[", "").replace("]", "").split(",")
                    if v.strip() != ""
                ]
            except Exception:
                return []

    return []


def add_client_accuracy_columns(df_temp):
    client_lists = df_temp["client_accuracies"].apply(parse_client_accuracies)
    max_clients = client_lists.apply(len).max()

    if pd.isna(max_clients) or max_clients == 0:
        return df_temp

    for cid in range(int(max_clients)):
        df_temp[f"client_{cid}_accuracy"] = client_lists.apply(
            lambda values: values[cid] if cid < len(values) else None
        )

    return df_temp


def visualize(
    list_filenames,
    list_experiments,
    accuracy_label="global_accuracy",
    client_id=None,
    drift_rounds=None,
    title="",
    output_name="figure",
    xlim=None,
    ylim=None,
    show_legend=True,
    legend_above=True,
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    all_exp_dfs = []

    for exp_id in list_filenames:
        files = glob.glob(
            f"fl-experiments/results/scenario_2_cifar_360/*_exp{exp_id}_federated_results_automated_tmp.csv"
        )

        if len(files) == 0:
            print(f"No files found for experiment {exp_id}")
            continue

        for file in files:
            df_temp = pd.read_csv(file)
            df_temp.columns = df_temp.columns.str.strip()
            df_temp["experiment_id"] = str(exp_id)

            if "client_accuracies" in df_temp.columns:
                df_temp = add_client_accuracy_columns(df_temp)

            all_exp_dfs.append(df_temp)

    if len(all_exp_dfs) == 0:
        raise ValueError("No CSV files found.")

    df = pd.concat(all_exp_dfs, ignore_index=True)
    df["experiment_id"] = df["experiment_id"].astype(str)

    tmp_df = df[df["experiment_id"].isin(list_experiments)].copy()

    if client_id is not None:
        accuracy_label = f"client_{client_id}_accuracy"

    tmp_df["method"] = tmp_df["experiment_id"].map(
        lambda x: label_dictionary.get(str(x), f"{x} - Unknown").split(" - ", 1)[1]
    )

    tmp_df = tmp_df.sort_values(["method", "round"])

    fig, ax = plt.subplots(figsize=(4.4, 3.1))

    sns.lineplot(
        data=tmp_df,
        x="round",
        y=accuracy_label,
        hue="method",
        hue_order=METHOD_ORDER,
        palette=PALETTE,
        errorbar=("ci", 95),
        linewidth=1.9,
        err_kws={"alpha": 0.12},
        ax=ax,
    )

    if drift_rounds is not None:
        for drift_round in drift_rounds:
            ax.axvline(
                drift_round,
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.6,
                zorder=0,
            )

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Communication round")

    if client_id is None:
        ax.set_ylabel("Global accuracy (%)")
    else:
        ax.set_ylabel(f"Client {client_id} accuracy (%)")

    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.5)

    sns.despine()

    if show_legend:
        if legend_above:
            ax.legend(
                title=None,
                frameon=False,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=3,
                handlelength=1.5,
                columnspacing=0.8,
            )
        else:
            ax.legend(
                title=None,
                frameon=False,
                loc="lower right",
                ncol=2,
                handlelength=1.7,
                columnspacing=0.8,
            )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    plt.tight_layout()

    Path("figures").mkdir(exist_ok=True)

    fig.savefig(f"figures/{output_name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{output_name}.png", dpi=600, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":

    list_filenames = ["326", "327", "328", "329", "330", "334"]

    drift_rounds = [40, 80, 120, 160, 200, 240, 280, 320]

    # visualize(
    #     list_filenames=list_filenames,
    #     list_experiments=list_filenames,
    #     accuracy_label="global_accuracy",
    #     client_id=None,
    #     drift_rounds=drift_rounds,
    #     title="",
    #     output_name="case_2_fmnist_global_accuracy_full",
    #     show_legend=True,
    #     legend_above=True,
    # )

    visualize(
        list_filenames=list_filenames,
        list_experiments=list_filenames,
        accuracy_label="global_accuracy",
        client_id=None,
        drift_rounds=drift_rounds,
        title="",
        output_name="case_2_cifar_global_accuracy_zoomed",
        xlim=(300, 360),
        ylim=(23, 35),
        show_legend=False,
    )