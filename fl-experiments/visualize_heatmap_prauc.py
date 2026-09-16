import glob 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize(list_filenames):
    dfs = []
    scores_dict = {}
    results_df = []
    for key, exp_drift in list_filenames.items():
        dfs = []  # reset dfs for each key
        for d in exp_drift:
            files = glob.glob(f"fl-experiments/results/experiment_results_2026_04_17_ER/*_exp_{d}_ap_scores_per_experiment.csv")
            for file in files:
                df_temp = pd.read_csv(file)
                dfs.append(df_temp)
        if dfs:  # check if any files found
            df = pd.concat(dfs, ignore_index=True).groupby(["experiment_id"])["ap_score"].mean().reset_index()
            scores_dict[key] = df['ap_score'].values
        else:
            print(f"No files found for {key}")
            continue
    
    if not scores_dict:
        print("No data to plot")
        return
    
    # Create DataFrame with drift as index
    results_df = pd.DataFrame(scores_dict).T
    results_df.columns = ['SS-FCL', 'KD-FCL', 'SS-KD-FCL', 'FedAvg', 'FedProx']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(results_df, annot=True, cmap='Blues', fmt='.3f', cbar_kws={'label': 'AP Score'})
    plt.title('AP Scores Heatmap')
    plt.xlabel('Model')
    plt.ylabel('Drift Percentage')
    plt.tight_layout()
    plt.savefig('fl-experiments/results/figures/ap_scores_heatmap_fmnist.png')
    plt.show()
        


if __name__ == "__main__":
    #fmnist
    list_filenames = {"20% drift": ["261", "262", "263", "264","265"], 
                      "80% drift": ["271", "272", "273", "274","275"]}
    
    # cifar
    # list_filenames = {"20% drift": ["266", "267", "268", "269","270"], 
    #                   "80% drift": ["276", "277", "278", "279","280"]}
    visualize(list_filenames)