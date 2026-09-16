from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# _m2 => select hyperparameters
# _m3 => sample selection, smo with the best parameters, and the combination of both
# file_names = ["_m2", "_m3", "_m4","_m5","_m6"]

label_dictionary={
        "211":"211 - SS-FCL",
        "212":"212 - KD-FCL",
        "213":"213 - SS-KD-FCL",
        "214":"214 - FedAvg",
        "251":"251 - FedProx",

        # "215":"215 - SS-FCL",
        # "216":"216 - KD-FCL",
        # "217":"217 - SS-KD-FCL",
        # "218":"218 - FedAvg",
        # "252":"252 - FedProx",

        "219":"219 - SS-FCL",
        "220":"220 - KD-FCL",
        "221":"221 - SS-KD-FCL",
        "222":"222 - FedAvg",
        "253":"253 - FedProx",

        "223":"223 - SS-FCL",
        "224":"224 - KD-FCL",
        "225":"225 - SS-KD-FCL",
        "226":"226 - FedAvg",
        "254":"254 - FedProx",

        # "227":"227 - SS-FCL",
        # "228":"228 - KD-FCL",
        # "229":"229 - SS-KD-FCL",
        # "230":"230 - FedAvg",
        # "255":"255 - FedProx",

        # "231":"231 - SS-FCL",
        # "232":"232 - KD-FCL",
        # "233":"233 - SS-KD-FCL",
        # "234":"234 - FedAvg",
        # "256":"256 - FedProx",

        # "235":"235 - SS-FCL",
        # "236":"236 - KD-FCL",
        # "237":"237 - SS-KD-FCL",
        # "238":"238 - FedAvg",
        # "257":"257 - FedProx",

        # "239":"239 - SS-FCL",
        # "240":"240 - KD-FCL",
        # "241":"241 - SS-KD-FCL",
        # "242":"242 - FedAvg",
        # "258":"258 - FedProx",

        # "243":"243 - SS-FCL",
        # "244":"244 - KD-FCL",
        # "245":"245 - SS-KD-FCL",
        # "246":"246 - FedAvg",
        # "259":"259 - FedProx",

        # "247":"247 - SS-FCL",
        # "248":"248 - KD-FCL",
        # "249":"249 - SS-KD-FCL",
        # "250":"250 - FedAvg",
        # "260":"260 - FedProx",

        "261":"261 - SS-FCL",
        "262":"262 - KD-FCL",
        "263":"263 - SS-KD-FCL",
        "264":"264 - FedAvg",
        "265":"265 - FedProx",

        "266":"266 - SS-FCL",
        "267":"267 - KD-FCL",
        "268":"268 - SS-KD-FCL",
        "269":"269 - FedAvg",
        "270":"270 - FedProx",

        "271":"271 - SS-FCL",
        "272":"272 - KD-FCL",
        "273":"273 - SS-KD-FCL",
        "274":"274 - FedAvg",
        "275":"275 - FedProx",

        "276":"276 - SS-FCL",
        "277":"277 - KD-FCL",
        "278":"278 - SS-KD-FCL",
        "279":"279 - FedAvg",
        "280":"280 - FedProx",

        "282":"282 - KD Fashion-MNIST",
        "284":"284 - FedAvg Fashion-MNIST",

        "301":"301 - SS-FCL",
        "302":"302 - KD-FCL",
        "303":"303 - SS-KD-FCL",
        "304":"304 - FedAvg",
        "305":"305 - FedProx",

        "306":"306 - SS-FCL",
        "307":"307 - KD-FCL",
        "308":"308 - SS-KD-FCL",
        "309":"309 - FedAvg",
        "310":"310 - FedProx",

        "311":"311 - SS-FCL",
        "312":"312 - KD-FCL",
        "313":"313 - SS-KD-FCL",
        "314":"314 - FedAvg",
        "315":"315 - FedProx",

        "316":"316 - SS-FCL",
        "317":"317 - KD-FCL",
        "318":"318 - SS-KD-FCL",
        "319":"319 - FedAvg",
        "320":"320 - FedProx",

        "321":"321 - SS-FCL",
        "322":"322 - KD-FCL",
        "323":"323 - SS-KD-FCL",
        "324":"324 - FedAvg",
        "325":"325 - FedProx",

        "326":"326 - SS-FCL",
        "327":"327 - KD-FCL",
        "328":"328 - SS-KD-FCL",
        "329":"329 - FedAvg",
        "330":"330 - FedProx",

        # "211":"211 - SS-FCL without drift",
        # "212":"212 - KD-FCL without drift",
        # "213":"213 - SS-KD-FCL without drift",
        # "214":"214 - FedAvg without drift",
        # "251":"251 - FedProx without drift",

        # "215":"215 - SS-FCL without drift",
        # "216":"216 - KD-FCL without drift",
        # "217":"217 - SS-KD-FCL without drift",
        # "218":"218 - FedAvg without drift",
        # "252":"252 - FedProx without drift",

        # "219":"219 - SS-FCL with 20% drift",
        # "220":"220 - KD-FCL with 20% drift",
        # "221":"221 - SS-KD-FCL with 20% drift",
        # "222":"222 - FedAvg with 20% drift",
        # "253":"253 - FedProx with 20% drift",

        # "223":"223 - SS-FCL with 20% drift",
        # "224":"224 - KD-FCL with 20% drift",
        # "225":"225 - SS-KD-FCL with 20% drift",
        # "226":"226 - FedAvg with 20% drift",
        # "254":"254 - FedProx with 20% drift",

        # "227":"227 - SS-FCL with 50% drift",
        # "228":"228 - KD-FCL with 50% drift",
        # "229":"229 - SS-KD-FCL with 50% drift",
        # "230":"230 - FedAvg with 50% drift",
        # "255":"255 - FedProx with 50% drift",

        # "231":"231 - SS-FCL with 50% drift",
        # "232":"232 - KD-FCL with 50% drift",
        # "233":"233 - SS-KD-FCL with 50% drift",
        # "234":"234 - FedAvg with 50% drift",
        # "256":"256 - FedProx with 50% drift",

        # "235":"235 - SS-FCL with 80% drift",
        # "236":"236 - KD-FCL with 80% drift",
        # "237":"237 - SS-KD-FCL with 80% drift",
        # "238":"238 - FedAvg with 80% drift",
        # "257":"257 - FedProx with 80% drift",

        # "239":"239 - SS-FCL with 80% drift",
        # "240":"240 - KD-FCL with 80% drift",
        # "241":"241 - SS-KD-FCL with 80% drift",
        # "242":"242 - FedAvg with 80% drift",
        # "258":"258 - FedProx with 80% drift",

        # "243":"243 - SS-FCL with 100% drift",
        # "244":"244 - KD-FCL with 100% drift",
        # "245":"245 - SS-KD-FCL with 100% drift",
        # "246":"246 - FedAvg with 100% drift",
        # "259":"259 - FedProx with 100% drift",

        # "247":"247 - SS-FCL with 100% drift",
        # "248":"248 - KD-FCL with 100% drift",
        # "249":"249 - SS-KD-FCL with 100% drift",
        # "250":"250 - FedAvg with 100% drift",
        # "260":"260 - FedProx with 100% drift",
    }

def visualize(list_filenames: list[str], list_experiments: list[str], accuracy_label: str, client_id: int = None, drift_rounds: list[int] = None) -> None:
    df_=[]
    all_exp_dfs = []
    # Read and clean data
   
    for d in list_filenames:
        files = glob.glob(f"fl-experiments/results/scenario_8_batching/*_exp{d}_federated_results_automated_tmp.csv")
        df_ = []
        for file in files:
            df_temp = pd.read_csv(file)
            df_temp['experiment_id'] = str(d)
            for clientid in range(df_temp["num_selected_clients"].values[0]):
                df_temp[f'client_{clientid}_accuracy'] = df_temp['client_accuracies'].apply(lambda a: float(a.split(",",-1)[clientid].replace("[","").replace("]","").strip()))
            df_.append(df_temp)
        # client_labels = [f'client_{client_id}_accuracy' for client_id in range(df_temp["num_selected_clients"].values[0])]
        # all_exp_dfs.append(pd.concat(df_).groupby(['experiment_id', 'round'])[[accuracy_label] + client_labels].mean().reset_index())
        all_exp_dfs.append(pd.concat(df_))
    df = pd.concat(all_exp_dfs)
    df.columns = df.columns.str.strip()
    print(df.groupby(["experiment_id","round"])["global_accuracy"].size())

    tmp_df = df[df["experiment_id"].apply(lambda a: a in list_experiments)]
    # tmp_df = tmp_df[['experiment_id','round','global_accuracy']].groupby(['round'])['global_accuracy'].mean().reset_index()
    # tmp_df.to_csv(f"327_without_GL.csv")
    if client_id is not None:
        accuracy_label = f"client_{client_id}_accuracy"
        # tmp_df[accuracy_label] = tmp_df[f"client_{client_id}_accuracy"]
    
    print(tmp_df.groupby(["experiment_id","round"])[accuracy_label].size())
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Line plot with confidence intervals
    tmp_df_sorted = tmp_df.sort_values(by='experiment_id')

    sns.lineplot(data=tmp_df_sorted, x="round", y=accuracy_label, hue=tmp_df_sorted["experiment_id"].map(lambda x: label_dictionary[x].split(' - ', 1)[1] if ' - ' in label_dictionary[x] else label_dictionary[x]), ax=ax, errorbar=('ci', 95), palette=['#1f77b4', '#ff7f0e', '#2ca02c', "#817e7e", "#F02525"], err_kws={'alpha': 0.2})
    ax.set_xlabel("Round")
    ax.set_ylabel(accuracy_label.replace('_',' ').capitalize())
    ax.legend(title='')
    # ax.get_legend().remove()
    # ax.set_ylim(52, 58)
    # ax.set_yticks(range(52, 59, 1))
    # ax.set_xlim(320, 370)
    
    if drift_rounds is not None:
        for drift_round in drift_rounds:
            ax.axvline(drift_round,color="gray",linestyle="dashed")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    list_filenames = ["321", "322","323","324","325"]
    # list_filenames = ["322","324","325"]

    visualize(list_filenames, list_filenames, accuracy_label="global_accuracy", client_id=None, drift_rounds=[40,80,120,160,200,240,280,320]) 