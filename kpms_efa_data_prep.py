import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
import numpy.linalg as npla

def map_cluster(syllable, cluster_map):
    for c_id, sylls in cluster_map.items():
        if syllable in sylls:
            return c_id
    return np.nan

def calc_usage_entropy(syllable_seq):
    probs = syllable_seq.value_counts(normalize=True)
    return entropy(probs, base=2)

def calc_transition_metrics(seq, valid_clusters, duration_sec):
    if not seq:
        return {}
        
    # Collapse consecutive identical clusters to isolate distinct behavioral bouts
    bouts = [seq[0]]
    for c in seq[1:]:
        if c != bouts[-1]:
            bouts.append(c)
            
    n_trans = len(bouts) - 1
    usage_probs = pd.Series(seq).value_counts(normalize=True)
    bout_counts = pd.Series(bouts).value_counts()
    
    res = {}
    for i in valid_clusters:
        res[f"C{i}_bout_freq"] = bout_counts.get(i, 0) / duration_sec if duration_sec > 0 else 0.0

    if n_trans <= 0:
        return res

    trans_df = pd.DataFrame(list(zip(bouts[:-1], bouts[1:])), columns=['from', 'to'])
    counts = pd.crosstab(
        pd.Categorical(trans_df['from'], categories=valid_clusters),
        pd.Categorical(trans_df['to'], categories=valid_clusters),
        dropna=False
    )

    joint_probs = counts / n_trans
    row_sums = counts.sum(axis=1)
    trans_matrix = counts.div(row_sums, axis=0).fillna(0)

    local_entropies = trans_matrix.apply(lambda r: entropy(r, base=2), axis=1)
    aligned_usage = usage_probs.reindex(valid_clusters).fillna(0)
    
    res["entropy_rate"] = (aligned_usage * local_entropies).sum()
    
    evals = npla.eigvals(trans_matrix.values)
    sorted_evals = np.sort(np.abs(evals))[::-1] 
    res["eigen2"] = sorted_evals[1] if len(sorted_evals) > 1 else np.nan

    for i in valid_clusters:
        res[f"C{i}_local_entropy"] = local_entropies.loc[i]
        for j in valid_clusters:
            if i != j: 
                res[f"bigram_C{i}_to_C{j}"] = joint_probs.loc[i, j]

    return res

def build_efa_dataset(data_dir, out_prefix, fps, cohort_meta, cluster_map):
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        raise ValueError(f"No CSVs found in {data_dir}! Double check the directory path.")
        
    all_trials = []
    valid_clusters = sorted(list(cluster_map.keys()))

    for f in files:
        f_path = os.path.join(data_dir, f)
        trial_data = pd.read_csv(f_path)
        
        req_cols = ["syllable", "centroid x", "centroid y", "heading"]
        if not all(c in trial_data.columns for c in req_cols):
            raise KeyError(f"Missing coordinate columns in {f}. Did you export the right tracking file?")

        trial_data["velocity_px_s"] = np.sqrt(trial_data["centroid x"].diff()**2 + trial_data["centroid y"].diff()**2).fillna(0) * fps
        trial_data["angular_velocity"] = np.concatenate(([0], np.diff(trial_data["heading"]))) * fps
        trial_data["cluster"] = trial_data["syllable"].apply(lambda x: map_cluster(x, cluster_map))
        trial_data = trial_data.dropna(subset=["cluster"]) 

        if trial_data.empty:
            raise ValueError(f"No data left for {f}! Check if the syllable mapping is correct or if the mouse was off-camera.")

        seq = trial_data["cluster"].tolist()
        duration = len(trial_data) / fps
        
        n_trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        adv_metrics = calc_transition_metrics(seq, valid_clusters, duration)

        features = ["RelDist_Port_I", "RelDist_Port_E", "RelDist_Port_C", 
                   "RelDist_Port_A", "RelDist_Port_B", "RelDist_Port_D", 
                   "velocity_px_s", "angular_velocity", "heading"]
        avail_features = [m for m in features if m in trial_data.columns]

        c_stats = trial_data.groupby("cluster")[avail_features].agg(["mean", "std", "min", "max"])
        c_stats.columns = ["_".join(c) for c in c_stats.columns]
        c_stats["proportion_time"] = trial_data["cluster"].value_counts(normalize=True)
        c_stats["duration"] = trial_data.groupby("cluster").size() / fps
        
        mouse_id = f.replace(".csv", "").split("_", 1)[-1] if "_" in f else f.replace(".csv", "")
            
        row = {"mouse_id": mouse_id}
        row.update(cohort_meta)
        
        row.update({
            "usage_entropy": calc_usage_entropy(pd.Series(seq)),
            "transitions_per_sec": n_trans / duration if duration > 0 else np.nan,
            "entropy_rate": adv_metrics.get("entropy_rate", np.nan),
            "eigen2": adv_metrics.get("eigen2", np.nan)
        })
        
        for k, v in adv_metrics.items():
            if k not in ["entropy_rate", "eigen2"]: 
                row[k] = v

        for c_id, c_data in c_stats.iterrows():
            for metric, val in c_data.items():
                row[f"C{int(c_id)}_{metric}"] = val

        all_trials.append(row)

    if not all_trials:
        raise ValueError("No valid trials survived processing! Double check your input directory and filtering logic.")

    batch_features = pd.DataFrame(all_trials)
    out_path = f"{out_prefix}efa_restructured.csv"

    if os.path.exists(out_path):
        master = pd.read_csv(out_path)
        combined = pd.concat([master, batch_features], ignore_index=True)
    else:
        combined = batch_features

    combined.to_csv(out_path, index=False)


if __name__ == "__main__":
    # --- Configuration ---
    data_dir = "/path/to/your/data/efa_batch"
    out_prefix = "/path/to/your/data/output/experiment_name_"
    
    # Defines metadata appended to all trials in this run
    cohort_meta = {
        "performance": "good",
        "train_test": "test",
        "trialoutcome": "error",
        "trialtype": "CD"
    }
    
    cluster_mapping = {
        1: [13],
        2: [9, 19, 0, 4, 10, 15, 3, 1, 16],
        3: [6, 7, 11],
        4: [2, 12, 18],
        5: [5],
        6: [17, 8, 14],
    }
    
    build_efa_dataset(data_dir, out_prefix, fps=30, cohort_meta=cohort_meta, cluster_map=cluster_mapping)