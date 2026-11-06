import pandas as pd
import numpy as np
import os
from scipy.stats import kruskal, mannwhitneyu, entropy
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import numpy.linalg as npla

def map_cluster(syllable, cluster_map):
    for cluster, sylls in cluster_map.items():
        if syllable in sylls:
            return cluster
    return np.nan

def calc_usage_entropy(syllables):
    probs = syllables.value_counts(normalize=True)
    return entropy(probs, base=2)

def calc_transitions(sequence, valid_clusters, duration_sec):
    if not sequence:
        return {}
        
    bouts = [sequence[0]]
    for c in sequence[1:]:
        if c != bouts[-1]:
            bouts.append(c)
            
    n_trans = len(bouts) - 1
    usage_probs = pd.Series(sequence).value_counts(normalize=True)
    bout_counts = pd.Series(bouts).value_counts()
    
    res = {}
    for i in valid_clusters:
        res[f"bout_freq_{i}"] = bout_counts.get(i, 0) / duration_sec if duration_sec > 0 else 0.0

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
    entropy_rate = (aligned_usage * local_entropies).sum()

    evals = npla.eigvals(trans_matrix.values)
    sorted_evals = np.sort(np.abs(evals))[::-1] 
    eigen2 = sorted_evals[1] if len(sorted_evals) > 1 else np.nan

    res["entropy_rate"] = entropy_rate
    res["eigen2"] = eigen2

    for i in valid_clusters:
        res[f"local_entropy_{i}"] = local_entropies.loc[i]
        for j in valid_clusters:
            if i != j: 
                res[f"bigram_{i}_to_{j}"] = joint_probs.loc[i, j]

    return res

def load_kinematics(data_dir, fps, cluster_map=None):
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSVs found! Double check if your folder path is pointing to the right directory.")
        
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        req_cols = ["syllable", "centroid x", "centroid y", "heading"]
        if not all(c in df.columns for c in req_cols):
            raise KeyError(f"Missing coordinate columns in {f}. Did you export the right tracking file?")
            
        df["velocity_px_s"] = np.sqrt(df["centroid x"].diff()**2 + df["centroid y"].diff()**2).fillna(0) * fps
        df["angular_velocity"] = np.concatenate(([0], np.diff(df["heading"]))) * fps
        df["frame_index"] = df.index
        
        if cluster_map:
            df["cluster"] = df["syllable"].apply(lambda x: map_cluster(x, cluster_map))
            
        df["group"] = f.split("_")[0]
        df.rename(columns={"centroid x": "centroid_x", "centroid y": "centroid_y"}, inplace=True)
        df["name"] = f.replace(".csv", "")
        df = df.dropna(subset=["cluster"])
        dfs.append(df)
        
    return pd.concat(dfs, ignore_index=True)

def build_trial_stats(kinematics, valid_clusters, fps, use_ports):
    res = []
    for (group, name), df in kinematics.groupby(["group", "name"]):
        seq = df["cluster"].tolist()
        duration = len(df) / fps
        
        usage_ent = calc_usage_entropy(pd.Series(seq))
        adv_metrics = calc_transitions(seq, valid_clusters, duration)

        n_trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        tps = n_trans / duration if duration > 0 else np.nan
    
        features = ["velocity_px_s", "angular_velocity", "heading"]

        if use_ports:
            p_cols = [f"RelDist_Port_{l}" for l in "IEABCD"]
            avail = [c for c in p_cols if c in df.columns]
            if len(avail) < len(p_cols):
                raise KeyError(f"Missing port columns for {name}. Turn off the port distances config if this slice doesn't have them.")
            features = avail + features

        c_stats = df.groupby("cluster")[features].agg(["mean", "std", "min", "max"])
        c_stats.columns = ["_".join(c) for c in c_stats.columns]
        c_stats["proportion_time"] = df["cluster"].value_counts(normalize=True)
        c_stats["duration"] = df.groupby("cluster")["frame_index"].count() / fps
        
        for c, row in c_stats.iterrows():
            r_dict = row.to_dict()
            r_dict.update({
                "group": group,
                "name": name,
                "cluster": c,
                "usage_entropy": usage_ent,
                "transitions_per_sec": tps,
                "local_entropy": adv_metrics.get(f"local_entropy_{c}", np.nan),
                "bout_freq": adv_metrics.get(f"bout_freq_{c}", 0.0),
                "entropy_rate": adv_metrics.get("entropy_rate", np.nan),
                "eigen2": adv_metrics.get("eigen2", np.nan)
            })
            
            for i in valid_clusters:
                for j in valid_clusters:
                    if i != j:
                        r_dict[f"bigram_{i}_to_{j}"] = adv_metrics.get(f"bigram_{i}_to_{j}", 0.0)
                        
            res.append(r_dict)
            
    return pd.DataFrame(res)

def run_kruskal_pipeline(stats_df, out_prefix):
    variables = [c for c in stats_df.columns if c not in ["group", "name", "cluster"]]
    clusters = sorted(stats_df["cluster"].unique())
    groups = stats_df["group"].unique()
    
    if len(groups) < 2:
        raise ValueError("Not enough experimental groups to run statistics! You need at least 2 distinct cohorts.")
        
    kw_res, ph_res = [], []

    for v in variables:
        if v in ["usage_entropy", "transitions_per_sec", "entropy_rate", "eigen2"] or v.startswith("bigram_"):
            data = [stats_df[stats_df["group"] == g][v].dropna() for g in groups]
            if all(len(d) > 0 for d in data):
                if pd.concat(data).nunique() <= 1:
                    continue 
                
                stat, p = kruskal(*data)
                kw_res.append({"Cluster": "all", "Variable": v, "Kruskal_statistic": stat, "p_value_uncorrected": p})
                
                for g1, g2 in combinations(groups, 2):
                    d1 = stats_df[stats_df["group"] == g1][v].dropna()
                    d2 = stats_df[stats_df["group"] == g2][v].dropna()
                    if len(d1) > 0 and len(d2) > 0:
                        stat_u, p_u = mannwhitneyu(d1, d2, alternative="two-sided")
                        eff = 1 - (2 * stat_u) / (len(d1) * len(d2))
                        ph_res.append({
                            "Cluster": "all", "Variable": v, "Group1": g1, "Group1_mean": d1.mean(), "n_group1": len(d1),
                            "Group2": g2, "Group2_mean": d2.mean(), "n_group2": len(d2),
                            "U_statistic": stat_u, "Effect_size_r_rb": eff, "p_value_uncorrected": p_u
                        })
        else:
            for c in clusters:
                sub = stats_df[stats_df["cluster"] == c]
                data = [sub[sub["group"] == g][v].dropna() for g in groups]
                if all(len(d) > 0 for d in data):
                    if pd.concat(data).nunique() <= 1:
                        continue 
                    
                    stat, p = kruskal(*data)
                    kw_res.append({"Cluster": c, "Variable": v, "Kruskal_statistic": stat, "p_value_uncorrected": p})
                    
                    for g1, g2 in combinations(groups, 2):
                        d1 = sub[sub["group"] == g1][v].dropna()
                        d2 = sub[sub["group"] == g2][v].dropna()
                        if len(d1) > 0 and len(d2) > 0:
                            stat_u, p_u = mannwhitneyu(d1, d2, alternative="two-sided")
                            eff = 1 - (2 * stat_u) / (len(d1) * len(d2))
                            ph_res.append({
                                "Cluster": c, "Variable": v, "Group1": g1, "Group1_mean": d1.mean(), "n_group1": len(d1),
                                "Group2": g2, "Group2_mean": d2.mean(), "n_group2": len(d2),
                                "U_statistic": stat_u, "Effect_size_r_rb": eff, "p_value_uncorrected": p_u
                            })

    kw_df = pd.DataFrame(kw_res)
    ph_df = pd.DataFrame(ph_res)

    if not kw_df.empty:
        kw_df["p_value_corrected"] = multipletests(kw_df["p_value_uncorrected"], method="fdr_bh")[1]
        kw_df["Significant (FDR<0.05)"] = kw_df["p_value_corrected"] < 0.05
        kw_df.to_csv(f"{out_prefix}kruskal_results.csv", index=False)

    if not ph_df.empty:
        ph_df["p_value_corrected"] = multipletests(ph_df["p_value_uncorrected"], method="fdr_bh")[1]
        ph_df["Significant (FDR<0.05)"] = ph_df["p_value_corrected"] < 0.05
        ph_df.to_csv(f"{out_prefix}posthoc_mannwhitney_fdr.csv", index=False)

def run_analysis(data_dir, out_prefix, use_ports, cluster_map):
    valid_clusters = sorted(list(cluster_map.keys()))
    
    kinematics = load_kinematics(data_dir, fps=30, cluster_map=cluster_map)
    kinematics.to_csv(f"{out_prefix}moseq_combined.csv", index=False)
    
    stats = build_trial_stats(kinematics, valid_clusters, fps=30, use_ports=use_ports)
    stats = stats[stats["cluster"].isin(valid_clusters)]
    stats.to_csv(f"{out_prefix}moseq_stats.csv", index=False)
    
    run_kruskal_pipeline(stats, out_prefix)

if __name__ == "__main__":
    data_directory = "/path/to/your/data/slices"
    export_prefix = "experiment_name_"
    include_location_metrics = False
    
    cohort_map = {
        1: [13],
        2: [9, 19, 0, 4, 10, 15, 3, 1, 16],
        3: [6, 7, 11],
        4: [2, 12, 18],
        5: [5],
        6: [17, 8, 14],
    }
    
    run_analysis(data_directory, export_prefix, include_location_metrics, cohort_map)