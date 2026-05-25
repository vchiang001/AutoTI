"""
Supplementary Behavioral Keypoint-MoSeq Statistics Pipeline
Tested on Python 3.8+, pandas 2.1+, scipy 1.11+, and statsmodels 0.14+
"""
import os
import numpy as np
import pandas as pd
import numpy.linalg as npla
from scipy.stats import kruskal, mannwhitneyu, entropy
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# --- Configuration ---
RAW_DATA_DIR = "data/raw_moseq_slices"
OUT_PREFIX = "results/kpms_rq3_"

# Group raw syllables into generalized behaviors
SYL_MAP = {
    1: [13],
    2: [9, 19, 0, 4, 10, 15, 3, 1, 16],
    3: [6, 7, 11],
    4: [2, 12, 18],
    5: [5],
    6: [17, 8, 14],
}

FPS = 30
CORR_THRESH = 0.90
KW_ALPHA = 0.05
FDR_Q = 0.10

# Organizes surviving variables into categories for hierarchical FDR correction
CAT_RULES = {
    "Usage": ["proportion", "duration", "bout"],
    "Entropy": ["entropy"],
    "Kinematic": ["velocity", "heading"],
    "Transition": ["bigram", "transitions"],
}

# --- 1. Feature Extraction & Mathematics ---
def map_syl(syl, cluster_map):
    for c_label, s_list in cluster_map.items():
        if syl in s_list: 
            return c_label
    return np.nan

def calc_adv_metrics(seq, valid_clusts, dur_sec):
    if not seq: 
        return {}

    # Collapse consecutive identical clusters to analyze transitions
    collapsed = [seq[0]]
    for c in seq[1:]:
        if c != collapsed[-1]: 
            collapsed.append(c)

    total_trans = len(collapsed) - 1
    usage = pd.Series(seq).value_counts(normalize=True)
    bout_counts = pd.Series(collapsed).value_counts()

    metrics = {f"bout_freq_{i}": bout_counts.get(i, 0) / dur_sec if dur_sec > 0 else 0.0 for i in valid_clusts}

    if total_trans <= 0: 
        return metrics

    trans_df = pd.DataFrame(list(zip(collapsed[:-1], collapsed[1:])), columns=['from', 'to'])
    counts = pd.crosstab(
        pd.Categorical(trans_df['from'], categories=valid_clusts),
        pd.Categorical(trans_df['to'], categories=valid_clusts), 
        dropna=False
    )

    joint_probs = counts / total_trans
    row_sums = counts.sum(axis=1)
    trans_mat = counts.div(row_sums, axis=0).fillna(0)

    loc_ent = trans_mat.apply(lambda r: entropy(r, base=2), axis=1)
    metrics["entropy_rate"] = (usage.reindex(valid_clusts).fillna(0) * loc_ent).sum()

    evals = npla.eigvals(trans_mat.values)
    sorted_evals = np.sort(np.abs(evals))[::-1]
    metrics["eigen2"] = sorted_evals[1] if len(sorted_evals) > 1 else np.nan

    for i in valid_clusts:
        metrics[f"local_entropy_{i}"] = loc_ent.loc[i]
        for j in valid_clusts:
            if i != j: 
                metrics[f"bigram_{i}_to_{j}"] = joint_probs.loc[i, j]

    return metrics

# --- 2. Data Loading & Trial Construction ---
def load_moseq_trials(folder, cluster_map, fps):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    if not files: 
        raise ValueError("We couldn't find any CSV files in the raw data directory!")

    all_trials = []
    for f in files:
        df = pd.read_csv(os.path.join(folder, f))
        if not all(col in df.columns for col in ["syllable", "centroid x", "centroid y", "heading"]):
            continue

        df["velocity_px_s"] = np.sqrt(df["centroid x"].diff()**2 + df["centroid y"].diff()**2).fillna(0) * fps
        df["angular_velocity"] = np.concatenate(([0], np.diff(df["heading"]))) * fps
        df["frame_index"] = df.index
        df["cluster"] = df["syllable"].apply(lambda x: map_syl(x, cluster_map))
        
        df["group"] = f.split("_")[0]
        df["name"] = f.replace(".csv", "")
        df.rename(columns={"centroid x": "centroid_x", "centroid y": "centroid_y"}, inplace=True)

        all_trials.append(df.dropna(subset=["cluster"]))

    if not all_trials: 
        raise ValueError("No valid trials remained after parsing the tracking columns! Check the MoSeq outputs.")
        
    return pd.concat(all_trials, ignore_index=True)

def calc_trial_metrics(track_data, fps):
    valid_clusts = sorted(track_data["cluster"].dropna().unique().tolist())
    results = []

    for (grp, mouse), t_df in track_data.groupby(["group", "name"]):
        seq = t_df["cluster"].tolist()
        dur_sec = len(t_df) / fps
        syl_probs = pd.Series(seq).value_counts(normalize=True)
        usg_ent = entropy(syl_probs, base=2)

        adv_metrics = calc_adv_metrics(seq, valid_clusts, dur_sec)

        total_trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        trans_sec = total_trans / dur_sec if dur_sec > 0 else np.nan

        # Safely handle port distances if present in the data
        kinematics = ["RelDist_Port_I", "RelDist_Port_E", "RelDist_Port_A", "RelDist_Port_B", "RelDist_Port_C", "RelDist_Port_D", "velocity_px_s", "angular_velocity", "heading"]
        avail_cols = [c for c in kinematics if c in t_df.columns]
        
        c_stats = t_df.groupby("cluster")[avail_cols].agg(["mean", "std", "min", "max"])
        c_stats.columns = ["_".join(col) for col in c_stats.columns]

        c_stats["proportion_time"] = t_df["cluster"].value_counts(normalize=True)
        c_stats["duration"] = t_df.groupby("cluster")["frame_index"].count() / fps

        for clust, row in c_stats.iterrows():
            row_dict = row.to_dict()
            row_dict.update({
                "group": grp, 
                "name": mouse, 
                "cluster": clust,
                "usage_entropy": usg_ent, 
                "transitions_per_sec": trans_sec,
                "local_entropy": adv_metrics.get(f"local_entropy_{clust}", np.nan),
                "bout_freq": adv_metrics.get(f"bout_freq_{clust}", 0.0),
                "entropy_rate": adv_metrics.get("entropy_rate", np.nan),
                "eigen2": adv_metrics.get("eigen2", np.nan)
            })

            for i in valid_clusts:
                for j in valid_clusts:
                    if i != j: 
                        row_dict[f"bigram_{i}_to_{j}"] = adv_metrics.get(f"bigram_{i}_to_{j}", 0.0)

            results.append(row_dict)

    return pd.DataFrame(results)

# --- 3. Animal-Level Aggregation ---
def aggregate_to_mouse(trial_stats, grouping_vars):
    df = trial_stats.copy()
    
    # Isolate animal ID from file string (assumes prefix formatting like Group_MouseID_Trial)
    df['name'] = df['name'].apply(lambda x: '_'.join(str(x).split('_')[:2]))

    num_cols = [c for c in df.select_dtypes(include='number').columns if c not in grouping_vars]
    txt_cols = [c for c in df.columns if c not in num_cols and c not in grouping_vars]

    agg_rules = {c: 'mean' for c in num_cols}
    agg_rules.update({c: 'first' for c in txt_cols})

    return df.groupby(grouping_vars).agg(agg_rules).reset_index()[df.columns]

# --- 4. Multi-Group Statistics & FDR Correction ---
def run_multi_stats(mouse_stats, corr_thresh, kw_alpha, fdr_q, cat_rules, out_prefix):
    exclude_cols = ["group", "name", "cluster"]
    num_vars = [c for c in mouse_stats.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(mouse_stats[c])]

    # 4A. Drop redundant features to reduce multi-test penalty
    corr_mat = mouse_stats[num_vars].corr(method='spearman').abs()
    up_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = []

    for c in up_tri.columns:
        if c in to_drop: 
            continue
        high_corrs = up_tri.index[up_tri[c] > corr_thresh].tolist()
        for hc in high_corrs:
            if hc in to_drop: 
                continue
            # Keep aggregate means over raw extremes, or keep the metric with higher variance
            if "mean" in c and "mean" not in hc: 
                kept, dropped = c, hc
            elif "mean" in hc and "mean" not in c: 
                kept, dropped = hc, c
            else:
                kept, dropped = (c, hc) if mouse_stats[c].var() > mouse_stats[hc].var() else (hc, c)
            if dropped not in to_drop: 
                to_drop.append(dropped)

    clean_df = mouse_stats.drop(columns=to_drop)
    surviving_vars = [c for c in num_vars if c not in to_drop]

    # 4B. Categorize surviving metrics
    cat_map = {cat: [] for cat in cat_rules.keys()}
    cat_map["Other"] = []

    for var in surviving_vars:
        assigned = False
        for cat, subs in cat_rules.items():
            if any(sub.lower() in var.lower() for sub in subs):
                cat_map[cat].append(var)
                assigned = True
                break
        if not assigned:
            cat_map["Other"].append(var)

    clusters = sorted(clean_df["cluster"].dropna().unique())
    cohorts = clean_df["group"].dropna().unique()

    all_kw, all_mw = [], []
    trial_wide_tags = ["entropy", "transitions_per_sec", "usage_entropy", "entropy_rate", "eigen2", "bigram"]

    # 4C. Execute Statistical Testing
    for cat, variables in cat_map.items():
        if not variables: 
            continue

        for var in variables:
            is_trial_wide = any(tag in var.lower() for tag in trial_wide_tags)

            if is_trial_wide:
                # Trial-wide metrics are calculated once per mouse
                mouse_avg = clean_df.groupby(["group", "name"])[var].mean().reset_index()
                scores = [mouse_avg[mouse_avg["group"] == g][var].dropna() for g in cohorts]

                if all(len(d) > 0 for d in scores):
                    if pd.concat(scores).nunique() <= 1: 
                        continue

                    stat, p_kw = kruskal(*scores)
                    all_kw.append({"Category": cat, "Cluster": "all", "Variable": var, "Kruskal_statistic": stat, "p_value_uncorrected": p_kw})

                    # Gatepost-hoc testing behind global significance
                    if p_kw < kw_alpha:
                        for c1, c2 in combinations(cohorts, 2):
                            d1, d2 = mouse_avg[mouse_avg["group"] == c1][var].dropna(), mouse_avg[mouse_avg["group"] == c2][var].dropna()
                            if len(d1) > 0 and len(d2) > 0:
                                st, p_mw = mannwhitneyu(d1, d2, alternative="two-sided")
                                rb = 1 - (2 * st) / (len(d1) * len(d2))
                                all_mw.append({"Category": cat, "Cluster": "all", "Variable": var, "Group1": c1, "Group1_mean": d1.mean(), "n_group1": len(d1), "Group2": c2, "Group2_mean": d2.mean(), "n_group2": len(d2), "U_statistic": st, "Effect_size_r_rb": rb, "p_value_uncorrected": p_mw})

            else:
                # Local metrics are assessed per cluster/behavioral state
                for cl in clusters:
                    sub_df = clean_df[clean_df["cluster"] == cl]
                    scores = [sub_df[sub_df["group"] == g][var].dropna() for g in cohorts]

                    if all(len(d) > 0 for d in scores):
                        if pd.concat(scores).nunique() <= 1: 
                            continue

                        stat, p_kw = kruskal(*scores)
                        all_kw.append({"Category": cat, "Cluster": cl, "Variable": var, "Kruskal_statistic": stat, "p_value_uncorrected": p_kw})

                        if p_kw < kw_alpha:
                            for c1, c2 in combinations(cohorts, 2):
                                d1, d2 = sub_df[sub_df["group"] == c1][var].dropna(), sub_df[sub_df["group"] == c2][var].dropna()
                                if len(d1) > 0 and len(d2) > 0:
                                    st, p_mw = mannwhitneyu(d1, d2, alternative="two-sided")
                                    rb = 1 - (2 * st) / (len(d1) * len(d2))
                                    all_mw.append({"Category": cat, "Cluster": cl, "Variable": var, "Group1": c1, "Group1_mean": d1.mean(), "n_group1": len(d1), "Group2": c2, "Group2_mean": d2.mean(), "n_group2": len(d2), "U_statistic": st, "Effect_size_r_rb": rb, "p_value_uncorrected": p_mw})

    # 4D. FDR Output
    kw_df = pd.DataFrame(all_kw)
    mw_df = pd.DataFrame(all_mw)

    if not kw_df.empty:
        kw_df["p_value_corrected"] = multipletests(kw_df["p_value_uncorrected"], method="fdr_bh")[1]
        kw_df[f"Significant (FDR<{fdr_q})"] = kw_df["p_value_corrected"] < fdr_q
        kw_df.to_csv(f"{out_prefix}kruskal_results.csv", index=False)

    if not mw_df.empty:
        mw_df["p_value_corrected"] = multipletests(mw_df["p_value_uncorrected"], method="fdr_bh")[1]
        mw_df[f"Significant (FDR<{fdr_q})"] = mw_df["p_value_corrected"] < fdr_q
        mw_df.to_csv(f"{out_prefix}posthoc_mannwhitney_fdr.csv", index=False)

# --- Master Execution ---
def run_pipeline():
    os.makedirs(os.path.dirname(OUT_PREFIX) or ".", exist_ok=True)

    # 1. Digest raw tracking files
    track_data = load_moseq_trials(RAW_DATA_DIR, SYL_MAP, FPS)
    
    # 2. Compute behavioral trial-level statistics
    trial_stats = calc_trial_metrics(track_data, FPS)
    
    # 3. Compress observations down to the individual animal
    mouse_stats = aggregate_to_mouse(trial_stats, grouping_vars=["name", "cluster"])
    mouse_stats.to_csv(f"{OUT_PREFIX}mouse_kpms_scores.csv", index=False)
    
    # 4. Multigroup testing pipeline
    run_multi_stats(mouse_stats, CORR_THRESH, KW_ALPHA, FDR_Q, CAT_RULES, OUT_PREFIX)

if __name__ == "__main__":
    run_pipeline()