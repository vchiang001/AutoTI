import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import numpy.linalg as npla
from sklearn.preprocessing import StandardScaler
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

# --- Configuration ---
RAW_TRIALS_DIR = "data/raw_efa_trials"
COMPILED_DATA_PATH = "data/compiled_efa_trials.csv"
FA_OUT_PREFIX = "results/efa_model_"

FPS = 30
N_FACTORS = 20  # Review the generated scree plot and adjust this value as needed

TRIAL_META = {
    "performance": "bad",           
    "train_test": "test",            
    "trialoutcome": "correct",       
    "trialtype": "DE"                
}

SYL_MAP = {
    1: [13],
    2: [9, 19, 0, 4, 10, 15, 3, 1, 16],
    3: [6, 7, 11],
    4: [2, 12, 18],
    5: [5],
    6: [17, 8, 14],
}

META_COLS = ['name', 'performance', 'train_test', 'trialoutcome', 'trialtype']

HEATMAP_VARS = [
    'usage_entropy', 'transitions_per_sec', 'entropy_rate', 'eigen2',
    'C1_bout_freq', 'C2_bout_freq', 'C3_bout_freq', 'C4_bout_freq', 'C5_bout_freq', 'C6_bout_freq',
    'C1_local_entropy', 'bigram_C1_to_C2', 'bigram_C1_to_C3', 'bigram_C1_to_C4', 'bigram_C1_to_C5', 'bigram_C1_to_C6',
    'C2_local_entropy', 'bigram_C2_to_C1', 'bigram_C2_to_C3', 'bigram_C2_to_C4', 'bigram_C2_to_C5', 'bigram_C2_to_C6',
    'C3_local_entropy', 'bigram_C3_to_C1', 'bigram_C3_to_C2', 'bigram_C3_to_C4', 'bigram_C3_to_C5', 'bigram_C3_to_C6',
    'C4_local_entropy', 'bigram_C4_to_C1', 'bigram_C4_to_C2', 'bigram_C4_to_C3', 'bigram_C4_to_C5', 'bigram_C4_to_C6',
    'C5_local_entropy', 'bigram_C5_to_C1', 'bigram_C5_to_C2', 'bigram_C5_to_C3', 'bigram_C5_to_C4', 'bigram_C5_to_C6',
    'C6_local_entropy', 'bigram_C6_to_C1', 'bigram_C6_to_C2', 'bigram_C6_to_C3', 'bigram_C6_to_C4', 'bigram_C6_to_C5',
    'C1_RelDist_Port_I_mean', 'C2_RelDist_Port_I_mean', 'C3_RelDist_Port_I_mean', 'C4_RelDist_Port_I_mean', 'C5_RelDist_Port_I_mean', 'C6_RelDist_Port_I_mean',
    'C1_angular_velocity_mean', 'C2_angular_velocity_mean', 'C3_angular_velocity_mean', 'C4_angular_velocity_mean', 'C5_angular_velocity_mean', 'C6_angular_velocity_mean',
    'C1_velocity_px_s_mean', 'C2_velocity_px_s_mean', 'C3_velocity_px_s_mean', 'C4_velocity_px_s_mean', 'C5_velocity_px_s_mean', 'C6_velocity_px_s_mean',
    'C1_heading_mean', 'C2_heading_mean', 'C3_heading_mean', 'C4_heading_mean', 'C5_heading_mean', 'C6_heading_mean',
    'C1_proportion_time', 'C2_proportion_time', 'C3_proportion_time', 'C4_proportion_time', 'C5_proportion_time', 'C6_proportion_time',
    'C1_duration', 'C2_duration', 'C3_duration', 'C4_duration', 'C5_duration', 'C6_duration'
]

# --- 1. Data Extraction & Feature Engineering ---
def map_syl(syl, cmap):
    for c_label, s_list in cmap.items():
        if syl in s_list:
            return c_label
    return np.nan

def calc_adv_metrics(seq, valid_clusts, dur_sec):
    if not seq:
        return {}
        
    collapsed = [seq[0]]
    for c in seq[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
            
    total_trans = len(collapsed) - 1
    usage = pd.Series(seq).value_counts(normalize=True)
    bouts = pd.Series(collapsed).value_counts()
    
    metrics = {f"C{i}_bout_freq": bouts.get(i, 0) / dur_sec if dur_sec > 0 else 0.0 for i in valid_clusts}

    if total_trans <= 0:
        return metrics

    trans_df = pd.DataFrame(list(zip(collapsed[:-1], collapsed[1:])), columns=['from', 'to'])
    counts = pd.crosstab(
        pd.Categorical(trans_df['from'], categories=valid_clusts),
        pd.Categorical(trans_df['to'], categories=valid_clusts),
        dropna=False
    )

    joint_probs = counts / total_trans
    trans_mat = counts.div(counts.sum(axis=1), axis=0).fillna(0)

    loc_ent = trans_mat.apply(lambda r: entropy(r, base=2), axis=1)
    metrics["entropy_rate"] = (usage.reindex(valid_clusts).fillna(0) * loc_ent).sum()

    evals = npla.eigvals(trans_mat.values)
    sorted_evals = np.sort(np.abs(evals))[::-1] 
    metrics["eigen2"] = sorted_evals[1] if len(sorted_evals) > 1 else np.nan

    for i in valid_clusts:
        metrics[f"C{i}_local_entropy"] = loc_ent.loc[i]
        for j in valid_clusts:
            if i != j: 
                metrics[f"bigram_C{i}_to_C{j}"] = joint_probs.loc[i, j]

    return metrics

def extract_trial_features(in_dir, out_path, cohort_meta, syl_map, fps):
    all_trials = []
    files = [f for f in os.listdir(in_dir) if f.endswith(".csv")]
    
    if not files:
        raise ValueError("No CSV tracking files found! Check your raw data directory path.")

    for f in files:
        df = pd.read_csv(os.path.join(in_dir, f))
        
        if not all(col in df.columns for col in ["syllable", "centroid x", "centroid y", "heading"]):
            continue

        df["velocity_px_s"] = np.sqrt(df["centroid x"].diff()**2 + df["centroid y"].diff()**2).fillna(0) * fps
        df["angular_velocity"] = np.concatenate(([0], np.diff(df["heading"]))) * fps
        df["cluster"] = df["syllable"].apply(lambda x: map_syl(x, syl_map))
        df.dropna(subset=["cluster"], inplace=True)

        if df.empty:
            continue

        seq = df["cluster"].tolist()
        dur_sec = len(df) / fps
        usg_ent = entropy(pd.Series(seq).value_counts(normalize=True), base=2)
        total_trans = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
        
        valid_clusts = [1, 2, 3, 4, 5, 6]
        adv_metrics = calc_adv_metrics(seq, valid_clusts, dur_sec)

        kinematics = ["RelDist_Port_I", "RelDist_Port_E", "RelDist_Port_C", "RelDist_Port_A", "RelDist_Port_B", "RelDist_Port_D", "velocity_px_s", "angular_velocity", "heading"]
        avail_cols = [m for m in kinematics if m in df.columns]

        c_stats = df.groupby("cluster")[avail_cols].agg(["mean", "std", "min", "max"])
        c_stats.columns = ["_".join(col) for col in c_stats.columns]
        c_stats["proportion_time"] = df["cluster"].value_counts(normalize=True)
        c_stats["duration"] = df.groupby("cluster").size() / fps

        base = f.replace(".csv", "")
        mouse_id = base.split("_", 1)[1] if "_" in base else base
        
        trial_dict = {"name": mouse_id, **cohort_meta}
        trial_dict.update({
            "usage_entropy": usg_ent,
            "transitions_per_sec": total_trans / dur_sec if dur_sec > 0 else np.nan,
            "entropy_rate": adv_metrics.get("entropy_rate", np.nan),
            "eigen2": adv_metrics.get("eigen2", np.nan)
        })
        
        for k, v in adv_metrics.items():
            if k not in ["entropy_rate", "eigen2"]: 
                trial_dict[k] = v

        for c_id, r_data in c_stats.iterrows():
            for m_name, val in r_data.items():
                trial_dict[f"C{int(c_id)}_{m_name}"] = val

        all_trials.append(trial_dict)

    if not all_trials:
        raise ValueError("No valid trials remained after parsing the tracking columns! Check the MoSeq outputs.")

    batch_df = pd.DataFrame(all_trials)
    
    if os.path.exists(out_path):
        pd.concat([pd.read_csv(out_path), batch_df], ignore_index=True).to_csv(out_path, index=False)
    else:
        batch_df.to_csv(out_path, index=False)

# --- 2. Exploratory Factor Analysis ---
def run_efa_modeling(data_path, out_prefix, meta_cols, target_vars, n_facts):
    trial_data = pd.read_csv(data_path)
    feat_cols = [c for c in trial_data.select_dtypes(include=[np.number]).columns if c not in meta_cols]
    
    trial_data[feat_cols] = trial_data[feat_cols].fillna(trial_data[feat_cols].mean())
    trial_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    trial_data.dropna(subset=feat_cols, inplace=True)

    # Drop flatlined variance
    variances = trial_data[feat_cols].var()
    flat_cols = variances[variances == 0].index.tolist()
    trial_data.drop(columns=flat_cols, inplace=True)
    feat_cols = [c for c in feat_cols if c not in flat_cols]

    # Drop pairwise multicollinearity (corr > 0.95)
    corr_mat = trial_data[feat_cols].corr().abs()
    up_tri = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    redundant = [c for c in up_tri.columns if any(up_tri[c] > 0.95)]
    trial_data.drop(columns=redundant, inplace=True)
    feat_cols = [c for c in feat_cols if c not in redundant]

    # Drop group collinearity (variables that sum to represent the same underlying matrix)
    while True:
        evals, evecs = npla.eigh(trial_data[feat_cols].corr().values)
        if np.min(evals) > 1e-4:
            break 
        worst_var = feat_cols[np.argmax(np.abs(evecs[:, np.argmin(evals)]))]
        trial_data.drop(columns=[worst_var], inplace=True)
        feat_cols.remove(worst_var)

    scaled_df = pd.DataFrame(StandardScaler().fit_transform(trial_data[feat_cols]), columns=feat_cols)

    # Ensure dataset is robust enough for decomposition
    chi2, p_bart = calculate_bartlett_sphericity(scaled_df)
    if p_bart >= 0.05:
        raise ValueError(f"Bartlett test failed (p={p_bart:.4f}). Variables are too independent to cluster into factors.")
        
    _, kmo_mod = calculate_kmo(scaled_df)
    if kmo_mod < 0.6:
        raise ValueError(f"KMO test failed (score={kmo_mod:.3f}). Sampling adequacy is too low for the number of variables.")

    # Generate Scree Plot for visual verification
    fa_init = FactorAnalyzer(rotation=None)
    fa_init.fit(scaled_df)
    ev, _ = fa_init.get_eigenvalues()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ev)+1), ev, marker='o')
    plt.axhline(1, color='red', linestyle='--', label='Kaiser Criterion')
    plt.title('Scree Plot')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}scree_plot.png', dpi=300)
    plt.close()

    # Fit final model
    fa = FactorAnalyzer(n_factors=n_facts, rotation='varimax')
    fa.fit(scaled_df)

    loadings = pd.DataFrame(fa.loadings_, index=feat_cols, columns=[f'Factor_{i+1}' for i in range(n_facts)])
    loadings.to_csv(f'{out_prefix}loadings.csv')
    
    pd.DataFrame(fa.get_factor_variance(), index=['SS Loadings', 'Proportion Variance', 'Cumulative Variance'], columns=loadings.columns).to_csv(f'{out_prefix}variance.csv')
    pd.DataFrame(fa.get_communalities(), index=feat_cols, columns=['Communality']).to_csv(f'{out_prefix}communalities.csv')

    avail_vars = [v for v in target_vars if v in loadings.index]
    plt.figure(figsize=(14, 22)) 
    sns.heatmap(loadings.loc[avail_vars], cmap='RdBu_r', center=0, annot=False, cbar_kws={'label': 'Factor Loading'})
    plt.title(f'Factor Loadings (Selected Variables, n={n_facts})')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}loadings_heatmap.png', dpi=300) 
    plt.close()

    meta_df = trial_data[[c for c in meta_cols if c in trial_data.columns]].reset_index(drop=True)
    scores_df = pd.DataFrame(fa.transform(scaled_df), columns=[f'Factor_{i+1}_Score' for i in range(n_facts)])
    pd.concat([meta_df, scores_df], axis=1).to_csv(f'{out_prefix}trial_scores.csv', index=False)

def run_efa_pipeline():
    os.makedirs(os.path.dirname(COMPILED_DATA_PATH) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(FA_OUT_PREFIX) or ".", exist_ok=True)
    
    extract_trial_features(RAW_TRIALS_DIR, COMPILED_DATA_PATH, TRIAL_META, SYL_MAP, FPS)
    run_efa_modeling(COMPILED_DATA_PATH, FA_OUT_PREFIX, META_COLS, HEATMAP_VARS, N_FACTORS)

if __name__ == "__main__":
    run_efa_pipeline()