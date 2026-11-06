import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, sem, t
from statsmodels.stats.multitest import multipletests
from itertools import combinations

def drop_collinear(df, threshold, protected_vars):
    # Drops redundant variables to preserve statistical power
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_mat = df[num_cols].corr(method='spearman').abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    
    drops = set()
    for c in upper.columns:
        if c in drops or c in protected_vars:
            continue
            
        high_corr = upper.index[upper[c] > threshold].tolist()
        for hc in high_corr:
            if hc in drops or hc in protected_vars:
                continue
                
            if "mean" in c and "mean" not in hc:
                drops.add(hc)
            elif "mean" in hc and "mean" not in c:
                drops.add(c)
            elif df[c].var() > df[hc].var():
                drops.add(hc)
            else:
                drops.add(c)
                
    return df.drop(columns=list(drops))

def get_complete_mice(df, mouse_col, trial_type_col):
    # Only keep mice that completed all required trial types
    req_conds = set(df[trial_type_col].dropna().unique())
    mouse_conds = df.groupby(mouse_col)[trial_type_col].apply(set)
    
    valid_mice = [m for m, c in mouse_conds.items() if req_conds.issubset(c)]
    
    if len(valid_mice) < 3:
        raise ValueError("Not enough complete mice! Double check if your performance filter is too strict.")
        
    return df[df[mouse_col].isin(valid_mice)]

def run_stats(df, target_map, mouse_col, trial_type_col, cluster_col, f_alpha, fdr_q):
    trial_types = list(df[trial_type_col].dropna().unique())
    friedman_res, wilcoxon_res = [], []
    
    for cluster, variables in target_map.items():
        c_df = df[df[cluster_col] == cluster]
        if c_df.empty:
            continue
            
        for v in variables:
            if v not in c_df.columns:
                continue
                
            pivot = c_df.pivot_table(index=mouse_col, columns=trial_type_col, values=v, aggfunc='mean').dropna()
            n_mice = len(pivot)
            
            if n_mice < 3:
                continue
                
            data_arrays = [pivot[t].values for t in trial_types]
            
            if len(trial_types) >= 3:
                stat, p_val = friedmanchisquare(*data_arrays)
                w_eff = stat / (n_mice * (len(trial_types) - 1))
                
                friedman_res.append({
                    "Cluster": cluster, "Variable": v,
                    "Friedman_statistic": stat, "Kendalls_W_effect_size": w_eff, "p_value_uncorrected": p_val
                })
                run_posthoc = p_val < f_alpha
            else:
                run_posthoc = True
                
            if run_posthoc:
                for g1, g2 in combinations(trial_types, 2):
                    d1, d2 = pivot[g1], pivot[g2]
                    stat_w, p_w = wilcoxon(d1, d2, alternative="two-sided")
                    r_eff = 1 - (4 * stat_w) / (n_mice * (n_mice + 1))
                    
                    diffs = d1 - d2
                    mean_diff = diffs.mean()
                    sem_diff = sem(diffs)
                    ci_l, ci_u = t.interval(0.95, df=n_mice-1, loc=mean_diff, scale=sem_diff) if sem_diff > 0 else (mean_diff, mean_diff)
                    
                    wilcoxon_res.append({
                        "Cluster": cluster, "Variable": v, "Comparison": f"{g1} vs {g2}",
                        "Group1": g1, "Group1_mean": d1.mean(),
                        "Group2": g2, "Group2_mean": d2.mean(),
                        "Paired_Mean_Difference": mean_diff,
                        "Paired_SEM_Difference": sem_diff,
                        "Diff_95CI_Lower": ci_l, "Diff_95CI_Upper": ci_u,
                        "N_Mice_Pairs": n_mice, "Wilcoxon_Statistic": stat_w, 
                        "Effect_size_r": r_eff, "p_value_uncorrected": p_w
                    })
                    
    f_df = pd.DataFrame(friedman_res)
    if not f_df.empty:
        f_df[f"Significant (p<{f_alpha})"] = f_df["p_value_uncorrected"] < f_alpha
        
    w_df = pd.DataFrame(wilcoxon_res)
    if not w_df.empty:
        w_df["p_value_corrected"] = multipletests(w_df["p_value_uncorrected"], method="fdr_bh")[1]
        w_df[f"Significant (FDR<{fdr_q})"] = w_df["p_value_corrected"] < fdr_q
        
    return f_df, w_df

def process_dataset(config):
    mouse_scores = pd.read_csv(config['in_path'])
    
    if mouse_scores[config['mouse_col']].astype(str).str.contains('_').any():
        mouse_scores[config['mouse_col']] = mouse_scores[config['mouse_col']].astype(str).str.split('_').str[-1]
        
    mouse_scores = get_complete_mice(mouse_scores, config['mouse_col'], config['trial_type_col'])
    
    protected_vars = {var for vars_list in config['target_map'].values() for var in vars_list}
    mouse_scores = drop_collinear(mouse_scores, config['corr_thresh'], protected_vars)
    
    mouse_scores.to_csv(f"{config['out_prefix']}cleaned_data.csv", index=False)
    
    f_df, w_df = run_stats(
        mouse_scores, config['target_map'], config['mouse_col'], 
        config['trial_type_col'], config['cluster_col'], 
        config['f_alpha'], config['fdr_q']
    )
    
    if not f_df.empty:
        f_df.to_csv(f"{config['out_prefix']}friedman_results.csv", index=False)
    if not w_df.empty:
        w_df.to_csv(f"{config['out_prefix']}posthoc_wilcoxon_fdr.csv", index=False)

if __name__ == "__main__":
    # --- Configuration ---
    pipeline_config = {
        "in_path": "/path/to/your/data/mouse_scores.csv",
        "out_prefix": "/path/to/your/data/output_prefix_",
        "mouse_col": "name",
        "trial_type_col": "group",
        "cluster_col": "cluster",
        "corr_thresh": 0.90,
        "f_alpha": 0.05,
        "fdr_q": 0.10,
        
        "target_map": {
            2: ["angular_velocity_mean", "angular_velocity_std", "angular_velocity_min", "angular_velocity_max",
                "heading_mean", "heading_std", "heading_min", "heading_max", "velocity_px_s_mean", 
                "velocity_px_s_std", "velocity_px_s_min", "velocity_px_s_max", "duration", "bout_freq"],
            4: ["angular_velocity_mean", "angular_velocity_std", "angular_velocity_min", "angular_velocity_max", 
                "heading_mean", "heading_std", "heading_min", "heading_max"], 
            5: ["heading_mean", "heading_std", "heading_min", "heading_max"],
            6: ["heading_mean", "heading_std", "heading_min", "heading_max", "duration", "bout_freq"]
        }
    }
    
    process_dataset(pipeline_config)