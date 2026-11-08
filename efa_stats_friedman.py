import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, sem, t
from statsmodels.stats.multitest import multipletests
from itertools import combinations

def run_efa_stats(mouse_scores_path, out_prefix, config):
    mouse_scores = pd.read_csv(mouse_scores_path)
    
    for col, val in config['filters'].items():
        if col in mouse_scores.columns:
            mouse_scores = mouse_scores[mouse_scores[col] == val]
            
    if config['group_combos']:
        mouse_scores[config['condition_col']] = mouse_scores[config['condition_col']].replace(config['group_combos'])
        
    if config['condition_col'] not in mouse_scores.columns:
        raise KeyError(f"Looks like the column '{config['condition_col']}' is missing. Check your CSV headers.")
        
    test_conditions = mouse_scores[config['condition_col']].dropna().unique()
    
    # Enforce strict paired design by dropping mice missing required conditions
    req_conditions = set(test_conditions)
    mouse_conditions = mouse_scores.groupby(config['mouse_col'])[config['condition_col']].apply(set)
    
    valid_mice = [m for m, conds in mouse_conditions.items() if req_conditions.issubset(conds)]
    mouse_scores = mouse_scores[mouse_scores[config['mouse_col']].isin(valid_mice)]
    
    if len(valid_mice) < 3:
        raise ValueError(f"Only {len(valid_mice)} mice completed all conditions! We need at least 3 for paired statistics.")
        
    if len(test_conditions) < 2:
        raise ValueError("Not enough trial conditions left to compare. Check if your filters are too strict.")
        
    factor_cols = [c for c in mouse_scores.columns if "Factor" in c and "Score" in c]
    friedman_res, posthoc_res = [], []
    
    for f in factor_cols:
        pairing_matrix = mouse_scores.pivot_table(
            index=config['mouse_col'], 
            columns=config['condition_col'], 
            values=f, 
            aggfunc='mean'
        ).dropna()
        
        n_mice = len(pairing_matrix)
        if n_mice < 3:
            continue
            
        paired_arrays = [pairing_matrix[c].values for c in test_conditions]
        k_conds = len(test_conditions)
        
        if k_conds >= 3:
            stat, p_val = friedmanchisquare(*paired_arrays)
            w_eff = stat / (n_mice * (k_conds - 1))
            
            friedman_res.append({
                "Factor": f, 
                "Friedman_statistic": stat, 
                "Kendalls_W": w_eff,  
                "p_uncorrected": p_val
            })
            run_posthoc = p_val < config['friedman_alpha']
        else:
            run_posthoc = True
            
        if run_posthoc:
            for c1, c2 in combinations(test_conditions, 2):
                d1, d2 = pairing_matrix[c1], pairing_matrix[c2]
                stat_w, p_w = wilcoxon(d1, d2, alternative="two-sided")
                r_eff = 1 - (4 * stat_w) / (n_mice * (n_mice + 1))
                
                diffs = d1 - d2
                mean_diff = diffs.mean()
                sem_diff = sem(diffs)
                ci_l, ci_u = t.interval(0.95, df=n_mice-1, loc=mean_diff, scale=sem_diff) if sem_diff > 0 else (mean_diff, mean_diff)
                
                posthoc_res.append({
                    "Factor": f, 
                    "Comparison": f"{c1}_vs_{c2}",
                    "Condition1": c1, "Condition1_mean": d1.mean(), 
                    "Condition2": c2, "Condition2_mean": d2.mean(), 
                    "Mean_Diff": mean_diff, "SEM_Diff": sem_diff,
                    "CI_Lower": ci_l, "CI_Upper": ci_u,
                    "N_Mice": n_mice, "Wilcoxon_W": stat_w, 
                    "Effect_size_r": r_eff, "p_uncorrected": p_w
                })
                
    f_df = pd.DataFrame(friedman_res)
    w_df = pd.DataFrame(posthoc_res)
    
    filter_tag = "_".join([str(v) for v in config['filters'].values()])
    prefix = f"{out_prefix}compare_{config['condition_col']}_in_{filter_tag}"
    
    if not f_df.empty:
        f_df[f"Significant (p<{config['friedman_alpha']})"] = f_df["p_uncorrected"] < config['friedman_alpha']
        f_df.to_csv(f"{prefix}_friedman.csv", index=False)
        
    if not w_df.empty:
        w_df["p_corrected"] = multipletests(w_df["p_uncorrected"], method="fdr_bh")[1]
        w_df[f"Significant (FDR<{config['fdr_q']})"] = w_df["p_corrected"] < config['fdr_q']
        w_df.to_csv(f"{prefix}_posthoc_paired_fdr.csv", index=False)


if __name__ == "__main__":
    # --- Configuration ---
    mouse_scores_file = "/path/to/your/data/efa_scores_per_mouse.csv"
    output_prefix = "/path/to/your/data/experiment_efa_"
    
    run_config = {
        "mouse_col": "name",
        "condition_col": "trial_tpe_out",
        "friedman_alpha": 0.05,
        "fdr_q": 0.10,
        "group_combos": {},
        "filters": {
            # "performance": "good",
            # "trialtype": "BC",
            # "trialoutcome": "correct",
            # "train_test": "test"
        }
    }
    
    run_efa_stats(mouse_scores_file, output_prefix, run_config)