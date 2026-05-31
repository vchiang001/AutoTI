import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations

# --- Configuration ---
RAW_TRIALS_PATH = "data/raw_efa_trial_scores.csv"
OUTPUT_PREFIX = "results/efa_stats_"

# Drop bad trials and isolate experimental testing phases
TRIAL_FILTERS = {
    "performance": ["good", "bad"],
    "train_test": ["test"],
    "trialoutcome": ["correct", "error"],
    "trialtype": ["BD"]  
}

# Variables to aggregate the trial-level behavioral data up to the animal level
ANIMAL_GROUPING_VARS = ["name", "trialtype", "trialoutcome"]

TARGET_COMPARISON_COL = "perf_tritpe_out"
COHORT_MAP = {} 
KW_ALPHA = 0.05 
FDR_Q = 0.1   

# --- 1. Animal-Level Aggregation ---
def calculate_mouse_averages(data_path, filters, grouping_vars):
    trial_data = pd.read_csv(data_path)
    
    for col, allowed_vals in filters.items():
        if col in trial_data.columns:
            trial_data = trial_data[trial_data[col].isin(allowed_vals)]
            
    if trial_data.empty:
        raise ValueError("No trials left! Double-check if your trial filters are too strict.")

    # Clean session strings to isolate just the mouse ID 
    trial_data['name'] = trial_data['name'].apply(lambda x: str(x).split('_')[0])
    
    # Separate numeric metrics from metadata to apply correct aggregation math
    num_cols = [c for c in trial_data.select_dtypes(include='number').columns if c not in grouping_vars]
    txt_cols = [c for c in trial_data.columns if c not in num_cols and c not in grouping_vars]
    
    agg_rules = {c: 'mean' for c in num_cols}
    agg_rules.update({c: 'first' for c in txt_cols})
    
    mouse_scores = trial_data.groupby(grouping_vars).agg(agg_rules).reset_index()
    
    # Realign columns with the original raw data structure
    return mouse_scores[trial_data.columns]

# --- 2. Multi-Group Statistics ---
def run_efa_stats(mouse_scores, target_col, cohort_map, kw_alpha, fdr_q, out_prefix):
    if cohort_map:
        mouse_scores[target_col] = mouse_scores[target_col].replace(cohort_map)

    if target_col not in mouse_scores.columns:
        raise ValueError(f"Couldn't find the target comparison column: {target_col}")

    cohorts = mouse_scores[target_col].dropna().unique()
    if len(cohorts) < 2:
        raise ValueError("We need at least two cohorts to run stats. Check your grouping variables.")

    factor_cols = [c for c in mouse_scores.columns if "Factor" in c and "Score" in c]
    if not factor_cols:
        raise ValueError("No valid Factor Score columns found to analyze.")

    kw_res = []
    mw_res = []

    for factor in factor_cols:
        scores_by_cohort = {c: mouse_scores[mouse_scores[target_col] == c][factor].dropna() for c in cohorts}
        
        # Require at least one valid animal per experimental cohort to run the test
        if all(len(scores) > 0 for scores in scores_by_cohort.values()):
            stat, p_val = kruskal(*scores_by_cohort.values())
            kw_res.append({
                "Variable": factor, 
                "Kruskal_statistic": stat, 
                "p_value_uncorrected": p_val
            })
            
            # Gate post-hoc pairwise testing behind global significance to prevent p-hacking
            if p_val < kw_alpha:
                for c1, c2 in combinations(cohorts, 2):
                    g1_scores, g2_scores = scores_by_cohort[c1], scores_by_cohort[c2]
                    
                    if len(g1_scores) > 0 and len(g2_scores) > 0:
                        stat_mw, p_mw = mannwhitneyu(g1_scores, g2_scores, alternative="two-sided")
                        r_rb = 1 - (2 * stat_mw) / (len(g1_scores) * len(g2_scores))
                        
                        mw_res.append({
                            "Variable": factor, 
                            "Comparison": f"{c1} vs {c2}",
                            "Group1": c1, "Group1_mean": g1_scores.mean(), "n_group1": len(g1_scores),
                            "Group2": c2, "Group2_mean": g2_scores.mean(), "n_group2": len(g2_scores),
                            "U_statistic": stat_mw, "Effect_size_r_rb": r_rb, 
                            "p_value_uncorrected": p_mw
                        })

    kw_df = pd.DataFrame(kw_res)
    mw_df = pd.DataFrame(mw_res)

    if not kw_df.empty:
        kw_df["p_value_corrected"] = multipletests(kw_df["p_value_uncorrected"], method="fdr_bh")[1]
        kw_df[f"Significant (FDR<{fdr_q})"] = kw_df["p_value_corrected"] < fdr_q
        kw_df.to_csv(f"{out_prefix}kruskal.csv", index=False)

    if not mw_df.empty:
        mw_df["p_value_corrected"] = multipletests(mw_df["p_value_uncorrected"], method="fdr_bh")[1]
        mw_df[f"Significant (FDR<{fdr_q})"] = mw_df["p_value_corrected"] < fdr_q
        mw_df.to_csv(f"{out_prefix}posthoc_fdr.csv", index=False)

# --- Master Execution ---
def run_pipeline():
    mouse_scores = calculate_mouse_averages(RAW_TRIALS_PATH, TRIAL_FILTERS, ANIMAL_GROUPING_VARS)
    
    # Save the intermediate animal-level scores for transparency
    mouse_scores.to_csv(f"{OUTPUT_PREFIX}scores_per_mouse.csv", index=False)
    
    run_efa_stats(mouse_scores, TARGET_COMPARISON_COL, COHORT_MAP, KW_ALPHA, FDR_Q, OUTPUT_PREFIX)

if __name__ == "__main__":
    run_pipeline()