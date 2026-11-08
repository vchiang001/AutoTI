import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import factor_analyzer.factor_analyzer

# Patches a known compatibility issue in factor_analyzer's internal array validation
orig_check_array = factor_analyzer.factor_analyzer.check_array
def patched_check(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return orig_check_array(*args, **kwargs)
factor_analyzer.factor_analyzer.check_array = patched_check

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo


def run_efa_pipeline(data_path, out_prefix, meta_cols, target_heatmap_vars, corr_alpha=0.95, forced_factors=None):
    trial_data = pd.read_csv(data_path)
    avail_meta = [c for c in meta_cols if c in trial_data.columns]
    features = [c for c in trial_data.select_dtypes(include=[np.number]).columns if c not in meta_cols]
    
    trial_data[features] = trial_data[features].fillna(trial_data[features].mean())

    # Drop flatlined features
    variances = trial_data[features].var()
    zero_var = variances[variances == 0].index.tolist()
    trial_data.drop(columns=zero_var, inplace=True)
    features = [c for c in features if c not in zero_var]

    trial_data[features] = trial_data[features].replace([np.inf, -np.inf], np.nan)
    trial_data.dropna(subset=features, inplace=True)

    # Filter highly collinear variables to preserve matrix stability
    corr_mat = trial_data[features].corr().abs()
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    collinear = [c for c in upper.columns if any(upper[c] > corr_alpha)]
    trial_data.drop(columns=collinear, inplace=True)
    features = [c for c in features if c not in collinear]

    # Drop complex group collinearity using eigenvalue decomposition
    while True:
        c_mat = trial_data[features].corr().values
        evals, evecs = np.linalg.eigh(c_mat)
        
        if np.min(evals) > 1e-4:
            break 
            
        worst_idx = np.argmax(np.abs(evecs[:, np.argmin(evals)]))
        worst_var = features[worst_idx]
        trial_data.drop(columns=[worst_var], inplace=True)
        features.remove(worst_var)

    if not features:
        raise ValueError("No variables survived the correlation filters! Double check if your threshold is too strict.")

    scaler = StandardScaler()
    scaled_kinematics = pd.DataFrame(scaler.fit_transform(trial_data[features]), columns=features)

    _, p_bartlett = calculate_bartlett_sphericity(scaled_kinematics)
    if p_bartlett >= 0.05:
        raise ValueError("Bartlett's test failed! The variables are too independent to be grouped into latent factors.")

    _, kmo_score = calculate_kmo(scaled_kinematics)
    if kmo_score < 0.6:
        raise ValueError(f"KMO score is only {kmo_score:.2f}. You likely have too many variables compared to your trial count.")

    # Parallel Analysis for factor selection
    efa_init = FactorAnalyzer(rotation=None)
    efa_init.fit(scaled_kinematics)
    actual_evals, _ = efa_init.get_eigenvalues()

    n_trials, n_vars = scaled_kinematics.shape
    n_sims = 100
    rand_evals = []

    for _ in range(n_sims):
        noise = np.random.normal(size=(n_trials, n_vars))
        noise_corr = np.corrcoef(noise, rowvar=False)
        rand_evals.append(np.linalg.eigvalsh(noise_corr)[::-1])

    noise_thresholds = np.percentile(np.array(rand_evals), 95, axis=0)

    pa_metrics = []
    math_recommended_factors = 0
    for i, (act, noise_th) in enumerate(zip(actual_evals, noise_thresholds)):
        status = "KEEP" if act > noise_th else "DROP"
        if status == "KEEP":
            math_recommended_factors += 1
        pa_metrics.append({
            "Factor_Number": i + 1,
            "Actual_Eigenvalue": act,
            "Noise_Threshold_95th": noise_th,
            "Recommendation": status
        })

    pd.DataFrame(pa_metrics).to_csv(f"{out_prefix}parallel_analysis.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(actual_evals)+1), actual_evals, marker='o', color='blue', label='Actual Data')
    plt.plot(range(1, len(noise_thresholds)+1), noise_thresholds, linestyle='--', color='red', label='Noise Threshold')
    plt.axhline(1, color='green', linestyle=':', label='Kaiser Criterion')
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{out_prefix}scree_pa_plot.png', dpi=300)
    plt.close()

    n_factors = forced_factors if forced_factors else math_recommended_factors
    if n_factors <= 0:
        raise ValueError("Math recommends 0 factors. The dataset might not contain strong latent structures.")

    efa_model = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    efa_model.fit(scaled_kinematics)

    factor_cols = [f'Factor_{i+1}' for i in range(n_factors)]
    loadings = pd.DataFrame(efa_model.loadings_, index=features, columns=factor_cols)
    loadings.to_csv(f'{out_prefix}efa_loadings.csv')

    pd.DataFrame(efa_model.get_factor_variance(), 
                 index=['SS Loadings', 'Proportion Variance', 'Cumulative Variance'],
                 columns=factor_cols).to_csv(f'{out_prefix}efa_variance.csv')

    pd.DataFrame(efa_model.get_communalities(), index=features, columns=['Communality']).to_csv(f'{out_prefix}efa_communalities.csv')

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16

    avail_vars = [v for v in target_heatmap_vars if v in loadings.index]
    plt.figure(figsize=(14, 22)) 
    ax = sns.heatmap(loadings.loc[avail_vars], cmap='RdBu_r', center=0, annot=False, 
                     cbar_kws={'label': 'Factor Loading', 'shrink': 0.4, 'aspect': 40, 'pad': 0.04})
    
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}efa_loadings_heatmap.png', dpi=300) 
    plt.close()
    plt.rcdefaults()

    scores = pd.DataFrame(efa_model.transform(scaled_kinematics), columns=[f'{c}_Score' for c in factor_cols])
    meta_df = trial_data[avail_meta].reset_index(drop=True)
    pd.concat([meta_df, scores], axis=1).to_csv(f'{out_prefix}efa_trial_scores.csv', index=False)


if __name__ == "__main__":
    # --- Configuration ---
    input_csv = "/path/to/your/data/efa_restructured.csv"
    export_prefix = "/path/to/your/data/experiment_name_"
    
    metadata = ['name', 'performance', 'train_test', 'trialoutcome', 'trialtype']
    
    heatmap_features = [
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
    
    # Leave forced_factors as None to automatically use the Parallel Analysis recommendation
    run_efa_pipeline(input_csv, export_prefix, metadata, heatmap_features, corr_alpha=0.95, forced_factors=None)