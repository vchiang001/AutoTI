import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from statsmodels.stats.multitest import fdrcorrection

# --- Configuration ---
DATA_PATH = "data/mouse_kpms_scores.csv"
OUT_PREFIX = "results/umap_analysis_"

UMAP_DIMS = 3 
TARGET_COHORTS = ['goodBDcorrect'] 

FDR_ALPHA = 0.1          
P_THRESH = 0.05  

def get_stars(p_val):
    if p_val <= 0.001: return '***'
    elif p_val <= 0.01: return '**'
    elif p_val <= P_THRESH: return '*'
    return ''

def run_umap_pipeline(data_path, out_prefix, dims, target_cohorts, fdr_alpha, p_thresh):
    moseq_data = pd.read_csv(data_path).fillna(0)
    
    if moseq_data.empty:
        raise ValueError("The dataset is completely empty! Check your file path.")

    # Isolate trial-wide metrics from cluster-specific metrics to widen the dataset
    txt_cols = ['name', 'group', 'cluster']
    global_cols = ['usage_entropy', 'transitions_per_sec', 'entropy_rate', 'eigen2'] + [c for c in moseq_data.columns if 'bigram' in c.lower()]
    cluster_cols = [c for c in moseq_data.drop(columns=txt_cols, errors='ignore').columns if c not in global_cols]

    global_metrics = moseq_data.groupby(['name', 'group'])[global_cols].first().reset_index()
    
    cluster_metrics = moseq_data.pivot(index=['name', 'group'], columns='cluster', values=cluster_cols)
    cluster_metrics.columns = [f"C{int(c_id)}_{feat}" for feat, c_id in cluster_metrics.columns]
    cluster_metrics = cluster_metrics.reset_index()

    master_df = pd.merge(global_metrics, cluster_metrics, on=['name', 'group']).fillna(0)
    feat_names = master_df.drop(columns=['name', 'group']).columns

    # Standardize scale constraints before manifold projection
    scaled_feats = StandardScaler().fit_transform(master_df[feat_names])
    pca = PCA(n_components=0.95, random_state=42)
    pca_feats = pca.fit_transform(scaled_feats)

    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='b')
    plt.title('PCA - Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Variance Explained')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"{out_prefix}pca_variance.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{out_prefix}pca_variance.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    mapper = umap.UMAP(n_components=dims, random_state=42)
    umap_emb = mapper.fit_transform(pca_feats)

    for i in range(dims):
        master_df[f'UMAP {i+1}'] = umap_emb[:, i]

    centroids = master_df.groupby('group')[[f'UMAP {i+1}' for i in range(dims)]].mean()
    cohorts = master_df['group'].unique()
    palette = dict(zip(cohorts, sns.color_palette("Set2", len(cohorts))))

    fig = plt.figure(figsize=(12, 10))
    if dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        for c in cohorts:
            sub = master_df[master_df['group'] == c]
            ax.scatter(sub['UMAP 1'], sub['UMAP 2'], sub['UMAP 3'], color=palette[c], label=c, s=50, alpha=0.3)
            cx, cy, cz = centroids.loc[c]
            ax.scatter(cx, cy, cz, color=palette[c], edgecolor='black', linewidth=1.5, s=300, alpha=1.0)
        ax.set_zlabel('UMAP 3')
    else:
        ax = fig.add_subplot(111)
        for c in cohorts:
            sub = master_df[master_df['group'] == c]
            ax.scatter(sub['UMAP 1'], sub['UMAP 2'], color=palette[c], label=c, s=50, alpha=0.3)
            cx, cy = centroids.loc[c, 'UMAP 1'], centroids.loc[c, 'UMAP 2']
            ax.scatter(cx, cy, color=palette[c], edgecolor='black', linewidth=1.5, s=300, alpha=1.0)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.title('Behavioural UMAP', fontsize=16)
    plt.legend(title='Cohort', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(f"{out_prefix}umap_{dims}D.svg", format='svg', bbox_inches='tight')
    plt.savefig(f"{out_prefix}umap_{dims}D.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    dist_df = pd.DataFrame(squareform(pdist(centroids.values, metric='euclidean')), index=centroids.index, columns=centroids.index)
    c_grid = sns.clustermap(dist_df, cmap='viridis_r', annot=True, fmt=".2f", figsize=(9, 9), cbar_kws={'label': 'Euclidean Distance'})
    c_grid.fig.suptitle('UMAP Distances', y=1.02, fontsize=16)
    c_grid.savefig(f"{out_prefix}dendrogram.svg", format='svg', bbox_inches='tight')
    c_grid.savefig(f"{out_prefix}dendrogram.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
    plt.close()

    for target in target_cohorts:
        if target not in centroids.index:
            raise ValueError(f"Whoops! I cannot find the cohort '{target}' in your data to run correlations against.")

        target_vec = centroids.loc[target].values
        master_df[f'dist_to_{target}'] = np.linalg.norm(umap_emb - target_vec, axis=1)

        feats_tested, corrs, pvals = [], [], []
        for f in feat_names:
            if master_df[f].std() == 0:
                continue
            r, p = spearmanr(master_df[f], master_df[f'dist_to_{target}'])
            if not np.isnan(p):
                feats_tested.append(f)
                corrs.append(-r)  
                pvals.append(p)

        pass_fdr, p_corr = fdrcorrection(pvals, alpha=fdr_alpha)
        
        res_df = pd.DataFrame({'Feature': feats_tested, 'Similarity_Score': corrs, 'P_Value': p_corr})
        res_df['Is_Significant'] = pass_fdr & (res_df['P_Value'] <= p_thresh)

        sig_df = res_df[res_df['Is_Significant']].copy()
        pos_sig = sig_df[sig_df['Similarity_Score'] > 0].sort_values('Similarity_Score', ascending=False)
        neg_sig = sig_df[sig_df['Similarity_Score'] < 0].sort_values('Similarity_Score', ascending=True)

        pos_sig['Rank'] = [f"Top {i+1}" for i in range(len(pos_sig))]
        neg_sig['Rank'] = [f"Bottom {i+1}" for i in range(len(neg_sig))]
        
        pd.concat([pos_sig, neg_sig]).sort_values('Similarity_Score', ascending=False).to_csv(f"{out_prefix}correlations_{target}.csv", index=False)

        if not sig_df.empty:
            top_feats = pd.concat([pos_sig.head(10), neg_sig.head(10)]).sort_values('Similarity_Score', ascending=False).reset_index(drop=True)
            
            fig, ax = plt.subplots(figsize=(11, 8))
            norm = plt.Normalize(vmin=-1, vmax=1)
            sns.barplot(x='Similarity_Score', y='Feature', data=top_feats, palette=plt.cm.coolwarm(norm(top_feats['Similarity_Score'])), ax=ax)
            
            for idx, row in top_feats.iterrows():
                s, p = row['Similarity_Score'], row['P_Value']
                stars = get_stars(p)
                ax.text(s + (0.02 if s >= 0 else -0.02), idx, stars, color='black', va='center', ha='left' if s >= 0 else 'right', fontsize=14, fontweight='bold')
            
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=norm), ax=ax)
            cbar.set_label('Correlation Magnitude', rotation=270, labelpad=15)

            ax.set_title(f'Correlates of Similarity to {target}')
            ax.set_xlabel(f'Similarities between feature and {target} behavioural profile')
            ax.set_ylabel('Behavioral Feature')
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)
            
            x_min, x_max = ax.get_xlim()
            ax.set_xlim(x_min - 0.1, x_max + 0.1)
            
            plt.savefig(f"{out_prefix}correlations_bar_{target}.svg", format='svg', bbox_inches='tight')
            plt.savefig(f"{out_prefix}correlations_bar_{target}.jpeg", format='jpeg', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    run_umap_pipeline(DATA_PATH, OUT_PREFIX, UMAP_DIMS, TARGET_COHORTS, FDR_ALPHA, P_THRESH)