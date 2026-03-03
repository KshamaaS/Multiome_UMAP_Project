#!/usr/bin/env python3
"""
Evaluate Dual-UMAP Quality: Quantitative Metrics + Diagnostic Plots

This script evaluates your dual-UMAP results with multiple metrics:
1. Cell type separation (silhouette score)
2. Neighborhood preservation 
3. Clustering agreement between modalities
4. Visual diagnostics

Usage:
    python evaluate_dual_umap_quality.py --input /path/to/processed_mdata.h5mu --output /path/to/evaluation_dir
    
    Or import and use programmatically:
    from evaluate_dual_umap_quality import evaluate_dual_umap
    evaluate_dual_umap(input_file, output_dir)
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_samples
import scanpy as sc
import mudata as md


def neighborhood_preservation(high_d, low_d, k=15):
    """Calculate what % of k-nearest neighbors are preserved."""
    nbrs_high = NearestNeighbors(n_neighbors=k+1).fit(high_d)
    _, indices_high = nbrs_high.kneighbors(high_d)
    
    nbrs_low = NearestNeighbors(n_neighbors=k+1).fit(low_d)
    _, indices_low = nbrs_low.kneighbors(low_d)
    
    preservation = []
    for i in range(len(high_d)):
        high_neighbors = set(indices_high[i][1:])
        low_neighbors = set(indices_low[i][1:])
        overlap = len(high_neighbors & low_neighbors) / k
        preservation.append(overlap)
    
    return np.mean(preservation)


def evaluate_dual_umap(input_path, output_dir):
    """
    Evaluate dual-UMAP quality with quantitative metrics and diagnostic plots.
    
    Parameters
    ----------
    input_path : str
        Path to the processed MuData file (*.h5mu) OR directory containing it
    output_dir : str
        Directory where evaluation results will be saved
        
    Returns
    -------
    dict
        Dictionary containing all computed metrics
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("DUAL-UMAP QUALITY EVALUATION")
    print("="*70)
    
    # Determine input file path
    if os.path.isfile(input_path):
        input_file = input_path
    elif os.path.isdir(input_path):
        # Look for processed_mdata.h5mu in the directory
        input_file = os.path.join(input_path, 'processed_mdata.h5mu')
        if not os.path.exists(input_file):
            raise FileNotFoundError(
                f"Could not find 'processed_mdata.h5mu' in directory: {input_path}\n"
                f"Please provide the full path to the .h5mu file."
            )
    else:
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    
    # Load results
    print(f"\nLoading processed data from: {input_file}")
    mdata = md.read_h5mu(input_file)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Cells: {mdata.n_obs:,}")
    
    # Fix cell type column - try multiple possible column names
    print("\nSearching for cell type annotations...")
    print(f"Available columns: {list(mdata.obs.columns)}")
    
    cell_type_col = None
    possible_cols = ['cell_type', 'cellAnnot', 'rna:cellAnnot', 'atac:cellAnnot', 'CellType', 'cell_type_annotation']
    
    for col in possible_cols:
        if col in mdata.obs.columns:
            unique_types = mdata.obs[col].nunique()
            if unique_types > 1:
                cell_type_col = col
                print(f"\n✓ Found cell types in column: '{col}'")
                print(f"  Unique cell types: {unique_types}")
                break
    
    if cell_type_col is None:
        print("\n✗ ERROR: Could not find valid cell type annotations!")
        print("\nDebugging info:")
        print("  Available columns:", list(mdata.obs.columns))
        print("\n  Checking each column for unique values:")
        for col in mdata.obs.columns:
            if mdata.obs[col].dtype == 'object' or mdata.obs[col].dtype.name == 'category':
                n_unique = mdata.obs[col].nunique()
                print(f"    {col}: {n_unique} unique values")
                if n_unique > 1 and n_unique < 100:
                    print(f"      Sample values: {list(mdata.obs[col].unique()[:5])}")
        raise ValueError("Could not find valid cell type annotations")
    
    # Use the found column
    cell_types = mdata.obs[cell_type_col]
    mdata.obs['cell_type'] = cell_types  # Standardize name
    
    print(f"\n  Cell type distribution:")
    type_counts = cell_types.value_counts()
    for ct, count in type_counts.head(10).items():
        print(f"    {ct}: {count:,}")
    if len(type_counts) > 10:
        print(f"    ... and {len(type_counts) - 10} more")
    
    # Get data
    rna_umap = mdata.obsm['X_umap_rna']
    atac_umap = mdata.obsm['X_umap_atac']
    integrated_umap = mdata.obsm['X_umap']
    
    # Encode cell types as integers for metrics
    le = LabelEncoder()
    cell_type_labels = le.fit_transform(cell_types)
    
    print("\n" + "="*70)
    print("QUANTITATIVE EVALUATION METRICS")
    print("="*70)
    
    # ============================================
    # 1. SILHOUETTE SCORE (Cell Type Separation)
    # ============================================
    print("\n1. SILHOUETTE SCORE (Cell Type Separation)")
    print("   Higher is better (range: -1 to 1, >0.5 is good)")
    print("   Measures how well cell types are separated in UMAP space")
    
    # Sample for faster computation if needed
    if len(cell_types) > 10000:
        sample_idx = np.random.choice(len(cell_types), 10000, replace=False)
        print(f"   (Using {len(sample_idx):,} sampled cells for speed)")
    else:
        sample_idx = np.arange(len(cell_types))
    
    print("   Computing silhouette scores...")
    silhouette_rna = silhouette_score(rna_umap[sample_idx], cell_type_labels[sample_idx])
    silhouette_atac = silhouette_score(atac_umap[sample_idx], cell_type_labels[sample_idx])
    silhouette_integrated = silhouette_score(integrated_umap[sample_idx], cell_type_labels[sample_idx])
    
    print(f"\n   RNA UMAP:        {silhouette_rna:.4f}")
    print(f"   ATAC UMAP:       {silhouette_atac:.4f}")
    print(f"   Integrated UMAP: {silhouette_integrated:.4f}")
    
    # Interpretation
    if silhouette_integrated > max(silhouette_rna, silhouette_atac):
        improvement = silhouette_integrated - max(silhouette_rna, silhouette_atac)
        print(f"\n   ✓ EXCELLENT: Integrated UMAP has best separation! (+{improvement:.4f})")
    elif silhouette_integrated >= 0.5:
        print(f"\n   ✓ GOOD: Integrated UMAP shows strong separation")
    elif silhouette_integrated >= 0.3:
        print(f"\n   ⚠ MODERATE: Integrated UMAP shows acceptable separation")
    else:
        print(f"\n   ⚠ WARNING: Integrated UMAP has weak separation")
    
    # ============================================
    # 2. NEIGHBORHOOD PRESERVATION
    # ============================================
    print("\n" + "-"*70)
    print("2. NEIGHBORHOOD PRESERVATION")
    print("   Higher is better (0-1, >0.7 is good)")
    print("   Measures how well local structure is preserved from high-D to UMAP")
    
    # Sample for speed
    if len(cell_types) > 5000:
        sample_idx2 = np.random.choice(len(cell_types), 5000, replace=False)
        print(f"   (Using {len(sample_idx2):,} sampled cells for speed)")
    else:
        sample_idx2 = np.arange(len(cell_types))
    
    print("\n   Computing neighborhood preservation (this may take 1-2 minutes)...")
    
    # Get high-dimensional representations
    rna_pca = mdata.obsm['X_rna_pca'][sample_idx2]
    atac_lsi = mdata.obsm['X_atac_lsi'][sample_idx2]
    combined = mdata.obsm['X_combined'][sample_idx2]
    
    np_rna = neighborhood_preservation(rna_pca, rna_umap[sample_idx2])
    np_atac = neighborhood_preservation(atac_lsi, atac_umap[sample_idx2])
    np_integrated = neighborhood_preservation(combined, integrated_umap[sample_idx2])
    
    print(f"\n   RNA UMAP:        {np_rna:.4f}")
    print(f"   ATAC UMAP:       {np_atac:.4f}")
    print(f"   Integrated UMAP: {np_integrated:.4f}")
    
    if np_integrated > 0.7:
        print(f"\n   ✓ EXCELLENT: Integrated UMAP preserves local structure very well")
    elif np_integrated > 0.5:
        print(f"\n   ✓ GOOD: Integrated UMAP preserves reasonable local structure")
    else:
        print(f"\n   ⚠ WARNING: Integrated UMAP may over-compress structure")
    
    # ============================================
    # 3. MODALITY AGREEMENT (Clustering Consistency)
    # ============================================
    print("\n" + "-"*70)
    print("3. MODALITY AGREEMENT (Clustering Consistency)")
    print("   Higher is better (0-1)")
    print("   Measures how similarly RNA and ATAC cluster cells")
    
    # Perform Leiden clustering on each modality
    print("\n   Computing Leiden clusters for each modality...")
    
    sc.tl.leiden(mdata.mod['rna'], resolution=1.0, key_added='leiden_rna')
    sc.tl.leiden(mdata.mod['atac'], resolution=1.0, key_added='leiden_atac')
    sc.tl.leiden(mdata, resolution=1.0, key_added='leiden_integrated')
    
    clusters_rna = mdata.mod['rna'].obs['leiden_rna']
    clusters_atac = mdata.mod['atac'].obs['leiden_atac']
    clusters_integrated = mdata.obs['leiden_integrated']
    
    print(f"   RNA clusters: {clusters_rna.nunique()}")
    print(f"   ATAC clusters: {clusters_atac.nunique()}")
    print(f"   Integrated clusters: {clusters_integrated.nunique()}")
    
    # Calculate agreement metrics
    ari_rna_atac = adjusted_rand_score(clusters_rna, clusters_atac)
    nmi_rna_atac = normalized_mutual_info_score(clusters_rna, clusters_atac)
    
    ari_integrated_rna = adjusted_rand_score(clusters_integrated, clusters_rna)
    ari_integrated_atac = adjusted_rand_score(clusters_integrated, clusters_atac)
    
    print(f"\n   RNA vs ATAC Agreement:")
    print(f"     Adjusted Rand Index (ARI): {ari_rna_atac:.4f}")
    print(f"     Normalized Mutual Info (NMI): {nmi_rna_atac:.4f}")
    
    print(f"\n   Integrated vs RNA:  {ari_integrated_rna:.4f}")
    print(f"   Integrated vs ATAC: {ari_integrated_atac:.4f}")
    
    if ari_rna_atac > 0.5:
        print(f"\n   ✓ EXCELLENT: Modalities show high agreement")
    elif ari_rna_atac > 0.3:
        print(f"\n   ✓ GOOD: Modalities show reasonable agreement")
    else:
        print(f"\n   ⚠ NOTE: Modalities capture different aspects of biology (may be expected)")
    
    # ============================================
    # 4. GENERATE SUMMARY TABLE
    # ============================================
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    
    summary_df = pd.DataFrame({
        'Metric': [
            'Silhouette Score',
            'Neighborhood Preservation',
            'RNA-ATAC Agreement (ARI)',
            'RNA-ATAC Agreement (NMI)'
        ],
        'RNA': [f"{silhouette_rna:.4f}", f"{np_rna:.4f}", '-', '-'],
        'ATAC': [f"{silhouette_atac:.4f}", f"{np_atac:.4f}", '-', '-'],
        'Integrated': [f"{silhouette_integrated:.4f}", f"{np_integrated:.4f}", 
                       f"{ari_rna_atac:.4f}", f"{nmi_rna_atac:.4f}"]
    })
    
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary
    summary_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
    print(f"\n✓ Saved: evaluation_metrics.csv")
    
    # ============================================
    # 5. VISUAL DIAGNOSTICS
    # ============================================
    print("\n" + "="*70)
    print("GENERATING VISUAL DIAGNOSTICS")
    print("="*70)
    
    # Plot 1: Per-cell-type silhouette scores
    print("\n1. Per-cell-type silhouette scores...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (umap_data, title) in enumerate([
        (rna_umap[sample_idx], 'RNA UMAP'),
        (atac_umap[sample_idx], 'ATAC UMAP'),
        (integrated_umap[sample_idx], 'Integrated UMAP')
    ]):
        silhouette_vals = silhouette_samples(umap_data, cell_type_labels[sample_idx])
        
        cell_type_scores = pd.DataFrame({
            'cell_type': cell_types.values[sample_idx],
            'silhouette': silhouette_vals
        })
        
        cell_type_scores.boxplot(by='cell_type', column='silhouette', ax=axes[idx], rot=90)
        axes[idx].set_title(title)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('Silhouette Score')
        axes[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.sca(axes[idx])
        plt.xticks(rotation=45, ha='right', fontsize=8)
    
    plt.suptitle('Silhouette Scores by Cell Type', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'silhouette_by_celltype.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: silhouette_by_celltype.png")
    
    # Plot 2: Cluster comparison
    print("\n2. Cluster assignments comparison...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: UMAPs colored by clusters
    for idx, (umap_data, clusters, title) in enumerate([
        (rna_umap, clusters_rna, 'RNA Clusters'),
        (atac_umap, clusters_atac, 'ATAC Clusters'),
        (integrated_umap, clusters_integrated, 'Integrated Clusters')
    ]):
        scatter = axes[0, idx].scatter(umap_data[:, 0], umap_data[:, 1], 
                                       c=clusters.astype('category').cat.codes,
                                       cmap='tab20', s=1, alpha=0.6)
        axes[0, idx].set_title(title, fontweight='bold')
        axes[0, idx].axis('off')
    
    # Row 2: UMAPs colored by true cell types
    n_types = cell_types.nunique()
    if n_types <= 20:
        cmap_choice = 'tab20'
    else:
        cmap_choice = 'rainbow'
    
    for idx, (umap_data, title) in enumerate([
        (rna_umap, 'RNA UMAP'),
        (atac_umap, 'ATAC UMAP'),
        (integrated_umap, 'Integrated UMAP')
    ]):
        scatter = axes[1, idx].scatter(umap_data[:, 0], umap_data[:, 1],
                                       c=cell_type_labels, cmap=cmap_choice, s=1, alpha=0.6)
        axes[1, idx].set_title(f'{title} (True Cell Types)', fontweight='bold')
        axes[1, idx].axis('off')
    
    plt.suptitle('Cluster Assignments vs True Cell Types', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: cluster_comparison.png")
    
    # Plot 3: Metric comparison bar plot
    print("\n3. Metric comparison bar plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Silhouette Score', 'Neighborhood Preservation', 'Modality Agreement']
    rna_scores = [silhouette_rna, np_rna, 0]
    atac_scores = [silhouette_atac, np_atac, 0]
    integrated_scores = [silhouette_integrated, np_integrated, ari_rna_atac]
    
    for idx, (metric, rna_val, atac_val, int_val) in enumerate(zip(
        metrics, rna_scores, atac_scores, integrated_scores
    )):
        ax = axes[idx]
        
        if idx < 2:
            bars = ax.bar(['RNA', 'ATAC', 'Integrated'], 
                         [rna_val, atac_val, int_val],
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
        else:
            bars = ax.bar(['ARI', 'NMI'], [ari_rna_atac, nmi_rna_atac],
                         color=['#9467bd', '#8c564b'])
            ax.set_ylabel('Agreement Score')
            ax.set_ylim([0, 1])
            ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Good threshold')
        
        ax.set_title(metric, fontweight='bold')
        ax.legend()
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved: metrics_comparison.png")
    
    # ============================================
    # FINAL ASSESSMENT
    # ============================================
    print("\n" + "="*70)
    print("FINAL QUALITY ASSESSMENT")
    print("="*70)
    
    score = 0
    max_score = 5
    
    # Criterion 1: Silhouette score
    if silhouette_integrated > 0.5:
        print("\n✓ [1/5] Excellent cell type separation in integrated UMAP")
        score += 1
    elif silhouette_integrated > 0.3:
        print("\n⚠ [0.5/5] Moderate cell type separation in integrated UMAP")
        score += 0.5
    else:
        print("\n✗ [0/5] Weak cell type separation in integrated UMAP")
    
    # Criterion 2: Neighborhood preservation
    if np_integrated > 0.7:
        print("✓ [1/5] Excellent local structure preservation")
        score += 1
    elif np_integrated > 0.5:
        print("⚠ [0.5/5] Moderate local structure preservation")
        score += 0.5
    else:
        print("✗ [0/5] Poor local structure preservation")
    
    # Criterion 3: Better than individual modalities
    if silhouette_integrated > max(silhouette_rna, silhouette_atac):
        print("✓ [1/5] Integrated UMAP outperforms individual modalities")
        score += 1
    elif silhouette_integrated >= 0.9 * max(silhouette_rna, silhouette_atac):
        print("⚠ [0.5/5] Integrated UMAP comparable to best modality")
        score += 0.5
    else:
        print("✗ [0/5] Integrated UMAP underperforms individual modalities")
    
    # Criterion 4: Modality agreement
    if ari_rna_atac > 0.5:
        print("✓ [1/5] High agreement between RNA and ATAC")
        score += 1
    elif ari_rna_atac > 0.3:
        print("⚠ [0.5/5] Moderate agreement between RNA and ATAC")
        score += 0.5
    else:
        print("⚠ [0/5] Low agreement (modalities may capture different biology)")
    
    # Criterion 5: Integration quality
    avg_agree = (ari_integrated_rna + ari_integrated_atac) / 2
    if avg_agree > 0.5:
        print("✓ [1/5] Integrated clusters well-balanced between modalities")
        score += 1
    elif avg_agree > 0.3:
        print("⚠ [0.5/5] Integrated clusters moderately balanced")
        score += 0.5
    else:
        print("✗ [0/5] Integrated clusters dominated by one modality")
    
    print("\n" + "="*70)
    print(f"OVERALL QUALITY SCORE: {score:.1f} / {max_score}")
    print("="*70)
    
    if score >= 4:
        print("\n🌟 EXCELLENT: Your dual-UMAP integration is performing very well!")
        print("   The integrated representation successfully combines both modalities.")
    elif score >= 3:
        print("\n✓ GOOD: Your dual-UMAP integration is performing reasonably well.")
        print("   Results are solid with minor room for improvement.")
    elif score >= 2:
        print("\n⚠ ACCEPTABLE: Integration works but has room for improvement.")
        print("   Consider tuning UMAP parameters or integration strategy.")
    else:
        print("\n⚠ NEEDS IMPROVEMENT: Consider tuning parameters or checking data quality.")
        print("   Review preprocessing steps and integration approach.")
    
    print(f"\n✓ All evaluation results saved to: {output_dir}")
    print("\nFiles generated:")
    print("  - evaluation_metrics.csv")
    print("  - silhouette_by_celltype.png")
    print("  - cluster_comparison.png")
    print("  - metrics_comparison.png")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    
    # Return metrics as dictionary
    results = {
        'silhouette_rna': silhouette_rna,
        'silhouette_atac': silhouette_atac,
        'silhouette_integrated': silhouette_integrated,
        'neighborhood_preservation_rna': np_rna,
        'neighborhood_preservation_atac': np_atac,
        'neighborhood_preservation_integrated': np_integrated,
        'ari_rna_atac': ari_rna_atac,
        'nmi_rna_atac': nmi_rna_atac,
        'ari_integrated_rna': ari_integrated_rna,
        'ari_integrated_atac': ari_integrated_atac,
        'overall_score': score,
        'max_score': max_score
    }
    
    return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Evaluate dual-UMAP quality with quantitative metrics and diagnostic plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_dual_umap_quality.py --input processed_mdata.h5mu --output evaluation_results
  python evaluate_dual_umap_quality.py -i /path/to/data.h5mu -o /path/to/output
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to the processed MuData file (*.h5mu) or directory containing it'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Directory where evaluation results will be saved'
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    try:
        results = evaluate_dual_umap(args.input, args.output)
        return 0
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())