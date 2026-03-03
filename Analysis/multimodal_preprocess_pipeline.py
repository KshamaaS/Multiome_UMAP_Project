import argparse
import muon as mu
import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def ensure_dir(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def generate_toy_data(n_cells=500, n_genes=300, n_peaks=1000):
    """Generate synthetic RNA + ATAC toy dataset."""
    X_rna = np.random.poisson(1.5, size=(n_cells, n_genes))
    adata_rna = sc.AnnData(X=X_rna)
    adata_rna.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata_rna.obs_names = [f"cell_{i}" for i in range(n_cells)]

    X_atac = np.random.binomial(1, 0.05, size=(n_cells, n_peaks))
    adata_atac = sc.AnnData(X=X_atac)
    adata_atac.var_names = [f"peak_{i}" for i in range(n_peaks)]
    adata_atac.obs_names = adata_rna.obs_names

    mdata = mu.MuData({"rna": adata_rna, "atac": adata_atac})
    return mdata


def preprocess_rna(adata, n_top_genes=2000):
    """Preprocess RNA modality in-place."""
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")
    adata._inplace_subset_var(adata.var["highly_variable"])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)


def preprocess_atac(adata, n_top_peaks=2000):
    """Preprocess ATAC modality in-place."""
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.umap(adata)


def multimodal_preprocess(input_path=None, output_dir="results", n_cells=500, n_genes=300, n_peaks=1000, 
                          n_top_genes=2000, n_top_peaks=2000, use_toy=False, 
                          n_rna_dims=30, n_atac_dims=30, integration_strategy='concat_pca', 
                          rna_weight=0.7, n_neighbors=15, min_dist=0.3):
    """Main multimodal preprocessing pipeline with improved integration."""

    ensure_dir(output_dir)

    # Step 1: Load or create data
    if use_toy or input_path is None:
        print("Generating toy synthetic multiome dataset...")
        mdata = generate_toy_data(n_cells, n_genes, n_peaks)
        toy_path = os.path.join(output_dir, "toy_multiome_synthetic.h5mu")
        mdata.write_h5mu(toy_path)
        print(f"Toy dataset saved to {toy_path}")
    else:
        print(f"Loading existing dataset from {input_path}")
        mdata = mu.read_h5mu(input_path)
        print(f"✓ Loaded: {mdata.n_obs:,} cells")
        print(f"  RNA: {mdata.mod['rna'].shape}")
        print(f"  ATAC: {mdata.mod['atac'].shape}")

    # ============================================
    # CHECK IF DATA IS ALREADY PREPROCESSED
    # ============================================
    is_preprocessed = ('X_rna_pca' in mdata.obsm and 'X_atac_lsi' in mdata.obsm)
    
    if is_preprocessed:
        print("\n" + "="*60)
        print("✓ DATA ALREADY PREPROCESSED - USING EXISTING EMBEDDINGS")
        print("="*60)
        print("  Using existing high-quality embeddings:")
        print(f"    RNA PCA: {mdata.obsm['X_rna_pca'].shape}")
        print(f"    ATAC LSI: {mdata.obsm['X_atac_lsi'].shape}")
        
        rna_pca = mdata.obsm['X_rna_pca']
        atac_lsi = mdata.obsm['X_atac_lsi']
        
    else:
        # ============================================
        # ORIGINAL PREPROCESSING FOR RAW DATA
        # ============================================
        print("\n" + "="*60)
        print("PREPROCESSING RAW DATA")
        print("="*60)
        
        # Step 2: RNA preprocessing
        print("Preprocessing RNA modality...")
        preprocess_rna(mdata["rna"], n_top_genes)
        sc.pl.pca_variance_ratio(mdata["rna"], log=True, show=False)
        plt.savefig(os.path.join(output_dir, "rna_pca_variance.png"))
        plt.close()
        
        # Step 3: ATAC preprocessing
        print("Preprocessing ATAC modality...")
        preprocess_atac(mdata["atac"], n_top_peaks)
        sc.pl.pca_variance_ratio(mdata["atac"], log=True, show=False)
        plt.savefig(os.path.join(output_dir, "atac_pca_variance.png"))
        plt.close()
        
        # Store at MuData level
        rna_pca = mdata.mod['rna'].obsm['X_pca']
        atac_lsi = mdata.mod['atac'].obsm['X_pca']  # For consistency
        mdata.obsm['X_rna_pca'] = rna_pca
        mdata.obsm['X_atac_lsi'] = atac_lsi

    # ============================================
    # IMPROVED INTEGRATION STRATEGY
    # ============================================
    print("\n" + "="*60)
    print("IMPROVED MULTIMODAL INTEGRATION")
    print("="*60)
    
    print(f"\nIntegration parameters:")
    print(f"  Strategy: {integration_strategy}")
    print(f"  RNA dimensions: {n_rna_dims}")
    print(f"  ATAC dimensions: {n_atac_dims}")
    print(f"  RNA weight: {rna_weight}" if integration_strategy == 'weighted' else "")
    
    # Step 1: Select informative dimensions
    print(f"\n1. Selecting dimensions...")
    rna_selected = rna_pca[:, :n_rna_dims]
    atac_selected = atac_lsi[:, :n_atac_dims]
    print(f"   RNA: {rna_selected.shape}")
    print(f"   ATAC: {atac_selected.shape}")
    
    # Step 2: Standardize each modality
    print(f"\n2. Standardizing modalities...")
    scaler_rna = StandardScaler()
    rna_scaled = scaler_rna.fit_transform(rna_selected)
    
    scaler_atac = StandardScaler()
    atac_scaled = scaler_atac.fit_transform(atac_selected)
    
    print(f"   RNA: mean={rna_scaled.mean():.4f}, std={rna_scaled.std():.4f}")
    print(f"   ATAC: mean={atac_scaled.mean():.4f}, std={atac_scaled.std():.4f}")
    
    # Step 3: Combine with chosen strategy
    print(f"\n3. Combining modalities using '{integration_strategy}'...")
    
    if integration_strategy == 'equal':
        # Equal weighting
        combined = np.hstack([0.5 * rna_scaled, 0.5 * atac_scaled])
        print(f"   Equal weighting: 0.5 RNA + 0.5 ATAC")
        
    elif integration_strategy == 'weighted':
        # Weighted combination
        atac_weight = 1.0 - rna_weight
        combined = np.hstack([rna_weight * rna_scaled, atac_weight * atac_scaled])
        print(f"   Weighted: {rna_weight} RNA + {atac_weight} ATAC")
        
    elif integration_strategy == 'concat_pca':
        # Concatenate then reduce with PCA (RECOMMENDED)
        combined_full = np.hstack([rna_scaled, atac_scaled])
        pca_joint = PCA(n_components=50, random_state=42)
        combined = pca_joint.fit_transform(combined_full)
        var_explained = pca_joint.explained_variance_ratio_[:10].sum()
        print(f"   Concat+PCA: {combined.shape}")
        print(f"   Variance explained (top 10 PCs): {var_explained:.2%}")
        
    else:
        raise ValueError(f"Unknown integration strategy: {integration_strategy}")
    
    print(f"   Final combined embedding: {combined.shape}")
    mdata.obsm['X_combined'] = combined
    
    # ============================================
    # COMPUTE UMAPs
    # ============================================
    print("\n" + "="*60)
    print("COMPUTING UMAPs")
    print("="*60)
    
    print(f"\nUMAP parameters: n_neighbors={n_neighbors}, min_dist={min_dist}")
    
    # RNA UMAP
    print("\n1. Computing RNA UMAP...")
    mdata.mod['rna'].obsm['X_pca'] = rna_scaled
    sc.pp.neighbors(mdata.mod['rna'], n_neighbors=n_neighbors, use_rep='X_pca')
    sc.tl.umap(mdata.mod['rna'], min_dist=min_dist)
    mdata.obsm['X_umap_rna'] = mdata.mod['rna'].obsm['X_umap'].copy()
    print(f"   ✓ RNA UMAP: {mdata.obsm['X_umap_rna'].shape}")
    
    # ATAC UMAP
    print("\n2. Computing ATAC UMAP...")
    mdata.mod['atac'].obsm['X_lsi'] = atac_scaled
    sc.pp.neighbors(mdata.mod['atac'], n_neighbors=n_neighbors, use_rep='X_lsi')
    sc.tl.umap(mdata.mod['atac'], min_dist=min_dist)
    mdata.obsm['X_umap_atac'] = mdata.mod['atac'].obsm['X_umap'].copy()
    print(f"   ✓ ATAC UMAP: {mdata.obsm['X_umap_atac'].shape}")
    
    # Integrated UMAP
    print("\n3. Computing Integrated UMAP...")
    sc.pp.neighbors(mdata, n_neighbors=n_neighbors, use_rep='X_combined')
    sc.tl.umap(mdata, min_dist=min_dist)
    print(f"   ✓ Integrated UMAP: {mdata.obsm['X_umap'].shape}")

    # ============================================
    # GENERATE COMPARISON VISUALIZATION
    # ============================================
    print("\n" + "="*60)
    print("GENERATING 3-PANEL UMAP COMPARISON")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Check if we have cell type annotations
    cell_type_col = None
    for col in ['cell_type', 'cellAnnot', 'rna:cellAnnot', 'atac:cellAnnot']:
        if col in mdata.obs.columns and mdata.obs[col].nunique() > 1:
            cell_type_col = col
            break
    
    if cell_type_col:
        cell_types = mdata.obs[cell_type_col].astype('category')
        n_types = len(cell_types.cat.categories)
        print(f"Coloring by {n_types} cell types from '{cell_type_col}'")
        
        # Choose color palette
        if n_types <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_types))
        elif n_types <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_types))
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, n_types))
        
        # Plot RNA UMAP
        for i, ct in enumerate(cell_types.cat.categories):
            mask = cell_types == ct
            axes[0].scatter(
                mdata.obsm['X_umap_rna'][mask, 0],
                mdata.obsm['X_umap_rna'][mask, 1],
                c=[colors[i]], label=ct, s=1, alpha=0.6
            )
        
        # Plot ATAC UMAP
        for i, ct in enumerate(cell_types.cat.categories):
            mask = cell_types == ct
            axes[1].scatter(
                mdata.obsm['X_umap_atac'][mask, 0],
                mdata.obsm['X_umap_atac'][mask, 1],
                c=[colors[i]], s=1, alpha=0.6
            )
        
        # Plot Integrated UMAP
        for i, ct in enumerate(cell_types.cat.categories):
            mask = cell_types == ct
            axes[2].scatter(
                mdata.obsm['X_umap'][mask, 0],
                mdata.obsm['X_umap'][mask, 1],
                c=[colors[i]], s=1, alpha=0.6
            )
        
        # Add legend if not too many types
        if n_types <= 20:
            handles, labels = axes[2].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                      fontsize=8, markerscale=3)
    else:
        # No cell types - plot all cells in gray
        print("No cell type annotations - plotting all cells")
        axes[0].scatter(mdata.obsm['X_umap_rna'][:, 0], mdata.obsm['X_umap_rna'][:, 1],
                       c='gray', s=1, alpha=0.5)
        axes[1].scatter(mdata.obsm['X_umap_atac'][:, 0], mdata.obsm['X_umap_atac'][:, 1],
                       c='gray', s=1, alpha=0.5)
        axes[2].scatter(mdata.obsm['X_umap'][:, 0], mdata.obsm['X_umap'][:, 1],
                       c='gray', s=1, alpha=0.5)
    
    axes[0].set_title('RNA UMAP', fontsize=14, fontweight='bold')
    axes[1].set_title('ATAC UMAP', fontsize=14, fontweight='bold')
    axes[2].set_title(f'Integrated UMAP ({integration_strategy})', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.axis('off')
    
    plt.tight_layout()
    comparison_file = os.path.join(output_dir, "three_umaps_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {comparison_file}")

    # ============================================
    # SAVE RESULTS
    # ============================================
    out_path = os.path.join(output_dir, "processed_mdata.h5mu")
    mdata.write_h5mu(out_path)
    print(f"\n✓ Processed MuData object saved to: {out_path}")
    
    # Save individual UMAP arrays
    np.save(os.path.join(output_dir, 'umap_rna.npy'), mdata.obsm['X_umap_rna'])
    np.save(os.path.join(output_dir, 'umap_atac.npy'), mdata.obsm['X_umap_atac'])
    np.save(os.path.join(output_dir, 'umap_integrated.npy'), mdata.obsm['X_umap'])
    print("✓ Saved individual UMAP embeddings as .npy files")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print("\nIntegration improvements applied:")
    print("  ✓ Dimension selection (reduced noise)")
    print("  ✓ Standardization (balanced modalities)")
    print(f"  ✓ Strategy: {integration_strategy}")
    print("  ✓ Optimized UMAP parameters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuData Multimodal Preprocessing Pipeline")

    parser.add_argument("--input_path", type=str, default=None, help="Path to input .h5mu file (optional if use_toy=True)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save outputs")
    parser.add_argument("--use_toy", action="store_true", help="Generate synthetic data instead of loading file")
    parser.add_argument("--n_cells", type=int, default=500, help="Number of cells for toy data")
    parser.add_argument("--n_genes", type=int, default=300, help="Number of genes for toy data")
    parser.add_argument("--n_peaks", type=int, default=1000, help="Number of peaks for toy data")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Top variable genes to keep")
    parser.add_argument("--n_top_peaks", type=int, default=2000, help="Top variable peaks to keep")
    
    # NEW: Integration parameters
    parser.add_argument("--n_rna_dims", type=int, default=30, help="Number of RNA PCA dimensions to use")
    parser.add_argument("--n_atac_dims", type=int, default=30, help="Number of ATAC LSI dimensions to use")
    parser.add_argument("--integration_strategy", type=str, default='concat_pca', 
                       choices=['equal', 'weighted', 'concat_pca'],
                       help="Integration strategy: equal, weighted, or concat_pca (recommended)")
    parser.add_argument("--rna_weight", type=float, default=0.7, help="RNA weight for 'weighted' strategy")
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for UMAP")
    parser.add_argument("--min_dist", type=float, default=0.3, help="UMAP min_dist parameter")

    args = parser.parse_args()

    multimodal_preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_peaks=args.n_peaks,
        n_top_genes=args.n_top_genes,
        n_top_peaks=args.n_top_peaks,
        use_toy=args.use_toy,
        n_rna_dims=args.n_rna_dims,
        n_atac_dims=args.n_atac_dims,
        integration_strategy=args.integration_strategy,
        rna_weight=args.rna_weight,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist
    )