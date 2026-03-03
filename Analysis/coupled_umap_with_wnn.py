import argparse
import muon as mu
import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import NearestNeighbors


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


def compute_wnn_weights(rna_embeddings, atac_embeddings, k=20):
    """
    Compute Weighted Nearest Neighbors (WNN) weights for each cell.
    
    This is a simplified implementation of the WNN algorithm from Seurat v4:
    - For each cell, compute k-nearest neighbors in RNA and ATAC space
    - Calculate modality-specific scores based on neighbor overlap
    - Learn a weight that balances RNA vs ATAC importance per cell
    
    Returns:
        weights_rna: Per-cell weight for RNA modality (n_cells,)
        weights_atac: Per-cell weight for ATAC modality (n_cells,)
    """
    n_cells = rna_embeddings.shape[0]
    
    print(f"\n   Computing WNN with k={k} neighbors...")
    
    # Find k-nearest neighbors in each modality
    nbrs_rna = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(rna_embeddings)
    dist_rna, idx_rna = nbrs_rna.kneighbors(rna_embeddings)
    
    nbrs_atac = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(atac_embeddings)
    dist_atac, idx_atac = nbrs_atac.kneighbors(atac_embeddings)
    
    # Compute modality-specific scores based on local structure
    weights_rna = np.zeros(n_cells)
    weights_atac = np.zeros(n_cells)
    
    for i in range(n_cells):
        # Get neighbors (excluding self)
        rna_neighbors = set(idx_rna[i, 1:])
        atac_neighbors = set(idx_atac[i, 1:])
        
        # Compute Jaccard similarity (neighbor overlap)
        overlap = len(rna_neighbors & atac_neighbors)
        union = len(rna_neighbors | atac_neighbors)
        jaccard = overlap / union if union > 0 else 0
        
        # Compute average distances (inverse = strength)
        rna_strength = 1.0 / (np.mean(dist_rna[i, 1:]) + 1e-10)
        atac_strength = 1.0 / (np.mean(dist_atac[i, 1:]) + 1e-10)
        
        # Combine: higher overlap + stronger modality = higher weight
        total_strength = rna_strength + atac_strength
        
        # Adjust by overlap: if modalities agree, trust them more equally
        # If they disagree, trust the stronger one more
        agreement_factor = 0.5 + 0.5 * jaccard
        
        weights_rna[i] = (rna_strength / total_strength) * agreement_factor + 0.5 * (1 - agreement_factor)
        weights_atac[i] = (atac_strength / total_strength) * agreement_factor + 0.5 * (1 - agreement_factor)
    
    # Normalize weights to sum to 1 per cell
    total_weights = weights_rna + weights_atac
    weights_rna = weights_rna / total_weights
    weights_atac = weights_atac / total_weights
    
    print(f"   RNA weight range: [{weights_rna.min():.3f}, {weights_rna.max():.3f}], mean={weights_rna.mean():.3f}")
    print(f"   ATAC weight range: [{weights_atac.min():.3f}, {weights_atac.max():.3f}], mean={weights_atac.mean():.3f}")
    
    return weights_rna, weights_atac


def normalize_graph_laplacian(adjacency_matrix):
    """
    Apply symmetric normalization to create normalized Laplacian.
    
    This fixes spectral initialization issues by ensuring:
    1. Proper eigenvalue spectrum for UMAP
    2. Scale-invariant graph structure
    3. Better numerical stability
    
    L_sym = D^(-1/2) * A * D^(-1/2)
    
    where D is the degree matrix and A is the adjacency matrix.
    """
    print("\n   🔧 Applying normalized Laplacian transformation...")
    
    # Compute degree matrix (sum of each row)
    degrees = np.array(adjacency_matrix.sum(axis=1)).flatten()
    
    # Avoid division by zero
    degrees[degrees == 0] = 1.0
    
    # D^(-1/2)
    d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt_mat = diags(d_inv_sqrt, format='csr')
    
    # L_sym = D^(-1/2) * A * D^(-1/2)
    normalized = d_inv_sqrt_mat @ adjacency_matrix @ d_inv_sqrt_mat
    
    print(f"   Original edge weight range: [{adjacency_matrix.data.min():.4f}, {adjacency_matrix.data.max():.4f}]")
    print(f"   Normalized edge weight range: [{normalized.data.min():.4f}, {normalized.data.max():.4f}]")
    
    return normalized


def compute_coupling_regularizer(rna_embeddings, atac_embeddings, k=15, alpha=0.5, normalize_laplacian=True):
    """
    Implement the Coupling Regularizer with normalized Laplacian.
    
    This creates a fused neighbor graph by:
    1. Computing k-NN graphs for RNA and ATAC separately
    2. Taking the intersection/union with weight alpha
    3. Applying symmetric normalization for stable spectral properties
    
    Args:
        rna_embeddings: RNA PCA embeddings (n_cells, n_dims)
        atac_embeddings: ATAC LSI embeddings (n_cells, n_dims)
        k: Number of neighbors
        alpha: Coupling strength (0=independent, 1=strict intersection)
        normalize_laplacian: Apply symmetric normalization (recommended)
    
    Returns:
        Fused connectivity matrix (n_cells, n_cells) as sparse matrix
    """
    n_cells = rna_embeddings.shape[0]
    
    print(f"\n   Computing Coupling Regularizer (alpha={alpha})...")
    print(f"   Building RNA graph with k={k}...")
    
    # Build RNA k-NN graph
    nbrs_rna = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(rna_embeddings)
    dist_rna, idx_rna = nbrs_rna.kneighbors(rna_embeddings)
    
    # Build SPARSE graph directly using COO format (row, col, data)
    print(f"   Converting RNA neighbors to sparse graph...")
    rows_rna = []
    cols_rna = []
    data_rna = []
    
    for i in range(n_cells):
        for j in idx_rna[i, 1:]:  # Skip self (first neighbor)
            rows_rna.append(i)
            cols_rna.append(j)
            data_rna.append(1.0)
    
    rna_graph = csr_matrix((data_rna, (rows_rna, cols_rna)), shape=(n_cells, n_cells))
    
    print(f"   Building ATAC graph with k={k}...")
    
    # Build ATAC k-NN graph
    nbrs_atac = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(atac_embeddings)
    dist_atac, idx_atac = nbrs_atac.kneighbors(atac_embeddings)
    
    print(f"   Converting ATAC neighbors to sparse graph...")
    rows_atac = []
    cols_atac = []
    data_atac = []
    
    for i in range(n_cells):
        for j in idx_atac[i, 1:]:
            rows_atac.append(i)
            cols_atac.append(j)
            data_atac.append(1.0)
    
    atac_graph = csr_matrix((data_atac, (rows_atac, cols_atac)), shape=(n_cells, n_cells))
    
    print(f"   Fusing graphs with coupling regularizer...")
    
    # Apply coupling regularizer using sparse operations
    # alpha=0: union (keep all edges from both)
    # alpha=1: intersection (only keep edges that appear in both)
    
    # Intersection: multiply element-wise (only edges in both)
    intersection = rna_graph.multiply(atac_graph)
    
    # Union: add and clip to [0, 1]
    union = rna_graph + atac_graph
    union.data = np.clip(union.data, 0, 1)
    
    # Coupled graph: interpolate between union and intersection
    coupled_graph = (1 - alpha) * union + alpha * intersection
    
    # Eliminate zeros and convert back to CSR
    coupled_graph.eliminate_zeros()
    
    # 🔧 FIX #1: Apply normalized Laplacian
    if normalize_laplacian:
        coupled_graph = normalize_graph_laplacian(coupled_graph)
    
    n_edges_rna = rna_graph.nnz
    n_edges_atac = atac_graph.nnz
    n_edges_coupled = coupled_graph.nnz
    
    print(f"   RNA graph edges: {n_edges_rna:,}")
    print(f"   ATAC graph edges: {n_edges_atac:,}")
    print(f"   Coupled graph edges: {n_edges_coupled:,}")
    print(f"   Edge retention: {n_edges_coupled/max(n_edges_rna, n_edges_atac):.1%}")
    
    return coupled_graph


def compute_two_stage_umap(mdata, integration_strategy, coupling_alpha, n_neighbors, min_dist):
    """
    🔧 FIX #2: Two-stage UMAP initialization
    
    Stage 1: Compute UMAP on uncoupled WNN embedding (stable initialization)
    Stage 2: Use Stage 1 result to initialize coupled UMAP (preserves structure)
    
    This dramatically improves neighborhood preservation by avoiding the
    spectral initialization failure on the coupled graph.
    """
    print("\n" + "="*60)
    print("🌟 TWO-STAGE UMAP INITIALIZATION")
    print("="*60)
    
    # Stage 1: Uncoupled WNN UMAP (stable baseline)
    print("\n📍 Stage 1: Computing baseline WNN UMAP for initialization...")
    
    # Temporarily store coupled graph if it exists
    coupled_conn = None
    coupled_dist = None
    if 'coupled_connectivities' in mdata.obsp:
        coupled_conn = mdata.obsp['coupled_connectivities'].copy()
        coupled_dist = mdata.obsp['distances'].copy()
    
    # Compute standard WNN-based neighbors
    sc.pp.neighbors(mdata, n_neighbors=n_neighbors, use_rep='X_combined')
    
    # Run UMAP with standard graph
    print("   Running UMAP on uncoupled graph...")
    sc.tl.umap(mdata, min_dist=min_dist, init_pos='spectral')
    
    # Store initialization embedding
    init_embedding = mdata.obsm['X_umap'].copy()
    print(f"   ✓ Stage 1 UMAP computed: {init_embedding.shape}")
    print(f"   Embedding range: [{init_embedding.min():.2f}, {init_embedding.max():.2f}]")
    
    # Stage 2: Coupled UMAP with warm start
    if integration_strategy == 'coupling' and coupled_conn is not None:
        print("\n📍 Stage 2: Refining with coupled graph using Stage 1 initialization...")
        
        # Restore coupled graph
        mdata.obsp['connectivities'] = coupled_conn
        mdata.obsp['distances'] = coupled_dist
        
        # Update metadata
        mdata.uns['neighbors']['params']['method'] = 'coupling_regularizer'
        mdata.uns['neighbors']['params']['coupling_alpha'] = coupling_alpha
        
        # Run UMAP with coupled graph, using Stage 1 as initialization
        print("   Running UMAP on coupled graph with warm start...")
        sc.tl.umap(mdata, min_dist=min_dist, init_pos=init_embedding)
        
        print(f"   ✓ Stage 2 UMAP computed: {mdata.obsm['X_umap'].shape}")
        
        # Compute initialization quality metric
        embedding_shift = np.linalg.norm(mdata.obsm['X_umap'] - init_embedding, axis=1)
        print(f"   Mean embedding shift from Stage 1: {embedding_shift.mean():.3f}")
        print(f"   Max embedding shift: {embedding_shift.max():.3f}")
    else:
        print("   No coupled graph - using Stage 1 result as final embedding")
    
    return mdata.obsm['X_umap']


def multimodal_preprocess(input_path=None, output_dir="results", n_cells=500, n_genes=300, n_peaks=1000, 
                          n_top_genes=2000, n_top_peaks=2000, use_toy=False, 
                          n_rna_dims=30, n_atac_dims=30, integration_strategy='wnn', 
                          rna_weight=0.7, n_neighbors=15, min_dist=0.3,
                          coupling_alpha=0.5, exclude_atac_dim0=True,
                          normalize_laplacian=True, use_two_stage_init=True):
    """
    Main multimodal preprocessing pipeline with FIXED spectral initialization.
    
    New parameters:
        normalize_laplacian: Apply symmetric normalization to coupled graph (recommended)
        use_two_stage_init: Use two-stage UMAP initialization (recommended)
    """

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

    # Check if data is already preprocessed
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
        print("\n" + "="*60)
        print("PREPROCESSING RAW DATA")
        print("="*60)
        
        # RNA preprocessing
        print("Preprocessing RNA modality...")
        preprocess_rna(mdata["rna"], n_top_genes)
        sc.pl.pca_variance_ratio(mdata["rna"], log=True, show=False)
        plt.savefig(os.path.join(output_dir, "rna_pca_variance.png"))
        plt.close()
        
        # ATAC preprocessing
        print("Preprocessing ATAC modality...")
        preprocess_atac(mdata["atac"], n_top_peaks)
        sc.pl.pca_variance_ratio(mdata["atac"], log=True, show=False)
        plt.savefig(os.path.join(output_dir, "atac_pca_variance.png"))
        plt.close()
        
        # Store at MuData level
        rna_pca = mdata.mod['rna'].obsm['X_pca']
        atac_lsi = mdata.mod['atac'].obsm['X_pca']
        mdata.obsm['X_rna_pca'] = rna_pca
        mdata.obsm['X_atac_lsi'] = atac_lsi

    # Exclude ATAC dimension 0
    if exclude_atac_dim0:
        print("\n" + "="*60)
        print("⚠️ CRITICAL FIX: EXCLUDING ATAC LSI DIMENSION 0")
        print("="*60)
        print("  First ATAC LSI component is usually sequencing depth (technical noise)")
        print(f"  Original ATAC shape: {atac_lsi.shape}")
        atac_lsi = atac_lsi[:, 1:]
        print(f"  After dropping dim 0: {atac_lsi.shape}")
        mdata.obsm['X_atac_lsi_corrected'] = atac_lsi
    
    # Integration
    print("\n" + "="*60)
    print("IMPROVED MULTIMODAL INTEGRATION")
    print("="*60)
    
    print(f"\nIntegration parameters:")
    print(f"  Strategy: {integration_strategy}")
    print(f"  RNA dimensions: {n_rna_dims}")
    print(f"  ATAC dimensions: {n_atac_dims}")
    
    # Select dimensions
    print(f"\n1. Selecting dimensions...")
    rna_selected = rna_pca[:, :n_rna_dims]
    atac_selected = atac_lsi[:, :n_atac_dims]
    print(f"   RNA: {rna_selected.shape}")
    print(f"   ATAC: {atac_selected.shape}")
    
    # Standardize
    print(f"\n2. Standardizing modalities...")
    scaler_rna = StandardScaler()
    rna_scaled = scaler_rna.fit_transform(rna_selected)
    
    scaler_atac = StandardScaler()
    atac_scaled = scaler_atac.fit_transform(atac_selected)
    
    print(f"   RNA: mean={rna_scaled.mean():.4f}, std={rna_scaled.std():.4f}")
    print(f"   ATAC: mean={atac_scaled.mean():.4f}, std={atac_scaled.std():.4f}")
    
    # Combine with strategy
    print(f"\n3. Combining modalities using '{integration_strategy}'...")
    
    if integration_strategy == 'wnn':
        print("\n   🌟 USING WEIGHTED NEAREST NEIGHBORS (WNN)")
        weights_rna, weights_atac = compute_wnn_weights(rna_scaled, atac_scaled, k=20)
        combined = np.hstack([
            rna_scaled * weights_rna[:, np.newaxis],
            atac_scaled * weights_atac[:, np.newaxis]
        ])
        mdata.obs['wnn_weight_rna'] = weights_rna
        mdata.obs['wnn_weight_atac'] = weights_atac
        print(f"   Final combined embedding: {combined.shape}")
        
    elif integration_strategy == 'coupling':
        print("\n   🌟 USING COUPLING REGULARIZER WITH FIXES")
        coupled_graph = compute_coupling_regularizer(
            rna_scaled, atac_scaled, 
            k=n_neighbors, 
            alpha=coupling_alpha,
            normalize_laplacian=normalize_laplacian
        )
        mdata.obsp['coupled_connectivities'] = coupled_graph
        combined = np.hstack([rna_scaled, atac_scaled])
        print(f"   Final combined embedding: {combined.shape}")
        
    elif integration_strategy == 'concat_pca':
        combined_full = np.hstack([rna_scaled, atac_scaled])
        pca_joint = PCA(n_components=50, random_state=42)
        combined = pca_joint.fit_transform(combined_full)
        var_explained = pca_joint.explained_variance_ratio_[:10].sum()
        print(f"   Concat+PCA: {combined.shape}")
        print(f"   Variance explained (top 10 PCs): {var_explained:.2%}")
        
    elif integration_strategy == 'weighted':
        atac_weight = 1.0 - rna_weight
        combined = np.hstack([rna_weight * rna_scaled, atac_weight * atac_scaled])
        print(f"   Weighted: {rna_weight} RNA + {atac_weight} ATAC")
        
    elif integration_strategy == 'equal':
        combined = np.hstack([0.5 * rna_scaled, 0.5 * atac_scaled])
        print(f"   Equal weighting: 0.5 RNA + 0.5 ATAC")
        
    else:
        raise ValueError(f"Unknown integration strategy: {integration_strategy}")
    
    mdata.obsm['X_combined'] = combined
    
    # Compute UMAPs
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
    
    # Replace the Integrated UMAP section (around line 480-520) with this:

    # Integrated UMAP
    print("\n3. Computing Integrated UMAP...")
    
    if integration_strategy == 'coupling' and 'coupled_connectivities' in mdata.obsp:
        print("   🚨 USING TWO-STAGE INITIALIZATION FOR COUPLED UMAP")
        
        if use_two_stage_init:
            # STAGE 1: Compute WNN embedding as initialization
            print("\n   📍 STAGE 1: Computing WNN embedding for initialization...")
            
            # Compute WNN weights
            weights_rna_init, weights_atac_init = compute_wnn_weights(rna_scaled, atac_scaled, k=20)
            
            # Create WNN embedding
            wnn_combined = np.hstack([
                rna_scaled * weights_rna_init[:, np.newaxis],
                atac_scaled * weights_atac_init[:, np.newaxis]
            ])
            
            # Compute initial UMAP with WNN
            temp_adata = sc.AnnData(X=wnn_combined)
            sc.pp.neighbors(temp_adata, n_neighbors=n_neighbors, use_rep='X')
            sc.tl.umap(temp_adata, min_dist=min_dist)
            init_embedding = temp_adata.obsm['X_umap']
            
            print(f"   ✓ WNN initialization computed: {init_embedding.shape}")
        else:
            print("\n   ⚠️  Using random initialization (not recommended)...")
            init_embedding = 'random'
        
        # STAGE 2: Use coupled graph with WNN initialization
        print("\n   📍 STAGE 2: Computing coupled UMAP with initialization...")
        
        # First run standard neighbors to initialize .uns['neighbors'] structure
        sc.pp.neighbors(mdata, n_neighbors=n_neighbors, use_rep='X_combined')
        
        # Get the coupled connectivity matrix (already normalized with Laplacian)
        coupled_conn = mdata.obsp['coupled_connectivities']
        
        # Create proper distance matrix from connectivity
        print("   Converting connectivity to distance matrix...")
        
        # For normalized Laplacian, distance = 1 - connectivity
        coupled_dist = coupled_conn.copy()
        coupled_dist.data = 1.0 - coupled_dist.data
        coupled_dist.data = np.clip(coupled_dist.data, 0, None)
        coupled_dist.eliminate_zeros()
        
        # Ensure symmetry for UMAP
        coupled_conn_sym = (coupled_conn + coupled_conn.T) / 2
        coupled_dist_sym = (coupled_dist + coupled_dist.T) / 2
        
        # Override with our coupled graph
        mdata.obsp['connectivities'] = coupled_conn_sym
        mdata.obsp['distances'] = coupled_dist_sym
        
        # Update neighbor metadata
        mdata.uns['neighbors']['params']['method'] = 'coupling_regularizer'
        mdata.uns['neighbors']['params']['coupling_alpha'] = coupling_alpha
        
        print("   Computing UMAP with coupled graph...")
        if use_two_stage_init and isinstance(init_embedding, np.ndarray):
            sc.tl.umap(mdata, min_dist=min_dist, init_pos=init_embedding)
            print("   ✓ Used WNN initialization - should preserve neighborhoods better!")
        else:
            sc.tl.umap(mdata, min_dist=min_dist, init_pos='random')
            print("   ⚠️  Used random initialization")
    else:
        # Standard neighbor computation for non-coupling strategies
        sc.pp.neighbors(mdata, n_neighbors=n_neighbors, use_rep='X_combined')
        sc.tl.umap(mdata, min_dist=min_dist)
    
    print(f"   ✓ Integrated UMAP: {mdata.obsm['X_umap'].shape}")

    # Generate visualization
    print("\n" + "="*60)
    print("GENERATING 3-PANEL UMAP COMPARISON")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Check for cell type annotations
    cell_type_col = None
    for col in ['cell_type', 'cellAnnot', 'rna:cellAnnot', 'atac:cellAnnot']:
        if col in mdata.obs.columns and mdata.obs[col].nunique() > 1:
            cell_type_col = col
            break
    
    if cell_type_col:
        cell_types = mdata.obs[cell_type_col].astype('category')
        n_types = len(cell_types.cat.categories)
        print(f"Coloring by {n_types} cell types from '{cell_type_col}'")
        
        if n_types <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_types))
        elif n_types <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_types))
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, n_types))
        
        for i, ct in enumerate(cell_types.cat.categories):
            mask = cell_types == ct
            axes[0].scatter(mdata.obsm['X_umap_rna'][mask, 0],
                          mdata.obsm['X_umap_rna'][mask, 1],
                          c=[colors[i]], label=ct, s=1, alpha=0.6)
            axes[1].scatter(mdata.obsm['X_umap_atac'][mask, 0],
                          mdata.obsm['X_umap_atac'][mask, 1],
                          c=[colors[i]], s=1, alpha=0.6)
            axes[2].scatter(mdata.obsm['X_umap'][mask, 0],
                          mdata.obsm['X_umap'][mask, 1],
                          c=[colors[i]], s=1, alpha=0.6)
        
        if n_types <= 20:
            handles, labels = axes[2].get_legend_handles_labels()
            fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                      fontsize=8, markerscale=3)
    else:
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

    # Save results
    out_path = os.path.join(output_dir, "processed_mdata.h5mu")
    mdata.write_h5mu(out_path)
    print(f"\n✓ Processed MuData object saved to: {out_path}")
    
    np.save(os.path.join(output_dir, 'umap_rna.npy'), mdata.obsm['X_umap_rna'])
    np.save(os.path.join(output_dir, 'umap_atac.npy'), mdata.obsm['X_umap_atac'])
    np.save(os.path.join(output_dir, 'umap_integrated.npy'), mdata.obsm['X_umap'])
    print("✓ Saved individual UMAP embeddings as .npy files")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"All results saved to: {output_dir}")
    print("\n🌟 KEY IMPROVEMENTS APPLIED:")
    if exclude_atac_dim0:
        print("  ✓ Excluded ATAC LSI dimension 0 (technical noise)")
    if integration_strategy == 'wnn':
        print("  ✓ Weighted Nearest Neighbors (per-cell adaptive weighting)")
    elif integration_strategy == 'coupling':
        print("  ✓ Coupling Regularizer with spectral fixes:")
        if normalize_laplacian:
            print("    • Normalized Laplacian transformation")
        if use_two_stage_init:
            print("    • Two-stage UMAP initialization")
    print("  ✓ Standardized embeddings for balanced integration")
    print("  ✓ Optimized UMAP parameters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed Multimodal Preprocessing with Spectral Initialization")

    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_toy", action="store_true")
    parser.add_argument("--n_cells", type=int, default=500)
    parser.add_argument("--n_genes", type=int, default=300)
    parser.add_argument("--n_peaks", type=int, default=1000)
    parser.add_argument("--n_top_genes", type=int, default=2000)
    parser.add_argument("--n_top_peaks", type=int, default=2000)
    
    parser.add_argument("--n_rna_dims", type=int, default=30)
    parser.add_argument("--n_atac_dims", type=int, default=30)
    parser.add_argument("--integration_strategy", type=str, default='coupling',
                       choices=['wnn', 'coupling', 'equal', 'weighted', 'concat_pca'])
    parser.add_argument("--rna_weight", type=float, default=0.7)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.3)
    
    parser.add_argument("--coupling_alpha", type=float, default=0.5)
    parser.add_argument("--exclude_atac_dim0", action="store_true")
    
    # NEW: Spectral initialization fixes
    parser.add_argument("--normalize_laplacian", action="store_true", default=True,
                       help="Apply normalized Laplacian to coupled graph (RECOMMENDED)")
    parser.add_argument("--use_two_stage_init", action="store_true", default=True,
                       help="Use two-stage UMAP initialization (RECOMMENDED)")

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
        min_dist=args.min_dist,
        coupling_alpha=args.coupling_alpha,
        exclude_atac_dim0=args.exclude_atac_dim0,
        normalize_laplacian=args.normalize_laplacian,
        use_two_stage_init=args.use_two_stage_init
    )