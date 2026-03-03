Project: Dual-UMAP Alignment for Multiome Data
Overview
This project explores methods to generate separate but aligned embeddings for single-cell multiome data (RNA + ATAC) using UMAP. Typically, tools produce either a single integrated UMAP or two independent ones that may not align. Our goal is to develop a pipeline that:
1.Runs UMAP separately on RNA and ATAC modalities.

2.Encourages alignment between the two embeddings (via coupling loss, Procrustes, or existing frameworks).

3.Evaluates alignment quality with quantitative metrics.

4.Produces visual and numerical comparisons across methods.

Dataset: GSE140203 (SHARE-seq, RNA + ATAC from the same single cells).
