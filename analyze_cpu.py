"""
Analysis pipeline for token embeddings across checkpoints

produces: 
    - UMAP visualizations per checkpoint 
    - K-NN & linear-probe metrics with train/test splits 
    - silhouette score per POS 
    - average intra-class and inter-class cosine similarity per POS 
    - top-k nearest neighbors for selected example tokens 
    - token trajectory plots (PCA 2D) showing movements across checkpoints 
    - saves json results and plots to analysis_results/ folder 
"""