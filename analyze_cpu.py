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

import os 
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np 
import matplotlib as plt 
import torch 
import umap 
import warnings 
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


warnings.filterwarnings('ignore', category=UserWarning) # Suppress user warning from appearing in the console 
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress Future warnings about a library from cluttering the output in the console 
torch.set_num_threads(2) # Limit pytorch to using only 2 CPU threads for parallel operations (like matrix multiplication)

# Default paths 
CHECKPOINT_DIR = "pilot_gpt2_cpu"
TOKEN_POS_MAP = "token_pos_map.json"
OUTDIR = "analysis_results"

# Exemplar tokens to track (common English words)
DEFAULT_EXEMPLAR_WORDS = ["the", "dog", "run", "was", "city", "new", "music", "time"]
TOP_K_NEIGHBORS = 10

# Dimensionality reduction settings
PCA_DIM = 50
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


def load_token_pos_map(path):
    """Load token ID -> POS mapping from JSON file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Token POS map not found: {path} \nRun build_token_pos_map_cpu.py first")
    
    with open(path, 'r', encoding='utf-8') as f: # Open the json file in read mode with UTF-8 encoding 
        mapping = json.load(f) # parse the json content into a python dictionary 

    # Convert all dictionary keys (they were saved as strings in the json file) to integers. 
    mapping = {int(k): v for k, v in mapping.items()} #  For example, {"1234": "NOUN"} becomes {1234: "NOUN"}

    print(f" Loaded POS mapping for {len(mapping):,} tokens")

    # POS distribution count 
    pos_counts = Counter(mapping.values()) # Count how many tokens belong to each POS category.
    print(f" Top 5 POS categories: {dict(pos_counts.most_common(5))}") # Print the 5 most common POS categories and their counts.

    return mapping 



    