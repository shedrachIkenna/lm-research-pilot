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


