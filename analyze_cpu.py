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


def load_training_metadata(checkpoint_dir):
    """Load training metadata from checkpoint directory if available"""
    
    metadata_path = Path(checkpoint_dir) / "training_metadata.json" # the metadata_path becomes checkpoint_dir/training_metadata.json. 

    if metadata_path.exists(): #  Checks if the metadata file exists 
        with open(metadata_path, 'r') as f: # if it does, open it in read mode
            return json.load(f) # parse and return the JSON content 
    
    return None # if file doesn't exist, return None instead of raising a error 

def get_checkpoints(base_dir):
    """Get sorted list of checkpoint directories"""

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {base_dir}\nRun train_cpu.py first")
    
    checkpoints = [] # list to save checkpoints directories paths 
    
    for item in Path(base_dir).iterdir(): # iterates through all the items in the base directory 
        if item.is_dir() and item.name.startswith("checkpoint-"): # finds directories whose name starts with "checkpoint-"
            checkpoints.append(str(item)) # Adds the path of those directories as a string to the checkpoints list 

    # Add final checkpoint to the list 
    final_path = Path(base_dir) / "final" # construct the path to the final directory
    if final_path.is_dir(): # check if it exists and is a directory 
        checkpoints.append(str(final_path)) # add it to the checkpoints list 

    # Sorting function 
    def checkpoint_key(path):
        name = os.path.basename(path) # Extracts the directory name from the full path (e.g., "outputs/checkpoint-100" → "checkpoint-100").
        if name == 'final': # if the directory name is == final 
            return (float('inf'), name) # return infinity as its sort key so it always comes last 
        
        if name.startswith("checkpoint-"): # for directories that starts with "checkpoint-"
            try:
                num = int(name.split("-")[1]) # split on "-" and convert index[1] (which is the number part to type int)
                return (num, name) # return a tuple (num, name) eg. (100, "checkpoint")
            except (IndexError, ValueError):
                pass
        
        return (0, name) # fallback for any unexpected directory names - sorts them to the beginning 
    
    # Apply sorting using the checkpoint_key function as the key 
    checkpoints = sorted(checkpoints, key=checkpoint_key)

    # 
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {base_dir}") # Raise an error if no checkpoints were found.
    
    print(f" Found {len(checkpoints)} checkpoints")
    return checkpoints


def extract_embeddings_from_checkpoint(checkpoint_path):
    """Load model and return embedding matrix (vocab_size, emb_dim) at that checkpoint"""
    try:
        # load the full GPT2 model from the saved checkpoint
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)  # from_pretrained reconstructs the model architecture and loads the saved weights 
        """
        Extract model embedding at that checkpoint.To do that we have to do the following: 
            - access the core transformer module using model.transformer 
            - access the Word Token Embedding layer (the look up table that converts token IDs to vectors) using .wte 
            - get the actual weight matrix (a pytorch tensor of shape [vocab_size, embedding_dim]) using .weight
            - remove the tensor from the computational graph (no gradient needed) using .detach() function 
            - move the tensor to CPU memory (if it was on GPU) using .cpu() function 
            - convert the pytorch tensor to a NumPy array using .numpy() function 
        """
        emb = model.transformer.wte.weight.detach().cpu().numpy() # does every step in the block comment above 

        return emb 
    except Exception as e: 
        raise RuntimeError(f"Failed to load model from {checkpoint_path}: {e}")
        

    