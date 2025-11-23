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
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, silhouette_score, silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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
        
def build_label_arrays(token_pos_map, vocab_size):
    """Build arrays for labeled tokens"""
    # create a sorted list of token IDS that have POS mappings and only key those within the model's vocabulary 
    token_ids = sorted([tid for tid in token_pos_map.keys() if tid < vocab_size]) # This is a safety-check. If the POS map was built with a different tokenizer or vocab size, then some token IDs might be out of range 

    # Create a NumPy array of POS tags corresponding to each token ID. 
    y_labels = np.array([token_pos_map[tid] for tid in token_ids]) # For example, if token_ids = [52, 318, 423], this might produce ['NOUN', 'VERB', 'ADJ']

    # Encode labels numerically.
    le = LabelEncoder() # initialize label encoder 
    y_encoded = le.fit_transform(y_labels) # convert string labels to intergers. For example, So ['NOUN', 'VERB', 'ADJ'] becomes [1, 2, 0]

    """
    Return values 
        - token IDS -> List of token IDs that have POS labels 
        - y_labels -> string POS tags (e.g., ['NOUN', 'VERB', ...])
        - y_encoded -> Integer encoded POS tags (e.g., [1, 2, ...])
        - le -> the LabelEncoder object that will be used for decoding interger back to string later on 
    """

    return token_ids, y_labels, y_encoded, le 


def compute_probes_with_split(X, y_labels, text_size=0.2, random_state=42):
    """
    Compute k-NN and logistic reqression probes with train/text splits 
    Handles cases of insufficient data or rare classes 

    Probes are simple classifiers that tests how well the embeddings encode linguistic information 
    """
    # Data validation 
    class_counts = Counter(y_labels) # counts how many samples exists for each POS category 
    min_class_size = min(class_counts.values()) # get the smallest POS sample category 

    """
    Use train/test split if 
        - We have at least 20 samples (otherwise test set would be too small). why? we don't want a test set that is too small
        - At least 2 samples per class (needed for stratified splitting). why? If we tried stratified split on a set 
            with a sample class = 1, it will fail because you cannot split a class with only one example. 
    """
    use_split = len(X) >= 20 and min_class_size >= 2

    # No splitting 
    if not use_split: # if train/test isn't used because the dataset is small, train on everything without splitting 
        # k-NN intuition: Predicts a token's POS by looking at its k nearest neighbors in embedding space. If most neighbors are NOUNs, predict NOUN.
        knn = KNeighborsClassifier(n_neighbors=min(5, len(X) - 1)) 
        knn.fit(X, y_labels)
        knn_train_acc = accuracy_score(y_labels, knn.predict(X))

        # Logistic Regression intuition: Learns linear decision boundary between POS categories in the embedding space 
        lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
        lr.fit(X, y_labels)
        lr_train_acc = accuracy_score(y_labels, lr.predict(X))

        return {
            'knn_train': knn_train_acc,
            'knn_test': None, # Returns None since no test set was used 
            'lr_train': lr_train_acc,
            'lr_test': None, # Returns None since no test set was used 
            'split_used': False
        }
    
    # Using train-test split using stratified split
    # Stratified splits tries to maintain class proportions in both sets (train and test splits)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_labels, text_size=text_size, random_state=random_state, stratify=y_labels)
    except ValueError: # if stratified test fails, fallback to random splitting without stratification 
        X_train, X_test, y_train, y_test = train_test_split(X, y_labels, text_size=text_size, random_state=random_state)

    # k_NN probe 
    knn = KNeighborsClassifier(n_neighbors=min(5, len(X_train) - 1))
    # Train k-NN on training set, evaluates on both training and test sets
    knn.fit(X_train, y_train) # train k-NN on training set
    knn_train_acc = accuracy_score(y_train, knn.predict(X_train)) # Evaluate on training set
    knn_test_acc = accuracy_score(y_test, knn.predict(X_test)) # Evaluate on test set 

    # Logistic Regression Probe 
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    # Train Logistic regression on training set, evaluate on both training and test sets 
    lr.fit(X_train, y_train) # train logistic regression on training set
    lr_train_acc = accuracy_score(y_train, lr.predict(X_train)) # Evaluate on training set 
    lr_test_acc = accuracy_score(y_test, lr.predict(X_test)) # Evaluate on test set 

    return {
        'knn_train': knn_train_acc,
        'knn_test': knn_test_acc,
        'lr_train': lr_train_acc,
        'lr_test': lr_test_acc,
        'split_used': True
    }

def compute_silhouette_per_pos(X, y_encoded, label_names):
    """Compute Overall and per-class silhouette scores"""

    results = {}
    # Compute overall silhouette scores 
    try:
        if len(np.unique(y_encoded)) > 1: # Checks if there are at least 2 different label categories. (silhouette needs at least 2 clusters)
            overall = silhouette_score(X, y_encoded) # computes the silhouette scores across all samples 
            results['overall_silhouette'] = float(overall) # store overall results as float 
        else:
            results['overall_silhouette'] = float('nan') # if only one category exists meaning that we can't compute the score. store as nan
    except Exception as e: # If computation fails for any reason (insufficient samples, numerical issues), stores NaN.
        results['overall_silhouette'] = float('nan')

    # Compute per-class silhouette scores 
    try: 
        # Computes silhouette score for each individual sample
        sample_sil = silhouette_samples(X, y_encoded) # Returns an array where sample_sil[i] is the silhouette score for sample i

        for idx, label in enumerate(label_names):
            # Create a boolean mask selecting all samples belonging to that category.
            mask = (y_encoded == idx) # For example, if idx=1 corresponds to "NOUN", mask is True for all NOUN tokens.
            if mask.sum() >= 2: # If the category has at least 2 samples (needed for meaningful silhouette)
                results[label] = float(sample_sil[mask].mean()) # compute the mean 
            else:
                results[label] = float("nan") # store as NAN 
    
    except Exception as e:
        # If computation fails, sets all per-class scores to NaN.
        for label in label_names:
            results[label] = float('nan')   
    
    return results # Returns dictionary with overall and per-class silhouette scores.
