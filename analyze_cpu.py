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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, silhouette_score, silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


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

def compute_intra_inter_cosine(X_full, token_ids, y_labels):
    """
    Compute mean intra-class and inter-class cosine similarity per POS 
    X_full: full embedding matrix (vocab_size, emb_dim)
    token_ids: list of labeled token indices 
    y_labels: POS labels for those tokens 
    """

    # Build POS to index mapping 
    """
    Creates a dictionary mapping each POS category to a list of indices (positions in token_ids). For example: 
        {
            "NOUN": [0, 3, 7, 12, ...],    # indices where NOUNs appear
            "VERB": [1, 5, 9, 15, ...],    # indices where VERBs appear
            "ADJ": [2, 4, 8, 11, ...]      # indices where ADJs appear
        }
    """
    pos2indices = defaultdict(list)
    for i, tid in enumerate(token_ids):
        pos2indices[y_labels[i]].append(i) # pos2indices is a dictionary mapping each POS category to a list of indices
    
    # Extracts embeddings for only the labeled tokens
    X_labeled = X_full[token_ids] #  If token_ids = [52, 318, 423], this gets rows 52, 318, and 423 from the full embedding matrix.
    
    # Compute pairwise cosine similarity between all labeled tokens 
    """
    Returns a matrix where cos_matrix[i, j] is the cosine similarity between token i and token j. Shape: (num_labeled_tokens, num_labeled_tokens).
    Cosine similarity ranges from -1 to +1:
        +1: Vectors point in exactly the same direction (very similar)
        0: Vectors are orthogonal (unrelated)
        -1: Vectors point in opposite directions (very dissimilar)
    """
    cos_matrix = cosine_similarity(X_labeled) # Returns the cosine similarity matrix between all labeled tokens as explained above

    results = {}
    for pos, idxs in pos2indices.items():
        if len(idxs) < 2:
            results[pos] = {"intra_mean": float("nan"), "inter_mean": float("nan")} # If a POS category has fewer than 2 tokens, can't compute meaningful similarities, so stores NaN.
            continue

        # Intra class similarity - Compute all pairwise similarities within each POS category 
        intra_vals = []
        for i in idxs:
            for j in idxs:
                if i != j: # Exclude self-similarity since every token has similarity 1.0 with itself
                    intra_vals.append(cos_matrix[i, j]) # For example, if this category has tokens at indices [0, 3, 7], collects similarities: (0,3), (0,7), (3,0), (3,7), (7,0), (7,3)
        
        intra_mean = float(np.mean(intra_vals)) if intra_vals else float('nan')

        # Inter class similarity - Compute all pairwise POS similarities across  cross-categories 
        other_idxs = [i for i in range(len(token_ids)) if i not in idxs] # Get indices of other POS categories 

        if other_idxs:
            inter_vals = cos_matrix[np.ix_(idxs, other_idxs)].ravel()
            inter_mean = float(np.mean(inter_vals))
        else:
            inter_mean = float('nan')

        results[pos] = {"intra_mean": intra_mean, "inter_mean": inter_mean}
    
    return results

def get_nearest_neighbors(emb_matrix, tokenizer, exemplar_words, top_k=10):
    """
    Find nearest neighbors of exemplar words in embedding space 
    Returns: 
        dict: word -> {token_id, neighbors: [...]}
    """
    # Normalize Embeddings 
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    normed = emb_matrix / (norms + 1e-12)

    # Nearest Neighbors Search 
    # Use a k-NN model that finds the nearest neighbors using cosine distance 
    nbrs = NearestNeighbors(n_neighbors=top_k+5, algorithm='auto', metric='cosine')
    nbrs.fit(normed)

    results = {}
    for word in exemplar_words:
        # Tokenize word 
        toks = tokenizer(" " + word, add_special_tokens=False)["input_ids"] # Tokenizes the word with a space prefix (critical for GPT-2 tokenizer to match vocabulary correctly). Returns list of token IDs.
        if not toks or toks[0] >= emb_matrix.shape[0]:
            results[word] = {"token_id": None, "neighbors": []}
            continue

        tid = toks[0] # Gets the token ID for this word (takes first token if word splits into multiple).
        # Find The Nearest Neighbors 
        distances, indices = nbrs.kneighbors(normed[tid].reshape(1, -1), return_distance=True)
        distances = distances[0]
        indices = indices[0]

        # Get Neighbor List 
        neighbors = []
        for dist, idx in zip(distances, indices): # Iterates through neighbors. Skips the word itself (every token is its own nearest neighbor with distance 0).
            if idx == tid: # Skip self
                continue

            cos_sim = 1.0 - float(dist) #Converts cosine distance back to cosine similarity: similarity = 1 - distance
            token_str = tokenizer.decode([idx]).strip() # Decodes the neighbor's token ID back to readable text

            # Add neighbor info to list and stop once we have top_k neighbors (excluding self).
            neighbors.append({
                "token_id": int(idx),
                "token_str": token_str,
                "cosine": cos_sim
            })

            if len(neighbors) >= top_k:
                break

        results[word] = {
            "token_id": int(tid),
            "neighbors": neighbors
        }

    return results

def plot_umap_visualization(emb_pca, token_ids, y_labels, output_path, checkpoint_name, metrics):
    """
    Create UMAP visualization colored by POS tags
    What is UMAP? Uniform Manifold Approximation and Projection:
         —Its a dimensionality reduction technique that preserves both local and global structure better than t-SNE or PCA.
    """
    try:
        X_labeled = emb_pca[token_ids] # Extract token embeddings that have POS labels 

        # Configure and initilize UMAP 
        reducer = umap.UMAP(
            n_components = 2, # Reduce to 2D for visualization (x, y coordinate)
            random_state = 42, # Ensures reproducibility (same input → same output)
            n_neighbors = min(UMAP_N_NEIGHBORS, len(token_ids) - 1), # How many neighbors UMAP considers for local structure. Uses the configured value or fewer if there aren't enough tokens. Must be less than total points.
            min_dis=UMAP_MIN_DIST # Minimum distance between points in 2D space (controls how tightly UMAP packs points)
        )

        emb_2d = reducer.fit_transform(X_labeled)

        unique_labels = sorted(list(set(y_labels))) # Gets all unique POS tags and sorts them alphabetically
        label2idx = {lab: i for i, lab in enumerate(unique_labels)} # Creates a mapping from POS tag → integer (e.g., {"ADJ": 0, "NOUN": 1, "VERB": 2})
        colors = [label2idx[y_labels[i]] for i in range(len(y_labels))] # Converts each token's POS label to its corresponding integer for coloring. Example: If token 0 is a NOUN, colors[0] = 1

        # Create figure
        plt.figure(figsize=(10, 8)) #Create a figure with 10×8 inch dimensions.

        # Create scatter plot 
        scatter = plt.scatter(
                emb_2d[:, 0], #  x-coordinates (UMAP dimension 1)
                emb_2d[:, 1], #  y-coordinates (UMAP dimension 2)
                c=colors, # Color by POS tag (integer values)
                s=15, # Point size
                alpha=0.6, # 60% opacity (helps see overlapping points)
                cmap='tab20' if len(unique_labels) <= 20 else 'viridis' # Uses 'tab20' colormap if ≤20 POS categories (distinct colors), otherwise 'viridis' (continuous gradient)
        )
        # Add title with metrics 
        title_lines = [f"UMAP - {checkpoint_name}"] # Add title stating the checkpoint name 
        # Show probe accuracies 
        if metrics['split_used']: # If train/test split was used, shows test accuracies
            title_lines.append(
                f"k-NN: test={metrics['knn_test']:.3f} | LR: test={metrics['lr_test']:.3f}"
            )
        else: # Otherwise, shows train accuracies
            title_lines.append(
                f"k-NN: {metrics['knn_train']:.3f} | LR: {metrics['lr_train']:.3f}"
            )

        plt.title('\n'.join(title_lines), fontsize=11) # Set the title by joining lines with newlines

        # Color bar legend 
        cbar = plt.colorbar(scatter, ticks=range(len(unique_labels)))
        cbar.set_label('POS Tag', rotation=270, labelpad=15)
        cbar.ax.set_yticklabels(unique_labels, fontsize=8)

        # Adds axis labels
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        # Automatically adjusts spacing to prevent label overlap
        plt.tight_layout()

        # Save figure 
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f" Saved UMAP to {output_path}")
        return emb_2d # returns the 2D coordinates

    except Exception as e: 
        print(f" UMAP Failed: {e}") # If anything fails (insufficient data, UMAP errors, etc.), prints error message and returns None instead of crashing.
        return None
    
def plot_trajectory(checkpoint_embeddings, exemplar_words, tokenizer, output_path):
    """
    Plot token trajectories across checkpoints using PCA 
    checkpoint_embeddings: dict of checkpoint_name -> embedding_matrix 
    """
    try:
        # Tokenize each of the exemplar words 
        exemplar_token_ids = {}
        for word in exemplar_words:
            toks = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
            if toks:
                exemplar_token_ids[word] = toks[0]
        
        if not exemplar_token_ids:
            print(" No valid exemplar tokens found")
            return 
        
        # Sorts checkpoint names by length first, then alphabetically.
        checkpoint_names = sorted(checkpoint_embeddings.keys(), key=lambda x: (len(x), x))

        # initialize a dictionary to store each word's trajectory (list of 2D coordinates across checkpoints)
        trajectories = defaultdict(list)

        # iterate through checkpoints and gets the embedding matrix for each checkpoint 
        for ckpt_name in checkpoint_names:
            emb = checkpoint_embeddings[ckpt_name]

            # Standardize and scale embeddings to have a mean = 0 and standard deviation = 1 
            scaler = StandardScaler()
            emb_scaled = scaler.fit_transform(emb)

            # Use PCA to reduce the high dimensions to 2D for visualization 
            pca = PCA(n_components=2, random_state=42)
            emb_2d = pca.fit_transform(emb_scaled)

            for word, tid in exemplar_token_ids.items(): # iterates through exemplar_token_ids
                if tid < emb_2d.shape[0]: # is token ID valid? 
                    trajectories[word].append(emb_2d[tid]) # append coordinates
                else: 
                    trajectories[word].append([np.nan, np.nan]) # appends NaN (missing data)

                """
                tracjectories output example 
                {
                    "king": [[x0, y0], [x1, y1], [x2, y2], ...],  # coordinates at each checkpoint
                    "run": [[x0, y0], [x1, y1], [x2, y2], ...],
                    "happy": [[x0, y0], [x1, y1], [x2, y2], ...]
                }
             
                """
        # Create plot 
        # Creates a figure and gets a colormap with 10 distinct colors (cycles if more than 10 words).
        plt.figure(figsize=(10, 8))
        colors = plt.cm.get_cmap('tab10')

        # Iterates through each word and its trajectory coordinates, converting to NumPy array
        for i, (word, coords) in enumerate(trajectories.items()):
            coords = np.array(coords)
            if len(coords) > 0:
                # Plot trajectories as a line-connecting points 
                plt.plot(
                    coords[:, 0], # All x-coordinates (PC 1)
                    coords[:, 1], # All y-coordinates (PC 2)
                    'o-', # Line with circle markers at each point
                    label=word, # label for each word 
                    color=colors(i % 10), # # Each word gets a unique color from the colormap
                    linewidth=2, # Line thickness 
                    markersize=8 # Point size
                )
                # Add directional arrows 
                # Iterates through consecutive checkpoint pairs, skipping if either coordinate is NaN.
                for j in range(len(coords) - 1):
                    if not (np.isnan(coords[j]).any() or np.isnan(coords[j+1]).any()):
                        # Calculates the change in x and y between consecutive checkpoints (the arrow direction).
                        dx = coords[j+1, 0] - coords[j, 0]
                        dy = coords[j+1, 1] - coords[j, 1]

                        # Draws an arrow from checkpoint j to j+1:
                        plt.arrow(
                            coords[j, 0], coords[j, 1], # Starts at (coords[j, 0], coords[j, 1])
                            dx, dy, # Extends by (dx, dy)
                            head_width=0.05, # Arrow head size 
                            head_length=0.05, # Arrow head size 
                            fc=colors(i % 10), # Fill color
                            ec=colors(i % 10), # Edge color
                            alpha=0.6 #  60% opacity (semi-transparent)
                        )

        # Adds legend, title, axis labels, and a semi-transparent grid.
        plt.legend(loc='best', fontsize=10)
        plt.title('Token Trajectories Across Training (PCA 2D)', fontsize=12)
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.grid(True, alpha=0.3)

        # Adjusts layout, saves the figure, closes it to free memory, and prints confirmation.
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saved trajectory plot to {output_path}")
    
    except Exception as e: # Catches any errors and prints a message instead of crashing.
        print(f" Trajectory plot failed: {e}")

def plot_temporal_metrics(metrics, output_path):
    """
    Plot probe accuracies over training
    The visualization answers: How does linguistic structure emerge over time during training?
    """
    try: 
        checkpoint_names = [m['checkpoint'] for m in metrics]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1 = axes[0]
        if metrics[0].get('split_used', False):
            knn_test = [m['knn_test'] for m in metrics]
            lr_test = [m['lr_test'] for m in metrics]

            ax1.plot(checkpoint_names, knn_test, 'o-', label='k-NN (test)', linewidth=2)
            ax1.plot(checkpoint_names, lr_test, 's-', label='Linear (test)', linewidth=2)

            ax1.set_ylabel('Test Accuracy')
            ax1.set_title('Probing Accuracy Over Training (Test Set)')
        else:
            knn_train = [m['knn_train'] for m in metrics]
            lr_train = [m['lr_train'] for m in metrics]
            ax1.plot(checkpoint_names, knn_train, 'o-', label='k-NN', linewidth=2)
            ax1.plot(checkpoint_names, lr_train, 's-', label='Linear', linewidth=2)
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Probing Accuracy Over Training')
        
        ax1.set_xlabel('Checkpoint')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        ax2 = axes[1]

        silhouette_scores = [m.get('silhouette_overall', np.nan) for m in metrics]
    
        ax2.plot(checkpoint_names, silhouette_scores, 'o-', color='green', linewidth=2)
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Cluster Quality Over Training')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f" Saved temporal plot to {output_path}")
    
    except Exception as e:
        print(f" Temporal plot failed: {e}")
            
