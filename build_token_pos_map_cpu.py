"""
Tokenize text using GPT-2 tokenizer 
Create token_id for each token 
map each token_id (word) to its POS 
"""

import json 
import sys
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset
from transformers import GPT2TokenizerFast
import spacy 
from tqdm import tqdm

# CPU/pilot-friendly settings 
NUM_WORDS_TO_PROCESS = 50_000 
OUTPATH = "token_pos_map.json" # output file 
MIN_OCCURENCES = 3 # if a token occurs fewer than 3 times in the entire dataset, POS tag won't be assigned 

def build_token_pos_mapping(
        num_words: int = NUM_WORDS_TO_PROCESS, 
        output_path: str = OUTPATH,
        min_occurences: int = MIN_OCCURENCES, 
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        tokenizer_name: str = "gpt2",
        spacy_model: str = "en_core_web_sm"
    ):
    """
    Maps token to POS from dataset 
    Only maps tokens that represents complete words 

    Args: 
        num_words: Number of words to process (defined default as 50k)
        output_path: file path to save the mapping JSON file 
        min_occurences: Minimum occurences required to assign a POS 
        dataset_name: HuggingFace dataset name 
        dataset_config: Dataset configuration 
        tokenizer_name: HuggingFace tokenizer name 
        spacy_model: spaCy model to use 

    Returns: 
        dict: token_id -> POS mapping (Example: "513": "NUM","4960": "ADJ","6578": "VERB") 
    """
    print(f"Loading tokenizer '{tokenizer_name}' and spaCy model '{spacy_model}'...")
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name) # Load GPT-2 tokenizer 

    try:
        # Load spacy model "en_core_web_sm"
        nlp = spacy.load(spacy_model, disable=['ner', 'parser', 'lemmatizer']) # disable ner/parser/lemmatizer functionality. We only need its POS tokenization feature 
    except OSError: # Catches error if spacy model isn't installed 
        print(f"spaCy model '{spacy_model}' not found. Downloading...")
        import subprocess
        # Use the current Python interpreter (from venv) instead of system python
        # Download the missing model. 
        subprocess.run([sys.executable, "-m", "spacy", "download", spacy_model], check=True) # sys.executable ensures the python interpreter from the current virtual environment 
        nlp = spacy.load(spacy_model, disable=["ner", "parser", "lemmatizer"])
    
    print(f"Loading dataset '{dataset_name}' ({dataset_config})...")

    # load the dataset using the HuggingFace datasets library (We choose the Wikitext dataset)
    # We also want only the training portion of the dataset 
    ds = load_dataset(dataset_name, dataset_config, split="train") # wikitext-2-raw-v1 is the configuration/subset of the dataset (wikitext) that we want

    # Counter and stats variables 
    token_pos_counts = defaultdict(Counter) # count part-of-speech tags for each token 
    """
    Example 
        token_pos_counts["running"]["VERB"] += 1
        token_pos_counts["running"]["NOUN"] += 1
        Result: {"running": Counter({"VERB": 1, "NOUN": 1})}
    """
    processed_words = 0 # how many single-token words have been processed 
    skipped_empty = 0 # empty text counted 
    skipped_multitoken = 0 # count of words that are tokenized into > 1 token and were skipped. Example: "New York" or "ice cream"

    print(f"Processing up to {num_words:,} words...")
    
    # iterate through each text item in ds (the dataset)
    for ex in tqdm(ds, desc="Processing dataset"): # tqdm shows a progress bar with the description "Processing dataset"
        text = ex[text].strip() # removes white space from current text item 
        if not text: 
            skipped_empty += 1 
            continue

        

    
    
