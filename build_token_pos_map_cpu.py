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
        nun_words: int = NUM_WORDS_TO_PROCESS, 
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