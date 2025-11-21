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
import argparse

# CPU/pilot-friendly settings 
NUM_WORDS_TO_PROCESS = 50_000 
OUTPATH = "token_pos_map.json" # output file 
MIN_OCCURRENCES = 3 # if a token occurs fewer than 3 times in the entire dataset, POS tag won't be assigned 

def build_token_pos_mapping(
        num_words: int = NUM_WORDS_TO_PROCESS, 
        output_path: str = OUTPATH,
        min_occurrences: int = MIN_OCCURRENCES, 
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
        min_occurrences: Minimum occurences required to assign a POS 
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
        text = ex["text"].strip() # removes white space from current text item 
        if not text: 
            skipped_empty += 1 
            continue

        # Process text with spacy to get POS for each token 
        doc = nlp(text)

        for token in doc: 
            # skip whitespace, puntuation and symbols 
            if token.is_space or token.is_punct or token.pos_ == "SYM": 
                continue

            word = token.text # Extract the token's word text 
            pos = token.pos_ # Extracts the token's POS tag 

            # Additional filter: skip tokens that are clearly not linguistic content
            # (catches cases where spaCy mistagged punctuation/symbols as content words)
            if word in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' or word.strip() == '':
                continue

            # Tokenize the word to get token IDs
            # CRITICAL: Add space prefix for GPT-2 tokenizer to match vocab correctly
            word_with_space = " " + word # This is because GPT-2's tokenizer was trained on text where most words have a leading space.
            toks = tokenizer(word_with_space, add_special_tokens=False) # returns a dictionary with multiple fields 
            """
            Example:
                toks = {
                    "input_ids": [3797],           -> The actual token IDs
                    "attention_mask": [1],         -> Mask for padding (not used here)
                    .
                    .
                    .
                    possibly other fields...
                }
            """
            ids = toks["input_ids"] # extracts just the list of token IDs from that dictionary.

            # record the POS only if the word is a single token. skip multi-token words and subwords 
            # subwords and multi-token words usually have len(ids) > 1
            if len(ids) == 1:
                tid = ids[0]
                token_pos_counts[tid][pos] += 1 
                processed_words += 1 
            else: 
                # increase count of skipped subwords (words that are split into subwords)
                skipped_multitoken += 1 
            
            if processed_words >= num_words: # Stops processing words once the target number of words is reached 
                break

        if processed_words >= num_words: # Stops processing words once the target number of words is reached 
            break
    
    # final token -> POS mappings with minimun occurence threshold 
    token_to_pos = {} # # Dictionary where {token id -> Pos tags} will be saved 
    low_count_tokens = 0 # Counter for tokens that don't meet the minimum occurence threshold criteria 

    for tid, counter in token_pos_counts.items():
        total_count = sum(counter.values()) # Iterates through each token ID and calculates how many times it was seen 

        # if the token appears enough times, find its most common POS tag 
        if total_count >= min_occurrences:
            most_common_pos, count = counter.most_common(1)[0]

            # Calculates confidence: calculates the percentage of times this token had its most common POS 
            confidence = count / total_count

            # includes mappings only if confidence exceeds 50%, otherwise count is as low confidence count (don't map)
            if confidence > 0.5: 
                token_to_pos[int(tid)] = most_common_pos
            else:
                low_count_tokens += 1 
        
        else:
            low_count_tokens += 1 
    
    # Save output to file 
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    

    # Save mappings as json file 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(token_to_pos, f, indent=2)

    # Summary of processed statistics 
    print(f"\n{'='*60}")
    print(f"✓ Saved token->POS map to {output_path}")
    print(f"  Processed single-token words: {processed_words:,}")
    print(f"  Unique tokens with POS: {len(token_to_pos):,}")
    print(f"  Tokens below threshold or low confidence: {low_count_tokens}")
    print(f"  Multi-token words skipped: {skipped_multitoken:,}")
    print(f"  Empty lines skipped: {skipped_empty}")
    print(f"{'='*60}\n")

    # Show sample mappings grouped by POS
    if token_to_pos:
        print("Sample token->POS mappings (complete words only):")
        
        # Group by POS for better visualization
        pos_to_tokens = defaultdict(list)
        for tid, pos in token_to_pos.items():
            pos_to_tokens[pos].append(tid)
        
        # Show samples from each POS category
        for pos in sorted(pos_to_tokens.keys())[:8]:  # Show first 8 POS categories
            tokens = pos_to_tokens[pos][:5]  # Show up to 5 examples per POS
            print(f"\n  {pos:8s}:", end="")
            for tid in tokens:
                token_str = tokenizer.decode([tid]).strip()
                print(f" {token_str}", end=",")
    
    print("\n")
    return token_to_pos

def main():
    # Python command-line interface setup
    # Create an argument parser. The description appears when the user runs build_token_pos_map_cpu.py --help 
    parser = argparse.ArgumentParser(
        description = "Build token-to-POS mappings for linguistic analysis (Complete words only)"
    )

    # Add --num-words flag. User can run build_token_pos_map_cpu --num-words 50000 to override the default 
    parser.add_argument(
        "--num-words",
        type=int, 
        default=NUM_WORDS_TO_PROCESS,
        help=f"Number of single-token words to process (default: {NUM_WORDS_TO_PROCESS:,})"
    )

    # Adds --output flag to specify where to save the JSON file 
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPATH,
        help=f"Output JSON path (default: {OUTPATH})"
    )

    # Add --min-occurences flag the specify the minimum occurence of a token before it can be assigned a POS tag 
    parser.add_argument(
        "--min-occurrences",
        type=int, 
        default=MIN_OCCURRENCES,
        help=f"Minimum occurences to assign POS (default: {MIN_OCCURRENCES})"
    )

    # Adds --dataset flag to let user choose a different dataset 
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name (default: wikitext)"
    )

    # Adds --dataset-config flag for dataset configurations options 
    parser.add_argument(
        "--dataset-config",
        type=str, 
        default="wikitext-2-raw-v1", 
        help="Dataset configuration (default: wikitext-2-raw-v1)"
    )

    # parses all command-line arguments provided by the user. Returns an object where each argument is an attribute (e.g., args.num_words, args.output)
    args = parser.parse_args()

    # Call the main function (build_token_pos_mapping) with the parsed arguments
    build_token_pos_mapping(
        num_words=args.num_words,
        output_path=args.output,
        min_occurrences=args.min_occurrences,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
    )

if __name__ == "__main__":
    main()


    
    
