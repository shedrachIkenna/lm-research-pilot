"""
CPU-friendly tiny GPT2-style pilot training using HuggingFace Trainer. 
Saves checkpoint (checkpoint-*) in pilot_gpt2_cpu folder from downstream analysis 
Integrates with build_token_pos_map_cpu.py for linguistic evaluation 
"""

import os 
import json 
import argparse 
from pathlib import Path 
import torch 
from datasets import load_dataset
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed
)

# Default settings 
OUTPUT_DIR = "pilot_gpt2_cpu"
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SEED = 42 

# Saves all relevant information needed to reproduce training run 
def save_training_metadata(output_dir: str, config: GPT2Config, args: TrainingArguments, dataset_size: int, num_examples: int):
    """
    Save training configuration and metadata for reproducibility 
    """
    metadata = {
        """
        Extract and store key model architecture details from the GPT2Config object: 
            vocabulary size, maximum sequence length (n_positions), embedding dimension, 
            number of transformer layers, and number of attention heads.
        """
        "model_config": {
            "vocab_size": config.vocab_size,
            "n_positions": config.n_positions,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
        },
        """
        Extract and store training hyperparameters from the TrainingArguments object: 
        learning rate, batch size per device, gradient accumulation steps, 
        calculate the effective batch size (physical batch size × accumulation steps), 
        maximum training steps, number of epochs, weight decay regularization parameter
        random seed for reproducibility.
        """
        "training_config": {
            "learning_rate": args.learning_rate,
            "per_device_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "max_steps": args.max_steps,
            "num_train_epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            "seed": SEED,
        },
        """
        Record dataset details: 
            the dataset name (DATASET_NAME), 
            configuration (DATASET_CONFIG), 
            original dataset size, number of processed training examples, and the block size used for chunking sequences.
        """
        "dataset_info": {
            "name": DATASET_NAME,
            "config": DATASET_CONFIG,
            "raw_dataset_size": dataset_size,
            "num_training_examples": num_examples,
            "block_size": config.n_positions,
        }
    }

    metadata_path = Path(output_dir) / "training_metadata.json" # Create file path to create training_metadata.json file 

    # Write the training metadata to the training_metadata.json file
    with open(metadata_path, "w") as f: 
        json.dump(metadata, f, indent=2)

    print(f" Saved training metadata to {metadata_path}")
    
def verify_tokenizer_compactibility(tokenizer_name: str = "gpt2"):
    """
    Verify that the tokenizer matches the one used to create token_pos_map
    """

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)

    # Check if metadata exists 
    if os.path.exists("token_pos_map_metadata.json"):
        with open("token_pos_map_metadata.json") as f:
            metadata = json.load(f)

        original_tokenizer = metadata.get("tokenizer_name")
        original_vocab_size = metadata.get("vocab_size")

        # Verify match 
        if original_tokenizer != tokenizer_name:
            raise ValueError(
                f"Tokenizer mismatch! "
                f"POS map used '{original_tokenizer}' tokenizer"
                f"but you're using '{tokenizer_name}' "
            )
        
        if original_vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"Vocab size mismatch! "
                f"POS map used vocab_size={original_vocab_size} "
                f"but current tokenizer has {tokenizer.vocab_size}"
            )
        
        print("Tokenizer verified: matches POS map")
    
    return tokenizer

def load_and_prepare_dataset(dataset_name: str, dataset_config: str, num_samples: int, tokenizer, block_size: int):
    """
    Load and prepare dataset for language modeling 
    """
    # Load dataset from Hugging Face (datasets library) and get the training split 
    print(f"Loading dataset: '{dataset_name}' ({dataset_config})...")
    ds = load_dataset(dataset_name, dataset_config, split="train")

    # Get the original dataset size 
    original_size = len(ds)


    # Subsampling 
    # If requested number of samples (num_samples) < dataset_size, select only the num_samples examples 
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))
        print(f"  Subsampled {num_samples:,} / {original_size:,} examples")
    # Otherwise, select the entire dataset 
    else:
        print(f"  Using all {original_size:,} examples")
    
    # Funtion that Tokenizes text: Converts text into token IDs 
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_attention_mask=False,
            add_special_tokens=False
        )
    
    # Apply the tokenize_function to the entire dataset 
    print("Tokenizing dataset...")
    tokenized = ds.map(
        tokenize_function,
        batched=True, # processes multiple examples at once for efficiency 
        batch_size=2000, # processes 2000 examples per batch 
        remove_columns=ds.column_names, 
        desc="Tokenizing"
    )

    # Group into fixed blocks: concatenate and chunk tokenized texts into fixed-size blocks
    def group_texts(examples):
        concatenated = sum(examples["input_ids"], []) # Concatenate all token ID lists in the batch into one long list
        total_length = len(concatenated) # Gets the total number of tokens after concatenation.
        if total_length >= block_size:
            # If there are enough tokens, truncates to a multiple of block_size
            total_length = (total_length // block_size) * block_size # For example, if total_length=1050 and block_size=512, this becomes 1024 (2 complete blocks, dropping the last 26 tokens).
        
        # Split the concatenated tokens into chunks of exactly block_size tokens. This creates a list of fixed-length sequences.
        result = {
            "input_ids": [
                concatenated[i:i + block_size] 
                for i in range(0, total_length, block_size)
            ]
        }

        # labels = input_ids (the model predicts the next token, so each token serves as both input and label with appropriate shifting during training).
        result["labels"] = result["input_ids"].copy()

        return result # Returns the dictionary containing chunked input_ids and labels

    # Apply the group_texts function 
    print(f"Grouping texts into blocks of size {block_size}...")
    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        batch_size = 1000, 
        remove_columns=tokenized.column_names,
        desc="Grouping"
    )

    # Show how many fixed_length training blocks were created 
    print(f"Created {len(lm_dataset):,} training examples (fixed-length blocks)")

    # Return the prepared dataset and the original size (before any subsampling).
    return lm_dataset, original_size


def create_model(tokenizer, n_positions: int = 32, n_embd: int = 64, n_layer: int = 1, n_head: int = 2):
    """
    Create a small GPT-2 model configured for CPU training 
    """
    # Model configuration 
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )

    # initialize model 
    model = GPT2LMHeadModel(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f" Created model with {total_params:,} parameters ({trainable_params:,} trainable)")
    print(f" Layers: {n_layer}, Embedding dim: {n_embd}, Heads: {n_head}")
    
    return model, config



    


    
