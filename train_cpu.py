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
    