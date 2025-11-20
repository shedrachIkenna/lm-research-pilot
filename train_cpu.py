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