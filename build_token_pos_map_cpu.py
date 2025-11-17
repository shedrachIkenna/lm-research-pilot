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