# Emergence of Linguistic Structure in Small Transformers

A lightweight, reproducible experimental pipeline designed to analyze the training dynamics of small language models. This project investigates when and how Part-of-Speech (POS) information crystallizes in the latent embedding space during the optimization process.

> **Figure 1:** PCA trajectories of exemplar tokens across training checkpoints. Note the distinct movement as semantic roles solidify.  
> *(see `results/pca_trajectories.png`)*

---

## Research Motivation

Inspired by Papadimitriou et al. (2024) on vocabulary embeddings, This project implements a complete, replicable research pipeline for studying how linguistic structure emerges inside the embedding space of small transformer language models.

This pipeline is designed to run entirely on CPU, making it suitable for low-resource experimentation, teaching, and conceptual replication studies.

---

## Research Questions

This project provides a small-scale framework for investigating:

- When does linguistic structure emerge during training?
    - Track probing accuracy over time.
- Which POS categories form clusters earliest?
    - Compare early and late checkpoints
- Is the embedding structure linearly separable?
    - k-NN vs logistic regression probes.
- How do individual tokens evolve?
    - Token trajectory visualization.
- Do semantic neighborhoods stabilize?
    - Nearest-neighbor tracking over training.


---

## Technical Pipeline

The repository is organized into three stages:

### 1. Data & Ground Truth Generation  

**Script:** `build_token_pos_map_cpu.py`

- Ingests raw text (WikiText-2).  
- Uses spaCy to generate linguistic ground truth (POS tags).  
- Rigorous filtering: maps only *single-token* words to POS tags to avoid subword-token noise (e.g., ignoring fragments like `ing` or `ed`).

### 2. Dynamics Training

**Script:** `train_cpu.py`  

- Architecture: decoder-only Transformer (GPT-2 style) using Hugging Face.  
- Checkpointing: custom Trainer config to save model snapshots at specified intervals (e.g., every 50 steps) to capture early dynamics.  
- Reproducibility: logs full config and hyperparameters to `training_metadata.json`.

### 3. Latent Space Analysis

**Script:** `analysis.py`  

Extracts `W_E` (embedding matrices) from checkpoints and computes:

- Linear & k-NN probing (train/test splits) to quantify linear separability of POS information.  
- Silhouette scores to measure cluster tightness and separation.  
- Trajectory analysis: tracks exemplar tokens through 2D PCA space over training.  
- Intra/inter-class cosine similarity to quantify semantic-region density.

---


## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
