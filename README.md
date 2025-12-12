# Scalable Duplicate Detection with BMSMP+

This repository contains the code for **Branded Multi-Component Similarity Method with Preselection+ (BMSMP+)**, a duplicate detection pipeline for TV offers across multiple web shops. The method builds on MSMP+ and introduces **brand-first blocking** and **Strong Model Words (SMW)** to reduce comparisons while improving duplicate detection performance.

## What’s inside
- **preprocessing/cleaning.py** (unit normalization, title cleaning, brand extraction)
- **lsh/minhashingp.y** (shingling (binary vectors), minhashing)
- **lsh/lsh.py** (lsh algorithm for candidate pair extraction)
- **msm/msm.py** (hard blocks + composite similarity for final clustering)
- **main.py** (Scripts to reproduce evaluation plots (FC vs metrics))

## Repository structure (suggested)
- `src/` : implementation (cleaning, shingling, minhash, LSH, MSM, evaluation)
- `data/` : input data (if included) or loaders
- `graphs/` : generated plots (not tracked; keep folder via `.gitkeep`)

## Setup
Create and activate an environment, then install requirements:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

