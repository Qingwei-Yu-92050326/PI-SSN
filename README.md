# PI‑SSN: Physics-Informed Sparse Shallow Network for Alfalfa Quality Estimation

This repository contains the official implementation of PI‑SSN, a multi‑temporal deep learning model for predicting alfalfa nutritional quality (N, CP, ADF, NDF) from UAV multispectral data. The model integrates:

- **High‑dimensional spectral features** (orthogonal features derived from 10 bands)
- **Temporal attention decoupling** (adaptive weighting of growth stages)
- **Physics‑informed constraints** (carbon‑nitrogen trade‑off, L1 sparsity)

## Pre‑trained Models

The pre‑trained model weights and inference code are available at:

## Requirements

- Python 3.8+
- PyTorch 1.10+
- numpy, pandas, scikit‑learn

## Usage

1. **Prepare your data**  
   For each sample, compute the 9 orthogonal features for three growth stages (see `extract_orthogonal_features` in the code). Concatenate them into a 27‑dimensional vector.

2. **Train the model**  
   Replace the synthetic data section in `main()` with your own data loading, then run the script.

3. **Inference**  
   Use the pre‑trained weights as shown in the script.

## Data Availability

The raw spectral data and ground‑truth values are not publicly available due to ongoing collaborative projects, but can be requested from the corresponding author.

## Citation

If you use this code, please cite our paper:

> [Your citation]