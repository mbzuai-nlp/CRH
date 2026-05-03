#!/usr/bin/env bash
set -euo pipefail

# MEAN CENTER
python baseline_exp/exp3_mean_center_steer_experiments_batch_gemma.py
python baseline_exp/exp3_mean_center_steer_experiments_batch_llama.py

# PROBE
python baseline_exp/exp3_probe_steer_experiments_batch_gemma.py
python baseline_exp/exp3_probe_steer_experiments_batch_llama.py

# PCA
python baseline_exp/exp3_pca_steer_experiments_batch_gemma.py
python baseline_exp/exp3_pca_steer_experiments_batch_llama.py