# The Cylindrical Representation Hypothesis for Language Model Steering

---

<div align="center">
  <a href="https://huggingface.co/datasets/LangGao/CRH_Data"><img src="https://img.shields.io/static/v1?label=CRH-Data&message=HF&color=yellow"></a>  
  <a href=""><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:CRH&color=red&logo=arxiv"></a>  
</div>

---


This is the official repository for the paper **"The Cylindrical Representation Hypothesis for Language Model Steering"**. 

## Overview
The Cylindrical Representation Hypothesis (CRH) extends LRH by allowing overlapping concept contributions, yielding a sample-specific axis-orthogonal geometry that explains irregular steering outcomes via a central axis, a normal plane, and sensitive sectors.


## Project Structure
- `baseline_exp/`: baseline steering experiments and verification runs.
- `penalty_exp/`: penalty-based steering controls for orthogonal components.
- `location_exp/`: prompt steering batch experiments (Gemma/LLaMA).
- `causal_intervention/`: causal intervention tests.
- `visualization/`: plotting scripts for CRH implications and metrics.
- `output_labelling/`: tools for labeling model outputs.
- `CRH_Data/`: local data artifacts.
- `extract_vecs_with_actis.py`: vector extraction pipeline.
- `env.yaml`: environment specification.
- `steering_results.tar.gz`: this folder contains all steering results in json format.


## Data
Local data layout:
```
CRH_Data/
├── alpaca_50.json
├── alpaca_805_prompts.json
├── alpaca_eval.json
├── data_pairs_gemma2b_805_filtered.csv
├── data_pairs_llama7b_805_filtered.csv
└── data_indices/
    ├── concepts_100.csv
    ├── data_indices_gemma.json
    ├── data_indices_gemma_stats.json
    ├── data_indices_llama7b.json
    └── data_indices_llama7b_stats.json
```

Optional external data: see project page for diff vectors and helix artifacts used by some visualization scripts.

## Experiments and Scripts
- `baseline_exp/`: functional verification runs for steering behavior.
- `penalty_exp/`: steering penalty studies and noise-rate analysis.
- `location_exp/`: prompt steering batches and steerability diagnostics.
- `causal_intervention/`: causal ablation and intervention tests.
- `extract_vecs_with_actis.py`: extraction utilities for difference vectors.

## Getting Started
1) Create the conda environment:
```bash
conda env create -f env.yaml
conda activate crh
```

2) Prepare data:
- Place the local dataset under `CRH_Data/` (see structure above).
- If you use optional external artifacts (diff vectors/helix), download them from the project page and set paths in scripts that require them.

3) Run experiments:
- Use scripts under `baseline_exp/`, `penalty_exp/`, `location_exp/`, or `causal_intervention/` depending on the experiment type.
- Example entry points include batch steering runs and steerability diagnostics.

4) Label outputs (if needed):
- Use utilities under `output_labelling/` to label generated outputs before visualization.

## Visualization Scripts Overview
This section summarizes three visualization scripts for CRH analysis, including inputs and outputs.

### 1) `visualization/draw_label34_distri_fullratio.py`
Purpose: analyze label distributions across step and penalty settings.

Outputs:
- Heatmap of corrupted outputs
- Heatmap of outputs with target concept

Inputs:
- Path: `data/penalty_res/`, in `steering_results.tar.gz`.
- Format: JSON (a single file can contain multiple penalties)
- Switch file via `labeled-results-path`


### 2) `visualization/sincos_rel.py`
Purpose: analyze the relationship between
```
steerability / ||steering vector||
```
and
```
sin^m(theta) * cos^n(theta)
```
for paper section 6.1 (Implication 2).

Outputs:
- Pearson Correlation curve
- p-value curve

Inputs:
- JSON files outside `penalty_res/` with `penalty = 0`
- Switch file via `labeled_results_path`

### 3) `visualization/cannot_determine.py`
Purpose: analyze the relationship between diff-vector similarity and steerability similarity (Implication 3).

Outputs:
- Scatter plot
- Window-averaged trend line
- Spearman correlation

Inputs:
- JSON files outside `penalty_res/` with `penalty = 0`
- Diffvec base path required


## Citation

If you find CRH useful for your research and applications, please cite using this BibTeX:

```bibtex
@misc{xie2024medtrinity25mlargescalemultimodaldataset,
      title={The Cylindrical Representation Hypothesis for Language Model Steering}, 
      author={Lang Gao, Jinghui Zhang, Wei Liu, Fengxian Ji, Chenxi Wang, Zirui Song, Akash Ghosh, Youssef Mohamed, Preslav Nakov, Xiuying Chen},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={},
      url={}, 
}
```