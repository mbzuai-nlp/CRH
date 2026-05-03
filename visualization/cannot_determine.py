import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import json
import re
from tqdm import tqdm
import os
from datetime import datetime
from scipy import stats
import argparse
import torch
from pathlib import Path
import matplotlib.ticker as ticker


BASE_FONT_SIZE = 32
plt.rcParams['font.family'] = 'STIXGeneral'  # Serif font similar to Times New Roman
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = BASE_FONT_SIZE
plt.rcParams['axes.titlesize'] = BASE_FONT_SIZE + 2
plt.rcParams['axes.labelsize'] = BASE_FONT_SIZE
plt.rcParams['xtick.labelsize'] = BASE_FONT_SIZE - 2
plt.rcParams['ytick.labelsize'] = BASE_FONT_SIZE - 2
plt.rcParams['legend.fontsize'] = BASE_FONT_SIZE - 2
plt.rcParams['axes.linewidth'] = 1.5      # Thicker axis lines
plt.rcParams['xtick.major.width'] = 1.5   # Thicker tick marks
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'    # Ticks point inward
plt.rcParams['ytick.direction'] = 'in'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Posthoc scatter plot analysis')
parser.add_argument('--labeled_results_path', type=str,
                    default="./baseline_results/probe/llama/exp3_probe_layer24_labelled.json",
                    help='Path to labeled results JSON file')
parser.add_argument('--output_dir', type=str,
                    default=".",
                    help='Output directory for results')
parser.add_argument('--diffmean_power', type=float, default=2,
                    help='Power a for normalization: y = steerability / |diffmean|^a')
parser.add_argument('--filter_zero_steerability', action='store_true', default=True,
                    help='If set, filter out samples where steerability=0')
parser.add_argument('--sin_cos_power_p', type=float, default=2.0,
                    help='Power p for x-axis: sin(alpha)^p * cos(alpha)^t')
parser.add_argument('--sin_cos_power_t', type=float, default=-1.0,
                    help='Power t for x-axis: sin(alpha)^p * cos(alpha)^t')
parser.add_argument('--power_search_step', type=float, default=0.05,
                    help='Step size for power search in the third subplot')
parser.add_argument('--power_search_max_offset', type=float, default=5.0,
                    help='Maximum offset from DIFFMEAN_POWER for power search range')
parser.add_argument('--p_search_step', type=float, default=0.5,
                    help='Step size for p parameter search. If None, will use power_search_step / 2')
parser.add_argument('--diff_vec_base_path', type=str, default=None,
                    help='Base path for diff vectors. If None, will try to infer from labeled_results_path')
parser.add_argument('--model_type', type=str, default=None,
                    help='Model type: "gemma2b" or "llama7b". If None, will try to infer from diff_vec_base_path')
args = parser.parse_args()

# Configure paths
LABELED_RESULTS_PATH = args.labeled_results_path
OUTPUT_DIR = args.output_dir
DIFF_VEC_BASE_PATH = args.diff_vec_base_path
MODEL_TYPE = args.model_type

# Path inference logic
if DIFF_VEC_BASE_PATH is None:
    if 'llama' in LABELED_RESULTS_PATH.lower():
        DIFF_VEC_BASE_PATH = "diff_vecs_with_actis/llama2-7b-chat"
    else:
        DIFF_VEC_BASE_PATH = "diff_vecs_with_actis/gemma2b"

if MODEL_TYPE is None:
    if 'gemma2b' in LABELED_RESULTS_PATH.lower() or 'gemma' in LABELED_RESULTS_PATH.lower():
        MODEL_TYPE = 'gemma2b'
    elif 'llama7b' in LABELED_RESULTS_PATH.lower() or 'llama' in LABELED_RESULTS_PATH.lower():
        MODEL_TYPE = 'llama7b'
    else:
        gemma_path = Path(DIFF_VEC_BASE_PATH) / 'gemma2b'
        llama_path = Path(DIFF_VEC_BASE_PATH) / 'llama7b'
        if gemma_path.exists():
            MODEL_TYPE = 'gemma2b'
            DIFF_VEC_BASE_PATH = str(gemma_path)
        elif llama_path.exists():
            MODEL_TYPE = 'llama7b'
            DIFF_VEC_BASE_PATH = str(llama_path)
        else:
            MODEL_TYPE = 'gemma2b'
            print("Warning: failed to auto-detect model type; defaulting to gemma2b")

# If user provides the llama2-7b-chat or gemma2b directory, use it as base directly
if DIFF_VEC_BASE_PATH is not None:
    if 'llama2-7b-chat' in DIFF_VEC_BASE_PATH:
        MODEL_TYPE = 'llama7b'
    elif DIFF_VEC_BASE_PATH.endswith('gemma2b'):
        MODEL_TYPE = 'gemma2b'

# Configure parameters
PENALTY = 0
DIFFMEAN_LENGTH_COEFF = 1
DIFFMEAN_POWER = args.diffmean_power
FILTER_ZERO_STEERABILITY = args.filter_zero_steerability
SIN_COS_POWER_P = args.sin_cos_power_p
SIN_COS_POWER_T = args.sin_cos_power_t
POWER_SEARCH_STEP = args.power_search_step
POWER_SEARCH_MAX_OFFSET = args.power_search_max_offset
P_SEARCH_STEP = args.p_search_step if args.p_search_step is not None else POWER_SEARCH_STEP / 2.0
USE_UNIFORM_SAMPLING = False
UNIFORM_SAMPLING_BINS = 25
UNIFORM_SAMPLING_MAX_PER_BIN = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load labeled results
print("Loading labeled results...")
try:
    with open(LABELED_RESULTS_PATH, 'r', encoding='utf-8') as f:
        labeled_results = json.load(f)
    print(f"Loaded {len(labeled_results)} concepts")
except Exception as e:
    print(f"Error: Failed to load labeled results: {e}")
    labeled_results = None

def calculate_steerability(output_list: List[List]) -> float:
    if not output_list:
        return 0.0
    max_steer_length = 0.0
    first_label2_steer_length = None
    for item in output_list:
        if len(item) < 4:
            continue
        steer_length = item[1]
        label = item[3]
        if isinstance(steer_length, (int, float)):
            max_steer_length = max(max_steer_length, float(steer_length))
        if label == 2 and first_label2_steer_length is None:
            if isinstance(steer_length, (int, float)):
                first_label2_steer_length = float(steer_length)
    if first_label2_steer_length is None:
        return 0.0
    return max_steer_length - first_label2_steer_length

def calculate_steerability_last_non2(output_list: List[List]) -> float:
    if not output_list:
        return 0.0
    max_steer_length = 0.0
    first_label2_idx = None
    for i, item in enumerate(output_list):
        if len(item) < 4:
            continue
        steer_length = item[1]
        label = item[3]
        if isinstance(steer_length, (int, float)):
            max_steer_length = max(max_steer_length, float(steer_length))
        if label == 2 and first_label2_idx is None:
            first_label2_idx = i
    if first_label2_idx is None:
        return 0.0
    if first_label2_idx == 0:
        return max_steer_length
    prev_item = output_list[first_label2_idx - 1]
    if len(prev_item) >= 2:
        prev_steer_length = prev_item[1]
        if isinstance(prev_steer_length, (int, float)):
            return max_steer_length - float(prev_steer_length)
    return 0.0

def load_diff_vector(layer: int, qid: int, cid: int, base_path: str, model_type: str = 'gemma2b') -> Optional[torch.Tensor]:
    if model_type == 'gemma2b':
        base = Path(base_path)
        if base.name == "gemma2b":
            file_path = base / str(layer) / f"question{qid}" / f"{qid}-{cid}.pt"
        else:
            file_path = base / "gemma2b" / str(layer) / f"question{qid}" / f"{qid}-{cid}.pt"
    elif model_type == 'llama7b':
        base = Path(base_path)
        if "llama2-7b-chat" in base.parts:
            file_path = base / str(layer) / f"question{qid}" / f"{qid}-{cid}.pt"
        else:
            file_path = base / "llama7b" / "llama2-7b-chat" / str(layer) / f"question{qid}" / f"{qid}-{cid}.pt"
    else:
        file_path = Path(base_path) / "gemma2b" / str(layer) / f"question{qid}" / f"{qid}-{cid}.pt"
    
    if not file_path.exists():
        return None
    try:
        pt_data = torch.load(str(file_path), map_location='cpu', weights_only=False)
        if 'vectors' not in pt_data or 'diff_vector' not in pt_data['vectors']:
            return None
        diff_vec = pt_data['vectors']['diff_vector']
        if isinstance(diff_vec, torch.Tensor):
            diff_vec = diff_vec.detach().cpu()
        else:
            diff_vec = torch.tensor(diff_vec)
        vec_norm = torch.norm(diff_vec).item()
        if vec_norm < 1e-10:
            return None
        return diff_vec
    except Exception as e:
        return None

def collect_data_for_penalty(penalty_value):
    all_data = {'sin_thetas': [], 'cos_thetas': [], 'steerabilities': [], 'steerabilities_last_non2': [], 'diffmean_lengths': []}
    stats_counts = {
        "exp_seen": 0,
        "missing_qid": 0,
        "missing_sin_cos": 0,
        "no_penalty_results": 0,
        "no_matched_penalty": 0,
        "missing_diffmean_length": 0,
        "kept": 0,
    }
    path_layer = None
    m = re.search(r'layer(\d+)', LABELED_RESULTS_PATH)
    if m:
        try:
            path_layer = int(m.group(1))
        except (TypeError, ValueError):
            path_layer = None
    for concept_data in tqdm(labeled_results, desc=f"Processing concepts (penalty={penalty_value})"):
        cid = concept_data.get('cid')
        layer = concept_data.get('layer')
        if layer is None and path_layer is not None:
            layer = path_layer
        if cid is None or layer is None:
            continue
        experiments = concept_data.get('experiments', [])
        for exp in experiments:
            stats_counts["exp_seen"] += 1
            qid = exp.get('qid')
            if qid is None:
                stats_counts["missing_qid"] += 1
                continue
            sin_theta = exp.get('sin_theta')
            cos_theta = exp.get('cos_theta')
            if sin_theta is None or cos_theta is None:
                stats_counts["missing_sin_cos"] += 1
                continue
            penalty_results = exp.get('penalty_results', [])
            if not penalty_results:
                stats_counts["no_penalty_results"] += 1
                candidate_outputs = None
                for v in exp.values():
                    if not isinstance(v, list) or len(v) == 0:
                        continue
                    first = v[0]
                    if isinstance(first, dict) and any(k in first for k in ['step', 'label', 'steer_length']):
                        candidate_outputs = v
                        break
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        candidate_outputs = v
                        break
                if candidate_outputs is not None:
                    penalty_results = [{'penalty': penalty_value, 'outputs': candidate_outputs}]
            matched_result = None
            for res in penalty_results:
                res_penalty = res.get('penalty')
                if res_penalty is not None and abs(res_penalty - penalty_value) < 1e-6:
                    matched_result = res
                    break
            if matched_result is None:
                stats_counts["no_matched_penalty"] += 1
            diffmean_length = None
            for res in penalty_results:
                res_penalty = res.get('penalty')
                if res_penalty is not None and abs(res_penalty - 0.0) < 1e-6:
                    outputs = res.get('outputs', [])
                    for item in outputs:
                        if isinstance(item, dict):
                            step = item.get('step')
                            steer_length = item.get('steer_length')
                        elif isinstance(item, list) and len(item) >= 2:
                            step = item[0]
                            steer_length = item[1]
                        else:
                            continue
                        if step == 1 and steer_length is not None:
                            diffmean_length = (steer_length / 2.5) * DIFFMEAN_LENGTH_COEFF
                            break
                    if diffmean_length is not None:
                        break
            if matched_result is not None:
                if diffmean_length is None:
                    stats_counts["missing_diffmean_length"] += 1
                output_list = matched_result.get('outputs', [])
                formatted_output_list = []
                for item in output_list:
                    if isinstance(item, dict):
                        formatted_output_list.append([
                            item.get('step'),
                            item.get('steer_length'),
                            item.get('output'),
                            item.get('label')
                        ])
                    else:
                        formatted_output_list.append(item)
                steerability = calculate_steerability(formatted_output_list)
                steerability_last_non2 = calculate_steerability_last_non2(formatted_output_list)
                all_data['sin_thetas'].append(sin_theta)
                all_data['cos_thetas'].append(cos_theta)
                all_data['steerabilities'].append(steerability)
                all_data['steerabilities_last_non2'].append(steerability_last_non2)
                all_data['diffmean_lengths'].append(diffmean_length)
                stats_counts["kept"] += 1
    print(
        f"[collect_data] exp_seen={stats_counts['exp_seen']} "
        f"missing_qid={stats_counts['missing_qid']} "
        f"missing_sin_cos={stats_counts['missing_sin_cos']} "
        f"no_penalty_results={stats_counts['no_penalty_results']} "
        f"no_matched_penalty={stats_counts['no_matched_penalty']} "
        f"missing_diffmean_length={stats_counts['missing_diffmean_length']} "
        f"kept={stats_counts['kept']}"
    )
    return all_data

def collect_pairwise_data_for_penalty(penalty_value, diff_vec_base_path: Optional[str] = None, model_type: Optional[str] = None):
    cosine_similarities = []
    steerability_diffs = []
    steerability_i_list = []
    steerability_j_list = []
    if diff_vec_base_path is None:
        diff_vec_base_path = DIFF_VEC_BASE_PATH
    if model_type is None:
        model_type = MODEL_TYPE
    failed_load_count = 0
    zero_vec_count = 0
    successful_pairs = 0
    samples_with_diffvec = 0
    concepts_with_pairs = 0
    concepts_seen = 0
    concepts_with_samples = 0
    nonzero_bins = {"0": 0, "1": 0, "2": 0, "3+": 0}
    path_layer = None
    m = re.search(r'layer(\d+)', LABELED_RESULTS_PATH)
    if m:
        try:
            path_layer = int(m.group(1))
        except (TypeError, ValueError):
            path_layer = None
    for concept_data in tqdm(labeled_results, desc=f"Processing concepts for pairwise data (penalty={penalty_value})"):
        concepts_seen += 1
        cid = concept_data.get('cid')
        layer = concept_data.get('layer')
        if layer is None and path_layer is not None:
            layer = path_layer
        if cid is None or layer is None:
            continue
        experiments = concept_data.get('experiments', [])
        concept_samples = []
        for exp in experiments:
            qid = exp.get('qid')
            if qid is None:
                continue
            penalty_results = exp.get('penalty_results', [])
            if not penalty_results:
                candidate_outputs = None
                for v in exp.values():
                    if not isinstance(v, list) or len(v) == 0:
                        continue
                    first = v[0]
                    if isinstance(first, dict) and any(k in first for k in ['step', 'label', 'steer_length']):
                        candidate_outputs = v
                        break
                    if isinstance(first, (list, tuple)) and len(first) >= 2:
                        candidate_outputs = v
                        break
                if candidate_outputs is not None:
                    penalty_results = [{'penalty': penalty_value, 'outputs': candidate_outputs}]
            matched_result = None
            for res in penalty_results:
                res_penalty = res.get('penalty')
                if res_penalty is not None and abs(res_penalty - penalty_value) < 1e-6:
                    matched_result = res
                    break
            if matched_result is None:
                continue
            output_list = matched_result.get('outputs', [])
            formatted_output_list = []
            for item in output_list:
                if isinstance(item, dict):
                    formatted_output_list.append([
                        item.get('step'),
                        item.get('steer_length'),
                        item.get('output'),
                        item.get('label')
                    ])
                else:
                    formatted_output_list.append(item)
            steerability = calculate_steerability(formatted_output_list)
            diff_vec = load_diff_vector(layer, qid, cid, diff_vec_base_path, model_type=model_type)
            if diff_vec is None:
                failed_load_count += 1
                continue
            samples_with_diffvec += 1
            concept_samples.append({
                'qid': qid,
                'diff_vec': diff_vec,
                'steerability': steerability
            })
        if len(concept_samples) > 0:
            concepts_with_samples += 1
            nonzero_count = sum(1 for s in concept_samples if s.get('steerability') not in (None, 0.0))
            if nonzero_count == 0:
                nonzero_bins["0"] += 1
            elif nonzero_count == 1:
                nonzero_bins["1"] += 1
            elif nonzero_count == 2:
                nonzero_bins["2"] += 1
            else:
                nonzero_bins["3+"] += 1
        if len(concept_samples) >= 2:
            concepts_with_pairs += 1
            for i in range(len(concept_samples)):
                for j in range(i + 1, len(concept_samples)):
                    sample_i = concept_samples[i]
                    sample_j = concept_samples[j]
                    if i == j: continue
                    if sample_i['qid'] == sample_j['qid']: continue
                    diff_vec_i = sample_i['diff_vec']
                    diff_vec_j = sample_j['diff_vec']
                    norm_i = torch.norm(diff_vec_i).item()
                    norm_j = torch.norm(diff_vec_j).item()
                    if norm_i < 1e-10 or norm_j < 1e-10:
                        zero_vec_count += 1
                        continue
                    diff_vec_i_norm = diff_vec_i / norm_i
                    diff_vec_j_norm = diff_vec_j / norm_j
                    cos_sim = torch.dot(diff_vec_i_norm.flatten(), diff_vec_j_norm.flatten()).item()
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    steer_diff = abs(sample_i['steerability'] - sample_j['steerability'])
                    cosine_similarities.append(cos_sim)
                    steerability_diffs.append(steer_diff)
                    steerability_i_list.append(sample_i['steerability'])
                    steerability_j_list.append(sample_j['steerability'])
                    successful_pairs += 1
    if failed_load_count > 0:
        print(f"\nWarning: {failed_load_count} samples failed to load diff vectors")
    print(
        f"[pairwise] concepts_seen={concepts_seen} "
        f"concepts_with_samples={concepts_with_samples} "
        f"concepts_with_pairs={concepts_with_pairs} "
        f"samples_with_diffvec={samples_with_diffvec} "
        f"successful_pairs={successful_pairs} "
        f"zero_vec={zero_vec_count}"
    )
    print(
        f"[pairwise] nonzero_steer_per_concept: "
        f"0={nonzero_bins['0']} "
        f"1={nonzero_bins['1']} "
        f"2={nonzero_bins['2']} "
        f"3+={nonzero_bins['3+']}"
    )
    return (np.array(cosine_similarities), np.array(steerability_diffs), 
            np.array(steerability_i_list), np.array(steerability_j_list))

# Main program starts
if isinstance(PENALTY, (list, tuple, np.ndarray)):
    raise ValueError("PENALTY must be a single value for scatter plot analysis")

    print(f"\nProcessing all concepts and questions...")
all_data = collect_data_for_penalty(PENALTY)

if len(all_data['sin_thetas']) > 0 and len(all_data['steerabilities']) > 0:
    min_len = min(len(all_data['sin_thetas']), len(all_data['cos_thetas']), 
                  len(all_data['steerabilities']), len(all_data['steerabilities_last_non2']),
                  len(all_data['diffmean_lengths']))
    sin_theta = np.array(all_data['sin_thetas'][:min_len])
    cos_theta = np.array(all_data['cos_thetas'][:min_len])
    steerabilities = np.array(all_data['steerabilities'][:min_len])
    diffmean_lengths = np.array(all_data['diffmean_lengths'][:min_len])

    zero_steer_count = int(np.sum(steerabilities == 0.0))
    total_steer_count = len(steerabilities)
    zero_steer_ratio = (zero_steer_count / total_steer_count) if total_steer_count > 0 else 0.0
    print(f"[steerability] zero={zero_steer_count}/{total_steer_count} ({zero_steer_ratio:.2%})")
    
    valid_mask = ~(np.isnan(diffmean_lengths) | np.isinf(diffmean_lengths))
    steerabilities = steerabilities[valid_mask]
    
    # Collect sample-pair data
    print("\nCollecting sample-pair data within the same concept...")
    cosine_similarities, steerability_diffs, steerability_i_list, steerability_j_list = collect_pairwise_data_for_penalty(
        PENALTY, diff_vec_base_path=DIFF_VEC_BASE_PATH, model_type=MODEL_TYPE
    )
    
    both_success_mask = (steerability_i_list != 0.0) & (steerability_j_list != 0.0)
    print(f"[pairwise] raw_pairs={len(cosine_similarities)} nonzero_steer_pairs={np.sum(both_success_mask)}")
    
    # ================= Plot styling (final revision) =================
    fig, ax2 = plt.subplots(figsize=(8.5, 6.5))
    
    if len(cosine_similarities) > 0 and len(steerability_diffs) > 0:
        valid_pair_mask = ~(np.isnan(cosine_similarities) | np.isinf(cosine_similarities) | 
                          np.isnan(steerability_diffs) | np.isinf(steerability_diffs))
        valid_pair_mask_no_failcases = valid_pair_mask & both_success_mask
        print(
            f"[pairwise] valid_pairs={np.sum(valid_pair_mask)} "
            f"valid_nonzero_pairs={np.sum(valid_pair_mask_no_failcases)}"
        )
        
        x_data = cosine_similarities[valid_pair_mask_no_failcases]
        y_data = steerability_diffs[valid_pair_mask_no_failcases]
        print(f"[pairwise] final_points={len(x_data)}")
        if len(x_data) > 3000:
            sample_idx = np.random.choice(len(x_data), size=3000, replace=False)
            x_data = x_data[sample_idx]
            y_data = y_data[sample_idx]
        
        if len(x_data) > 0:
            # --- 1. Scatter: single color, clean style ---
            # No KDE density; plot raw points
            scatter = ax2.scatter(
                x_data, y_data, 
                color='#528FAD',     # Classic academic blue
                s=90,                
                alpha=0.6,           # Semi-transparent to reveal density in overlaps
                edgecolors='white',  # Thin white edge to separate points
                linewidths=0.4,      
                zorder=1,
                label='Sample Pairs'
            )

            # --- 2. Trend line (Mean ± SEM) ---
            n_bins = 20
            bin_edges = np.linspace(np.min(x_data), np.max(x_data), n_bins + 1)
            bin_indices = np.digitize(x_data, bin_edges) - 1
            
            bin_centers = []
            bin_means = []
            bin_sems = []
            
            for b in range(n_bins):
                mask = bin_indices == b
                if np.any(mask):
                    vals = y_data[mask]
                    center = 0.5 * (bin_edges[b] + bin_edges[b + 1])
                    bin_centers.append(center)
                    bin_means.append(np.mean(vals))
                    bin_sems.append(np.std(vals) / np.sqrt(len(vals)))
            
            bin_centers = np.array(bin_centers)
            bin_means = np.array(bin_means)
            bin_sems = np.array(bin_sems)
            
            # Shaded band (light orange, matching the line)
            ax2.fill_between(
                bin_centers, 
                bin_means - bin_sems, 
                bin_means + bin_sems, 
                color='#F7AA58', # Orange
                alpha=0.25,      # Light transparency
                linewidth=0,
                zorder=2,
                label=None
            )
            
            # Mean line (bright orange, white centers)
            ax2.plot(
                bin_centers,
                bin_means,
                color='#F7AA58', 
                linewidth=3.5,
                marker='o',
                markersize=9,
                markerfacecolor='white', 
                markeredgecolor='#F7AA58', 
                markeredgewidth=2.0,
                zorder=3,
                label='Binned Mean'
            )
            
            # --- 3. Layout and axis control (remove whitespace) ---
            ax2.set_xlabel('Cosine Similarity (Sample Pairs)', labelpad=10)
            ax2.set_ylabel(r'$\left|\mathrm{Minimal\ Strength\ Difference}\right|$', labelpad=10)
            ax2.yaxis.set_label_coords(-0.10, 0.42)
            
            # Tight axis limits to remove padding
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            
            # 1% buffer
            x_pad = (x_max - x_min) * 0.01
            y_pad = (y_max - y_min) * 0.02
            if len(bin_centers) > 0:
                ax2.set_xlim(np.min(bin_centers) - x_pad, np.max(bin_centers) + x_pad)
            else:
                ax2.set_xlim(x_min - x_pad, x_max + x_pad)
            ax2.set_ylim(y_min - y_pad, y_max + y_pad) 
            
            # Grid lines
            ax2.grid(True, linestyle='--', alpha=0.3, color='gray', zorder=0)
            ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            
            # --- 4. Statistics ---
            if len(x_data) > 1:
                pearson_corr, pearson_p = stats.pearsonr(x_data, y_data)
                
                stats_text = (
                    f"Pearson: {pearson_corr:.3f} ($p$={pearson_p:.3f})"
                )
                ax2.text(0.04, 0.96, stats_text, 
                       transform=ax2.transAxes, 
                       verticalalignment='top',
                       fontsize=BASE_FONT_SIZE-2,
                       color='dimgray',
                       bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
            ax2.legend(
                frameon=True,
                fancybox=True,
                framealpha=0.65,
                loc='upper left',
                bbox_to_anchor=(0.00, 0.9),
                fontsize=BASE_FONT_SIZE - 6,
            )

        else:
            ax2.text(0.5, 0.5, 'No valid non-failcase pairs', ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No pairwise data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = "analysis_final_single_color.pdf"
    output_path = os.path.join(OUTPUT_DIR, "impl3", output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()

else:
    print("No data found!")
