import numpy as np
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import os
from datetime import datetime
from scipy import stats
import argparse
import re

# ==============================================================================
# 1. Argument parsing
# ==============================================================================
parser = argparse.ArgumentParser(description='Posthoc scatter plot analysis')
parser.add_argument('--labeled_results_path', type=str,
                    default="./data/baseline_results/diffmean/gemma/exp3_diffmean_layer9_labelled.json",
                    help='Path to labeled results JSON file')
parser.add_argument('--output_dir', type=str,
                    default=".",
                    help='Output directory for results (default: current directory)')
parser.add_argument('--output_name', type=str, default=None,
                    help='Custom output filename (without directory); defaults to auto model-layer name')
parser.add_argument('--diffmean_power', type=float, default=2,
                    help='Power a for normalization: y = steerability / |diffmean|^a')
parser.add_argument('--filter_zero_steerability', action='store_true', default=False,
                    help='If set, filter out samples where steerability=0')
parser.add_argument('--sin_cos_power_p', type=float, default=2.0,
                    help='Power p for x-axis: sin(alpha)^p * cos(alpha)^t')
parser.add_argument('--sin_cos_power_t', type=float, default=-1.0,
                    help='Power t for x-axis: sin(alpha)^p * cos(alpha)^t')
parser.add_argument('--power_search_step', type=float, default=0.05,
                    help='Step size for power search in the third subplot')
parser.add_argument('--power_search_max_offset', type=float, default=8,
                    help='Maximum offset from DIFFMEAN_POWER for power search range')
parser.add_argument('--p_search_step', type=float, default=None,
                    help='Step size for p parameter search. If None, will use power_search_step / 2')
parser.add_argument('--power_range_start', type=float, default=None,
                    help='Start of diffmean power search range (inclusive)')
parser.add_argument('--power_range_end', type=float, default=None,
                    help='End of diffmean power search range (inclusive)')
parser.add_argument('--exclude_first_step_label2', action='store_true', default=False,
                    help='Exclude samples where step=1 label=2 (default: included)')
args = parser.parse_args()

# ==============================================================================
# 2. Global font config
# ==============================================================================
def setup_fonts(base_size: int = 34):
    """Configure global fonts and sizes for consistent plots."""
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.titlesize'] = base_size + 4
    plt.rcParams['axes.labelsize'] = base_size + 2
    plt.rcParams['xtick.labelsize'] = base_size
    plt.rcParams['ytick.labelsize'] = base_size
    plt.rcParams['legend.fontsize'] = base_size
    plt.rcParams['figure.titlesize'] = base_size + 6

# ==============================================================================
# 3. Helper functions
# ==============================================================================
def extract_model_layer(path: str):
    lower_path = path.lower()
    model_match = re.search(r'(gemma|llama)', lower_path)
    model = model_match.group(1) if model_match else "unknown"
    layer_match = re.search(r'layer(\d+)', lower_path)
    layer = f"layer{layer_match.group(1)}" if layer_match else "layer"
    return model, layer

def calculate_steerability_from_all_outputs(all_output_lists: List[List[List]]) -> float:
    """Compute steerability as the slope of a linear fit over step vs. label-2 ratio."""
    if not all_output_lists or len(all_output_lists) == 0:
        return 0.0
    
    step_data = {}
    for output_list in all_output_lists:
        if not output_list: continue
        
        for item in output_list[:50]:
            if len(item) < 4: continue
            
            step = item[0]  
            label = item[3]  
            if step is None: continue
            
            step_key = int(step) if isinstance(step, (int, float)) else step
            if step_key not in step_data:
                step_data[step_key] = {'total': 0, 'label2': 0}
            
            step_data[step_key]['total'] += 1
            if label == 2:
                step_data[step_key]['label2'] += 1
    
    if len(step_data) == 0: return 0.0
    
    steps = []
    label2_ratios = []
    for step_key in sorted(step_data.keys()):
        data = step_data[step_key]
        if data['total'] > 0:
            ratio = data['label2'] / data['total']
            steps.append(float(step_key))
            label2_ratios.append(ratio)
    
    if len(steps) < 2: return 0.0
    
    try:
        steps_array = np.array(steps)
        ratios_array = np.array(label2_ratios)
        slope, intercept = np.polyfit(steps_array, ratios_array, 1)
        return float(slope)
    except:
        return 0.0


def collect_data_for_penalty(penalty_value, labeled_results, diffmean_length_coeff):
    """Collect data for analysis."""
    all_data = {'sin_thetas': [], 'cos_thetas': [], 'steerabilities': [], 'diffmean_lengths': []}
    
    for concept_data in tqdm(labeled_results, desc=f"Processing concepts (penalty={penalty_value})"):
        cid = concept_data.get('cid')
        if cid is None: continue
        
        experiments = concept_data.get('experiments', [])
        all_output_lists = []
        qid_data = []  
        
        for exp in experiments:
            qid = exp.get('qid')
            if qid is None: continue
            
            sin_theta = exp.get('sin_theta')
            cos_theta = exp.get('cos_theta')
            if sin_theta is None or cos_theta is None: continue
            
            penalty_results = exp.get('penalty_results', [])
            
            # Backward-compatible handling for legacy format
            if not penalty_results:
                candidate_outputs = None
                for v in exp.values():
                    if isinstance(v, list) and len(v) > 0:
                        first = v[0]
                        if isinstance(first, dict) and any(k in first for k in ['step', 'label']):
                            candidate_outputs = v
                            break
                        elif isinstance(first, (list, tuple)) and len(first) >= 2:
                            candidate_outputs = v
                            break
                if candidate_outputs is not None:
                    penalty_results = [{'penalty': 0.0, 'outputs': candidate_outputs}]

            matched_result = None
            for res in penalty_results:
                res_penalty = res.get('penalty')
                if res_penalty is not None and abs(res_penalty - penalty_value) < 1e-6:
                    matched_result = res
                    break
            
            # Compute diffmean_length (from penalty=0, step=1)
            diffmean_length = None
            for res in penalty_results:
                res_penalty = res.get('penalty')
                if res_penalty is not None and abs(res_penalty - 0.0) < 1e-6:
                    outputs = res.get('outputs', [])
                    for item in outputs:
                        if isinstance(item, dict):
                            step = item.get('step')
                            s_len = item.get('steer_length')
                        elif isinstance(item, list) and len(item) >= 2:
                            step = item[0]
                            s_len = item[1]
                        else: continue
                        
                        if step == 1 and s_len is not None:
                            diffmean_length = (s_len / 2.5) * diffmean_length_coeff
                            break
                    if diffmean_length is not None: break
            
            if matched_result is not None:
                output_list = matched_result.get('outputs', [])
                formatted_output_list = []
                for item in output_list:
                    if isinstance(item, dict):
                        formatted_output_list.append([
                            item.get('step'), item.get('steer_length'), item.get('output'), item.get('label')
                        ])
                    else:
                        formatted_output_list.append(item)

                if formatted_output_list and EXCLUDE_FIRST_STEP_LABEL2:
                    has_step1_label2 = False
                    for item in formatted_output_list:
                        if len(item) < 4:
                            continue
                        step = item[0]
                        label = item[3]
                        try:
                            step_int = int(step)
                        except (TypeError, ValueError):
                            continue
                        if step_int == 1 and label == 2:
                            has_step1_label2 = True
                            break
                    if has_step1_label2:
                        continue
                
                all_output_lists.append(formatted_output_list)
                qid_data.append((qid, sin_theta, cos_theta, diffmean_length))
        
        if len(all_output_lists) > 0:
            steerability = calculate_steerability_from_all_outputs(all_output_lists)
            for qid, sin_theta, cos_theta, diffmean_length in qid_data:
                all_data['sin_thetas'].append(sin_theta)
                all_data['cos_thetas'].append(cos_theta)
                all_data['steerabilities'].append(steerability)
                all_data['diffmean_lengths'].append(diffmean_length)
    
    return all_data

# ==============================================================================
# 4. Main workflow
# ==============================================================================

# Configure parameters
PENALTY = 0 
DIFFMEAN_LENGTH_COEFF = 1  
DIFFMEAN_POWER = args.diffmean_power  
FILTER_ZERO_STEERABILITY = args.filter_zero_steerability 
SIN_COS_POWER_P = args.sin_cos_power_p  
SIN_COS_POWER_T = args.sin_cos_power_t 
EXCLUDE_FIRST_STEP_LABEL2 = args.exclude_first_step_label2
POWER_SEARCH_STEP = args.power_search_step 
POWER_SEARCH_MAX_OFFSET = args.power_search_max_offset 
P_SEARCH_STEP = args.p_search_step if args.p_search_step is not None else POWER_SEARCH_STEP / 2.0
POWER_RANGE_START = args.power_range_start if args.power_range_start is not None else 0.0
POWER_RANGE_END = args.power_range_end if args.power_range_end is not None else DIFFMEAN_POWER + POWER_SEARCH_MAX_OFFSET
LABELED_RESULTS_PATH = args.labeled_results_path
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)
setup_fonts(base_size=28) 

# Load data
print("Loading labeled results...")
try:
    with open(LABELED_RESULTS_PATH, 'r', encoding='utf-8') as f:
        labeled_results = json.load(f)
    print(f"Loaded {len(labeled_results)} concepts")
except Exception as e:
    print(f"Error: Failed to load labeled results: {e}")
    labeled_results = None

if labeled_results is None:
    raise SystemExit("No labeled results loaded.")

model_name, layer_name = extract_model_layer(LABELED_RESULTS_PATH)

# Process data
print(f"\nProcessing all concepts and questions...")
all_data = collect_data_for_penalty(PENALTY, labeled_results, DIFFMEAN_LENGTH_COEFF)
print(f"\nSuccessfully processed {len(all_data['sin_thetas'])} data points")

if len(all_data['sin_thetas']) > 0 and len(all_data['steerabilities']) > 0:
    # Align and clean data
    min_len = min(len(all_data['sin_thetas']), len(all_data['cos_thetas']), 
                  len(all_data['steerabilities']), len(all_data['diffmean_lengths']))
    
    sin_theta = np.array(all_data['sin_thetas'][:min_len])
    cos_theta = np.array(all_data['cos_thetas'][:min_len])
    steerabilities = np.array(all_data['steerabilities'][:min_len])
    diffmean_lengths = np.array(all_data['diffmean_lengths'][:min_len])
    
    # Basic filtering
    valid_mask = ~np.isnan(diffmean_lengths)
    sin_theta = sin_theta[valid_mask]
    cos_theta = cos_theta[valid_mask]
    steerabilities = steerabilities[valid_mask]
    diffmean_lengths = diffmean_lengths[valid_mask]
    
    if FILTER_ZERO_STEERABILITY:
        non_zero_mask = steerabilities != 0.0
        sin_theta = sin_theta[non_zero_mask]
        cos_theta = cos_theta[non_zero_mask]
        steerabilities = steerabilities[non_zero_mask]
        diffmean_lengths = diffmean_lengths[non_zero_mask]
    
    # Preprocess data for power search
    epsilon = 1e-10
    
    # Initialize figure
    fig, ax3 = plt.subplots(figsize=(9, 6))
    
    # Power search logic
    power_max = POWER_RANGE_END
    power_min = POWER_RANGE_START
    power_search_range = np.arange(power_min, power_max + POWER_SEARCH_STEP * 0.5, POWER_SEARCH_STEP)
    power_pearson_max = []
    power_pearson_pvalue = []
    power_values = []
    
    # Lock subset based on max power to keep samples consistent
    max_test_power = power_max
    diffmean_powered_max = diffmean_lengths ** max_test_power
    valid_normalization_mask_max = diffmean_powered_max > epsilon
    
    sin_theta_base = sin_theta[valid_normalization_mask_max]
    cos_theta_base = cos_theta[valid_normalization_mask_max]
    steerabilities_base = steerabilities[valid_normalization_mask_max]
    diffmean_lengths_base = diffmean_lengths[valid_normalization_mask_max]
    
    for test_power in tqdm(power_search_range, desc="Searching optimal correlations"):
        diffmean_powered_test = diffmean_lengths_base ** test_power
        # Double-check
        valid_test = diffmean_powered_test > epsilon
        if np.sum(valid_test) < 2: continue
        
        y_norm_test = steerabilities_base[valid_test] / diffmean_powered_test[valid_test]
        final_mask_test = ~(np.isnan(y_norm_test) | np.isinf(y_norm_test))
        y_plot_test = y_norm_test[final_mask_test]
        
        sin_t = sin_theta_base[valid_test][final_mask_test]
        cos_t = cos_theta_base[valid_test][final_mask_test]
        
        if len(y_plot_test) < 2: continue
        
        # Inner loop: search best p, t combo (p+t = test_power)
        p_min_test = 0.1
        p_max_test = test_power - 0.1
        if p_max_test <= p_min_test: continue
        
        p_search_range_test = np.arange(p_min_test, p_max_test + P_SEARCH_STEP * 0.5, P_SEARCH_STEP)
        best_r = -np.inf
        best_p_val = None
        
        for p_test in p_search_range_test:
            t_test = test_power - p_test
            if p_test <= 0 or t_test <= 0: continue
            
            p_minus_t = p_test - t_test
            sin_part = np.abs(sin_t)**test_power * np.sign(sin_t)**test_power # sin(a)^power
            
            # cos(a)^(p-t)
            cos_abs = np.abs(cos_t)
            cos_sign = np.sign(cos_t)
            if p_minus_t < 0:
                # Avoid divide-by-zero
                safe_cos = np.where(cos_abs < 1e-10, 1e-10, cos_abs)
                cos_part = (safe_cos ** abs(p_minus_t)) * (cos_sign ** abs(p_minus_t))
                cos_part = 1.0 / cos_part
            else:
                cos_part = (cos_abs ** p_minus_t) * (cos_sign ** p_minus_t)
                
            x_plot = sin_part * cos_part
            
            valid_xy = ~(np.isnan(x_plot) | np.isinf(x_plot))
            if np.sum(valid_xy) < 2: continue
            
            try:
                r, p_val = stats.pearsonr(x_plot[valid_xy], y_plot_test[valid_xy])
                if not np.isnan(r) and r > best_r:
                    best_r = r
                    best_p_val = p_val
            except:
                continue
        
        if best_r != -np.inf and best_p_val is not None:
            power_values.append(test_power)
            power_pearson_max.append(best_r)
            power_pearson_pvalue.append(best_p_val)
    
    # ==============================================================================
    # 5. Plotting (final polish)
    # ==============================================================================
    if len(power_values) > 0:
        ax3_twin = ax3.twinx()
        color_left = "#3F6F87"
        color_right = "#D47F30"
        
        # --- Left axis (Max Pearson) ---
        line1 = ax3.plot(
            power_values,
            power_pearson_max,
            "o-",
            label="Max Pearson Corr.",
            linewidth=3.0,
            markersize=9,
            color=color_left,
            markevery=8, # sparse markers
        )
        ax3.set_xlabel('Diffmean Power')
        ax3.set_ylabel('Max Pearson Corr.', color=color_left)
        ax3.tick_params(axis='y', labelcolor=color_left, color=color_left)
        
        # Spine coloring
        ax3.spines['left'].set_color(color_left)
        ax3.spines['left'].set_linewidth(1.5)
        ax3.spines['left'].set_zorder(5)
        ax3.spines['right'].set_visible(False)
        
        # Tighten X-axis range
        ax3.set_xlim(min(power_values), max(power_values))
        
        # --- Right axis (p-value) ---
        line2 = ax3_twin.plot(
            power_values,
            power_pearson_pvalue,
            "s-",
            label="p-value",
            linewidth=3.0,
            markersize=9,
            color=color_right,
            markevery=8, # sparse markers
        )
        ax3_twin.set_ylabel('p-value', color=color_right, labelpad=2)
        ax3_twin.tick_params(axis='y', which='both', colors=color_right, labelcolor=color_right)
        ax3_twin.set_yscale('log')
        for tick in ax3_twin.get_yticklabels():
            tick.set_color(color_right)
        ax3_twin.yaxis.get_offset_text().set_color(color_right)
        ax3_twin.yaxis.set_tick_params(colors=color_right, labelcolor=color_right)
        
        # Spine coloring
        ax3_twin.spines['right'].set_color(color_right)
        ax3_twin.spines['right'].set_linewidth(1.5)
        ax3_twin.spines['left'].set_visible(False)
        
        # --- Guides ---
        # 1. Max Peak
        idx_max_pearson = np.argmax(power_pearson_max)
        max_power_at = power_values[idx_max_pearson]
        max_pearson_value = power_pearson_max[idx_max_pearson]
        accent_green = "#5C8451"
        ax3.axvline(
            x=max_power_at,
            color=accent_green,
            linestyle='--',
            linewidth=3.0,
            alpha=0.9,
            zorder=3,
            label=f'Max Pearson Corr. at {max_power_at:.2f}',
        )
        ax3.plot(
            [max_power_at],
            [max_pearson_value],
            marker='o',
            markerfacecolor=accent_green,
            markeredgecolor=accent_green,
            markeredgewidth=2.2,
            markersize=11,
            zorder=3,
        )
        ax3_twin.plot(
            [max_power_at],
            [power_pearson_pvalue[idx_max_pearson]],
            marker='o',
            markerfacecolor=accent_green,
            markeredgecolor=accent_green,
            markeredgewidth=2.2,
            markersize=11,
            zorder=3,
        )
        y_min, y_max = ax3.get_ylim()
        x_min, x_max = ax3.get_xlim()
        y_mid = 0.5 * (y_min + y_max)
        x_offset = 0.015 * (x_max - x_min)
        ax3.text(
            max_power_at + x_offset,
            y_mid,
            f'{max_power_at:.2f}',
            color=accent_green,
            fontsize=plt.rcParams['legend.fontsize'],
            ha='left',
            va='center',
        )
        
        # --- Legend ---
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        
        # Attach to top axis (ax3_twin) so lines don't cover it
        legend = ax3_twin.legend(
            lines,
            labels,
            loc='lower right', 
            frameon=True,
            framealpha=0.6,   # opacity
            facecolor='white',
            edgecolor='0.7',
            fancybox=True,     # rounded corners
            # shadow=True,       # shadow
            fontsize=plt.rcParams['legend.fontsize']
        )
        legend.set_zorder(10)



        
        # --- Title and annotations ---
        ax3.grid(True, alpha=0.3)
        
    else:
        ax3.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax3.transAxes)
    
    plt.tight_layout()
    
    # Save figure
    if args.output_name:
        output_filename = args.output_name
        if not output_filename.lower().endswith(".pdf"):
            output_filename += ".pdf"
    else:
        output_filename = f"sensitive_analysis_alpha_power_{model_name}_{layer_name}.pdf"
    output_path = os.path.join(OUTPUT_DIR, "impl2", output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\nFigure saved to: {output_path}")
    
    # plt.show() # Comment out when running on a server

else:
    print("No data found!")
