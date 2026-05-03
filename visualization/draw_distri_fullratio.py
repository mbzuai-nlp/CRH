"""
Results analysis module: analyze the ratios of label=3 and label=2 across (step, penalty) pairs.

Label definitions:
- 1: output unchanged
- 2: output matches the target concept
- 3: corrupted output

Output: one figure with two subplots (left: label=2 probability, right: label=3 probability)
"""

import argparse
import os
import numpy as np
import json
import re
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm


def setup_fonts(base_size: int = 26):
    """Set global font config + sizes (base_size affects most elements)."""
    plt.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # ===== Global sizes =====
    plt.rcParams['font.size'] = base_size          # Default size
    plt.rcParams['axes.titlesize'] = base_size + 2 # Axis title (if used)
    plt.rcParams['axes.labelsize'] = base_size + 2 # Axis labels
    plt.rcParams['xtick.labelsize'] = base_size    # x tick
    plt.rcParams['ytick.labelsize'] = base_size    # y tick
    plt.rcParams['legend.fontsize'] = base_size
    plt.rcParams['figure.titlesize'] = base_size + 4



# ============================================================================
# Utilities
# ============================================================================

def collect_step_penalty_label_data(
    labeled_results: List[Dict],
    exclude_step1_label: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, float], Counter], int, int]:
    """
    Collect label distributions for all (step, penalty) pairs.

    Args:
        labeled_results: list of labeled result dicts

    Returns:
        step_penalty_labels: dict keyed by (step, penalty) with Counter of labels
    """
    step_penalty_labels = defaultdict(Counter)
    total_qids = 0
    excluded_qids = 0
    
    for concept_data in tqdm(labeled_results, desc="Collecting data"):
        cid = concept_data.get('cid')
        if cid is None:
            continue
        
        # Get all experiments for this concept
        experiments = concept_data.get('experiments', [])
        
        for exp in experiments:
            qid = exp.get('qid')
            if qid is None:
                continue
            
            # Iterate all penalty results
            penalty_results = exp.get('penalty_results', [])
            
            has_outputs = False
            exclude_qid = False
            for penalty_result in penalty_results:
                res_penalty = penalty_result.get('penalty')
                if res_penalty is None:
                    continue
                output_list = penalty_result.get('outputs', [])
                if not output_list:
                    continue
                has_outputs = True
                if exclude_step1_label is not None:
                    for output_item in output_list:
                        if isinstance(output_item, dict):
                            step = output_item.get('step')
                            label = output_item.get('label')
                        elif isinstance(output_item, list) and len(output_item) >= 4:
                            step = output_item[0]
                            label = output_item[3]
                        else:
                            continue
                        if step == 1 and label == exclude_step1_label:
                            exclude_qid = True
                            break
                    if exclude_qid:
                        break

            if not has_outputs:
                continue
            total_qids += 1
            if exclude_qid:
                excluded_qids += 1
                continue

            for penalty_result in penalty_results:
                res_penalty = penalty_result.get('penalty')
                if res_penalty is None:
                    continue
                output_list = penalty_result.get('outputs', [])
                if not output_list:
                    continue
                for output_item in output_list:
                    if isinstance(output_item, dict):
                        step = output_item.get('step')
                        label = output_item.get('label')
                        if step is not None and label is not None:
                            step_penalty_labels[(step, res_penalty)][label] += 1
                    elif isinstance(output_item, list) and len(output_item) >= 4:
                        step = output_item[0]
                        label = output_item[3]
                        if step is not None and label is not None:
                            step_penalty_labels[(step, res_penalty)][label] += 1
    
    return step_penalty_labels, total_qids, excluded_qids

def build_prob_matrix(
    step_penalty_labels: Dict[Tuple[int, float], Counter],
    steps_sorted: List[int],
    penalties_desc: List[float],
    target_label: int,
) -> np.ndarray:
    """
    Build matrix: rows are steps (bottom to top increasing), columns are penalties (left to right decreasing).
    Each cell is the target_label probability within that configuration.
    """
    data = np.full((len(steps_sorted), len(penalties_desc)), np.nan)

    for s_idx, step in enumerate(steps_sorted):
        for p_idx, penalty in enumerate(penalties_desc):
            key = (step, penalty)
            if key not in step_penalty_labels:
                continue
            counter = step_penalty_labels[key]
            total = sum(counter.values())
            if total == 0:
                continue
            prob = counter.get(target_label, 0) / total
            data[s_idx, p_idx] = prob
    return data


def plot_dual_heatmaps(
    data_label3: np.ndarray,
    data_label2: np.ndarray,
    steps_sorted: List[int],
    penalties_desc: List[float],
    output_path: str,
):
    """
    Changes:
    1. Swap left/right: left shows Label 2, right shows Label 3.
    2. Force the right plot to display the Y-axis label ("Step") and tick labels.
    """
    cmap3 = LinearSegmentedColormap.from_list(
        "custom_blue",
        ["#FFFFFF", "#4A839E"],
    )
    cmap4 = LinearSegmentedColormap.from_list(
        "custom_orange",
        ["#FFFFFF", "#E7923F"],
    )

    def get_vmin_vmax(data: np.ndarray):
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            return 0.0, 1.0
        vmin = float(valid.min())
        vmax = float(valid.max())
        if np.isclose(vmin, vmax):
            eps = 1e-6
            vmin = max(0.0, vmin - eps)
            vmax = min(1.0, vmax + eps)
        return vmin, vmax

    vmin3, vmax3 = get_vmin_vmax(data_label3)
    vmin2, vmax2 = get_vmin_vmax(data_label2)

    # ===== Tick sparsification config =====
    step_every = 5          
    penalty_every = 0.1     

    step_tick_pos = [i for i, s in enumerate(steps_sorted) if (s % step_every == 0)]
    step_tick_lab = [str(steps_sorted[i]) for i in step_tick_pos]

    penalty_tick_pos = []
    penalty_tick_lab = []
    for j, p in enumerate(penalties_desc):
        p_round = round(p, 2)
        if abs(p_round / penalty_every - round(p_round / penalty_every)) < 1e-6:
            penalty_tick_pos.append(j)
            penalty_tick_lab.append(f"{p_round:.1f}")

    # sharey=True hides right Y ticks by default; re-enable below
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Swap order: axes[0] for label2 (orange), axes[1] for label3 (blue)
    configs = [
        (axes[0], data_label2, cmap4, vmin2, vmax2, "Rate of Outputs Changed but Not Target"),
        (axes[1], data_label3, cmap3, vmin3, vmax3, "Rate of Corrupted Outputs"),
    ]

    for idx, (ax, data, cmap, vmin, vmax, title) in enumerate(configs):
        im = ax.imshow(
            data,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            origin="lower",
        )
        ax.set_title(title, pad=20)

        # ===== Axis labels =====
        ax.set_xlabel("Penalty", labelpad=-4)
        
        # Change: always show "Step" on both axes
        ax.set_ylabel("Step")

        # ===== Sparse ticks =====
        ax.set_xticks(penalty_tick_pos)
        ax.set_xticklabels(penalty_tick_lab)

        ax.set_yticks(step_tick_pos)
        ax.set_yticklabels(step_tick_lab)
        
        # Change: force Y tick labels to show (override sharey=True)
        ax.tick_params(labelleft=True)

        # ===== Gridlines =====
        ax.set_xticks(np.arange(len(penalties_desc)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(steps_sorted)) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        # ===== Colorbar =====
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t:.1%}" for t in ticks])

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.42)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()






def extract_model_layer(path: str) -> Tuple[str, str]:
    """Extract model name (gemma/llama) and layer from path/filename."""
    lower_path = path.lower()
    model_match = re.search(r'(gemma|llama)', lower_path)
    model = model_match.group(1) if model_match else "unknown"

    layer_match = re.search(r'layer(\d+)', lower_path)
    layer = f"layer{layer_match.group(1)}" if layer_match else "layer"
    return model, layer


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze label distribution across (step, penalty) pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--labeled-results-path',
        type=str,
        default="./data/diff_location/llama_steer_last_prompt_token/labeled-concepts/exp1_llama_layer24_last_prompt_token_labelled.json",
        help='Path to labeled results JSON file'
    )
    parser.add_argument(
        '--only-label2-concepts',
        action='store_true',
        help='Keep only concepts that contain label=2'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=".",
        help='Output directory for figures (default: current directory)'
    )
    parser.add_argument(
        '--include-step1-label2',
        action='store_true',
        help='Keep qids with step=1 and label=2 (excluded by default)',
    )
    parser.add_argument(
        '--include-step1-label3',
        action='store_true',
        help='Keep qids with step=1 and label=3 (excluded by default; affects right plot only)',
    )
    
    return parser.parse_args()


def main():
    """Main: load data and generate heatmaps."""
    
    # Parse command-line arguments
    args = parse_args()
    labeled_results_path = args.labeled_results_path
    
    # Load labeled results
    print("Loading labeled results...")
    try:
        with open(labeled_results_path, 'r', encoding='utf-8') as f:
            labeled_results = json.load(f)
        print(f"Loaded {len(labeled_results)} concepts")
    except Exception as e:
        print(f"Error: failed to load labeled results file: {e}")
        return
    
    # Collect data
    print("\nCollecting data...")
    if args.only_label2_concepts:
        def concept_has_label2(concept: Dict) -> bool:
            for exp in concept.get('experiments', []):
                for pres in exp.get('penalty_results', []):
                    for out in pres.get('outputs', []):
                        if isinstance(out, dict) and out.get('label') == 2:
                            return True
                        if isinstance(out, list) and len(out) >= 4 and out[3] == 2:
                            return True

        before_count = len(labeled_results)
        labeled_results = [c for c in labeled_results if concept_has_label2(c)]
        print(f"Filtered concepts: {before_count} -> {len(labeled_results)} (only keep concepts with label=2)")

    exclude_label2 = None if args.include_step1_label2 else 2
    exclude_label3 = None if args.include_step1_label3 else 3

    step_penalty_labels_label2, total_qids, excluded_label2 = collect_step_penalty_label_data(
        labeled_results,
        exclude_step1_label=exclude_label2,
    )
    step_penalty_labels_label3, total_qids_3, excluded_label3 = collect_step_penalty_label_data(
        labeled_results,
        exclude_step1_label=exclude_label3,
    )
    if total_qids_3 != total_qids:
        total_qids = max(total_qids, total_qids_3)

    print(f"Total cases (qids): {total_qids}")
    if exclude_label2 is not None:
        print(f"Excluded cases for left plot (step=1,label=2): {excluded_label2}")
    if exclude_label3 is not None:
        print(f"Excluded cases for right plot (step=1,label=3): {excluded_label3}")
    
    if len(step_penalty_labels_label2) == 0 and len(step_penalty_labels_label3) == 0:
        print("No data found!")
        return
    
    # Extract unique steps (ascending) and penalties (descending)
    all_keys = set(step_penalty_labels_label2.keys()) | set(step_penalty_labels_label3.keys())
    print(f"Collected data for {len(all_keys)} (step, penalty) pairs")
    steps_sorted = sorted(set(step for step, _ in all_keys))
    penalties_desc = sorted(set(penalty for _, penalty in all_keys), reverse=True)
    
    print(f"Step range: {min(steps_sorted)} - {max(steps_sorted)}")
    print(f"Penalty range: {min(penalties_desc):.2f} - {max(penalties_desc):.2f} (x-axis decreases left to right)")
    
    # Set fonts
    setup_fonts()

    os.makedirs(args.output_dir, exist_ok=True)
    model, layer = extract_model_layer(labeled_results_path)
    output_both = os.path.join(args.output_dir, f"viz", f"label23_prob_{model}_{layer}.pdf")

    heatmap_label3 = build_prob_matrix(step_penalty_labels_label3, steps_sorted, penalties_desc, target_label=3)
    heatmap_label2 = build_prob_matrix(step_penalty_labels_label2, steps_sorted, penalties_desc, target_label=2)

    plot_dual_heatmaps(
        heatmap_label3,
        heatmap_label2,
        steps_sorted,
        penalties_desc,
        output_path=output_both,
    )

    print("\nFigure saved to:")
    print(f"  {output_both}")


if __name__ == "__main__":
    main()
