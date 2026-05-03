"""
python3 visualization/draw_line_nopenalty.py \
  --baseline-methods diffmean,PCA,mean_center,probe \
  --model llama --layer layer24 \
  --output-name llama_baseline_layer24.pdf
"""



import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm


def setup_fonts(base_size: int = 28) -> None:
    """Set global font config and sizes."""
    plt.rcParams["font.family"] = "STIXGeneral"
    plt.rcParams["mathtext.fontset"] = "stix"
    
    # Font size settings
    plt.rcParams["font.size"] = base_size
    plt.rcParams["axes.titlesize"] = base_size + 2
    plt.rcParams["axes.labelsize"] = base_size + 2
    plt.rcParams["xtick.labelsize"] = base_size
    plt.rcParams["ytick.labelsize"] = base_size
    plt.rcParams["legend.fontsize"] = base_size - 2 # Legend slightly smaller
    plt.rcParams["figure.titlesize"] = base_size + 4


def extract_model_layer(path: str) -> Tuple[str, str]:
    lower_path = path.lower()
    model = "unknown"
    if "gemma" in lower_path:
        model = "gemma"
    elif "llama" in lower_path:
        model = "llama"
    layer = "layer"
    for part in lower_path.split("_"):
        if part.startswith("layer"):
            layer = part
            break
    return model, layer


def find_labelled_file(baseline_dir: str, model: str, layer: str) -> str:
    pattern = os.path.join(
        baseline_dir,
        model,
        "labeled-concepts",
        f"*{layer}*_labelled.json",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else ""


def collect_rates_per_concept(
    labeled_results: List[Dict],
    exclude_step1_label2: bool = True,
    exclude_step1_label3: bool = True,
) -> Dict[int, Dict[int, List[float]]]:
    """
    Collect list of rates per concept for each step.
    
    Structure:
    {
        step: {
            label_id: [rate_concept_1, rate_concept_2, ...],
        }
    }
    This lets us compute variance (std error).
    """
    # step -> label -> list_of_rates
    step_data = defaultdict(lambda: defaultdict(list))

    for concept_data in tqdm(labeled_results, desc="Collecting concept-wise data"):
        concept_counts_label2 = defaultdict(lambda: defaultdict(int))
        concept_counts_label3 = defaultdict(lambda: defaultdict(int))
        
        experiments = concept_data.get("experiments", [])
        for exp in experiments:
            penalty_results = exp.get("penalty_results", [])
            
            # Legacy-compatible handling
            if not penalty_results:
                candidate_outputs = None
                for v in exp.values():
                    if isinstance(v, list) and len(v) > 0:
                        first = v[0]
                        if isinstance(first, dict) and "step" in first and "label" in first:
                            candidate_outputs = v
                            break
                        elif isinstance(first, (list, tuple)) and len(first) >= 4:
                            candidate_outputs = v
                            break
                if candidate_outputs:
                    penalty_results = [{"penalty": None, "outputs": candidate_outputs}]

            for penalty_result in penalty_results:
                outputs = penalty_result.get("outputs", [])
                if not outputs:
                    continue

                has_step1_label2 = False
                has_step1_label3 = False
                for item in outputs:
                    if isinstance(item, dict):
                        step = item.get("step")
                        label = item.get("label")
                    elif isinstance(item, list) and len(item) >= 4:
                        step = item[0]
                        label = item[3]
                    else:
                        continue
                    try:
                        step_int = int(step)
                    except (TypeError, ValueError):
                        continue
                    if step_int == 1 and label == 2:
                        has_step1_label2 = True
                    if step_int == 1 and label == 3:
                        has_step1_label3 = True

                allow_label2 = not (exclude_step1_label2 and has_step1_label2)
                allow_label3 = not (exclude_step1_label3 and has_step1_label3)

                for item in outputs:
                    if isinstance(item, dict):
                        step = item.get("step")
                        label = item.get("label")
                    elif isinstance(item, list) and len(item) >= 4:
                        step = item[0]
                        label = item[3]
                    else:
                        continue
                    if step is None or label is None:
                        continue
                    try:
                        step_int = int(step)
                    except (TypeError, ValueError):
                        continue
                    if allow_label2:
                        concept_counts_label2[step_int][label] += 1
                        concept_counts_label2[step_int]["_total"] += 1
                    if allow_label3:
                        concept_counts_label3[step_int][label] += 1
                        concept_counts_label3[step_int]["_total"] += 1
        
        # Finalize current concept data and add to totals
        for step, counts in concept_counts_label2.items():
            total = counts.get("_total", 0)
            if total > 0:
                step_data[step][2].append(counts.get(2, 0) / total)
        for step, counts in concept_counts_label3.items():
            total = counts.get("_total", 0)
            if total > 0:
                step_data[step][3].append(counts.get(3, 0) / total)

    return step_data


def build_stats_series(
    step_data: Dict[int, Dict[int, List[float]]],
    target_label: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics: mean, lower bound, upper bound (95% CI).
    
    Returns:
        steps, means, lower_bounds, upper_bounds
    """
    steps_sorted = sorted(step_data.keys())
    means = []
    lowers = []
    uppers = []
    
    for step in steps_sorted:
        rates = step_data[step].get(target_label, [])
        if not rates:
            means.append(np.nan)
            lowers.append(np.nan)
            uppers.append(np.nan)
        else:
            arr = np.array(rates)
            mu = np.mean(arr)
            # Compute standard error (Standard Error)
            # SEM = std / sqrt(n)
            # 95% CI approx = 1.96 * SEM
            n = len(arr)
            if n > 1:
                sem = np.std(arr, ddof=1) / np.sqrt(n)
                ci = 1.96 * sem
            else:
                ci = 0.0
            
            means.append(mu)
            lowers.append(max(0.0, mu - ci)) # Clamp to >= 0
            uppers.append(min(1.0, mu + ci)) # Clamp to <= 1
            
    return np.array(steps_sorted), np.array(means), np.array(lowers), np.array(uppers)


def plot_dual_lines_with_confidence(
    baseline_series: List[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]],
    output_path: str,
    show_ci: bool,
) -> None:
    """Plot multiple baselines on two subplots (label2/label3)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    colors = plt.cm.tab10.colors

    for idx, series in enumerate(baseline_series):
        color = colors[idx % len(colors)]
        name = series["name"]
        for ax, key, title in (
            (axes[0], "label2", "Rate of Outputs with Label 2"),
            (axes[1], "label3", "Rate of Corrupted Outputs"),
        ):
            steps, means, lowers, uppers = series[key]
            ax.plot(steps, means, color=color, linewidth=3, label=name)
            if show_ci:
                ax.fill_between(
                    steps,
                    lowers,
                    uppers,
                    color=color,
                    alpha=0.15,
                    edgecolor="none",
                )
            ax.set_title(title, pad=12)
            ax.set_xlabel("Step")
            label_pad = 0 if key == "label3" else 6
            ax.set_ylabel("Rate", labelpad=label_pad)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
            ax.set_xlim(1, 25)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
            ax.tick_params(axis="y", labelleft=True)

    legend_kwargs = {
        "frameon": True,
        "framealpha": 0.6,
        "edgecolor": "0.7",
        "fontsize": setup_fonts.__defaults__[0] - 6,
        "ncol": 1,
        "handlelength": 1.8,
        "labelspacing": 0.4,
        "borderpad": 0.4,
    }
    axes[0].legend(loc="upper right", **legend_kwargs)
    axes[1].legend(loc="lower right", **legend_kwargs)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.35)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot label rates over step with confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--labeled-results-path",
        type=str,
        default="./baseline_results/exp3_diffmean/llama/exp3_diffmean_layer24_labelled.json",
        help="Path to labeled results JSON file",
    )
    parser.add_argument(
        "--baseline-root",
        type=str,
        default="./baseline_results",
        help="Root directory containing baseline methods",
    )
    parser.add_argument(
        "--baseline-methods",
        type=str,
        default="",
        help="Comma-separated baseline method names (e.g., exp3_diffmean,exp3_PCA)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name (gemma/llama) for baseline comparison",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="",
        help="Layer name (e.g., layer24) for baseline comparison",
    )
    parser.add_argument(
        "--no-ci",
        action="store_false",
        dest="show_ci",
        help="Disable 95% confidence bands",
    )
    parser.set_defaults(show_ci=True)
    parser.add_argument(
        "--include-step1-label2",
        action="store_true",
        help="Include qids where step=1 label=2 (default excludes)",
    )
    parser.add_argument(
        "--include-step1-label3",
        action="store_true",
        help="Include qids where step=1 label=3 (default excludes)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="label34_rate_over_step_ci.pdf",
        help="Output filename",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_fonts()

    display_names = {
        "exp3_PCA": "PCA",
        "exp3_diffmean": "DiffMean",
        "exp3_mean_center": "MC",
        "exp3_probe": "Probe",
    }

    baseline_series = []
    if args.baseline_methods:
        methods = [m.strip() for m in args.baseline_methods.split(",") if m.strip()]
        model = args.model
        layer = args.layer
        if not model or not layer:
            inferred_model, inferred_layer = extract_model_layer(args.labeled_results_path)
            model = model or inferred_model
            layer = layer or inferred_layer
        for method in methods:
            baseline_dir = os.path.join(args.baseline_root, method)
            labelled_path = find_labelled_file(baseline_dir, model, layer)
            if not labelled_path:
                print(f"Skip: no labelled file for {method} ({model}, {layer})")
                continue
            try:
                with open(labelled_path, "r", encoding="utf-8") as f:
                    labeled_results = json.load(f)
            except Exception as exc:
                print(f"Skip: failed to load {labelled_path}: {exc}")
                continue
            step_data = collect_rates_per_concept(
                labeled_results,
                exclude_step1_label2=not args.include_step1_label2,
                exclude_step1_label3=not args.include_step1_label3,
            )
            if not step_data:
                print(f"Skip: no data in {labelled_path}")
                continue
            stats_label2 = build_stats_series(step_data, target_label=2)
            stats_label3 = build_stats_series(step_data, target_label=3)
            baseline_series.append(
                {
                    "name": display_names.get(method, method),
                    "label2": stats_label2,
                    "label3": stats_label3,
                }
            )
    else:
        print("Loading labeled results...")
        try:
            with open(args.labeled_results_path, "r", encoding="utf-8") as f:
                labeled_results = json.load(f)
            print(f"Loaded {len(labeled_results)} concepts")
        except Exception as exc:
            print(f"Error: failed to load labeled results: {exc}")
            return
        step_data = collect_rates_per_concept(
            labeled_results,
            exclude_step1_label2=not args.include_step1_label2,
            exclude_step1_label3=not args.include_step1_label3,
        )
        if not step_data:
            print("No data found.")
            return
        stats_label2 = build_stats_series(step_data, target_label=2)
        stats_label3 = build_stats_series(step_data, target_label=3)
        baseline_series.append(
            {
                "name": "baseline",
                "label2": stats_label2,
                "label3": stats_label3,
            }
        )

    if not baseline_series:
        print("No baseline series to plot.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    plot_dual_lines_with_confidence(baseline_series, output_path=output_path, show_ci=args.show_ci)

    print("Saved plot with confidence intervals to:")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()
