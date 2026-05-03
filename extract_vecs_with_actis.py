"""
Batch extract and save difference vectors

Process summary:
1) Read a CSV containing qid,cid,question,concept,resp_no_concept,resp_with_concept
2) For each row:
   - Build prompts and extract the last token activation difference vector between "no concept" and "with concept"
   - Save to {output_dir}/question{qid}/{qid}-{cid}.pt with diff vector and metadata
3) Resume support: skip if output exists, start from the first missing item
4) Show overall progress with tqdm

python optim_vecs_only_diff_vecs.py \
  --csv_path CRH_Data/data_pairs_gemma2b_805_filtered.csv \
  --model_name /path/to/models/gemma-2b-it \
  --output_dir CRH_Data/diff_vecs_with_actis/gemma2b


python optim_vecs_only_diff_vecs.py \
  --csv_path CRH_Data/data_pairs_llama7b_805_filtered.csv \
  --model_name /path/to/models/Llama-2-7b-chat-hf \
  --output_dir CRH_Data/diff_vecs_with_actis/llama2-7b-chat


"""

import os
import sys
import csv
import argparse
from typing import Dict, List

import torch
from tqdm import tqdm

# === Import utility functions ===
CAUSAL_DIR = "./causal_intervention"

sys.path.append(CAUSAL_DIR)

from causal_intervention.utils import (  # noqa: E402
    load_model,
    get_device,
    get_model_blocks,
)


# ============== Helper functions ==============
def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch extract and save difference vectors from a CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_name", type=str, default="/path/to/model/gemma-2-2b-it",
                        help="Model path or name")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Input CSV; must contain qid,cid,question,concept,resp_no_concept,resp_with_concept")
    parser.add_argument("--output_dir", type=str, default="/path/to/output/opt_results",
                        help="Output directory; saved as question{qid}/{qid}-{cid}.pt")
    parser.add_argument("--layer_idx", type=int, default=None, help="Single layer index (0-based); if empty, use 1/2 and 3/4 depth layers")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu); auto-select by default")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    parser.add_argument("--clear_cache_freq", type=int, default=1, help="Clear GPU cache every N samples (default 1)")
    return parser.parse_args()


def build_prompt(question: str, response: str) -> str:
    """Simple prompt template; keep both samples consistent."""
    return f"User: {question}\nAssistant: {response}"


def extract_last_token_activation(model, tokenizer, text: str, layer_idx: int, device) -> torch.Tensor:
    """
    Get the activation of the last token at the specified layer (using hidden states output).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # List, length is num_layers + 1
    if layer_idx + 1 >= len(hidden_states):
        raise ValueError(f"layer_idx={layer_idx} out of hidden states range {len(hidden_states)-1}")
    return hidden_states[layer_idx + 1][:, -1, :].squeeze(0).detach()


class DeviceTokenizerWrapper:
    """
    Lightweight wrapper: move tokenizer outputs to the target device.
    Does not modify the original tokenizer state; suitable for multithreaded sharing.
    """
    def __init__(self, tokenizer, device):
        self._tok = tokenizer
        self._device = device

    def __call__(self, *args, **kwargs):
        from transformers.tokenization_utils_base import BatchEncoding
        result = self._tok(*args, **kwargs)
        if isinstance(result, BatchEncoding):
            moved = {}
            for k, v in result.items():
                moved[k] = v.to(self._device) if isinstance(v, torch.Tensor) else v
            return BatchEncoding(moved, encoding=getattr(result, "encodings", None))
        return result

    def encode(self, *args, **kwargs):
        return self._tok.encode(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._tok, item)




def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_default_layers(model) -> List[int]:
    """
    Default to layers at 1/2 and 3/4 of model depth (0-based).
    If the depth is too small, de-duplicate and clip to valid range.
    """
    num_layers = None
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    if num_layers is None:
        try:
            num_layers = len(get_model_blocks(model))
        except Exception:
            pass
    if not num_layers or num_layers <= 0:
        raise ValueError("Unable to determine model depth; please specify --layer_idx explicitly")
    mid = int(num_layers * 0.5)
    upper = int(num_layers * 0.75)
    candidates = [mid, upper]
    # Clip to [0, num_layers-1] and de-duplicate while preserving order
    seen = set()
    layers = []
    for l in candidates:
        l_clip = min(max(l, 0), num_layers - 1)
        if l_clip not in seen:
            layers.append(l_clip)
            seen.add(l_clip)
    return layers


def load_csv_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    required = {"qid", "cid", "question", "concept", "resp_no_concept", "resp_with_concept"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"CSV missing fields: {missing}")
    return rows


def find_start_index(rows: List[Dict[str, str]], output_dir: str) -> int:
    """
    Return the index of the first unfinished sample.
    """
    for idx, row in enumerate(rows):
        qid = row["qid"]
        cid = row["cid"]
        save_path = os.path.join(output_dir, f"question{qid}", f"{qid}-{cid}.pt")
        if not os.path.exists(save_path):
            return idx
    return len(rows)


def count_remaining_samples(rows: List[Dict[str, str]], output_dir: str, start_idx: int = 0) -> int:
    """
    Count remaining samples from start_idx, excluding existing outputs.
    """
    count = 0
    for idx in range(start_idx, len(rows)):
        row = rows[idx]
        qid = row["qid"]
        cid = row["cid"]
        save_path = os.path.join(output_dir, f"question{qid}", f"{qid}-{cid}.pt")
        if not os.path.exists(save_path):
            count += 1
    return count


def save_result(
    output_path: str,
    diff_vector: torch.Tensor,
    act_no: torch.Tensor,
    act_with: torch.Tensor,
    meta: Dict,
):
    ensure_dir(os.path.dirname(output_path))
    payload = {
        "vectors": {
            "diff_vector": diff_vector,  # diff_vector is already on CPU
            "act_no": act_no,            # Activation for no-concept sample
            "act_with": act_with,        # Activation for with-concept sample
        },
        "meta": meta,
    }
    torch.save(payload, output_path)


# ============== Main flow ==============
def main():
    args = parse_args()
    device = get_device(device=args.device)

    # Read CSV
    rows = load_csv_rows(args.csv_path)

    # Load model
    model, tokenizer_raw, device = load_model(args.model_name, device=device)
    tokenizer = DeviceTokenizerWrapper(tokenizer_raw, device)

    # Determine layer list
    if args.layer_idx is not None:
        layer_list = [args.layer_idx]
    else:
        layer_list = get_default_layers(model)
    print(f"Layers to process: {layer_list}")

    for layer_choice in layer_list:
        layer_output_root = os.path.join(args.output_dir, str(layer_choice))
        start_idx = find_start_index(rows, layer_output_root) if not args.overwrite else 0
        
        # Count remaining samples (excluding existing outputs)
        if args.overwrite:
            total_remaining = len(rows) - start_idx
        else:
            total_remaining = count_remaining_samples(rows, layer_output_root, start_idx)
        
        print(f"\n=== Processing layer {layer_choice} ===")
        print(f"Total samples: {len(rows)}, start index: {start_idx}, remaining: {total_remaining}")

        if total_remaining <= 0:
            print(f"Layer {layer_choice} already complete, skipping.")
            continue

        pbar = tqdm(total=total_remaining, desc=f"Layer {layer_choice} samples", unit="sample")

        for idx in range(start_idx, len(rows)):
            row = rows[idx]
            qid = row["qid"]
            cid = row["cid"]
            question = row["question"]
            concept = row["concept"]
            resp_no = row["resp_no_concept"]
            resp_with = row["resp_with_concept"]

            save_dir = os.path.join(layer_output_root, f"question{qid}")
            save_path = os.path.join(save_dir, f"{qid}-{cid}.pt")

            if os.path.exists(save_path) and not args.overwrite:
                pbar.update(1)
                continue

            # try:
            prompt_no = build_prompt(question, resp_no)
            prompt_with = build_prompt(question, resp_with)

            act_no = extract_last_token_activation(model, tokenizer, prompt_no, layer_choice, device)
            act_with = extract_last_token_activation(model, tokenizer, prompt_with, layer_choice, device)
            diff_vec_original = act_with - act_no  # Save raw (unnormalized) difference vector
            diff_norm_original = torch.norm(diff_vec_original).item()  # Save raw norm
            
            # Clear activations to release memory
            act_no_cpu = act_no.cpu()
            act_with_cpu = act_with.cpu()
            diff_vec_original_cpu = diff_vec_original.cpu()
            del act_no, act_with, diff_vec_original
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            meta = {
                "qid": qid,
                "cid": cid,
                "question": question,
                "concept": concept,
                "layer_idx": layer_choice,
                "diff_norm": diff_norm_original,
                "model_name": args.model_name,
            }

            save_result(
                output_path=save_path,
                diff_vector=diff_vec_original_cpu,
                act_no=act_no_cpu,
                act_with=act_with_cpu,
                meta=meta,
            )
            
            # Clear variables
            del act_no_cpu, act_with_cpu
            del diff_vec_original_cpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(1)
            
            # Periodically clear GPU cache
            if (idx - start_idx + 1) % args.clear_cache_freq == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        pbar.close()

    print("All processing completed.")


if __name__ == "__main__":
    main()
