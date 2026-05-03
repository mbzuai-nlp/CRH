#!/usr/bin/env python3
"""
Evaluate weight-grid outputs into classes {1,2,3} using OpenAI API.

It only fills labels that are null. Existing label=3 (gibberish) is preserved.

Usage:
  python3 output_labelling.py --input_json gemma-trimed/exp1_penalty_layer9.json --sample_concepts 10 --max_workers 100
"""

import argparse
import csv
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI


API_KEY = "YOUR_DEEPINFRA_API_KEY_HERE"
BASE_URL = "YOUR_API_BASE_URL_HERE"
MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"



SYSTEM_PROMPT = (
    "You are an evaluator for a steering experiment. "
    "Given a user query, a target concept, and a model output, "
    "assign ONE label from {1,2,3} using these rules:\n"
    "1 = The output is a normal answer to the query, but it does NOT reflect the target concept.\n"
    "2 = The output contains or relates to the target concept (successful steering).\n"
    "3 = Gibberish/corrupted output (nonsense, heavy repetition, broken symbols).\n"
    "Notes: The model output may be truncated due to max tokens; truncation can occur in any label. "
    "Judge based on the visible content only.\n"
    "Return ONLY the label as a single digit: 1, 2, or 3."
)


def build_messages(query, concept, output):
    user_prompt = (
        f"Query:\n{query}\n\n"
        f"Target concept:\n{concept}\n\n"
        f"Model output:\n{output}\n\n"
        "Label:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: int):
        self.rate_per_sec = rate_per_sec
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        if self.rate_per_sec <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    refill = elapsed * self.rate_per_sec
                    if refill > 0:
                        self.tokens = min(self.capacity, self.tokens + refill)
                        self.last_refill = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
            time.sleep(0.005)


def classify_one(query, concept, output, client, semaphore, rate_limiter, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            with semaphore:
                if rate_limiter is not None:
                    rate_limiter.acquire()
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=build_messages(query, concept, output),
                    max_tokens=4,
                    stream=False,
                )
            text = resp.choices[0].message.content.strip()
            if text not in {"1", "2", "3", "4"}:
                raise ValueError(f"Invalid label: {text!r}")
            return text
        except Exception:
            if attempt >= max_retries:
                return None
            time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="batch_weight_grid_results.json")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=250)
    parser.add_argument("--max_inflight", type=int, default=200)
    parser.add_argument("--max_qps", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=50)
    # 0 means processing all concepts
    parser.add_argument("--sample_concepts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concept_ids", type=str, default="")
    parser.add_argument("--concept_csv", type=str, default="res_labelling/concepts_100.csv")
    parser.add_argument("--label_json", type=str, default="")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    def iter_concepts(obj):
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "experiments" in obj:
            return [obj]
        return []

    concepts = iter_concepts(data)

    def has_any_label(c):
        exp_type = c.get("experiment_type")
        for exp in c.get("experiments", []):
            if exp_type == "penalty":
                for pres in exp.get("penalty_results", []):
                    for out in pres.get("outputs", []):
                        if out.get("label") is not None:
                            return True
            elif exp_type == "components":
                for cres in exp.get("component_results", []):
                    for out in cres.get("outputs", []):
                        if out.get("label") is not None:
                            return True
            elif exp_type == "step":
                for out in exp.get("outputs", []):
                    if out.get("label") is not None:
                        return True
            elif exp_type == "multiscale_step":
                for sres in exp.get("scale_results", []):
                    for out in sres.get("outputs", []):
                        if out.get("label") is not None:
                            return True
            else:
                for res in exp.get("results", []):
                    if "output" not in res:
                        continue
                    if res.get("label") is not None:
                        return True
        return False

    def has_unlabeled(c):
        exp_type = c.get("experiment_type")
        for exp in c.get("experiments", []):
            if exp_type == "penalty":
                for pres in exp.get("penalty_results", []):
                    for out in pres.get("outputs", []):
                        if out.get("label") is None:
                            return True
            elif exp_type == "components":
                for cres in exp.get("component_results", []):
                    for out in cres.get("outputs", []):
                        if out.get("label") is None:
                            return True
            elif exp_type == "step":
                for out in exp.get("outputs", []):
                    if out.get("label") is None:
                        return True
            elif exp_type == "multiscale_step":
                for sres in exp.get("scale_results", []):
                    for out in sres.get("outputs", []):
                        if out.get("label") is None:
                            return True
            else:
                for res in exp.get("results", []):
                    if "output" not in res:
                        continue
                    if res.get("label") is None:
                        return True
        return False

    if args.concept_ids:
        wanted = {int(x.strip()) for x in args.concept_ids.split(",") if x.strip()}
        concepts = [c for c in concepts if c.get("cid") in wanted]

    labeled_cids = set()
    if args.label_json:
        with open(args.label_json, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        label_concepts = iter_concepts(label_data)
        label_map = {}
        for idx, c in enumerate(label_concepts):
            key = c.get("cid", idx)
            label_map[key] = c
            if has_any_label(c):
                labeled_cids.add(key)

        def maybe_set_label(base_item, label_item):
            if base_item.get("label") is None:
                label = label_item.get("label")
                if label is not None:
                    base_item["label"] = label

        for idx, base_concept in enumerate(concepts):
            key = base_concept.get("cid", idx)
            label_concept = label_map.get(key)
            if not label_concept:
                continue
            exp_type = base_concept.get("experiment_type")
            label_exps = label_concept.get("experiments", [])
            for exp_idx, exp in enumerate(base_concept.get("experiments", [])):
                if exp_idx >= len(label_exps):
                    break
                label_exp = label_exps[exp_idx]
                if exp_type == "penalty":
                    label_pres = label_exp.get("penalty_results", [])
                    for pres_idx, pres in enumerate(exp.get("penalty_results", [])):
                        if pres_idx >= len(label_pres):
                            break
                        label_outputs = label_pres[pres_idx].get("outputs", [])
                        for out_idx, out in enumerate(pres.get("outputs", [])):
                            if out_idx >= len(label_outputs):
                                break
                            maybe_set_label(out, label_outputs[out_idx])
                elif exp_type == "components":
                    label_cres = label_exp.get("component_results", [])
                    for cres_idx, cres in enumerate(exp.get("component_results", [])):
                        if cres_idx >= len(label_cres):
                            break
                        label_outputs = label_cres[cres_idx].get("outputs", [])
                        for out_idx, out in enumerate(cres.get("outputs", [])):
                            if out_idx >= len(label_outputs):
                                break
                            maybe_set_label(out, label_outputs[out_idx])
                elif exp_type == "step":
                    label_outputs = label_exp.get("outputs", [])
                    for out_idx, out in enumerate(exp.get("outputs", [])):
                        if out_idx >= len(label_outputs):
                            break
                        maybe_set_label(out, label_outputs[out_idx])
                elif exp_type == "multiscale_step":
                    label_sres = label_exp.get("scale_results", [])
                    for sres_idx, sres in enumerate(exp.get("scale_results", [])):
                        if sres_idx >= len(label_sres):
                            break
                        label_outputs = label_sres[sres_idx].get("outputs", [])
                        for out_idx, out in enumerate(sres.get("outputs", [])):
                            if out_idx >= len(label_outputs):
                                break
                            maybe_set_label(out, label_outputs[out_idx])
                else:
                    label_results = label_exp.get("results", [])
                    for res_idx, res in enumerate(exp.get("results", [])):
                        if res_idx >= len(label_results):
                            break
                        maybe_set_label(res, label_results[res_idx])

    output_data = data
    if args.sample_concepts and args.sample_concepts > 0:
        # Strictly cap the number of concepts to sample_concepts.
        candidates = [c for c in concepts if has_unlabeled(c)]
        random.seed(args.seed)
        concepts = random.sample(candidates, min(args.sample_concepts, len(candidates)))
        if isinstance(data, list):
            output_data = concepts

    concept_map = {}
    try:
        with open(args.concept_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cid = int(row.get("concept_id", ""))
                except ValueError:
                    continue
                name = row.get("output_concept", "") or row.get("concept", "")
                if name:
                    concept_map[cid] = name
    except FileNotFoundError:
        raise FileNotFoundError(f"concept_csv not found: {args.concept_csv}")

    tasks = []
    for concept in concepts:
        cid = concept.get("cid")
        if cid in labeled_cids:
            continue
        concept_name = concept.get("concept") or (concept_map.get(cid) if cid is not None else None)
        if not concept_name:
            raise ValueError(f"Missing concept for cid={cid} (concept_csv={args.concept_csv})")
        exp_type = concept.get("experiment_type")
        for exp in concept.get("experiments", []):
            query = exp.get("question", "")
            if exp_type == "penalty":
                for pres in exp.get("penalty_results", []):
                    for out in pres.get("outputs", []):
                        label = out.get("label")
                        if label is None:
                            tasks.append((out, query, concept_name))
            elif exp_type == "components":
                for cres in exp.get("component_results", []):
                    for out in cres.get("outputs", []):
                        label = out.get("label")
                        if label is None:
                            tasks.append((out, query, concept_name))
            elif exp_type == "step":
                for out in exp.get("outputs", []):
                    label = out.get("label")
                    if label is None:
                        tasks.append((out, query, concept_name))
            elif exp_type == "multiscale_step":
                for sres in exp.get("scale_results", []):
                    for out in sres.get("outputs", []):
                        label = out.get("label")
                        if label is None:
                            tasks.append((out, query, concept_name))
            else:
                for res in exp.get("results", []):
                    if "output" not in res:
                        continue
                    label = res.get("label")
                    if label is None:
                        tasks.append((res, query, concept_name))

    completed = 0
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=60,
        max_retries=0,
    )
    semaphore = threading.Semaphore(max(1, args.max_inflight))
    rate_limiter = None
    if args.max_qps and args.max_qps > 0:
        rate_limiter = TokenBucket(rate_per_sec=args.max_qps, capacity=max(1, int(args.max_qps)))
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_map = {
            ex.submit(
                classify_one,
                query,
                concept_name,
                item.get("output", ""),
                client,
                semaphore,
                rate_limiter,
            ): item
            for item, query, concept_name in tasks
        }
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="Labelling"):
            res = future_map[fut]
            label = fut.result()
            if label is not None:
                res["label"] = int(label)
            completed += 1
            if completed % args.save_every == 0:
                target_path = args.output_json or args.input_json
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

    target_path = args.output_json or args.input_json
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"updated {target_path} with {len(tasks)} labels")


if __name__ == "__main__":
    main()
