#!/usr/bin/env python3
"""
Label gibberish outputs in weight-grid or penalty-structured JSON.

Adds/sets "label" in each result item:
  - label = 3 for gibberish
  - leave existing non-null labels unchanged

Usage:
  python3 label_gibberish.py --input_json gemma-trimed/exp1_penalty_layer13.json
"""

import argparse
import json
import re
from collections import Counter


def is_gibberish(text):
    t = (text or "").strip()
    if not t:
        return True

    non_word = len(re.findall(r"[^A-Za-z0-9\\s]", t))
    if len(t) >= 20 and non_word / len(t) > 0.4:
        return True

    words = re.findall(r"[A-Za-z]+", t.lower())
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio <= 0.25:
            return True

        top_freq = Counter(words).most_common(1)[0][1] / len(words)
        if top_freq >= 0.4:
            return True

    if re.search(r"(\\b\\w+\\b)(\\s+\\1){3,}", t.lower()):
        return True

    if re.search(r"([\\W_])\\1{6,}", t):
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="batch_weight_grid_results.json")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    changed = 0

    def label_output_obj(obj, text_key="output"):
        nonlocal changed
        if text_key not in obj:
            return
        if "label" in obj and obj["label"] is not None:
            return
        if is_gibberish(obj.get(text_key, "")):
            obj["label"] = 3
            changed += 1
        else:
            if "label" not in obj:
                obj["label"] = None

    if isinstance(data, list):
        for concept in data:
            exp_type = concept.get("experiment_type")
            if exp_type == "penalty":
                for exp in concept.get("experiments", []):
                    for pres in exp.get("penalty_results", []):
                        for out in pres.get("outputs", []):
                            label_output_obj(out, text_key="output")
            elif exp_type == "components":
                for exp in concept.get("experiments", []):
                    for cres in exp.get("component_results", []):
                        for out in cres.get("outputs", []):
                            label_output_obj(out, text_key="output")
            elif exp_type == "step":
                for exp in concept.get("experiments", []):
                    for out in exp.get("outputs", []):
                        label_output_obj(out, text_key="output")
            elif exp_type == "multiscale_step":
                for exp in concept.get("experiments", []):
                    for sres in exp.get("scale_results", []):
                        for out in sres.get("outputs", []):
                            label_output_obj(out, text_key="output")
            else:
                for exp in concept.get("experiments", []):
                    for res in exp.get("results", []):
                        label_output_obj(res, text_key="output")
    elif isinstance(data, dict):
        exp_type = data.get("experiment_type")
        if exp_type == "penalty":
            for exp in data.get("experiments", []):
                for pres in exp.get("penalty_results", []):
                    for out in pres.get("outputs", []):
                        label_output_obj(out, text_key="output")
        elif exp_type == "components":
            for exp in data.get("experiments", []):
                for cres in exp.get("component_results", []):
                    for out in cres.get("outputs", []):
                        label_output_obj(out, text_key="output")
        elif exp_type == "step":
            for exp in data.get("experiments", []):
                for out in exp.get("outputs", []):
                    label_output_obj(out, text_key="output")
        elif exp_type == "multiscale_step":
            for exp in data.get("experiments", []):
                for sres in exp.get("scale_results", []):
                    for out in sres.get("outputs", []):
                        label_output_obj(out, text_key="output")
        else:
            raise ValueError("Unsupported input JSON structure.")
    else:
        raise ValueError("Unsupported input JSON structure.")

    target_path = args.output_json or args.input_json
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"updated {target_path}, gibberish labelled: {changed}")


if __name__ == "__main__":
    main()
