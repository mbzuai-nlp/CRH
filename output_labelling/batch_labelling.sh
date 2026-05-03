set -euo pipefail

root_dir="CRH_paperviz/data/diff_location"
sample_concepts=100

for input_dir in "$root_dir"/*/; do
  [[ -d "$input_dir" ]] || continue
  output_root="${input_dir%/}/labeled-concepts"

  shopt -s nullglob
  json_files=("$input_dir"/*.json)
  shopt -u nullglob
  (( ${#json_files[@]} )) || continue

  for input_json in "${json_files[@]}"; do
    [[ "$input_json" == *_labelled.json ]] && continue
    base_name="$(basename "$input_json" .json)"
    labeled_json="${output_root}/${base_name}_labelled.json"
    out_dir="${output_root}/label_analysis_${base_name}"

    mkdir -p "$output_root"
    python3 res_labelling/label_gibberish.py --input_json "$input_json"
    python3 res_labelling/output_labelling.py --input_json "$input_json" --sample_concepts "$sample_concepts" --output_json "$labeled_json"
  done
done
