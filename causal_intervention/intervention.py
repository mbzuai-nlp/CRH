"""
Run steering-vector interventions and compare baselines.

Workflow per index:
1. Fetch helix/spiral point from probe outputs.
2. Build patched vector via P^+ and W; also gather original vec vector.
3. Create two baselines: random-norm-matched vector and PCA-truncated vector.
4. Measure losses and generated text for each vector with steering hooks.
5. Save aggregated metrics to CSV.
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Add steering_opt repo to import path
sys.path.append('./llm-steering-opt')

# train_probe helpers
from train_probe import restore_activation_from_spiral, generate_spiral_labels
from utils import (
    get_device, register_steering_hook, 
    calculate_loss_for_vector, find_diff_token_position
)
import steering_opt


def steer_generate_no_progress(prompt, model, tokenizer, device, max_new_tokens=20, **kwargs):
    """
    Minimal generate loop with steering hooks already registered elsewhere.
    Does not show a tqdm bar; applies temperature/top-p and stops on EOS.
    """
    model.eval()
    # Sampling controls
    temperature = kwargs.get('temperature', 0.1)
    top_p = kwargs.get('top_p', 1.0)
    
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):  # plain range to avoid a progress bar
            # Forward pass for next-token logits
            outputs = model(input_ids=generated_ids, use_cache=False)
            logits = outputs.logits[:, -1, :]  # last token logits
            
            # Guard against NaN/Inf
            logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
            
            # Temperature scaling
            if temperature != 1.0:
                temp = max(temperature, 1e-8)
                logits = logits / temp
                logits = torch.clamp(logits, min=-1e10, max=1e10)
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                probs = torch.softmax(logits, dim=-1)
                probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs = torch.clamp(probs, min=0.0, max=1.0)
                
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs <= top_p
                mask[..., 0] = True
                sorted_probs = sorted_probs * mask.float()
                prob_sum = sorted_probs.sum(dim=-1, keepdim=True)
                prob_sum = torch.clamp(prob_sum, min=1e-10)
                sorted_probs = sorted_probs / prob_sum
                sorted_probs = torch.clamp(sorted_probs, min=0.0, max=1.0)
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token_id = sorted_indices.gather(1, sampled_idx)
            else:
                probs = torch.softmax(logits, dim=-1)
                probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs = torch.clamp(probs, min=0.0, max=1.0)
                prob_sum = probs.sum(dim=-1, keepdim=True)
                prob_sum = torch.clamp(prob_sum, min=1e-10)
                probs = probs / prob_sum
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Append sampled token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
            # Early stop on EOS
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break
    
    # Strip the prompt portion from the output
    input_len = input_ids.shape[1]
    generated_len = generated_ids.shape[1]
    
    if generated_len > input_len:
        # Remove prompt tokens
        output_ids = generated_ids[:, input_len:]
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    else:
        # No generation happened
        output_text = ""
    
    input_text = prompt
    
    return {
        'input': input_text,
        'output': output_text,
        'full': tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    }


def get_spiral_point_at_idx(probe_dict, idx):
    """
    Retrieve the 3D spiral/helix point for a given index from the probe dict.
    """
    Y = probe_dict['Y']  # (n_samples, 3)
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    
    n_samples = Y.shape[0]
    if idx < 0 or idx >= n_samples:
        raise ValueError(f"idx {idx} out of range [0, {n_samples-1}]")
    
    return Y[idx]  # (3,)


def create_patched_vector(spiral_point, probe_dict, device=None):
    """
    Construct the patched steering vector: x = W^T P^+ helix(idx).
    Matches the helix coordinate to hidden dimension via probe matrices.
    """
    P = probe_dict['P']  # (3, k)
    P_pinv = probe_dict['P_pinv']  # (k, 3)
    W = probe_dict['W']  # (k, d)
    
    # Ensure numpy array for probe-space projection
    if isinstance(spiral_point, torch.Tensor):
        spiral_point = spiral_point.detach().cpu().numpy()
    if spiral_point.ndim == 1:
        spiral_point = spiral_point.reshape(1, -1)
    
    # Convert probe matrices to numpy for matmul
    P_pinv_np = P_pinv.detach().cpu().numpy() if isinstance(P_pinv, torch.Tensor) else P_pinv
    W_np = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else W
    
    # Project helix point into probe space: z = P^+ @ helix(idx)
    z = spiral_point @ P_pinv_np.T  # (1, k)
    
    # Map to hidden space: x_centered = z @ W = W^T @ z^T
    x_centered = z @ W_np  # (1, d)
    
    # Convert back to torch tensor
    if isinstance(x_centered, np.ndarray):
        patched_vector = torch.from_numpy(x_centered.squeeze(0)).float()
    else:
        patched_vector = x_centered.squeeze(0).float()
    
    if device is not None:
        patched_vector = patched_vector.to(device)
    
    return patched_vector


def create_random_patch_vector(ref_vector, device=None):
    """
    Baseline: random vector rescaled to match the norm of ref_vector.
    """
    if isinstance(ref_vector, torch.Tensor):
        ref_norm = torch.norm(ref_vector).item()
        random_vector = torch.randn_like(ref_vector)
        random_norm = torch.norm(random_vector).item()
        if random_norm > 1e-8:
            random_vector = random_vector / random_norm * ref_norm
    else:
        ref_norm = np.linalg.norm(ref_vector)
        random_vector = np.random.randn(*ref_vector.shape)
        random_norm = np.linalg.norm(random_vector)
        if random_norm > 1e-8:
            random_vector = random_vector / random_norm * ref_norm
        random_vector = torch.from_numpy(random_vector).float()
    
    if device is not None:
        random_vector = random_vector.to(device)
    elif isinstance(ref_vector, torch.Tensor):
        random_vector = random_vector.to(ref_vector.device)
    
    return random_vector


def create_pca_patch_vector(ref_vector, probe_dict, top_k=10, device=None):
    """
    Baseline: PCA-truncated reconstruction using top-k probe components.
    """
    W = probe_dict['W']  # (k, d)
    X_mean = probe_dict['mean']  # (d,)
    
    # Convert to numpy for PCA projection
    if isinstance(ref_vector, torch.Tensor):
        ref_vector_np = ref_vector.detach().cpu().numpy()
    else:
        ref_vector_np = ref_vector
    
    if isinstance(X_mean, torch.Tensor):
        X_mean_np = X_mean.detach().cpu().numpy()
    else:
        X_mean_np = X_mean
    
    if isinstance(W, torch.Tensor):
        W_np = W.detach().cpu().numpy()
    else:
        W_np = W
    
    # Center by probe mean
    ref_vector_centered = ref_vector_np - X_mean_np
    
    # PCA projection: z = ref_vector_centered @ W^T
    z = ref_vector_centered @ W_np.T  # (k,)
    
    # Keep only top-k components
    k_actual = min(top_k, len(z))
    z_topk = np.zeros_like(z)
    z_topk[:k_actual] = z[:k_actual]
    
    # Reconstruct from top-k: ref_vector_pca = z_topk @ W
    ref_vector_pca = z_topk @ W_np  # (d,)
    
    # Return reconstructed (centered) vector
    pca_vector = ref_vector_pca
    
    # Convert back to torch tensor
    if isinstance(pca_vector, np.ndarray):
        pca_vector = torch.from_numpy(pca_vector).float()
    else:
        pca_vector = torch.from_numpy(pca_vector).float()
    
    if device is not None:
        pca_vector = pca_vector.to(device)
    elif isinstance(ref_vector, torch.Tensor):
        pca_vector = pca_vector.to(ref_vector.device)
    
    return pca_vector


def run_intervention_experiment(
    model,
    tokenizer,
    probe_dict,
    vectors_tensor,
    datapoint,
    layer,
    diff_pos,
    idx,
    device=None,
    max_new_tokens=30,
    **kwargs
):
    """
    Run one intervention trial for a given idx and return losses/texts for all vectors.
    """
    device = get_device(model=model, device=device)
    model = model.to(device)
    
    # 1. Build patched vector from probe manifold point
    spiral_point = get_spiral_point_at_idx(probe_dict, idx)
    patched_vector = create_patched_vector(spiral_point, probe_dict, device=device)
    
    # 2. Reference vector from stored steering vectors
    if idx >= vectors_tensor.shape[0]:
        raise ValueError(f"idx {idx} out of range for vectors_tensor [0, {vectors_tensor.shape[0]-1}]")
    vec_vector = vectors_tensor[idx].to(device)
    
    # 2.5. Baseline vectors
    random_patch_vector = create_random_patch_vector(vec_vector, device=device)
    pca_patch_vector = create_pca_patch_vector(vec_vector, probe_dict, top_k=10, device=device)
    
    # 3. Evaluate patched vector loss
    patched_loss = calculate_loss_for_vector(
        model=model,
        tokenizer=tokenizer,
        datapoint=datapoint,
        layer=layer,
        vector=patched_vector,
        device=device,
        only_hook_prompt=kwargs.get('only_hook_prompt', True),
        coldness=kwargs.get('coldness', 0.7),
        eps=kwargs.get('eps', 1e-6),
        do_one_minus=kwargs.get('do_one_minus', True),
        normalize_token_length=kwargs.get('normalize_token_length', False)
    )
    
    # 4. Evaluate original vec loss
    vec_loss = calculate_loss_for_vector(
        model=model,
        tokenizer=tokenizer,
        datapoint=datapoint,
        layer=layer,
        vector=vec_vector,
        device=device,
        only_hook_prompt=kwargs.get('only_hook_prompt', True),
        coldness=kwargs.get('coldness', 0.7),
        eps=kwargs.get('eps', 1e-6),
        do_one_minus=kwargs.get('do_one_minus', True),
        normalize_token_length=kwargs.get('normalize_token_length', False)
    )
    
    # 4.5. Evaluate baseline losses
    random_patch_loss = calculate_loss_for_vector(
        model=model,
        tokenizer=tokenizer,
        datapoint=datapoint,
        layer=layer,
        vector=random_patch_vector,
        device=device,
        only_hook_prompt=kwargs.get('only_hook_prompt', True),
        coldness=kwargs.get('coldness', 0.7),
        eps=kwargs.get('eps', 1e-6),
        do_one_minus=kwargs.get('do_one_minus', True),
        normalize_token_length=kwargs.get('normalize_token_length', False)
    )
    
    pca_patch_loss = calculate_loss_for_vector(
        model=model,
        tokenizer=tokenizer,
        datapoint=datapoint,
        layer=layer,
        vector=pca_patch_vector,
        device=device,
        only_hook_prompt=kwargs.get('only_hook_prompt', True),
        coldness=kwargs.get('coldness', 0.7),
        eps=kwargs.get('eps', 1e-6),
        do_one_minus=kwargs.get('do_one_minus', True),
        normalize_token_length=kwargs.get('normalize_token_length', False)
    )
    
    # 5. Generate text with patched vector
    handle_patched = register_steering_hook(
        model, layer, patched_vector, alpha=1.0, target_pos=diff_pos
    )
    try:
        patched_result = steer_generate_no_progress(
            datapoint.prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 1.0)
        )
        patched_text = patched_result['output']
    finally:
        handle_patched.remove()
    
    # 6. Generate text with original vector
    handle_vec = register_steering_hook(
        model, layer, vec_vector, alpha=1.0, target_pos=diff_pos
    )
    try:
        vec_result = steer_generate_no_progress(
            datapoint.prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 1.0)
        )
        vec_text = vec_result['output']
    finally:
        handle_vec.remove()
    
    # 6.5. Generate text with baselines
    handle_random = register_steering_hook(
        model, layer, random_patch_vector, alpha=1.0, target_pos=diff_pos
    )
    try:
        random_result = steer_generate_no_progress(
            datapoint.prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 1.0)
        )
        random_patch_text = random_result['output']
    finally:
        handle_random.remove()
    
    handle_pca = register_steering_hook(
        model, layer, pca_patch_vector, alpha=1.0, target_pos=diff_pos
    )
    try:
        pca_result = steer_generate_no_progress(
            datapoint.prompt,
            model,
            tokenizer,
            device,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 1.0)
        )
        pca_patch_text = pca_result['output']
    finally:
        handle_pca.remove()
    
    return {
        'idx': idx,
        'patched_loss': patched_loss,
        'patched_text': patched_text,
        'vec_loss': vec_loss,
        'vec_text': vec_text,
        'random_patch_loss': random_patch_loss,
        'random_patch_text': random_patch_text,
        'pca_patch_loss': pca_patch_loss,
        'pca_patch_text': pca_patch_text
    }


def run_intervention_experiments(
    model,
    tokenizer,
    probe_path,
    vectors_path,
    prompt,
    src_response,
    dst_response,
    layer,
    prompt_ori,
    prompt_tgt,
    device=None,
    idx_range=None,
    max_new_tokens=30,
    output_csv=None,
    **kwargs
):
    """
    Batch over steering vectors, running interventions and aggregating results.
    """
    device = get_device(model=model, device=device)
    model = model.to(device)
    
    # Load probe
    print(f"  probe: {probe_path}")
    probe_dict = torch.load(probe_path, map_location=device)
    
    # Load vectors
    print(f"  vectors: {vectors_path}")
    vectors_tensor = torch.load(vectors_path, map_location=device)
    if isinstance(vectors_tensor, torch.Tensor):
        print(f"Vectors tensor  : {vectors_tensor.shape}")
    else:
        raise ValueError(f"        : {type(vectors_tensor)}")
    
    # Find differing token position between prompts
    diff_pos = find_diff_token_position(tokenizer, prompt_ori, prompt_tgt, device)
    if diff_pos is None:
        raise ValueError("  prompt            token  ")
    print(f"   token  : {diff_pos}")
    
    # Build training datapoint for steering evaluation
    datapoint = steering_opt.TrainingDatapoint(
        prompt=prompt,
        src_completions=[src_response],
        dst_completions=[dst_response],
        token=diff_pos,
    )
    
    # Align available probe points and steering vectors
    Y = probe_dict['Y']
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    n_samples = Y.shape[0]
    n_vectors = vectors_tensor.shape[0]
    
    max_idx = min(n_samples, n_vectors)
    if idx_range is None:
        # Default to full available range
        idx_range = range(max_idx)
        print(f"    idx range: 0 to {max_idx-1} (total {max_idx})")
    else:
        idx_range = range(idx_range[0], min(idx_range[1], max_idx))
        print(f"    idx range: {idx_range.start} to {idx_range.stop-1} (total {len(idx_range)})")
    
    # Iterate experiments
    results = []
    for idx in tqdm(idx_range, desc="Running interventions"):
        try:
            result = run_intervention_experiment(
                model=model,
                tokenizer=tokenizer,
                probe_dict=probe_dict,
                vectors_tensor=vectors_tensor,
                datapoint=datapoint,
                layer=layer,
                diff_pos=diff_pos,
                idx=idx,
                device=device,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            results.append(result)
        except Exception as e:
            print(f"\n  Error at idx={idx}: {e}")
            results.append({
                'idx': idx,
                'patched_loss': None,
                'patched_text': f"ERROR: {str(e)}",
                'vec_loss': None,
                'vec_text': f"ERROR: {str(e)}",
                'random_patch_loss': None,
                'random_patch_text': f"ERROR: {str(e)}",
                'pca_patch_loss': None,
                'pca_patch_text': f"ERROR: {str(e)}"
            })
    
    # Collect into DataFrame
    results_df = pd.DataFrame(results)
    
    # Optionally persist CSV
    if output_csv:
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
        results_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\nSaved results to: {output_csv}")
    
    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run steering vector intervention experiments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model_name', type=str, required=True,
                       help='HuggingFace model name or local path')
    parser.add_argument('--probe_path', type=str, required=True,
                       help='Path to probe .pt file')
    parser.add_argument('--vectors_path', type=str, required=True,
                       help='Path to steering vectors .pt file')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Prompt on which to apply steering')
    parser.add_argument('--src_response', type=str, required=True,
                       help='Reference source response string')
    parser.add_argument('--dst_response', type=str, required=True,
                       help='Target response string')
    parser.add_argument('--prompt_ori', type=str, required=True,
                       help='Original prompt for diff position detection')
    parser.add_argument('--prompt_tgt', type=str, required=True,
                       help='Target prompt for diff position detection')
    parser.add_argument('--layer', type=int, required=True,
                       help='Layer index to hook')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Where to save results CSV')
    parser.add_argument('--device', type=str, default=None,
                       help='Force device (cpu/cuda); None auto-selects')
    parser.add_argument('--idx_start', type=int, default=None,
                       help='Start index (inclusive); default 0')
    parser.add_argument('--idx_end', type=int, default=None,
                       help='End index (exclusive); default = all')
    parser.add_argument('--max_new_tokens', type=int, default=30,
                       help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=1.0,
                       help='Top-p nucleus sampling threshold')
    parser.add_argument('--coldness', type=float, default=0.7,
                       help='Loss coldness factor for calculate_loss_for_vector')
    parser.add_argument('--only_hook_prompt', action='store_true', default=True,
                       help='If set, apply steering only on prompt tokens')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model and tokenizer
    from utils import load_model
    model, tokenizer, device = load_model(args.model_name, device=args.device)
    
    # Build index range if requested
    idx_range = None
    if args.idx_start is not None or args.idx_end is not None:
        # Resolve start/end bounds
        start = args.idx_start if args.idx_start is not None else 0
        if args.idx_end is not None:
            idx_range = (start, args.idx_end)
        else:
            # If end is missing, derive from probe size
            probe_dict = torch.load(args.probe_path, map_location='cpu')
            Y = probe_dict['Y']
            if isinstance(Y, torch.Tensor):
                Y = Y.detach().cpu().numpy()
            n_samples = Y.shape[0]
            idx_range = (start, n_samples)
    # If idx_range is None, run_intervention_experiments uses full range
    
    # Run experiments
    results_df = run_intervention_experiments(
        model=model,
        tokenizer=tokenizer,
        probe_path=args.probe_path,
        vectors_path=args.vectors_path,
        prompt=args.prompt,
        src_response=args.src_response,
        dst_response=args.dst_response,
        layer=args.layer,
        prompt_ori=args.prompt_ori,
        prompt_tgt=args.prompt_tgt,
        device=device,
        idx_range=idx_range,
        max_new_tokens=args.max_new_tokens,
        output_csv=args.output_csv,
        temperature=args.temperature,
        top_p=args.top_p,
        coldness=args.coldness,
        only_hook_prompt=args.only_hook_prompt
    )
    
    # Summary statistics
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total rows: {len(results_df)}")
    
    # Patched loss
    if 'patched_loss' in results_df.columns:
        valid_patched = results_df['patched_loss'].notna()
        if valid_patched.sum() > 0:
            print(f"Patched loss - mean: {results_df.loc[valid_patched, 'patched_loss'].mean():.6f}")
            print(f"Patched loss - std:  {results_df.loc[valid_patched, 'patched_loss'].std():.6f}")
    
    # Vec loss
    if 'vec_loss' in results_df.columns:
        valid_vec = results_df['vec_loss'].notna()
        if valid_vec.sum() > 0:
            print(f"Vec loss - mean: {results_df.loc[valid_vec, 'vec_loss'].mean():.6f}")
            print(f"Vec loss - std:  {results_df.loc[valid_vec, 'vec_loss'].std():.6f}")
    
    # Random patch loss
    if 'random_patch_loss' in results_df.columns:
        valid_random = results_df['random_patch_loss'].notna()
        if valid_random.sum() > 0:
            print(f"Random patch loss - mean: {results_df.loc[valid_random, 'random_patch_loss'].mean():.6f}")
            print(f"Random patch loss - std:  {results_df.loc[valid_random, 'random_patch_loss'].std():.6f}")
    
    # PCA patch loss
    if 'pca_patch_loss' in results_df.columns:
        valid_pca = results_df['pca_patch_loss'].notna()
        if valid_pca.sum() > 0:
            print(f"PCA patch loss - mean: {results_df.loc[valid_pca, 'pca_patch_loss'].mean():.6f}")
            print(f"PCA patch loss - std:  {results_df.loc[valid_pca, 'pca_patch_loss'].std():.6f}")
    
    print("="*80)


if __name__ == "__main__":
    main()
