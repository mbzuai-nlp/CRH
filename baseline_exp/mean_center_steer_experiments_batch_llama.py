#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mean-center steering experiments (Llama-2-7b-chat).

Computes mean-center vectors from act_with/act_no on the training set,
then sweeps steering magnitudes on the test set. Supports multi-GPU
batch execution with incremental saves.

Optimization notes:
- Batches per-question steering vectors for high GPU utilization
- Automatic batching to reduce OOM risk
- Expected speedup: 10-50x depending on batch size and number of questions
"""

import torch
import numpy as np
import pandas as pd
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from multiprocessing import Process, Queue, Manager
import multiprocessing

# Add steering_opt library path
sys.path.append('./llm-steering-opt')
import steering_opt

# ============== Configuration ==============
MODEL_NAME = "YOUR-LLAMA-MODEL-PATH"  # e.g., "meta-llama/Llama-2-7b-chat-hf    "
LAYERS = [16, 24]  # layers to test
DATA_INDICES_FILE = "CRH_Data/data_indices/data_indices_llama7b.json"
CSV_FILE = "CRH_Data/data_pairs_llama7b_805_filtered.csv"
DIFF_VEC_BASE_PATH = "CRH_Data/diff_vecs_with_actis/llama2-7b-chat"
RESULTS_BASE_PATH = "./baseline_results/exp3_mean_center/llama"

# Experiment parameters
N_STEPS = 25  # steer25
MAX_NEW_TOKENS = 32
BATCH_SIZE = 128  # batch size (adjust for VRAM)

# GPU configuration
# Examples:
# - GPU_IDS = None uses all visible GPUs
# - GPU_IDS = [0, 1, 2, 3] uses specific GPUs
# - GPU_IDS = [0] uses a single GPU
GPU_IDS = None  # NoneallGPU [0, 1, 2, 3]

# Multiprocessing configuration
USE_MULTIPROCESSING = True  # TrueFalseGPU

# Per-GPU process guidance when USE_MULTIPROCESSING=True (single GPU examples):
# - 24GB: 2-4 processes
# - 12GB: 1-2 processes
# - 8GB: 1 process
NUM_PROCESSES_PER_GPU = 1  # single-GPU multi-process mode

# Auto-detect available GPUs
if GPU_IDS is None:
    if torch.cuda.is_available():
        GPU_IDS = list(range(torch.cuda.device_count()))
    else:
        GPU_IDS = []
        print("Warning: no CUDA detected, using CPU")
else:
    # Filter to available GPU IDs
    available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    GPU_IDS = [gpu_id for gpu_id in GPU_IDS if gpu_id in available_gpus]
    if not GPU_IDS:
        print("Warning: specified GPUs unavailable, using CPU")
        GPU_IDS = []

# Decide whether to use multiprocessing
# Use multiprocessing if USE_MULTIPROCESSING is True or multiple GPUs are available
USE_MULTI_GPU = (USE_MULTIPROCESSING or len(GPU_IDS) > 1) if (torch.cuda.is_available() and len(GPU_IDS) > 0) else False

print(f"Number of available GPUs: {len(GPU_IDS)}")
if GPU_IDS:
    print(f"Using GPUs: {GPU_IDS}")
    for gpu_id in GPU_IDS:
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
else:
    print("Using CPU mode")

# ============== Utility functions ==============

def build_prompt(question: str) -> str:
    """Build prompt template."""
    return f"User: {question}\nAssistant: "

def load_diff_vector(layer, qid, cid):
    """Load diff vector."""
    pt_file_path = f"{DIFF_VEC_BASE_PATH}/{layer}/question{qid}/{qid}-{cid}.pt"
    if not os.path.exists(pt_file_path):
        return None
    
    pt_data = torch.load(pt_file_path, map_location='cpu', weights_only=False)
    
    if 'vectors' not in pt_data or 'diff_vector' not in pt_data['vectors']:
        return None
    
    diff_vec = pt_data['vectors']['diff_vector']
    if isinstance(diff_vec, torch.Tensor):
        diff_vec = diff_vec.detach().cpu()
    else:
        diff_vec = torch.tensor(diff_vec)
    
    return diff_vec


def load_act_vectors(layer, qid, cid):
    """Load act_no/act_with vectors for mean-center computation."""
    pt_file_path = f"{DIFF_VEC_BASE_PATH}/{layer}/question{qid}/{qid}-{cid}.pt"
    if not os.path.exists(pt_file_path):
        return None, None

    pt_data = torch.load(pt_file_path, map_location="cpu", weights_only=False)
    vectors = pt_data.get("vectors", {})
    act_no = vectors.get("act_no")
    act_with = vectors.get("act_with")
    if act_no is None or act_with is None:
        return None, None
    if not isinstance(act_no, torch.Tensor):
        act_no = torch.tensor(act_no)
    if not isinstance(act_with, torch.Tensor):
        act_with = torch.tensor(act_with)
    return act_no.detach().cpu(), act_with.detach().cpu()

def compute_mean_center_vector_from_training(train_qids, layer, cid):
    """Compute mean-center vector (mean act_with - mean act_no)."""
    act_with_list = []
    act_no_list = []
    for qid in train_qids:
        act_no, act_with = load_act_vectors(layer, qid, cid)
        if act_no is None or act_with is None:
            continue
        act_with_list.append(act_with)
        act_no_list.append(act_no)
    
    if len(act_with_list) == 0 or len(act_no_list) == 0:
        return None
    
    mean_with = torch.stack(act_with_list).mean(dim=0)
    mean_no = torch.stack(act_no_list).mean(dim=0)
    return mean_with - mean_no

def compute_angle_components(diff_vec, base_vec):
    """
    Compute angle-related values between diff_vec and base_vec
    """
    diff_vec_norm = torch.norm(diff_vec).item()
    base_vec_norm = torch.norm(base_vec).item()
    
    if diff_vec_norm < 1e-10 or base_vec_norm < 1e-10:
        raise ValueError(f"One of the vectors has zero norm: diff_vec_norm={diff_vec_norm:.6e}, base_vec_norm={base_vec_norm:.6e}")
    
    # Compute cosine similarity without normalization using cos = (ab)/(|a||b|)
    dot_product = torch.dot(diff_vec.flatten(), base_vec.flatten()).item()
    cosine_similarity = dot_product / (diff_vec_norm * base_vec_norm)
    
    # Compute sin(theta) = sqrt(1 - cos^2(theta))
    sin_theta = np.sqrt(max(0, 1 - cosine_similarity**2))
    
    return diff_vec_norm, sin_theta, cosine_similarity

def make_prompt_only_steering_hook_hf(vector_, matrix=None, token=None):
    """
    Prompt-only steering hook.
    - Applies on prefill only; decode tokens are untouched.
    """
    if token is None:
        token = slice(None)
    applied = False

    def hook_fn(module, args):
        nonlocal applied
        x = args[0]
        if applied:
            return x
        applied = True

        vector = vector_.to(x) if isinstance(vector_, torch.Tensor) else vector_
        x_sliced = x[:, token].detach().clone()
        x[:, token] = x_sliced + vector

        if matrix is not None:
            affine_term = torch.zeros_like(x)
            affine_term[:, token] = torch.einsum("...n, mn -> ...m", x_sliced, matrix.to(x))
            x = x + affine_term

        return x

    return hook_fn

def steer_with_vector(model, tokenizer, question, steering_vector, layer_idx, max_new_tokens=25, device=None):
    """Single steering generation (optimized with inference_mode)"""
    if device is None:
        device = next(model.parameters()).device
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    steering_vector = steering_vector.to(device)
    
    hook_fn = make_prompt_only_steering_hook_hf(steering_vector, token=slice(None))
    hook_info = (layer_idx, hook_fn)
    
    with steering_opt.hf_hooks_contextmanager(model, [hook_info]):
        with torch.inference_mode():  # faster than torch.no_grad()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1
            )
    
    input_len = inputs['input_ids'].shape[1]
    generated_ids = outputs[0][input_len:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return output_text

def make_batch_steering_hook_hf(steering_vectors_batch, token=None):
    """
    Create a hook supporting batch-wise different steering vectors.
    Args:
        steering_vectors_batch: [batch_size, hidden_dim]
        token: token slice to steer (default: all tokens)
    """
    if token is None:
        token = slice(None)
    applied = False
    
    def hook_fn(module, args):
        nonlocal applied
        x = args[0]  # x shape: [batch_size, seq_len, hidden_dim]
        if applied:
            return x
        applied = True
        steering_vectors = steering_vectors_batch.to(x) if isinstance(steering_vectors_batch, torch.Tensor) else steering_vectors_batch
        
        # Ensure steering_vectors have correct shape[batch_size, hidden_dim]
        if steering_vectors.dim() == 1:
            # If only one vector, broadcast to batch size
            steering_vectors = steering_vectors.unsqueeze(0).expand(x.shape[0], -1)
        elif steering_vectors.shape[0] != x.shape[0]:
            raise ValueError(f"steering_vectors batch size ({steering_vectors.shape[0]}) must match input batch size ({x.shape[0]})")
        
        # Apply steering at specified token positions
        # x[:, token] shape: [batch_size, token_len, hidden_dim]
        # steering_vectors shape: [batch_size, hidden_dim]
        # Need to broadcast steering_vectors to [batch_size, token_len, hidden_dim]
        x_sliced = x[:, token].detach().clone()
        # Use unsqueeze(1) to reshape steering_vectors from[batch_size, hidden_dim][batch_size, 1, hidden_dim]
        # then broadcast to[batch_size, token_len, hidden_dim]
        steering_broadcast = steering_vectors.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x[:, token] = x_sliced + steering_broadcast
        
        return x
    
    return hook_fn

def steer_questions_batch_with_same_vector(model, tokenizer, questions, steering_vector, layer_idx, max_new_tokens=25, device=None):
    """
    True batching: process many questions with the same steering vector
    This is the real speedup: all questions share one hook and generate in batch
    """
    if len(questions) == 0:
        return []
    
    if device is None:
        device = next(model.parameters()).device
    
    # Build all prompts and batch tokenize
    prompts = [build_prompt(q) for q in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Ensure steering vector on GPU
    steering_vector = steering_vector.to(device)
    
    # Create hook (all questions share the same steering vector)
    hook_fn = make_prompt_only_steering_hook_hf(steering_vector, token=slice(None))
    hook_info = (layer_idx, hook_fn)
    
    # Batch generate
    with steering_opt.hf_hooks_contextmanager(model, [hook_info]):
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1
            )
    
    # Decode each output (only newly generated part, excluding input)
    outputs_list = []
    for i in range(len(questions)):
        # Use attention_mask to get real input length (excluding padding)
        input_len = inputs['attention_mask'][i].sum().item()
        # Decode only the newly generated part
        generated_ids = outputs[i][input_len:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        outputs_list.append(output_text)
    
    return outputs_list

def steer_questions_batch_with_different_vectors(model, tokenizer, questions, steering_vectors_list, layer_idx, max_new_tokens=25, batch_size=None, device=None):
    """
    Batch processing: each question uses its own steering vector
    Still fast: even with different steering vectors we batch them
    
    Args:
        model: 
        tokenizer: tokenizer
        questions: List of questions, length=batch_size
        steering_vectors_list: List of steering vectors length=batch_size, each with shape[hidden_dim]
        layer_idx: Layer index to apply steering
        max_new_tokens: Max generated tokens
        batch_size: If set, split large batches into smaller to avoid OOM
        device: Device, auto-detected if None
    
    Returns:
        Output text list of length batch_size
    """
    if len(questions) == 0:
        return []
    
    if len(questions) != len(steering_vectors_list):
        raise ValueError(f"questions length ({len(questions)}) must match steering_vectors_list length ({len(steering_vectors_list)})")
    
    if device is None:
        device = next(model.parameters()).device
    
    # If batch_size not set or >= number of questions, process in one batch
    if batch_size is None or batch_size >= len(questions):
        # Build all prompts and batch tokenize
        prompts = [build_prompt(q) for q in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Stack steering vectors list into tensor: [batch_size, hidden_dim]
        steering_vectors_batch = torch.stack([sv.to(device) if isinstance(sv, torch.Tensor) else torch.tensor(sv).to(device) 
                                              for sv in steering_vectors_list])
        
        # Create batch-wise hook
        hook_fn = make_batch_steering_hook_hf(steering_vectors_batch, token=slice(None))
        hook_info = (layer_idx, hook_fn)
        
        # Batch generate
        with steering_opt.hf_hooks_contextmanager(model, [hook_info]):
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1
                )
        
        # Decode each output (only newly generated part, excluding input)
        outputs_list = []
        for i in range(len(questions)):
            # Use attention_mask to get real input length (excluding padding)
            input_len = inputs['attention_mask'][i].sum().item()
            # Decode only the newly generated part
            generated_ids = outputs[i][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs_list.append(output_text)
        
        return outputs_list
    else:
        # Process in smaller batches
        outputs_list = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_steering_vectors = steering_vectors_list[i:i+batch_size]
            
            batch_outputs = steer_questions_batch_with_different_vectors(
                model, tokenizer, batch_questions, batch_steering_vectors, 
                layer_idx, max_new_tokens, batch_size=None, device=device  # Do not re-split during recursive call
            )
            outputs_list.extend(batch_outputs)
        
        return outputs_list

# ============== Sub-experiment: mean-center ==============

def run_experiment1_mean_center(model, tokenizer, layer, cid, train_qids, test_qids, df, verbose=True, device=None):
    """Run mean-center steering sweeps over a test set."""
    if device is None:
        device = next(model.parameters()).device
    diff_root = os.path.dirname(DIFF_VEC_BASE_PATH)
    model_name = os.path.basename(DIFF_VEC_BASE_PATH)
    mean_center_save_root = os.path.join(diff_root, "mean_center_vectors", model_name, str(layer))
    os.makedirs(mean_center_save_root, exist_ok=True)
    results = {
        'cid': cid,
        'layer': layer,
        'experiment_type': 'mean_center',
        'mean_center_vec_norm': None,
        'experiments': []
    }
    
    # Compute mean-center vector
    if verbose:
        print(f"  Computing mean-center vector: layer={layer}, cid={cid}, train_size={len(train_qids)}")
    mean_center_vec = compute_mean_center_vector_from_training(train_qids, layer, cid)
    if mean_center_vec is None:
        if verbose:
            print(f"  Warning: mean-center vector unavailable for cid={cid}")
        return None
    
    mean_center_vec_norm = torch.norm(mean_center_vec).item()
    if verbose:
        print(f"  mean-center vector norm={mean_center_vec_norm:.4f}")
    results['mean_center_vec_norm'] = float(mean_center_vec_norm)
    
    mean_center_normalized = mean_center_vec / torch.norm(mean_center_vec)

    mean_center_save_path = os.path.join(mean_center_save_root, f"{cid}.pt")
    torch.save(
        {
            "mean_center_vector": mean_center_vec.cpu(),
            "mean_center_vec_norm": mean_center_vec_norm,
        },
        mean_center_save_path,
    )
    
    # Preload diff vectors and questions for all test samples
    if verbose:
        print(f"  Preloading test data (test_size={len(test_qids)})")
    test_data = []
    for qid in test_qids:
        diff_vec = load_diff_vector(layer, qid, cid)
        if diff_vec is None:
            continue
        
        qid_rows = df[df['qid'] == qid]
        if len(qid_rows) == 0:
            continue
        
        question_text = qid_rows.iloc[0]['question']
        try:
            diff_vec_norm, sin_theta, cos_theta = compute_angle_components(diff_vec, mean_center_vec)
        except (ValueError, Exception) as e:
            if verbose:
                print(f"  Warning: failed angle metrics (qid={qid}, cid={cid}): {e}")
            continue
        
        test_data.append({
            'qid': int(qid),
            'question': question_text,
            'diff_vec': diff_vec,
            'diff_vec_norm': diff_vec_norm,
            'sin_theta': sin_theta,
            'cos_theta': cos_theta
        })
    
    if len(test_data) == 0:
        if verbose:
            print(f"  Warning: no valid test data, skip cid={cid}")
        return None
    
    if verbose:
        print(f"  Successfully loaded {len(test_data)} test questions")
    
    # Maximum steer length = 5 * ||mean_center||
    max_steer_length = 5 * mean_center_vec_norm
    
    if verbose:
        print(f"  Maximum steer length={max_steer_length:.4f}, N_STEPS={N_STEPS}")
    
    # All questions share the same steer_lengths
    steer_lengths = np.linspace(0, max_steer_length, N_STEPS + 1)[1:]
    question_steer_lengths = [steer_lengths for _ in test_data]
    
    # Precompute steering vectors for each step
    batch_configs = {}  # {steer_len_idx: {'steering_vector': tensor, 'question_indices': [...]}}
    if verbose:
        total_vectors = N_STEPS
        print(f"  Precomputing steering vectors: {N_STEPS} steps")
    for steer_len_idx in range(N_STEPS):
        steer_len = steer_lengths[steer_len_idx]
        steering_vector = mean_center_normalized * steer_len
        batch_configs[steer_len_idx] = {
            'steering_vector': steering_vector,
            'question_indices': list(range(len(test_data))),
            'steer_len_idx': steer_len_idx
        }
    
    if verbose:
        print(f"  Steering vectors precomputed, total {len(batch_configs)} configs")
    
    # Batch process all configurations
    # Initialize result structure for each question
    question_results = {}
    for test_idx, test_item in enumerate(test_data):
        question_results[test_idx] = {
            'qid': test_item['qid'],
            'question': test_item['question'],
            'diff_vec_norm': float(test_item['diff_vec_norm']),
            'sin_theta': float(test_item['sin_theta']),
            'cos_theta': float(test_item['cos_theta']),
            'mean_center_results': []
        }
    
    # Batch process each configuration
    total_configs = len(batch_configs)
    if verbose:
        print(f"  Start batch processing {total_configs} configs (steer_length)")
    
    config_pbar = None
    if verbose:
        config_pbar = tqdm(
            total=total_configs,
            desc=f"  Batch config (layer={layer}, cid={cid})",
            unit="configs",
            leave=False,
            ncols=100
        )
    
    for config_idx, (steer_len_idx, config_info) in enumerate(batch_configs.items()):
        steering_vector = config_info['steering_vector']
        question_indices = config_info['question_indices']
        
        if verbose and config_pbar:
            config_pbar.set_description(f"  Processing {config_idx+1}/{total_configs} (step={steer_len_idx+1})")
        
        # Collect questions and steering vectors
        questions = [test_data[idx]['question'] for idx in question_indices]
        
        try:
            outputs = steer_questions_batch_with_same_vector(
                model, tokenizer, questions, steering_vector, layer, MAX_NEW_TOKENS, device=device
            )
            
            # Assign results back to each question
            for q_idx, output in zip(question_indices, outputs):
                steer_len = question_steer_lengths[q_idx][steer_len_idx]
                question_results[q_idx]['mean_center_results'].append({
                    'step': steer_len_idx + 1,
                    'steer_length': float(steer_len),
                    'output': output
                })
        except Exception as e:
            # If batch fails, fall back to single processing
            if verbose:
                print(f"  Warning: batch processing failed: {e}")
            for q_idx in question_indices:
                question_text = test_data[q_idx]['question']
                steer_len = question_steer_lengths[q_idx][steer_len_idx]
                try:
                    output = steer_with_vector(model, tokenizer, question_text, steering_vector, layer, MAX_NEW_TOKENS, device=device)
                    question_results[q_idx]['mean_center_results'].append({
                        'step': steer_len_idx + 1,
                        'steer_length': float(steer_len),
                        'output': output
                    })
                except Exception as e2:
                    question_results[q_idx]['mean_center_results'].append({
                        'step': steer_len_idx + 1,
                        'steer_length': float(steer_len),
                        'output': f"[ERROR: {str(e2)}]"
                    })
        
        # Update progress bar
        if config_pbar:
            config_pbar.update(1)
    
    # Close progress bar
    if config_pbar:
        config_pbar.close()
    
    if verbose:
        print(f"  Batch processing finished: layer={layer}, cid={cid}, test_questions={len(test_data)}")
    
    # Organize results: sort by step
    for test_idx, result in question_results.items():
        result['mean_center_results'].sort(key=lambda x: x['step'])
        results['experiments'].append(result)
    
    return results

# ============== Multi-GPU worker function ==============

def worker_process(gpu_id, task_queue, result_queue, model_name, data_indices_file, csv_file, 
                   diff_vec_base_path, results_base_path, n_steps, max_new_tokens, batch_size):
    """Worker process: handle tasks on assigned GPU"""
    # Set current process GPU
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    print(f"[GPU {gpu_id}] Process started")
    
    # Load data
    with open(data_indices_file, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)
    df = pd.read_csv(csv_file)
    
    # Load model
    print(f"[GPU {gpu_id}] Load model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Try torch.compile acceleration
    try:
        if hasattr(torch, 'compile') and device.startswith('cuda'):
            model = torch.compile(model, mode='reduce-overhead')
    except Exception as e:
        print(f"[GPU {gpu_id}] torch.compile unavailable or failed: {e}")
    
    print(f"[GPU {gpu_id}] Model loaded")
    
    # Process tasks
    task_count = 0
    while True:
        task = task_queue.get()
        if task is None:  # end signal
            break
        
        layer, cid = task
        cid_str = str(cid)
        task_count += 1
        
        print(f"[GPU {gpu_id}] startProcess tasks {task_count}: layer={layer}, cid={cid}")
        
        if cid_str not in data_indices:
            print(f"[GPU {gpu_id}] Skip: cid={cid} not in data index")
            result_queue.put((layer, cid, None))
            continue
        
        # Get train and test sets
        if 'train' in data_indices[cid_str] and isinstance(data_indices[cid_str]['train'], dict):
            train_qids = data_indices[cid_str]['train'].get(str(layer), [])
        else:
            train_qids = data_indices[cid_str].get('train', [])
        
        test_qids = data_indices[cid_str].get('test', [])
        
        if not train_qids or not test_qids:
            print(f"[GPU {gpu_id}] Skip: layer={layer}, cid={cid} train or test set empty (train={len(train_qids) if train_qids else 0}, test={len(test_qids) if test_qids else 0})")
            result_queue.put((layer, cid, None))
            continue
        
        print(f"[GPU {gpu_id}] Data preparation done: layer={layer}, cid={cid}, train_qids={len(train_qids)}, test_qids={len(test_qids)}")
        
        # Run experiment
        import sys
        current_module = sys.modules[__name__]
        # Set global variables for this process
        current_module.N_STEPS = n_steps
        current_module.MAX_NEW_TOKENS = max_new_tokens
        current_module.BATCH_SIZE = batch_size
        current_module.DIFF_VEC_BASE_PATH = diff_vec_base_path
        
        try:
            print(f"[GPU {gpu_id}] startRun experiment: layer={layer}, cid={cid}")
            result = run_experiment1_mean_center(
                model, tokenizer, layer, cid, train_qids, test_qids, df, verbose=True, device=device
            )
            if result is not None:
                print(f"[GPU {gpu_id}] Task finished: layer={layer}, cid={cid}, ={len(result.get('experiments', []))}")
            else:
                print(f"[GPU {gpu_id}] Task returned None: layer={layer}, cid={cid}")
            result_queue.put((layer, cid, result))
        except Exception as e:
            print(f"[GPU {gpu_id}] Process tasks (layer={layer}, cid={cid}) error while processing: {e}")
            import traceback
            traceback.print_exc()
            result_queue.put((layer, cid, None))
        finally:
            # Restore global variables
            pass
    
    print(f"[GPU {gpu_id}] Process ended")

# ============== Main program ==============

def main():
    # Create results directory
    os.makedirs(RESULTS_BASE_PATH, exist_ok=True)
    
    # Load data
    print("Load data...")
    with open(DATA_INDICES_FILE, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)
    
    # Get all concept IDs
    all_cids = sorted([int(cid) for cid in data_indices.keys()])
    print(f"Total {len(all_cids)} concepts to process")
    
    # Compute total tasksalllayerconcept
    total_tasks = 0
    task_info = {}  # {(layer, cid): True} for tracking tasks
    for layer in LAYERS:
        exp1_output_file = f"{RESULTS_BASE_PATH}/exp3_mean_center_layer{layer}.json"
        print(f"exp1_output_file: {exp1_output_file}")
        existing_exp1_results = {}
        if os.path.exists(exp1_output_file):
            with open(exp1_output_file, 'r') as f:
                existing_results = json.load(f)
                for r in existing_results:
                    existing_exp1_results[r['cid']] = r
        
        for cid in all_cids:
            cid_str = str(cid)
            if cid_str not in data_indices:
                continue
            # Skip existing results
            if cid in existing_exp1_results:
                continue
            
            # Check if train or test set is empty
            if 'train' in data_indices[cid_str] and isinstance(data_indices[cid_str]['train'], dict):
                train_qids = data_indices[cid_str]['train'].get(str(layer), [])
            else:
                train_qids = data_indices[cid_str].get('train', [])
            test_qids = data_indices[cid_str].get('test', [])
            
            if train_qids and test_qids:
                total_tasks += 1
                task_info[(layer, cid)] = True
    
    print(f"Total tasks: {total_tasks} ( {len(LAYERS)} layer  {len(all_cids)} concept)")
    
    # Create global progress bar
    global_pbar = tqdm(
        total=total_tasks,
        desc="Overall progress",
        unit="tasks",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=120,
        miniters=1,
        mininterval=0.5
    )
    
    # process
    if USE_MULTI_GPU:
        if len(GPU_IDS) > 1:
            print(f"\nUsing multi-GPU multiprocessing: {len(GPU_IDS)} GPUs, {NUM_PROCESSES_PER_GPU} proc/GPU")
        else:
            print(f"\nUsing single-GPU multiprocessing: GPU {GPU_IDS[0]}, {NUM_PROCESSES_PER_GPU} proc")
        
        # tasks
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # layerconcepttasks
        tasks = []
        for layer in LAYERS:
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp3_mean_center_layer{layer}.json"
            existing_exp1_results = {}
            if os.path.exists(exp1_output_file):
                with open(exp1_output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    for r in existing_results:
                        existing_exp1_results[r['cid']] = r
            
            for cid in all_cids:
                cid_str = str(cid)
                if cid_str not in data_indices:
                    continue
                # Skip existing results
                if cid in existing_exp1_results:
                    continue
                tasks.append((layer, cid))
        
        print(f"Total tasks enqueued: {len(tasks)}")
        
        # Start worker processes
        # Create per-GPUNUM_PROCESSES_PER_GPU
        processes = []
        total_processes = len(GPU_IDS) * NUM_PROCESSES_PER_GPU
        
        # Print VRAM guidance
        if len(GPU_IDS) == 1:
            gpu_id = GPU_IDS[0]
            gpu_memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"  GPU {gpu_id}: {gpu_memory_gb:.1f} GB total")
            print(f"  Per-process budget (approx): {gpu_memory_gb / NUM_PROCESSES_PER_GPU:.1f} GB")
        else:
            print(f"  Per-GPU processes: {NUM_PROCESSES_PER_GPU}")
            for gpu_id in GPU_IDS:
                gpu_memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"  GPU {gpu_id}: {gpu_memory_gb:.1f} GB total, ~{gpu_memory_gb / NUM_PROCESSES_PER_GPU:.1f} GB per process")
        
        # Create per-GPUNUM_PROCESSES_PER_GPU
        for gpu_id in GPU_IDS:
            for proc_idx in range(NUM_PROCESSES_PER_GPU):
                p = Process(target=worker_process, args=(
                    gpu_id, task_queue, result_queue, MODEL_NAME, DATA_INDICES_FILE,
                    CSV_FILE, DIFF_VEC_BASE_PATH, RESULTS_BASE_PATH,
                    N_STEPS, MAX_NEW_TOKENS, BATCH_SIZE
                ))
                p.start()
                processes.append(p)
        
        print(f"Spawned {len(processes)} workers across {len(GPU_IDS)} GPUs ({NUM_PROCESSES_PER_GPU} per GPU)")
        
        # tasks
        for task in tasks:
            task_queue.put(task)
        
        # end signal
        for _ in processes:
            task_queue.put(None)
        
        # Collect resultsIncremental save
        layer_results = {layer: {} for layer in LAYERS}
        completed = 0
        
        # TODO: tune SAVE_INTERVAL to reduce I/O overhead
        SAVE_INTERVAL = 1  # Save every N completed tasks
        
        # Preload existing results for incremental save
        existing_results_all_layers = {}
        for layer in LAYERS:
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp3_mean_center_layer{layer}.json"
            existing_exp1_results = {}
            if os.path.exists(exp1_output_file):
                with open(exp1_output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    for r in existing_results:
                        existing_exp1_results[r['cid']] = r
            existing_results_all_layers[layer] = existing_exp1_results
        
        def save_results_incremental():
            """Persist incremental results for all layers."""
            for layer in LAYERS:
                exp1_output_file = f"{RESULTS_BASE_PATH}/exp3_mean_center_layer{layer}.json"
                existing_exp1_results = existing_results_all_layers[layer]
                
                # Merge new and existing results
                new_results = list(layer_results[layer].values())
                all_results = list(existing_exp1_results.values()) + new_results
                
                # Deduplicate
                seen_cids = set()
                unique_results = []
                for r in reversed(all_results):
                    if r['cid'] not in seen_cids:
                        seen_cids.add(r['cid'])
                        unique_results.append(r)
                unique_results.reverse()
                
                # Save
                with open(exp1_output_file, 'w', encoding='utf-8') as f:
                    json.dump(unique_results, f, ensure_ascii=False, indent=2)
                
                # Update existing results cache
                existing_results_all_layers[layer] = {r['cid']: r for r in unique_results}
        
        while completed < len(tasks):
            layer, cid, result = result_queue.get()
            if result is not None:
                if cid not in layer_results[layer]:
                    layer_results[layer][cid] = result
            completed += 1
            
            # Incremental save every SAVE_INTERVAL tasks
            if completed % SAVE_INTERVAL == 0:
                save_results_incremental()
                print(f"  [Incremental save] {completed}/{len(tasks)} tasks")
            
            # Progress update
            mode_str = f"({len(GPU_IDS)} GPU, {NUM_PROCESSES_PER_GPU} proc/GPU, {len(processes)} total)"
            global_pbar.set_description(f"Layer {layer} | CID {cid} {mode_str}")
            global_pbar.update(1)
        
        # All processes ended
        for p in processes:
            p.join()
        
        # Final save
        print("\nFinal save...")
        save_results_incremental()
        
        # Summary
        for layer in LAYERS:
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp3_mean_center_layer{layer}.json"
            final_count = len(existing_results_all_layers[layer])
            print(f"Layer {layer}: saved {final_count} concepts")
    
    else:
        # Single-process CPU/GPU execution
        device = f'cuda:{GPU_IDS[0]}' if GPU_IDS else 'cpu'
        print(f"\nDevice: {device}")
        
        # Load CSV data
        print("Load CSV data...")
        df = pd.read_csv(CSV_FILE)
        
        # Load model
        print(f"Load model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model = model.to(device)
        model.eval()
        
        # Try torch.compile acceleration (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and device.startswith('cuda'):
                print("Try torch.compile acceleration...")
                model = torch.compile(model, mode='reduce-overhead')
                print("torch.compile applied successfully")
        except Exception as e:
            print(f"torch.compile unavailable or failed: {e}")
        
        print(f"Model loaded to device: {device}")
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            gpu_id = int(device.split(':')[1])
            print(f"GPU memory usage: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        
        # Per-layer processing
        for layer in LAYERS:
            print(f"\n{'='*80}")
            print(f"Processing layer {layer}")
            print(f"{'='*80}")
            
            # Sub-experiment: mean-center
            print(f"\nSub-experiment: mean-center (layer {layer})")
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp3_mean_center_layer{layer}.json"
            
            # Load existing results
            existing_exp1_results = {}
            if os.path.exists(exp1_output_file):
                with open(exp1_output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    for r in existing_results:
                        existing_exp1_results[r['cid']] = r
                print(f"Loaded {len(existing_exp1_results)} existing results")
            
            exp1_results = []
            
            for cid in all_cids:
                cid_str = str(cid)
                if cid_str not in data_indices:
                    continue
                
                # Reuse existing result
                if cid in existing_exp1_results:
                    exp1_results.append(existing_exp1_results[cid])
                    continue
                
                # Resolve train/test splits
                if 'train' in data_indices[cid_str] and isinstance(data_indices[cid_str]['train'], dict):
                    # Layer-specific train split
                    train_qids = data_indices[cid_str]['train'].get(str(layer), [])
                else:
                    # Global train split
                    train_qids = data_indices[cid_str].get('train', [])
                
                test_qids = data_indices[cid_str].get('test', [])
                
                # Check if train or test set is empty
                if not train_qids:
                    continue
                if not test_qids:
                    continue
                
                # Update global progress description
                global_pbar.set_description(f"Layer {layer} | CID {cid}")
                
                result = run_experiment1_mean_center(
                    model, tokenizer, layer, cid, train_qids, test_qids, df, verbose=False, device=device
                )
                
                if result is not None:
                    exp1_results.append(result)
                    # Save incremental results
                    with open(exp1_output_file, 'w', encoding='utf-8') as f:
                        json.dump(exp1_results + list(existing_exp1_results.values()), f, ensure_ascii=False, indent=2)
                
                # progress
                global_pbar.update(1)
            
            # Finalize layer results
            print(f"\nSaved results to: {exp1_output_file}")
            # Merge new and existing results
            all_exp1_results = list(existing_exp1_results.values()) + exp1_results
            # Deduplicate
            seen_cids = set()
            unique_results = []
            for r in reversed(all_exp1_results):
                if r['cid'] not in seen_cids:
                    seen_cids.add(r['cid'])
                    unique_results.append(r)
            unique_results.reverse()
            
            with open(exp1_output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_results, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(unique_results)} concepts")
    
    # Close progress bar
    global_pbar.close()
    
    print(f"\n{'='*80}")
    print("All experiments completed")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Set multiprocessing start method (spawn for Windows, fork for Linux/macOS)
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set
    
    main()
