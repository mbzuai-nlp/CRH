#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt-only steering sweep for Llama-2-7b-chat on the location dataset.

Sweeps steering magnitudes, optional orthogonal penalties, and supports
multi-GPU batch execution; saves incremental results per layer/concept.
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

# Add custom steering utility to import path
sys.path.append('./llm-steering-opt')
import steering_opt

# ==== Experiment setup =====================================================
MODEL_NAME = "YOUR_MODEL_PATH_OR_NAME"  # e.g., "meta-llama/Llama-2-7b-chat-hf"
LAYERS = [16, 24]  # Target transformer blocks for steering hooks
DATA_INDICES_FILE = "CRH_Data/data_indices/data_indices_llama7b.json"
CSV_FILE = "CRH_Data/data_pairs_llama7b_805_filtered.csv"
DIFF_VEC_BASE_PATH = "CRH_Data/diff_vecs/llama2-7b-chat"
RESULTS_BASE_PATH = "./results/location/all_prompt_llama"

# Sweep hyperparameters
N_STEPS = 25  # Number of steering magnitudes to test per question
MAX_NEW_TOKENS = 32
BATCH_SIZE = 256  # Generation batch size when sharing steering vectors

# Scaling factor for the orthogonal component; 0 = keep full component
PENALTY_VALUES = np.linspace(0,1, 20).tolist()

# GPU configuration (None = use all visible GPUs)
GPU_IDS = None

# Whether to fan out tasks across multiple processes/GPUs
USE_MULTIPROCESSING = True

# Number of worker processes to launch per GPU when multiprocessing is enabled
NUM_PROCESSES_PER_GPU = 1

# Resolve GPU list
if GPU_IDS is None:
    if torch.cuda.is_available():
        GPU_IDS = list(range(torch.cuda.device_count()))
    else:
        GPU_IDS = []
        print("Warning: CUDA not available, falling back to CPU")
else:
    # Filter to available GPU IDs
    available_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    GPU_IDS = [gpu_id for gpu_id in GPU_IDS if gpu_id in available_gpus]
    if not GPU_IDS:
        print("Warning: no valid GPU IDs found, falling back to CPU")
        GPU_IDS = []

# Multi-GPU is enabled when multiprocessing is on or multiple GPUs are visible.
USE_MULTI_GPU = (USE_MULTIPROCESSING or len(GPU_IDS) > 1) if (torch.cuda.is_available() and len(GPU_IDS) > 0) else False

print(f"GPU count: {len(GPU_IDS)}")
if GPU_IDS:
    print(f"GPU IDs: {GPU_IDS}")
    for gpu_id in GPU_IDS:
        print(f"GPU {gpu_id}: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
else:
    print("CPU only")

# ==============      ==============

def build_prompt(question: str) -> str:
    """Wrap a user question into the chat-style prompt expected by the model."""
    return f"User: {question}\nAssistant: "

def make_prompt_only_steering_hook_hf(vector_, matrix=None, token=None):
    """
    Apply a steering vector only on the prompt tokens during the forward
    prefill phase (generation is unaffected). Optionally apply an affine
    matrix on the same slice.
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

def make_prompt_only_batch_steering_hook_hf(steering_vectors_batch, token=None):
    """
    Batch version of the prompt-only steering hook; each item in the batch
    receives its own steering vector during the prefill forward pass.
    """
    if token is None:
        token = slice(None)
    applied = False

    def hook_fn(module, args):
        nonlocal applied
        x = args[0]  # [batch, seq_len, hidden]
        if applied:
            return x
        applied = True

        steering_vectors = (
            steering_vectors_batch.to(x) if isinstance(steering_vectors_batch, torch.Tensor) else steering_vectors_batch
        )

        if steering_vectors.dim() == 1:
            steering_vectors = steering_vectors.unsqueeze(0).expand(x.shape[0], -1)
        elif steering_vectors.shape[0] != x.shape[0]:
            raise ValueError(
                f"steering_vectors batch size ({steering_vectors.shape[0]}) must match input batch size ({x.shape[0]})"
            )

        x_sliced = x[:, token].detach().clone()
        steering_broadcast = steering_vectors.unsqueeze(1)  # [batch, 1, hidden]
        x[:, token] = x_sliced + steering_broadcast
        return x

    return hook_fn

def load_diff_vector(layer, qid, cid):
    """Load a precomputed diff vector for the given layer/question/concept."""
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

def compute_diffmean_from_training(train_qids, layer, cid):
    """Average diff vectors from training questions to form a diff mean."""
    diff_vecs = []
    for qid in train_qids:
        diff_vec = load_diff_vector(layer, qid, cid)
        if diff_vec is not None:
            diff_vecs.append(diff_vec)
    
    if len(diff_vecs) == 0:
        return None
    
    # Compute diff mean
    diffmean = torch.stack(diff_vecs).mean(dim=0)
    return diffmean

def compute_angle_components(diff_vec, diffmean_vec):
    """
    Return norms and angular components between diff_vec and diffmean_vec.
    """
    diff_vec_norm = torch.norm(diff_vec).item()
    diffmean_vec_norm = torch.norm(diffmean_vec).item()
    
    if diff_vec_norm < 1e-10 or diffmean_vec_norm < 1e-10:
        raise ValueError(f"One of the vectors has zero norm: diff_vec_norm={diff_vec_norm:.6e}, diffmean_vec_norm={diffmean_vec_norm:.6e}")
    
    # cosine similarity: (a dot b) / (|a| * |b|)
    dot_product = torch.dot(diff_vec.flatten(), diffmean_vec.flatten()).item()
    cosine_similarity = dot_product / (diff_vec_norm * diffmean_vec_norm)
    
    # sin(theta) = sqrt(1 - cos^2(theta))
    sin_theta = np.sqrt(max(0, 1 - cosine_similarity**2))
    
    return diff_vec_norm, sin_theta, cosine_similarity

def steer_with_vector(model, tokenizer, question, steering_vector, layer_idx, max_new_tokens=25, device=None):
    """Generate one answer for a question while injecting a steering vector."""
    if device is None:
        device = next(model.parameters()).device
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    steering_vector = steering_vector.to(device)
    
    hook_fn = make_prompt_only_steering_hook_hf(steering_vector, token=slice(None))
    hook_info = (layer_idx, hook_fn)
    
    with steering_opt.hf_hooks_contextmanager(model, [hook_info]):
        with torch.inference_mode():  # same effect as torch.no_grad()
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
    Batch hook that applies a per-sample steering vector on the selected tokens.
    """
    if token is None:
        token = slice(None)
    
    def hook_fn(module, args):
        x = args[0]  # x shape: [batch_size, seq_len, hidden_dim]
        steering_vectors = steering_vectors_batch.to(x) if isinstance(steering_vectors_batch, torch.Tensor) else steering_vectors_batch
        
        # Ensure steering_vectors is [batch_size, hidden_dim]
        if steering_vectors.dim() == 1:
            # Expand single vector to batch size
            steering_vectors = steering_vectors.unsqueeze(0).expand(x.shape[0], -1)
        elif steering_vectors.shape[0] != x.shape[0]:
            raise ValueError(f"steering_vectors batch size ({steering_vectors.shape[0]}) must match input batch size ({x.shape[0]})")
        
        # Apply steering on selected token slice
        # x[:, token] shape: [batch_size, token_len, hidden_dim]
        x_sliced = x[:, token].detach().clone()
        # Broadcast to [batch_size, token_len, hidden_dim]
        steering_broadcast = steering_vectors.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x[:, token] = x_sliced + steering_broadcast
        
        return x
    
    return hook_fn

def steer_questions_batch_with_same_vector(model, tokenizer, questions, steering_vector, layer_idx, max_new_tokens=25, device=None):
    """
    Generate answers for a batch of questions using the same steering vector.
    """
    if len(questions) == 0:
        return []
    
    if device is None:
        device = next(model.parameters()).device
    
    # Build prompts and tokenize
    prompts = [build_prompt(q) for q in questions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Move steering vector to device
    steering_vector = steering_vector.to(device)
    
    # Register prompt-only steering hook
    hook_fn = make_prompt_only_steering_hook_hf(steering_vector, token=slice(None))
    hook_info = (layer_idx, hook_fn)
    
    with steering_opt.hf_hooks_contextmanager(model, [hook_info]):
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1
            )
    
    # Decode outputs
    outputs_list = []
    for i in range(len(questions)):
        # Use attention_mask to exclude padding
        input_len = inputs['attention_mask'][i].sum().item()
        generated_ids = outputs[i][input_len:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        outputs_list.append(output_text)
    
    return outputs_list

def steer_questions_batch_with_different_vectors(model, tokenizer, questions, steering_vectors_list, layer_idx, max_new_tokens=25, batch_size=None, device=None):
    """
    Generate answers where each question uses its own steering vector.
    """
    if len(questions) == 0:
        return []
    
    if len(questions) != len(steering_vectors_list):
        raise ValueError(f"questions length ({len(questions)}) must match steering_vectors_list length ({len(steering_vectors_list)})")
    
    if device is None:
        device = next(model.parameters()).device
    
    # If batch_size is None or large, process in one batch
    if batch_size is None or batch_size >= len(questions):
        # Build prompts and tokenize
        prompts = [build_prompt(q) for q in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Stack steering vectors to [batch_size, hidden_dim]
        steering_vectors_batch = torch.stack([sv.to(device) if isinstance(sv, torch.Tensor) else torch.tensor(sv).to(device) 
                                              for sv in steering_vectors_list])
        
        # Register batch-wise hook
        hook_fn = make_prompt_only_batch_steering_hook_hf(steering_vectors_batch, token=slice(None))
        hook_info = (layer_idx, hook_fn)
        
        with steering_opt.hf_hooks_contextmanager(model, [hook_info]):
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.1
                )
        
        # Decode outputs
        outputs_list = []
        for i in range(len(questions)):
            # Use attention_mask to exclude padding
            input_len = inputs['attention_mask'][i].sum().item()
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
                layer_idx, max_new_tokens, batch_size=None, device=device  # recurse with full batch
            )
            outputs_list.extend(batch_outputs)
        
        return outputs_list

# ============== Experiment 1: penalty sweep ==============

def run_experiment1_penalty(model, tokenizer, layer, cid, train_qids, test_qids, df, verbose=True, device=None):
    """Run the penalty sweep: vary steer length/penalty and collect outputs."""
    if device is None:
        device = next(model.parameters()).device
    results = {
        'cid': cid,
        'layer': layer,
        'experiment_type': 'penalty',
        'experiments': []
    }
    
    # Compute diffmean from training set
    if verbose:
        print(f"Computing diffmean (layer={layer}, cid={cid}) from {len(train_qids)} train samples...")
    diffmean_vec = compute_diffmean_from_training(train_qids, layer, cid)
    if diffmean_vec is None:
        if verbose:
            print(f"Warning: no diffmean found for cid={cid}")
        return None
    
    if verbose:
        print(f"Diffmean norm: {torch.norm(diffmean_vec).item():.4f}")
    
    diffmean_normalized = diffmean_vec / torch.norm(diffmean_vec)
    
    # Load test diff vectors
    if verbose:
        print(f"Loading test samples: {len(test_qids)}")
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
            diff_vec_norm, sin_theta, cos_theta = compute_angle_components(diff_vec, diffmean_vec)
        except (ValueError, Exception) as e:
            if verbose:
                print(f"Warning: failed for qid={qid}, cid={cid}: {e}")
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
            print(f"Warning: no valid test data for cid={cid}")
        return None
    
    if verbose:
        print(f"Collected {len(test_data)} test items")
    
    # Max steer length = 5 * ||diffmean||
    diffmean_vec_norm = torch.norm(diffmean_vec).item()
    max_steer_length = 5 * diffmean_vec_norm
    
    if verbose:
        print(f"Max steer length={max_steer_length:.4f}, N_STEPS={N_STEPS}, penalties={len(PENALTY_VALUES)}")
    
    # Build steer length schedule
    steer_lengths = np.linspace(max_steer_length/2, max_steer_length, N_STEPS + 1)[1:]
    question_steer_lengths = [steer_lengths for _ in test_data]
    
    # Precompute parallel/orthogonal components per test item
    if verbose:
        print("  Precomputing parallel/orthogonal components...")
    for test_item in test_data:
        diff_vec = test_item['diff_vec']
        diff_vec_flat = diff_vec.flatten()
        diffmean_flat = diffmean_normalized.flatten()
        cos_theta = test_item['cos_theta']
        sin_theta = test_item['sin_theta']
        
        # Project diff_vec onto diffmean direction
        projection_scalar = torch.dot(diff_vec_flat, diffmean_flat)
        parallel_component_flat = projection_scalar * diffmean_flat
        
        # Orthogonal component to diffmean
        orthogonal_component_flat = diff_vec_flat - parallel_component_flat
        
        orthogonal_norm = torch.norm(orthogonal_component_flat)
        if orthogonal_norm > 1e-10:
            orthogonal_unit_flat = orthogonal_component_flat / orthogonal_norm
        else:
            orthogonal_unit_flat = torch.zeros_like(orthogonal_component_flat)
        
        # Store components and norms
        test_item['parallel_component'] = parallel_component_flat.reshape(diffmean_normalized.shape)
        test_item['orthogonal_unit'] = orthogonal_unit_flat.reshape(diffmean_normalized.shape)
        test_item['parallel_norm'] = torch.norm(parallel_component_flat).item()
        test_item['orthogonal_norm'] = orthogonal_norm.item()
    
    # Build steering vectors per (penalty, steer_len_idx) for batch generation
    batch_configs = {}  # {(penalty_idx, steer_len_idx): {'steering_vectors': [...], 'question_indices': [...]}}
    
    # Create steering vectors using parallel/orthogonal decomposition
    if verbose:
        total_vectors = len(PENALTY_VALUES) * N_STEPS * len(test_data)
        print(f"Building steering vectors: {total_vectors} total")
    for penalty_idx, penalty in enumerate(PENALTY_VALUES):
        for steer_len_idx in range(N_STEPS):
            config_key = (penalty_idx, steer_len_idx)
            batch_configs[config_key] = {
                'steering_vectors': [],
                'question_indices': [],
                'penalty': penalty,
                'steer_len_idx': steer_len_idx
            }
            
            # Build steering vector for each test item at this steer length
            for test_idx, test_item in enumerate(test_data):
                # Scale relative to max_steer_length
                steer_len = question_steer_lengths[test_idx][steer_len_idx]
                cos_theta = test_item['cos_theta']
                sin_theta = test_item['sin_theta']
                orthogonal_unit = test_item['orthogonal_unit']
                
                # Component lengths
                v_parallel_length = steer_len * cos_theta
                v_orthogonal_length = steer_len * sin_theta * (1 - penalty)
                
                # Combine parallel and orthogonal components
                v_parallel = diffmean_normalized * v_parallel_length
                v_orthogonal = orthogonal_unit * v_orthogonal_length
                steering_vector = v_parallel + v_orthogonal
                
                batch_configs[config_key]['steering_vectors'].append(steering_vector)
                batch_configs[config_key]['question_indices'].append(test_idx)
    
    if verbose:
        print(f"Prepared {len(batch_configs)} steering configs")
    
    # Prepare result buckets per question
    question_results = {}
    for test_idx, test_item in enumerate(test_data):
        question_results[test_idx] = {
            'qid': test_item['qid'],
            'question': test_item['question'],
            'diff_vec_norm': float(test_item['diff_vec_norm']),
            'sin_theta': float(test_item['sin_theta']),
            'cos_theta': float(test_item['cos_theta']),
            'penalty_results': {penalty: {'penalty': penalty, 'outputs': []} for penalty in PENALTY_VALUES}
        }
    
    # Iterate over all (penalty, steer_len_idx) configs
    total_configs = len(batch_configs)
    if verbose:
        print(f"Running {total_configs} configs (penalty x steer_length)")
    
    config_pbar = None
    if verbose:
        config_pbar = tqdm(
            total=total_configs,
            desc=f"Running configs (layer={layer}, cid={cid})",
            unit="configs",
            leave=False,
            ncols=100
        )
    
    for config_idx, (config_key, config_info) in enumerate(batch_configs.items()):
        penalty_idx, steer_len_idx = config_key
        penalty = config_info['penalty']
        steering_vectors = config_info['steering_vectors']
        question_indices = config_info['question_indices']
        
        if verbose and config_pbar:
            config_pbar.set_description(f"{config_idx+1}/{total_configs} (penalty={penalty:.3f}, step={steer_len_idx+1})")
        
        # Build question list for this config
        questions = [test_data[idx]['question'] for idx in question_indices]
        
        # Check if all steering vectors are identical
        if len(steering_vectors) > 1:
            all_same = torch.allclose(steering_vectors[0], steering_vectors[-1], atol=1e-6)
            if all_same:
                # Verify all vectors are identical
                for sv in steering_vectors[1:-1]:
                    if not torch.allclose(steering_vectors[0], sv, atol=1e-6):
                        all_same = False
                        break
        else:
            all_same = True
        
        try:
            if all_same and len(steering_vectors) > 1:
                # Use shared-vector batch generation
                outputs = steer_questions_batch_with_same_vector(
                    model, tokenizer, questions, steering_vectors[0], layer, MAX_NEW_TOKENS, device=device
                )
            else:
                # Use per-sample vectors, possibly batched
                outputs = steer_questions_batch_with_different_vectors(
                    model, tokenizer, questions, steering_vectors, layer, MAX_NEW_TOKENS, batch_size=BATCH_SIZE, device=device
                )
            
            # Store outputs
            for q_idx, output in zip(question_indices, outputs):
                steer_len = question_steer_lengths[q_idx][steer_len_idx]
                question_results[q_idx]['penalty_results'][penalty]['outputs'].append({
                    'step': steer_len_idx + 1,
                    'steer_length': float(steer_len),
                    'output': output
                })
        except Exception as e:
            # Fallback to per-sample generation on failure
            if verbose:
                print(f"Warning: batch generation failed: {e}")
            for q_idx, steering_vector in zip(question_indices, steering_vectors):
                question_text = test_data[q_idx]['question']
                steer_len = question_steer_lengths[q_idx][steer_len_idx]
                try:
                    output = steer_with_vector(model, tokenizer, question_text, steering_vector, layer, MAX_NEW_TOKENS, device=device)
                    question_results[q_idx]['penalty_results'][penalty]['outputs'].append({
                        'step': steer_len_idx + 1,
                        'steer_length': float(steer_len),
                        'output': output
                    })
                except Exception as e2:
                    question_results[q_idx]['penalty_results'][penalty]['outputs'].append({
                        'step': steer_len_idx + 1,
                        'steer_length': float(steer_len),
                        'output': f"[ERROR: {str(e2)}]"
                    })
        
        # Update progress
        if config_pbar:
            config_pbar.update(1)
    
    # Close progress bar
    if config_pbar:
        config_pbar.close()
    
    if verbose:
        print(f"Completed layer={layer}, cid={cid} with {len(test_data)} test items")
    
    # Convert penalty_results dict to sorted list by step
    for test_idx, result in question_results.items():
        penalty_results_list = []
        for penalty in PENALTY_VALUES:
            penalty_data = result['penalty_results'][penalty]
            # Sort by step
            penalty_data['outputs'].sort(key=lambda x: x['step'])
            penalty_results_list.append(penalty_data)
        result['penalty_results'] = penalty_results_list
        results['experiments'].append(result)
    
    return results

# ============== GPU Worker ==============

def worker_process(gpu_id, task_queue, result_queue, model_name, data_indices_file, csv_file, 
                   diff_vec_base_path, results_base_path, n_steps, max_new_tokens, batch_size, penalty_values):
    """Worker loop pinned to one GPU; pulls tasks, runs experiment, returns results."""
    # Pin this worker to a single GPU
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    print(f"[GPU {gpu_id}] Worker starting")
    
    # Load indices and CSV
    with open(data_indices_file, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)
    df = pd.read_csv(csv_file)
    
    # Load model/tokenizer on this GPU
    print(f"[GPU {gpu_id}] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # Optional torch.compile for speed
    try:
        if hasattr(torch, 'compile') and device.startswith('cuda'):
            model = torch.compile(model, mode='reduce-overhead')
    except Exception as e:
        print(f"[GPU {gpu_id}] torch.compile failed: {e}")
    
    print(f"[GPU {gpu_id}] Ready")
    
    task_count = 0
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel to stop
            break
        
        layer, cid = task
        cid_str = str(cid)
        task_count += 1
        
        print(f"[GPU {gpu_id}] Task {task_count}: layer={layer}, cid={cid}")
        
        if cid_str not in data_indices:
            print(f"[GPU {gpu_id}] Missing cid={cid} in indices")
            result_queue.put((layer, cid, None))
            continue
        
        # Resolve train/test splits
        if 'train' in data_indices[cid_str] and isinstance(data_indices[cid_str]['train'], dict):
            train_qids = data_indices[cid_str]['train'].get(str(layer), [])
        else:
            train_qids = data_indices[cid_str].get('train', [])
        
        test_qids = data_indices[cid_str].get('test', [])
        
        if not train_qids or not test_qids:
            print(f"[GPU {gpu_id}] Skipping layer={layer}, cid={cid} (train={len(train_qids) if train_qids else 0}, test={len(test_qids) if test_qids else 0})")
            result_queue.put((layer, cid, None))
            continue
        
        print(f"[GPU {gpu_id}] Data ready: layer={layer}, cid={cid}, train_qids={len(train_qids)}, test_qids={len(test_qids)}")
        
        # Apply runtime overrides
        import sys
        current_module = sys.modules[__name__]
        # Update module-level globals
        current_module.N_STEPS = n_steps
        current_module.MAX_NEW_TOKENS = max_new_tokens
        current_module.BATCH_SIZE = batch_size
        current_module.PENALTY_VALUES = penalty_values
        current_module.DIFF_VEC_BASE_PATH = diff_vec_base_path
        
        try:
            print(f"[GPU {gpu_id}] Running experiment: layer={layer}, cid={cid}")
            result = run_experiment1_penalty(
                model, tokenizer, layer, cid, train_qids, test_qids, df, verbose=True, device=device
            )
            if result is not None:
                print(f"[GPU {gpu_id}] Done: layer={layer}, cid={cid}, experiments={len(result.get('experiments', []))}")
            else:
                print(f"[GPU {gpu_id}] No result: layer={layer}, cid={cid}")
            result_queue.put((layer, cid, result))
        except Exception as e:
            print(f"[GPU {gpu_id}] Error (layer={layer}, cid={cid}): {e}")
            import traceback
            traceback.print_exc()
            result_queue.put((layer, cid, None))
        finally:
            # Explicit cleanup hook (optional)
            pass
    
    print(f"[GPU {gpu_id}] Worker exiting")

# ==============     ==============

def main():
    # Prepare result directory
    os.makedirs(RESULTS_BASE_PATH, exist_ok=True)
    
    # Load index metadata
    print("Loading indices...")
    with open(DATA_INDICES_FILE, 'r', encoding='utf-8') as f:
        data_indices = json.load(f)
    
    # Collect concept IDs
    all_cids = sorted([int(cid) for cid in data_indices.keys()])
    print(f"Found {len(all_cids)} concepts")
    
    # Count pending tasks
    total_tasks = 0
    task_info = {}  # {(layer, cid): True}       
    for layer in LAYERS:
        exp1_output_file = f"{RESULTS_BASE_PATH}/exp1_penalty_layer{layer}.json"
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
            # Skip completed results
            if cid in existing_exp1_results:
                continue
            
            # Resolve train/test splits
            if 'train' in data_indices[cid_str] and isinstance(data_indices[cid_str]['train'], dict):
                train_qids = data_indices[cid_str]['train'].get(str(layer), [])
            else:
                train_qids = data_indices[cid_str].get('train', [])
            test_qids = data_indices[cid_str].get('test', [])
            
            if train_qids and test_qids:
                total_tasks += 1
                task_info[(layer, cid)] = True
    
    print(f"Pending tasks: {total_tasks} (layers={len(LAYERS)}, concepts={len(all_cids)})")
    
    # Global progress bar
    global_pbar = tqdm(
        total=total_tasks,
        desc="Progress",
        unit="tasks",
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ncols=120,
        miniters=1,
        mininterval=0.5
    )
    
    # Multi-GPU execution
    if USE_MULTI_GPU:
        if len(GPU_IDS) > 1:
            print(f"\nUsing {len(GPU_IDS)} GPUs, {NUM_PROCESSES_PER_GPU} processes per GPU")
        else:
            print(f"\nUsing GPU {GPU_IDS[0]} with {NUM_PROCESSES_PER_GPU} processes")
        
        # Shared queues across processes
        manager = Manager()
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # Build task list
        tasks = []
        for layer in LAYERS:
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp1_penalty_layer{layer}.json"
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
                # Skip completed results
                if cid in existing_exp1_results:
                    continue
                tasks.append((layer, cid))
        
        print(f"Total tasks enqueued: {len(tasks)}")
        
        # Launch worker processes per GPU
        processes = []
        total_processes = len(GPU_IDS) * NUM_PROCESSES_PER_GPU
        
        # Print GPU memory hints
        if len(GPU_IDS) == 1:
            gpu_id = GPU_IDS[0]
            gpu_memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"GPU {gpu_id} total memory: {gpu_memory_gb:.1f} GB")
            print(f"Per-process budget (approx): {gpu_memory_gb / NUM_PROCESSES_PER_GPU:.1f} GB")
        else:
            print(f"Per-GPU processes: {NUM_PROCESSES_PER_GPU}")
            for gpu_id in GPU_IDS:
                gpu_memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"GPU {gpu_id}: {gpu_memory_gb:.1f} GB total, ~{gpu_memory_gb / NUM_PROCESSES_PER_GPU:.1f} GB per process")
        
        # Start workers
        for gpu_id in GPU_IDS:
            for proc_idx in range(NUM_PROCESSES_PER_GPU):
                p = Process(target=worker_process, args=(
                    gpu_id, task_queue, result_queue, MODEL_NAME, DATA_INDICES_FILE,
                    CSV_FILE, DIFF_VEC_BASE_PATH, RESULTS_BASE_PATH,
                    N_STEPS, MAX_NEW_TOKENS, BATCH_SIZE, PENALTY_VALUES
                ))
                p.start()
                processes.append(p)
        
        print(f"Spawned {len(processes)} workers across {len(GPU_IDS)} GPUs ({NUM_PROCESSES_PER_GPU} per GPU)")
        
        # Enqueue tasks
        for task in tasks:
            task_queue.put(task)
        
        # Send stop signals
        for _ in processes:
            task_queue.put(None)
        
        # Accumulate results
        layer_results = {layer: {} for layer in LAYERS}
        completed = 0
        
        while completed < len(tasks):
            layer, cid, result = result_queue.get()
            if result is not None:
                if cid not in layer_results[layer]:
                    layer_results[layer][cid] = result
            completed += 1
            # Update progress bar
            mode_str = f"      ({len(GPU_IDS)}GPU {NUM_PROCESSES_PER_GPU}  ={len(processes)}  )"
            global_pbar.set_description(f"Layer {layer} | CID {cid} ({mode_str})")
            global_pbar.update(1)
        
        # Wait for all workers to finish
        for p in processes:
            p.join()
        
        # Save merged results per layer
        for layer in LAYERS:
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp1_penalty_layer{layer}.json"
            
            # Load existing results
            existing_exp1_results = {}
            if os.path.exists(exp1_output_file):
                with open(exp1_output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                    for r in existing_results:
                        existing_exp1_results[r['cid']] = r
            
            # Merge new and existing results
            new_results = list(layer_results[layer].values())
            all_results = list(existing_exp1_results.values()) + new_results
            
            # Deduplicate by cid (keep latest)
            seen_cids = set()
            unique_results = []
            for r in reversed(all_results):
                if r['cid'] not in seen_cids:
                    seen_cids.add(r['cid'])
                    unique_results.append(r)
            unique_results.reverse()
            
            # Persist to disk
            with open(exp1_output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_results, f, ensure_ascii=False, indent=2)
            print(f"Layer {layer}: {len(unique_results)} results saved")
    
    else:
        # Single-process CPU/GPU execution
        device = f'cuda:{GPU_IDS[0]}' if GPU_IDS else 'cpu'
        print(f"\nDevice: {device}")
        
        # Load CSV
        print("Loading CSV...")
        df = pd.read_csv(CSV_FILE)
        
        # Load model/tokenizer
        print(f"Model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        model = model.to(device)
        model.eval()
        
        # Optional torch.compile (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and device.startswith('cuda'):
                print("Compiling model...")
                model = torch.compile(model, mode='reduce-overhead')
                print("Model compiled")
        except Exception as e:
            print(f"torch.compile failed: {e}")
        
        print(f"Device ready: {device}")
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            gpu_id = int(device.split(':')[1])
            print(f"GPU memory: {torch.cuda.memory_allocated(gpu_id) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        
        # Run per-layer experiments
        for layer in LAYERS:
            print(f"\n{'='*80}")
            print(f"Layer {layer}")
            print(f"{'='*80}")
            
            # Experiment 1: penalty sweep
            print(f"\nExperiment 1 (penalty sweep), layer {layer}")
            exp1_output_file = f"{RESULTS_BASE_PATH}/exp1_penalty_layer{layer}.json"
            
            # Load existing results if available
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
                
                # Skip if missing splits
                if not train_qids:
                    continue
                if not test_qids:
                    continue
                
                # Update progress bar label
                global_pbar.set_description(f"Layer {layer} | CID {cid}")
                
                result = run_experiment1_penalty(
                    model, tokenizer, layer, cid, train_qids, test_qids, df, verbose=False, device=device
                )
                
                if result is not None:
                    exp1_results.append(result)
                    # Persist incremental results
                    with open(exp1_output_file, 'w', encoding='utf-8') as f:
                        json.dump(exp1_results + list(existing_exp1_results.values()), f, ensure_ascii=False, indent=2)
                
                # Update progress
                global_pbar.update(1)
            
            # Finalize layer results
            print(f"\nSaved experiment 1 results to: {exp1_output_file}")
            # Merge with existing results
            all_exp1_results = list(existing_exp1_results.values()) + exp1_results
            # Deduplicate by cid
            seen_cids = set()
            unique_results = []
            for r in reversed(all_exp1_results):
                if r['cid'] not in seen_cids:
                    seen_cids.add(r['cid'])
                    unique_results.append(r)
            unique_results.reverse()
            
            with open(exp1_output_file, 'w', encoding='utf-8') as f:
                json.dump(unique_results, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(unique_results)} unique results")
    
    # Close progress bar
    global_pbar.close()
    
    print(f"\n{'='*80}")
    print("All tasks completed")
    print(f"{'='*80}")

if __name__ == "__main__":
    # Use spawn start method for multiprocessing compatibility
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set
    
    main()
