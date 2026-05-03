"""
Steering Vector Optimization and Visualization Tool

This script optimizes steering vectors for language models and visualizes the results
using PCA and loss analysis.

Usage:
    python opt_steer_viz.py [OPTIONS]
    
Examples:
    # Basic usage with default parameters
    python opt_steer_viz.py
    
    # Specify custom parameters
    python opt_steer_viz.py --layer_idx 2 --max_iters 50 --lr 0.1
    
    # Use custom prompts and responses
    python opt_steer_viz.py --prompt_ori "Your prompt here" --prompt_tgt "Target prompt here"
    
    # Load responses from a JSON file
    python opt_steer_viz.py --responses_file responses.json
    
    # Use a configuration file
    python opt_steer_viz.py --config config.json
    
    # Save visualization without displaying
    python opt_steer_viz.py --save_path output.png --no_show
    
    # For causal intervention, we need to specify the source and target responses.
    python optim_vecs.py --prompt_ori "Describe the structure of an atom. Output in json." \
        --prompt_tgt "Describe the structure of an atom. Output in python." \
        --src_response '```json' \
        --dst_response '```python' \
        --save_path "./" \
        --w_start 0.01 --w_end 1.50 --w_step 0.03
    
For more information, run: python opt_steer_viz.py --help
"""

from utils import *


def parse_args():
    """Parse CLI arguments for steering vector optimization/visualization."""
    parser = argparse.ArgumentParser(
        description='Steering Vector Optimization and Visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model_name', type=str,
                        default='YOUR_MODEL_NAME_HERE',
                        help='HF model name or local path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda/cpu); auto if None')
    
    # Optimization config
    parser.add_argument('--layer_idx', type=int, default=1,
                        help='Layer index to hook steering vector')
    parser.add_argument('--max_iters', type=int, default=20,
                        help='Max optimization iterations')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    
    # Constraint / batch mode
    parser.add_argument('--batch_mode', action='store_true',
                        help='Enable processing multiple constraints/questions')
    parser.add_argument('--constrain_file', type=str,
                        default='reps/constrain.json',
                        help='JSON file with constraint entries (name/pos/neg)')
    parser.add_argument('--question_file', type=str,
                        default='reps/alpaca_eval.json',
                        help='JSON file containing question instructions')
    parser.add_argument('--n_questions', type=int, default=3,
                        help='Number of questions to sample per constraint')
    parser.add_argument('--max_response_tokens', type=int, default=40,
                        help='Max tokens to generate for responses')
    
    # Prompts
    parser.add_argument('--prompt_ori', type=str,
                        default='Describe the structure of an atom. Output in json.',
                        help='Original prompt (source style)')
    parser.add_argument('--prompt_tgt', type=str,
                        default='Describe the structure of an atom. Output in python.',
                        help='Target prompt (desired style)')
    
    # Responses
    default_src_response = '\nmodel\n```json\n{\n  "structure": {\n    "nucleus": {\n      "charge": 1,\n      "size": 1\n    },\n    "electrons": {\n      "count": 1\n    }'
    default_dst_response = '\nmodel\n```python\nclass Atom:\n    def __init__(self, atomic_number, mass_number):\n        self.atomic_number = atomic_number\n        self.mass_number = mass_number\n\n    def __repr__('
    
    parser.add_argument('--src_response', type=str,
                        default=default_src_response,
                        help='Source reference response string')
    parser.add_argument('--dst_response', type=str,
                        default=default_dst_response,
                        help='Target reference response string')
    parser.add_argument('--responses_file', type=str, default=None,
                        help='JSON file with {"src": "...", "dst": "..."} to override responses')
    
    # Weight sweep
    parser.add_argument('--w_start', type=float, default=0.1,
                        help='Starting steering weight')
    parser.add_argument('--w_end', type=float, default=3.0,
                        help='Ending steering weight')
    parser.add_argument('--w_step', type=float, default=0.1,
                        help='Step size for weight sweep')
    parser.add_argument('--w_values', type=str, default=None,
                        help='Comma list of explicit weights (overrides range)')
    
    # Output / display
    parser.add_argument('--save_path', type=str, default='reps/opt_traces',
                        help='Directory to save SVG/CSV results')
    parser.add_argument('--no_show', action='store_true',
                        help='Do not open figures, only save')
    
    # Reference saving
    parser.add_argument('--save_reference_only', action='store_true',
                        help='Save only reference (unoptimized) traces')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Optional JSON config to override args')
    
    return parser.parse_args()


def load_config(config_path):
    """Load a JSON config file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Entry point for CLI execution."""
    args = parse_args()
    
    # Apply optional config overrides
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model(args.model_name, args.device)
    
    # Build sweep weights
    if args.w_values:
        w_values = np.array([float(x.strip()) for x in args.w_values.split(',')])
    else:
        w_values = np.arange(args.w_start, args.w_end + args.w_step, args.w_step)
    
    # Batch mode (multi-constraint, multi-question)
    if args.batch_mode:
        print("\n" + "="*80)
        print("Batch Processing Mode")
        print("="*80)
        
        # Load constraint and question data
        print(f"\nLoading constraints from: {args.constrain_file}")
        with open(args.constrain_file, 'r', encoding='utf-8') as f:
            constrains = json.load(f)
        
        print(f"Loading questions from: {args.question_file}")
        with open(args.question_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        print(f"\nFound {len(constrains)} constraints and {len(questions_data)} questions")
        print(f"Will process {args.n_questions} questions per constraint")
        print()
        
        # Iterate constraints
        for constrain_idx, constrain in enumerate(constrains):
            constrain_name = constrain['name']
            constrain_pos = constrain['pos']
            constrain_neg = constrain['neg']
            
            print(f"\n{'='*80}")
            print(f"Processing Constraint {constrain_idx + 1}/{len(constrains)}: {constrain_name}")
            print(f"{'='*80}")
            print(f"  Positive: {constrain_pos}")
            print(f"  Negative: {constrain_neg}")
            
            # Create per-constraint output directory
            constrain_save_dir = os.path.join(args.save_path, constrain_name)
            os.makedirs(constrain_save_dir, exist_ok=True)
            
            # Sample questions for this constraint
            selected_indices = random.sample(range(len(questions_data)), min(args.n_questions, len(questions_data)))
            
            print(f"\n  Selected {len(selected_indices)} questions (indices: {selected_indices})")
            
            # Iterate selected questions
            for q_idx, question_idx in enumerate(selected_indices):
                question = questions_data[question_idx]['instruction']
                
                print(f"\n  [{q_idx + 1}/{len(selected_indices)}] Processing question {question_idx}: {question[:60]}...")
                
                # Build prompts
                prompt_ori = prepare_prompt(tokenizer, [{"role": "user", "content": question + " " + constrain_neg}])
                prompt_tgt = prepare_prompt(tokenizer, [{"role": "user", "content": question + " " + constrain_pos}])
                
                # Generate baseline responses without steering
                print("    Generating responses without steering...")
                src_response = generate_response(prompt_ori, model, tokenizer, device, max_new_tokens=args.max_response_tokens)
                dst_response = generate_response(prompt_tgt, model, tokenizer, device, max_new_tokens=args.max_response_tokens)
                
                print(f"    Source response length: {len(src_response)} chars")
                print(f"    Target response length: {len(dst_response)} chars")
                
                # Optimize steering vectors
                print("    Starting optimization...")
                try:
                    vectors_tensor, w_vals, base_vector, activation_distance, diff_pos, initial_losses, final_losses = \
                        get_optimized_vectors_with_fixed_lengths(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_ori=prompt_ori,
                            prompt_tgt=prompt_tgt,
                            src_response=src_response,
                            dst_response=dst_response,
                            layer=args.layer_idx,
                            w_values=w_values,
                            device=device,
                            max_iters=args.max_iters,
                            lr=args.lr,
                            save_reference_only=args.save_reference_only
                        )
                    
                    print(f"    Optimization completed! Activation distance: {activation_distance:.4f}")
                    
                    # Save vectors tensor
                    tensor_save_path = os.path.join(constrain_save_dir, f"{question_idx}_vectors.pt")
                    torch.save(vectors_tensor, tensor_save_path)
                    print(f"    Vectors tensor saved to: {tensor_save_path}")
                    
                    # Save visualization
                    save_file_path = os.path.join(constrain_save_dir, f"{question_idx}.svg")
                    print(f"    Saving visualization to: {save_file_path}")
                    
                    pca, vectors_3d = plot_combined_visualization(
                        vectors_tensor, w_vals, initial_losses, final_losses,
                        save_path=save_file_path,
                        show_plot=not args.no_show
                    )
                    
                    print(f"      Question {question_idx} completed!")
                    
                except Exception as e:
                    print(f"      Error processing question {question_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        print(f"\n{'='*80}")
        print("Batch Processing Completed!")
        print(f"{'='*80}\n")
    
    else:
        # Single run mode
        print("\nSingle Processing Mode")
        
        # Build prompts
        prompt_ori = prepare_prompt(tokenizer, [{"role": "user", "content": args.prompt_ori}])
        prompt_tgt = prepare_prompt(tokenizer, [{"role": "user", "content": args.prompt_tgt}])
        
        # Load or use provided reference responses
        if args.responses_file:
            with open(args.responses_file, 'r', encoding='utf-8') as f:
                responses = json.load(f)
                src_response = responses.get('src', args.src_response)
                dst_response = responses.get('dst', args.dst_response)
        else:
            src_response = args.src_response
            dst_response = args.dst_response
        
        print(f"\nConfiguration:")
        print(f"  Layer index: {args.layer_idx}")
        print(f"  Max iterations: {args.max_iters}")
        print(f"  Learning rate: {args.lr}")
        print(f"  W values range: {w_values.min():.2f} to {w_values.max():.2f} (step: {args.w_step:.2f}, total: {len(w_values)})")
        print(f"  Save reference only: {args.save_reference_only}")
        print(f"  Prompt original: {prompt_ori}...")
        print(f"  Prompt target: {prompt_tgt}...")
        print()
        
        # Run optimization or reference-only generation
        if args.save_reference_only:
            print("Generating reference vectors (no optimization)...")
        else:
            print("Starting optimization...")
        vectors_tensor, w_values, base_vector, activation_distance, diff_pos, initial_losses, final_losses = \
            get_optimized_vectors_with_fixed_lengths(
                model=model,
                tokenizer=tokenizer,
                prompt_ori=prompt_ori,
                prompt_tgt=prompt_tgt,
                src_response=src_response,
                dst_response=dst_response,
                layer=args.layer_idx,
                w_values=w_values,
                device=device,
                max_iters=args.max_iters,
                lr=args.lr,
                save_reference_only=args.save_reference_only
            )
        
        print(f"\nOptimization completed!")
        print(f"  Activation distance: {activation_distance:.4f}")
        print(f"  Different token position: {diff_pos}")
        
        # Save vectors tensor
        if os.path.isdir(args.save_path):
            tensor_save_path = os.path.join(args.save_path, "vectors.pt")
        else:
            tensor_save_path = os.path.splitext(args.save_path)[0] + "_vectors.pt"
        torch.save(vectors_tensor, tensor_save_path)
        print(f"Vectors tensor saved to: {tensor_save_path}")
        
        # Generate visualization
        print("\nGenerating visualization...")
        pca, vectors_3d = plot_combined_visualization(
            vectors_tensor, w_values, initial_losses, final_losses,
            save_path=args.save_path,
            show_plot=not args.no_show
        )
        
        print("\nDone!")


if __name__ == "__main__":
    main()
