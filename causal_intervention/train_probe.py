"""
Train a low-rank probe that maps activations to a 3D spiral manifold.

Inputs:
- vectors_tensor (n_samples, d) activation vectors
- Spiral labels Y in 3D.

Steps:
1) PCA: X -> Z (n_samples, k)
2) Generate 3D spiral targets Y (n_samples, 3)
3) Learn probe P (3, k) so that Z @ P^T ≈ Y
4) Save probe and optional visualizations.

Usage:
    python train_probe.py --vectors_path vectors.pt --output_dir ./probes --k 10 --visualize
"""

import os
import argparse
import json
import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


def generate_spiral_labels(n_samples, n_rotations=1.0, radius=1.0, radius_variation=0.0):
    # Generate 3D spiral labels: (t, r*cos(theta), r*sin(theta))
    t = np.linspace(0, 1, n_samples)
    theta = 2 * np.pi * t * n_rotations
    if radius_variation > 0:
        r = radius + radius_variation * t
    else:
        r = np.full(n_samples, radius)
    Y = np.zeros((n_samples, 3))
    Y[:, 0] = t
    Y[:, 1] = r * np.cos(theta)
    Y[:, 2] = r * np.sin(theta)
    return Y


def train_spiral_probe(vectors_tensor, k=5, n_rotations=1.0, radius=1.0, radius_variation=0.0, verbose=True):
    # Train a low-rank probe that maps PCA space to a 3D spiral
    if isinstance(vectors_tensor, torch.Tensor):
        X = vectors_tensor.detach().cpu().numpy()
    else:
        X = vectors_tensor

    n_samples, d = X.shape
    if verbose:
        print(f"Input shape: ({n_samples}, {d})")

    if verbose:
        print("\nStep 1: center activations...")
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    if verbose:
        print(f"Mean shape: {X_mean.shape}")

    if verbose:
        print(f"\nStep 2: PCA with k={k}...")
    pca = PCA(n_components=k)
    Z = pca.fit_transform(X_centered)
    W = pca.components_

    if verbose:
        print(f"PCA Z shape: {Z.shape}")
        print(f"PCA W shape: {W.shape}")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Explained variance sum: {np.sum(pca.explained_variance_ratio_):.4f}")

    if verbose:
        print(f"\nStep 3: build 3D spiral targets (n_rotations={n_rotations}, radius={radius})...")
    Y = generate_spiral_labels(n_samples, n_rotations=n_rotations, 
                               radius=radius, radius_variation=radius_variation)
    if verbose:
        print(f"Spiral label shape: {Y.shape}")

    if verbose:
        print("\nStep 4: solve probe P with least squares...")
    Z_torch = torch.from_numpy(Z).float()
    Y_torch = torch.from_numpy(Y).float()

    P_T, residuals, rank, s = torch.linalg.lstsq(Z_torch, Y_torch)
    P = P_T.T

    if verbose:
        print(f"Probe P shape: {P.shape}")
        print(f"Residuals: {residuals}")
        print(f"Rank: {rank}")

    if verbose:
        print("\nStep 5: evaluate probe fit...")
    Y_pred = (Z_torch @ P_T).detach().cpu().numpy()
    mse = np.mean((Y_pred - Y) ** 2)
    if verbose:
        print(f"Fit MSE: {mse:.6f}")

    if verbose:
        print("\nStep 6: compute pseudoinverse...")
    P_pinv = torch.linalg.pinv(P)
    if verbose:
        print(f"Pseudoinverse shape: {P_pinv.shape}")
        identity_check = P @ P_pinv
        print(f"P @ P_pinv diagonal: {torch.diag(identity_check)}")

    probe_dict = {
        'P': torch.from_numpy(P.detach().cpu().numpy()) if isinstance(P, torch.Tensor) else torch.from_numpy(P),
        'W': torch.from_numpy(W),
        'mean': torch.from_numpy(X_mean),
        'P_pinv': torch.from_numpy(P_pinv.detach().cpu().numpy()) if isinstance(P_pinv, torch.Tensor) else torch.from_numpy(P_pinv),
        'pca': pca,  # PCA object for visualization
        'Z': torch.from_numpy(Z),
        'Y': torch.from_numpy(Y),
        'Y_pred': torch.from_numpy(Y_pred),
        'mse': mse,
        'n_rotations': n_rotations,
        'radius': radius,
        'radius_variation': radius_variation,
        'k': k,
        'd': d,
        'n_samples': n_samples
    }
    return probe_dict


def auto_fit_spiral_params(vectors_tensor, k=5, n_rotations_range=(0.5, 1.5, 0.05), 
                           radius_range=(0.5, 2.0, 0.1), radius_variation_range=(0.0, 0.5, 0.05)):
    # Grid search for spiral parameters that minimize MSE
    print("="*80)
    print("Auto-fit spiral parameters")
    print("="*80)
    n_rotations_values = np.arange(n_rotations_range[0], n_rotations_range[1] + n_rotations_range[2], n_rotations_range[2])
    radius_values = np.arange(radius_range[0], radius_range[1] + radius_range[2], radius_range[2])
    radius_variation_values = np.arange(radius_variation_range[0], radius_variation_range[1] + radius_variation_range[2], radius_variation_range[2])
    total_combinations = len(n_rotations_values) * len(radius_values) * len(radius_variation_values)
    print("Search grid:")
    print(f"  n_rotations: {len(n_rotations_values)} ({n_rotations_range[0]:.2f} to {n_rotations_range[1]:.2f})")
    print(f"  radius: {len(radius_values)} ({radius_range[0]:.2f} to {radius_range[1]:.2f})")
    print(f"  radius_variation: {len(radius_variation_values)} ({radius_variation_range[0]:.2f} to {radius_variation_range[1]:.2f})")
    print(f"  total combinations: {total_combinations}")
    print()

    best_mse = float('inf')
    best_params = None
    best_probe_dict = None

    pbar = tqdm(total=total_combinations, desc="Auto-fit search")
    for n_rot in n_rotations_values:
        for rad in radius_values:
            for rad_var in radius_variation_values:
                try:
                    probe_dict = train_spiral_probe(
                        vectors_tensor,
                        k=k,
                        n_rotations=n_rot,
                        radius=rad,
                        radius_variation=rad_var,
                        verbose=False
                    )
                    mse = probe_dict['mse']
                    if mse < best_mse:
                        best_mse = mse
                        best_params = {
                            'n_rotations': n_rot,
                            'radius': rad,
                            'radius_variation': rad_var,
                            'mse': mse
                        }
                        best_probe_dict = probe_dict
                    pbar.set_postfix({'best_mse': f'{best_mse:.6f}', 
                                     'current': f'{mse:.6f}',
                                     'n_rot': f'{n_rot:.2f}',
                                     'rad': f'{rad:.2f}',
                                     'rad_var': f'{rad_var:.2f}'})
                except Exception as e:
                    print(f"\n  Error (n_rot={n_rot:.2f}, rad={rad:.2f}, rad_var={rad_var:.2f}): {e}")
                pbar.update(1)
    pbar.close()
    print("\n" + "="*80)
    print("Best parameters")
    print("="*80)
    print("Summary:")
    print(f"  n_rotations: {best_params['n_rotations']:.4f}")
    print(f"  radius: {best_params['radius']:.4f}")
    print(f"  radius_variation: {best_params['radius_variation']:.4f}")
    print(f"  MSE: {best_params['mse']:.6f}")
    print()
    return best_params, best_probe_dict


def calibrate_activation(x, probe_dict):
    # Map activation vector to spiral coordinates
    P = probe_dict['P']  # (3, k)
    W = probe_dict['W']  # (k, d)
    X_mean = probe_dict['mean']  # (d,)

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.ndim == 1:
        x = x.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    x_centered = x - X_mean.detach().cpu().numpy() if isinstance(X_mean, torch.Tensor) else x - X_mean
    W_np = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else W
    z = x_centered @ W_np.T
    P_np = P.detach().cpu().numpy() if isinstance(P, torch.Tensor) else P
    spiral_coords = z @ P_np.T
    if squeeze_output:
        spiral_coords = spiral_coords.squeeze(0)
    return spiral_coords


def restore_activation_from_spiral(spiral_coords, probe_dict):
    # Map spiral coordinates back to activation space
    P = probe_dict['P']  # (3, k)
    P_pinv = probe_dict['P_pinv']  # (k, 3)
    W = probe_dict['W']  # (k, d)
    X_mean = probe_dict['mean']  # (d,)

    if isinstance(spiral_coords, torch.Tensor):
        spiral_coords = spiral_coords.detach().cpu().numpy()
    if spiral_coords.ndim == 1:
        spiral_coords = spiral_coords.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False
    P_pinv_np = P_pinv.detach().cpu().numpy() if isinstance(P_pinv, torch.Tensor) else P_pinv
    z = spiral_coords @ P_pinv_np.T
    W_np = W.detach().cpu().numpy() if isinstance(W, torch.Tensor) else W
    x_centered = z @ W_np
    X_mean_np = X_mean.detach().cpu().numpy() if isinstance(X_mean, torch.Tensor) else X_mean
    x_calibrated = x_centered + X_mean_np
    if squeeze_output:
        x_calibrated = x_calibrated.squeeze(0)
    return x_calibrated


def visualize_spiral_probe(probe_dict, save_path=None, show_plot=True):
    # Visualize true spiral, predicted spiral, and PCA projection
    Y = probe_dict['Y']
    Y_pred = probe_dict['Y_pred']
    Z = probe_dict['Z']
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.detach().cpu().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.detach().cpu().numpy()

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=range(len(Y)), cmap='viridis', s=20, alpha=0.6)
    ax1.plot(Y[:, 0], Y[:, 1], Y[:, 2], 'k--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Linear Dimension')
    ax1.set_ylabel('Circular X')
    ax1.set_zlabel('Circular Y')
    ax1.set_title('True Spiral Labels')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], c=range(len(Y_pred)), cmap='viridis', s=20, alpha=0.6)
    ax2.plot(Y_pred[:, 0], Y_pred[:, 1], Y_pred[:, 2], 'k--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Linear Dimension')
    ax2.set_ylabel('Circular X')
    ax2.set_zlabel('Circular Y')
    mse = probe_dict['mse']
    ax2.set_title(f'Predicted Spiral (MSE={mse:.6f})')

    ax3 = fig.add_subplot(133)
    scatter = ax3.scatter(Z[:, 0], Z[:, 1], c=range(len(Z)), cmap='viridis', s=20, alpha=0.6)
    pca = probe_dict['pca']
    evr = getattr(pca, 'explained_variance_ratio_', [0, 0])
    ax3.set_xlabel(f'PC1 ({evr[0]:.2%})')
    ax3.set_ylabel(f'PC2 ({evr[1]:.2%})')
    ax3.set_title('PCA Projection (PC1 vs PC2)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Sample Index')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a low-rank spiral probe',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--vectors_path', type=str, required=True,
                       help='Path to vectors tensor .pt file')
    parser.add_argument('--output_dir', type=str, default='./probes',
                       help='Output directory for probe artifacts')
    parser.add_argument('--k', type=int, default=5,
                       help='PCA dimension k')
    parser.add_argument('--n_rotations', type=float, default=None,
                       help='Spiral rotations; if None, auto-fit')
    parser.add_argument('--radius', type=float, default=None,
                       help='Spiral radius; if None, auto-fit')
    parser.add_argument('--radius_variation', type=float, default=None,
                       help='Linear radius variation; if None, auto-fit')
    parser.add_argument('--auto_fit', action='store_true',
                       help='Force auto-fit of spiral parameters')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display plots')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("="*80)
    print("Loading vectors tensor...")
    print("="*80)
    vectors_tensor = torch.load(args.vectors_path)
    print(f"Path: {args.vectors_path}")
    if isinstance(vectors_tensor, torch.Tensor):
        print(f"Tensor shape: {vectors_tensor.shape}")
    else:
        raise ValueError(f"Unexpected tensor type: {type(vectors_tensor)}")
    need_auto_fit = args.auto_fit or (args.n_rotations is None) or (args.radius is None) or (args.radius_variation is None)
    if need_auto_fit:
        best_params, probe_dict = auto_fit_spiral_params(
            vectors_tensor,
            k=args.k,
            n_rotations_range=(0.6, 1.2, 0.1),
            radius_range=(0.1, 1.0, 0.1),
            radius_variation_range=(0.0, 0.5, 0.1)
        )
        args.n_rotations = best_params['n_rotations']
        args.radius = best_params['radius']
        args.radius_variation = best_params['radius_variation']
    else:
        print("\n" + "="*80)
        print("Training probe with provided parameters")
        print("="*80)
        probe_dict = train_spiral_probe(
            vectors_tensor,
            k=args.k,
            n_rotations=args.n_rotations,
            radius=args.radius,
            radius_variation=args.radius_variation
        )

    print("\n" + "="*80)
    print("Saving probe artifacts")
    print("="*80)

    # Convert sklearn fields and scalars to torch/tensor/numbers for saving
    probe_save_dict = {
        'P': probe_dict['P'],
        'W': probe_dict['W'],
        'mean': probe_dict['mean'],
        'P_pinv': probe_dict['P_pinv'],
        'Z': probe_dict['Z'],
        'Y': probe_dict['Y'],
        'Y_pred': probe_dict['Y_pred'],
        'mse': float(probe_dict['mse']),
        'n_rotations': float(probe_dict['n_rotations']),
        'radius': float(probe_dict['radius']),
        'radius_variation': float(probe_dict['radius_variation']),
        'k': int(probe_dict['k']),
        'd': int(probe_dict['d']),
        'n_samples': int(probe_dict['n_samples']),
        # PCA       sklearn           
        'pca_explained_variance_ratio': torch.from_numpy(probe_dict['pca'].explained_variance_ratio_),
        'pca_mean': torch.from_numpy(probe_dict['pca'].mean_),
        'pca_components': torch.from_numpy(probe_dict['pca'].components_),
    }
    probe_path = os.path.join(args.output_dir, 'probe.pt')
    torch.save(probe_save_dict, probe_path)
    print(f"Probe saved to: {probe_path}")

    # Save metadata JSON
    meta_info = {
        'vectors_path': args.vectors_path,
        'k': args.k,
        'n_rotations': args.n_rotations,
        'radius': args.radius,
        'radius_variation': args.radius_variation,
        'mse': float(probe_dict['mse']),
        'n_samples': int(probe_dict['n_samples']),
        'd': int(probe_dict['d']),
        'pca_explained_variance': float(np.sum(probe_dict['pca'].explained_variance_ratio_)),
    }
    meta_path = os.path.join(args.output_dir, 'probe_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    print(f"Metadata saved to: {meta_path}")

    if args.visualize:
        print("\n" + "="*80)
        print("Generating visualizations")
        print("="*80)
        viz_path = os.path.join(args.output_dir, 'spiral_probe_visualization.png')
        visualize_spiral_probe(probe_dict, save_path=viz_path, show_plot=not args.no_show)

    print("\n" + "="*80)
    print("Quick sanity check")
    print("="*80)
    test_vector = vectors_tensor[0]
    spiral_coords = calibrate_activation(test_vector, probe_dict)
    restored_vector = restore_activation_from_spiral(spiral_coords, probe_dict)
    print(f"Input vector shape: {test_vector.shape}")
    print(f"Spiral coords: {spiral_coords}")
    print(f"Restored vector shape: {restored_vector.shape}")
    if isinstance(test_vector, torch.Tensor):
        test_np = test_vector.detach().cpu().numpy()
    else:
        test_np = test_vector
    print(f"Reconstruction error (L2): {np.linalg.norm(test_np - restored_vector):.6f}")
    print("\n" + "="*80)
    print("Usage example")
    print("="*80)
    print("\nSteps:")
    print("  1. Load the probe:")
    print(f"     probe_dict = torch.load('{probe_path}')")
    print("  2. Map activation to spiral coords:")
    print(f"     spiral_coords = calibrate_activation(x, probe_dict)")
    print("  3. Restore activation from spiral coords:")
    print(f"     x_calibrated = restore_activation_from_spiral(spiral_coords, probe_dict)")


if __name__ == "__main__":
    main()
