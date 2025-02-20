"""
Utility functions for evaluating generated images against real images.
Includes LPIPS, SSIM, Sobel edge similarity, and distribution metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import lpips
from scipy.optimize import linear_sum_assignment
from skimage.filters import sobel

def load_and_preprocess_data(data_path, batch_size):
    """
    data shape: [40000, 1, 64, 64]
    """
    dataset = torch.load(data_path)

    # random image in 8x8 grid 
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            idx = torch.randint(dataset.shape[0], (1,)).item()
            img = dataset[idx][0].numpy()  # numpy array, shape: (64,64)
            ax[i, j].imshow(img, origin="lower", cmap="viridis")
            ax[i, j].axis("off")
    plt.suptitle("Random Samples from Dataset", fontsize=16)
    plt.show()

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, data_loader

def downsample_image(image: np.ndarray, block_size: int = 2) -> np.ndarray:
    """
    Downsample image by averaging blocks of pixels.
    Handles any block_size by padding if necessary.
    
    Args:
        image: Input image of shape (height, width)
        block_size: Size of blocks for downsampling
        
    Returns:
        np.ndarray: Downsampled image
    """
    h, w = image.shape
    
    # Calculate new dimensions that are divisible by block_size
    new_h = ((h + block_size - 1) // block_size) * block_size
    new_w = ((w + block_size - 1) // block_size) * block_size
    
    # Create padded image if necessary
    if new_h != h or new_w != w:
        padded = np.pad(
            image,
            ((0, new_h - h), (0, new_w - w)),
            mode='edge'  # 가장자리 값으로 패딩
        )
    else:
        padded = image
    
    # Reshape and compute mean
    return padded.reshape(new_h//block_size, block_size, 
                         new_w//block_size, block_size).mean(axis=(1,3))
def calculate_mse(a, b):
    return np.mean((a - b) ** 2)

def evaluate_mse(real_images: np.ndarray, generated_images: np.ndarray, block_sizes: list = [2, 4, 8, 16, 32]) -> dict:
    """
    Evaluate distribution similarity using multiple downsampling sizes.
    
    Args:
        real_images: Real images of shape (n_samples, 64, 64)
        generated_images: Generated images of shape (n_samples, 64, 64)
        block_sizes: List of downsampling block sizes
        
    Returns:
        dict: Dictionary containing MSE, MAE, KLD values for each block size
    """
    results = {}
    
    for block_size in block_sizes:
        real_downsampled = np.array([downsample_image(img, block_size) for img in real_images])
        gen_downsampled = np.array([downsample_image(img, block_size) for img in generated_images])
        
        real_all = real_downsampled.flatten()
        gen_all = gen_downsampled.flatten()
        
        results[block_size] = {
            'mse': calculate_mse(real_all, gen_all)
        }
    
    return results

def evaluate_lpips(real_images: np.ndarray,
                  gen_images: np.ndarray,
                  device: torch.device,
                  match_mode: str = "best_match") -> float:
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) between real and generated images.
    Lower score indicates higher perceptual similarity.
    
    Args:
        real_images: Real images of shape (n_samples, height, width)
        gen_images: Generated images of shape (n_samples, height, width)
        device: PyTorch device
        match_mode: Either "one_to_one" or "best_match"
        
    Returns:
        float: Mean LPIPS score
    """
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    def to_tensor_batch(imgs: np.ndarray) -> torch.Tensor:
        imgs_torch = torch.from_numpy(imgs).float().unsqueeze(1)
        return imgs_torch.repeat(1, 3, 1, 1)

    real_torch = to_tensor_batch(real_images).to(device)
    gen_torch = to_tensor_batch(gen_images).to(device)
    n = real_images.shape[0]

    if match_mode == "one_to_one":
        with torch.no_grad():
            dist = lpips_fn.forward(real_torch, gen_torch)
            dist = dist.squeeze().cpu().numpy()
        return dist.mean()

    elif match_mode == "best_match":
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            real_i = real_torch[i].unsqueeze(0).expand(n, -1, -1, -1)
            with torch.no_grad():
                dist_i = lpips_fn.forward(real_i, gen_torch)
                dist_i = dist_i.squeeze().cpu().numpy()
            distance_matrix[i] = dist_i

        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        min_sum = distance_matrix[row_ind, col_ind].sum()
        return min_sum / n

    else:
        raise ValueError("match_mode should be either 'one_to_one' or 'best_match'")

def evaluate_sobel_edge_mse(real_images: np.ndarray,
                          gen_images: np.ndarray,
                          match_mode: str = "best_match") -> float:
    """
    Calculate MSE between Sobel edge maps of real and generated images.
    Lower score indicates more similar edge structures.
    
    Args:
        real_images: Real images of shape (n_samples, height, width)
        gen_images: Generated images of shape (n_samples, height, width)
        match_mode: Either "one_to_one" or "best_match"
        
    Returns:
        float: Mean edge MSE
    """
    real_edges = np.array([sobel(r) for r in real_images])
    gen_edges = np.array([sobel(g) for g in gen_images])
    n = real_images.shape[0]

    if match_mode == "one_to_one":
        mse_sum = np.mean((real_edges - gen_edges)**2, axis=(1,2)).sum()
        return mse_sum / n

    elif match_mode == "best_match":
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            diff = real_edges[i] - gen_edges
            mse_vals = np.mean(diff**2, axis=(1,2))
            distance_matrix[i] = mse_vals

        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        min_sum = distance_matrix[row_ind, col_ind].sum()
        return min_sum / n

    else:
        raise ValueError("match_mode should be either 'one_to_one' or 'best_match'")
