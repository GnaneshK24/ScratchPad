"""
Utility functions for DriftSense Navigation-Error Recovery
Includes image processing, visualization, and evaluation metrics
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt


def create_dram_grid(height: int, width: int, pitch: int = 10) -> np.ndarray:
    """
    Generate a DRAM-style grid pattern with word-lines, bit-lines, and contact points.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        pitch: Spacing between grid lines in pixels (default: 10)
    
    Returns:
        2D numpy array representing the DRAM grid pattern
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Horizontal word-lines (darker, but brighter)
    for y in range(0, height, pitch):
        image[y:y+2, :] = 200
    
    # Vertical bit-lines
    for x in range(0, width, pitch):
        image[:, x:x+2] = 200
    
    # Contact/via dots at intersections
    for y in range(0, height, pitch):
        for x in range(0, width, pitch):
            image[y-1:y+2, x-1:x+2] = 255
    
    return image


def create_finfet_structure(height: int, width: int, fin_width: int = 4, 
                           fin_spacing: int = 8, gate_bars: int = 2) -> np.ndarray:
    """
    Generate a FinFET-style structure with parallel fins and gate bars.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        fin_width: Width of each fin in pixels (default: 4)
        fin_spacing: Spacing between fins in pixels (default: 8)
        gate_bars: Number of horizontal gate bar crossings (default: 2)
    
    Returns:
        2D numpy array representing the FinFET structure
    """
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Draw parallel vertical fin lines
    for x in range(0, width, fin_width + fin_spacing):
        image[:, x:x+fin_width] = 180
    
    # Draw horizontal gate bars (brighter/higher contrast)
    gate_positions = np.linspace(height // 3, 2 * height // 3, gate_bars + 2)[1:-1].astype(int)
    for y in gate_positions:
        image[y:y+3, :] = 255
    
    return image


def add_edge_brightening(image: np.ndarray, kernel_size: int = 3, strength: float = 0.3) -> np.ndarray:
    """
    Apply edge-brightening to simulate SEM image contrast behavior.
    Edges appear brighter in SEM micrographs due to edge effects.
    
    Reference: SEM contrast mechanisms (secondary electron emission from edges)
    
    Args:
        image: Input image (grayscale)
        kernel_size: Sobel kernel size (default: 3)
        strength: Strength of edge brightening (0-1, default: 0.3)
    
    Returns:
        Image with edge brightening applied
    """
    image_float = image.astype(np.float32) / 255.0
    
    # Compute edge magnitude using Sobel
    edges_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=kernel_size)
    edges_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=kernel_size)
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = edges / (edges.max() + 1e-8)  # Normalize
    
    # Apply brightening
    brightened = image_float + strength * edges
    brightened = np.clip(brightened, 0, 1) * 255
    
    return brightened.astype(np.uint8)


def add_independent_noise(image: np.ndarray, noise_level: float = 0.1, 
                         noise_type: str = 'gaussian') -> np.ndarray:
    """
    Add independent, realistic noise to simulate sensor/measurement noise.
    Each image gets its own independent noise pattern.
    
    Args:
        image: Input image (grayscale)
        noise_level: Noise standard deviation as fraction of image max (default: 0.1)
        noise_type: 'gaussian' (default) or 'poisson'
    
    Returns:
        Noisy image
    """
    image_float = image.astype(np.float32)
    max_val = image.max()
    noise_sigma = noise_level * max_val
    
    if noise_type == 'gaussian':
        # Independent Gaussian noise
        noise = np.random.normal(0, noise_sigma, image.shape)
    elif noise_type == 'poisson':
        # Poisson noise (photon noise)
        noise = np.random.poisson(noise_sigma / 255.0, image.shape) * 255
    else:
        noise = np.zeros_like(image_float)
    
    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 255)
    
    return noisy.astype(np.uint8)


def apply_blur(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to simulate optical degradation or measurement blur.
    
    Args:
        image: Input image
        kernel_size: Gaussian blur kernel size (must be odd, default: 3)
        sigma: Standard deviation of Gaussian kernel (default: 1.0)
    
    Returns:
        Blurred image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Apply rotation to image (simulate stage misalignment).
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
    
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return rotated


def apply_scaling(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Apply scaling to image (simulate magnification variation).
    
    Args:
        image: Input image
        scale_factor: Scale factor (>1 = zoom in, <1 = zoom out)
    
    Returns:
        Scaled image (preserving original size by cropping/padding)
    """
    h, w = image.shape[:2]
    new_size = int(w * scale_factor), int(h * scale_factor)
    
    if scale_factor > 1:
        # Zoom in and crop center
        scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        start_y = (scaled.shape[0] - h) // 2
        start_x = (scaled.shape[1] - w) // 2
        cropped = scaled[start_y:start_y+h, start_x:start_x+w]
        return cropped
    else:
        # Zoom out and pad with reflection
        scaled = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        pad_top = (h - scaled.shape[0]) // 2
        pad_left = (w - scaled.shape[1]) // 2
        padded = cv2.copyMakeBorder(scaled, pad_top, h-scaled.shape[0]-pad_top,
                                    pad_left, w-scaled.shape[1]-pad_left,
                                    cv2.BORDER_REFLECT)
        return padded


def compute_center_coordinates(image: np.ndarray) -> Tuple[float, float]:
    """Get the center coordinates of an image."""
    h, w = image.shape[:2]
    return w / 2.0, h / 2.0


def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def compute_accuracy(predictions: np.ndarray, ground_truths: np.ndarray, 
                    tolerance: float = 10.0) -> float:
    """
    Compute accuracy as percentage of predictions within tolerance of ground truth.
    
    Args:
        predictions: Array of predicted (x, y) coordinates
        ground_truths: Array of ground truth (x, y) coordinates
        tolerance: Pixel tolerance for "correct" prediction (default: 10.0)
    
    Returns:
        Accuracy percentage (0-100)
    """
    distances = np.sqrt(np.sum((predictions - ground_truths)**2, axis=1))
    correct = np.sum(distances <= tolerance)
    return (correct / len(predictions)) * 100


def save_annotations(annotations: Dict[str, Any], output_path: str) -> None:
    """Save annotations to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Saved annotations to {output_path}")


def load_annotations(annotation_path: str) -> Dict[str, Any]:
    """Load annotations from JSON file."""
    with open(annotation_path, 'r') as f:
        return json.load(f)


def visualize_prediction(reference: np.ndarray, search: np.ndarray,
                        predicted_pos: Tuple[float, float],
                        ground_truth_pos: Tuple[float, float] = None,
                        output_path: str = None) -> None:
    """
    Visualize reference image overlaid on search image with predicted location.
    
    Args:
        reference: Reference image (100x100 or similar)
        search: Search image (1000x1000)
        predicted_pos: Predicted (x, y) center of reference in search image
        ground_truth_pos: Ground truth (x, y) center (optional)
        output_path: Path to save visualization (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Show reference
    axes[0].imshow(reference, cmap='gray')
    axes[0].set_title('Reference Image')
    axes[0].axis('off')
    
    # Show search with predicted location marked
    axes[1].imshow(search, cmap='gray')
    ref_h, ref_w = reference.shape[:2]
    
    # Draw predicted bounding box
    x, y = predicted_pos
    rect_pred = plt.Rectangle((x - ref_w/2, y - ref_h/2), ref_w, ref_h,
                              linewidth=2, edgecolor='red', facecolor='none',
                              label='Predicted')
    axes[1].add_patch(rect_pred)
    
    # Draw ground truth bounding box if provided
    if ground_truth_pos:
        gx, gy = ground_truth_pos
        rect_gt = plt.Rectangle((gx - ref_w/2, gy - ref_h/2), ref_w, ref_h,
                               linewidth=2, edgecolor='green', facecolor='none',
                               label='Ground Truth')
        axes[1].add_patch(rect_gt)
    
    axes[1].set_title('Search Image with Prediction')
    axes[1].legend()
    axes[1].axis('off')
    
    # Show error (if ground truth available)
    if ground_truth_pos:
        error = euclidean_distance(predicted_pos, ground_truth_pos)
        axes[2].text(0.5, 0.7, f"Predicted: ({predicted_pos[0]:.1f}, {predicted_pos[1]:.1f})",
                    ha='center', fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.5, 0.5, f"Ground Truth: ({ground_truth_pos[0]:.1f}, {ground_truth_pos[1]:.1f})",
                    ha='center', fontsize=12, transform=axes[2].transAxes)
        axes[2].text(0.5, 0.3, f"Error: {error:.2f} pixels",
                    ha='center', fontsize=12, color='red', weight='bold',
                    transform=axes[2].transAxes)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    # Quick test
    print("Testing utility functions...")
    
    # Test DRAM grid generation
    dram = create_dram_grid(200, 200, pitch=10)
    print(f"DRAM grid shape: {dram.shape}, min: {dram.min()}, max: {dram.max()}")
    
    # Test FinFET structure
    finfet = create_finfet_structure(200, 200)
    print(f"FinFET shape: {finfet.shape}, min: {finfet.min()}, max: {finfet.max()}")
    
    # Test noise addition
    dram_noisy = add_independent_noise(dram, noise_level=0.1)
    print(f"DRAM with noise - min: {dram_noisy.min()}, max: {dram_noisy.max()}")
    
    # Test edge brightening
    dram_enhanced = add_edge_brightening(dram, strength=0.3)
    print(f"DRAM with edge brightening - min: {dram_enhanced.min()}, max: {dram_enhanced.max()}")
    
    print("✓ All utility functions tested successfully")
