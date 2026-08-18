"""
Utility script to generate sample FinFET base CAD images (10,000 x 10,000 pixels).

This creates synthetic base images that simulate real CAD layouts from which
the FinFET SEM dataset generator can crop and augment.

Usage:
    python create_finfet_base_images.py --output_dir finfet_base_images --num_images 3
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def create_finfet_base_image(size: int = 10000, seed: int = None) -> np.ndarray:
    """
    Create a synthetic FinFET base CAD image (10k x 10k).
    
    Structure:
    - Dense parallel vertical fin lines
    - Horizontal gate bars crossing at regular intervals
    - Realistic spacing and feature sizes
    
    Args:
        size: Image size (default: 10000)
        seed: Random seed for reproducibility
    
    Returns:
        Grayscale image (uint8)
    """
    if seed is not None:
        np.random.seed(seed)
    
    image = np.ones((size, size), dtype=np.uint8) * 50  # Dark background
    
    # Fin spacing (in pixels, scaled for 10k image)
    fin_pitch = 100  # Distance between adjacent fin centerlines
    fin_width = 30   # Width of each fin
    
    # Gate bar parameters
    gate_height = 40
    gate_pitch = 400  # Distance between adjacent gate bars
    gate_offset = 100  # Starting position of first gate bar
    
    # Draw parallel vertical fins (white/bright)
    for x in range(0, size, fin_pitch):
        x_start = max(0, x - fin_width // 2)
        x_end = min(size, x + fin_width // 2)
        image[:, x_start:x_end] = 200
    
    # Draw horizontal gate bars (very bright)
    for y in range(gate_offset, size, gate_pitch):
        y_start = max(0, y - gate_height // 2)
        y_end = min(size, y + gate_height // 2)
        image[y_start:y_end, :] = 255
    
    # Intersections: enhance contact points (brightest)
    for x in range(0, size, fin_pitch):
        for y in range(gate_offset, size, gate_pitch):
            x_start = max(0, x - fin_width // 2)
            x_end = min(size, x + fin_width // 2)
            y_start = max(0, y - gate_height // 2)
            y_end = min(size, y + gate_height // 2)
            image[y_start:y_end, x_start:x_end] = 255
    
    # Add slight random variation to simulate real CAD data
    noise = np.random.normal(0, 3, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight Gaussian blur to simulate lithography effects
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    return image


def create_dram_base_image(size: int = 10000, seed: int = None) -> np.ndarray:
    """
    Create a synthetic DRAM base CAD image (10k x 10k).
    
    Structure:
    - Orthogonal grid of word-lines (horizontal) and bit-lines (vertical)
    - Contact vias at every intersection
    - Realistic pitch and feature sizes
    
    Args:
        size: Image size (default: 10000)
        seed: Random seed for reproducibility
    
    Returns:
        Grayscale image (uint8)
    """
    if seed is not None:
        np.random.seed(seed)
    
    image = np.ones((size, size), dtype=np.uint8) * 50  # Dark background
    
    # Grid pitch (in pixels, scaled for 10k image)
    line_pitch = 200  # Distance between adjacent lines
    line_width = 15   # Width of each line
    contact_size = 25  # Size of contact via at intersections
    
    # Draw horizontal word-lines (brighter)
    for y in range(0, size, line_pitch):
        y_start = max(0, y - line_width // 2)
        y_end = min(size, y + line_width // 2)
        image[y_start:y_end, :] = 180
    
    # Draw vertical bit-lines (brighter)
    for x in range(0, size, line_pitch):
        x_start = max(0, x - line_width // 2)
        x_end = min(size, x + line_width // 2)
        image[:, x_start:x_end] = 180
    
    # Draw contact vias at intersections (brightest)
    for y in range(0, size, line_pitch):
        for x in range(0, size, line_pitch):
            y_start = max(0, y - contact_size // 2)
            y_end = min(size, y + contact_size // 2)
            x_start = max(0, x - contact_size // 2)
            x_end = min(size, x + contact_size // 2)
            image[y_start:y_end, x_start:x_end] = 255
    
    # Add slight random variation
    noise = np.random.normal(0, 3, image.shape)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # Apply slight Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    return image


def main():
    """Command-line interface for base image generation."""
    parser = argparse.ArgumentParser(
        description='Generate sample FinFET/DRAM base CAD images for dataset generation'
    )
    parser.add_argument('--architecture', type=str, default='finfet',
                       choices=['finfet', 'dram'],
                       help='Architecture type (default: finfet)')
    parser.add_argument('--output_dir', type=str, default='finfet_base_images',
                       help='Output directory (default: finfet_base_images)')
    parser.add_argument('--num_images', type=int, default=3,
                       help='Number of base images to generate (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_images} {args.architecture.upper()} base CAD images (10,000 x 10,000)...")
    print(f"Output directory: {output_path}")
    
    for i in tqdm(range(args.num_images), desc="Creating images"):
        # Create image with deterministic seed if provided
        img_seed = args.seed + i if args.seed is not None else None
        
        if args.architecture == 'finfet':
            image = create_finfet_base_image(size=10000, seed=img_seed)
        else:  # dram
            image = create_dram_base_image(size=10000, seed=img_seed)
        
        # Save image
        filename = f"{args.architecture}_base_{i:02d}.png"
        filepath = output_path / filename
        cv2.imwrite(str(filepath), image)
    
    print(f"\n✓ Successfully generated {args.num_images} base images")
    print(f"  Location: {output_path}/")
    print(f"\nNext step:")
    print(f"  python dataset_generator.py --finfet --output_count 30 --input_dir {args.output_dir} --output_dir output")


if __name__ == '__main__':
    main()
