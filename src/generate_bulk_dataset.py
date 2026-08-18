"""
Bulk Dataset Generator for FinFET SEM Image Pairs

Generates large-scale datasets with specific noise mode distributions.
Highly configurable - modify CONFIG at the top for quick parameter changes.

Usage:
    python generate_bulk_dataset.py
"""

import argparse
from pathlib import Path
from typing import Dict, List
import sys

from dataset_generator import FinFETSEMDatasetGenerator

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES FOR QUICK CUSTOMIZATION
# ============================================================================

CONFIG = {
    # Base images input directory
    "base_images_dir": "finfet_base_images",
    
    # Noise mode configuration: {noise_mode: num_pairs}
    # Easy to modify - just change the numbers!
    "noise_modes": {
        "clean": 100,      # Minimal noise - pristine conditions
        "low": 100,        # Slight noise - realistic SEM
        "medium": 100,     # Blur + astigmatism - harder
        "high": 100,       # Full degradation - extreme difficulty
    },
    
    # Output directory configuration
    "output_base_dir": "output_bulk_dataset",
    "output_subdir_pattern": "output_{noise_mode}",
    
    # Random seed for reproducibility (set to None for random)
    "seed": 42,
    
    # Image parameters
    "crop_size": 1000,
    
    # Verbosity
    "verbose": True,
    
    # ========================================================================
    # NOISE PARAMETERS - Tune these to customize each noise mode
    # ========================================================================
    "noise_parameters": {
        
        # CLEAN MODE: Minimal degradation
        "clean": {
            "reference_gaussian": 0.018,      # Gaussian noise level for reference
            "search_gaussian": 0.035,         # Gaussian noise level for search
            # Note: Rotation always 0 for clean mode
        },
        
        # LOW MODE: Slight noise and artifacts
        "low": {
            "reference_gaussian": 0.025,      # Gaussian noise for reference
            "reference_poisson": 0.016,       # Poisson noise for reference
            "reference_brightness_scale": 1.08,  # Brightness multiplier
            "reference_brightness_offset": 6,    # Brightness offset
            
            "search_gaussian": 0.055,         # Gaussian noise for search
            "search_poisson": 0.035,          # Poisson noise for search
            "search_brightness_scale": 1.14,  # Brightness multiplier
            "search_brightness_offset": 8,    # Brightness offset
            # Note: Rotation always 0 for low mode
        },
        
        # MEDIUM MODE: Blur + astigmatism
        "medium": {
            # Reference parameters
            "reference_blur_ksize": 3,        # Blur kernel size (must be odd)
            "reference_blur_sigma": 0.8,      # Blur sigma
            "reference_astigmatism_direction": "horizontal",  # "horizontal", "vertical", or "diagonal"
            "reference_astigmatism_strength": 0.8,
            "reference_gaussian": 0.06,
            "reference_poisson": 0.03,
            
            # Search parameters
            "search_blur_ksize": 5,           # Blur kernel size (must be odd)
            "search_blur_sigma": 1.2,
            "search_astigmatism_strength": 1.2,  # Random direction chosen at runtime
            "search_gaussian": 0.10,
            "search_poisson": 0.06,
            # Note: Rotation always 0 for medium mode
        },
        
        # HIGH MODE: Full SEM degradation
        "high": {
            # Reference parameters
            "reference_blur_ksize": 5,
            "reference_blur_sigma": 1.0,
            "reference_astigmatism_direction": "horizontal",
            "reference_astigmatism_strength": 1.1,
            "reference_gaussian": 0.08,
            "reference_poisson": 0.05,
            "reference_vignetting": 0.08,    # Edge darkening strength (0-1)
            
            # Search parameters
            "search_blur_ksize": 7,
            "search_blur_sigma": 2.0,
            "search_astigmatism_strength": 2.2,  # Random direction chosen at runtime
            "search_gaussian": 0.18,
            "search_poisson": 0.12,
            "search_vignetting": 0.22,      # Heavy edge darkening
            "rotation_range": 3.0,           # Max rotation in degrees (±3°)
        },
    },
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


def generate_dataset_for_mode(
    base_images_dir: str,
    noise_mode: str,
    num_pairs: int,
    output_dir: str,
    noise_params: Dict = None,
    seed: int = None,
    verbose: bool = True,
) -> Dict:
    """
    Generate dataset for a single noise mode.
    
    Args:
        base_images_dir: Directory containing base CAD images
        noise_mode: 'clean', 'low', 'medium', or 'high'
        num_pairs: Number of image pairs to generate
        output_dir: Output directory for this noise mode
        noise_params: Dictionary with noise parameters for all modes
        seed: Random seed (None for random)
        verbose: Print progress information
    
    Returns:
        Dictionary with generation statistics
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Generating {num_pairs} pairs - NOISE MODE: {noise_mode.upper()}")
        print(f"{'='*70}")
    
    try:
        # Initialize generator with noise parameters
        generator = FinFETSEMDatasetGenerator(
            input_dir=base_images_dir,
            seed=seed,
            noise_mode=noise_mode,
            noise_params=noise_params or {}
        )
        
        # Generate dataset
        generator.generate_dataset(
            output_count=num_pairs,
            output_dir=output_dir
        )
        
        return {
            "noise_mode": noise_mode,
            "status": "success",
            "num_pairs": num_pairs,
            "output_dir": output_dir,
        }
    
    except Exception as e:
        if verbose:
            print(f"✗ Error generating {noise_mode} dataset: {e}")
        
        return {
            "noise_mode": noise_mode,
            "status": "failed",
            "num_pairs": num_pairs,
            "error": str(e),
        }


def main(config: Dict = None):
    """
    Generate bulk datasets with all noise modes.
    
    Args:
        config: Configuration dictionary (uses global CONFIG if None)
    """
    if config is None:
        config = CONFIG
    
    base_images_dir = config["base_images_dir"]
    noise_modes = config["noise_modes"]
    noise_parameters = config["noise_parameters"]
    output_base_dir = config["output_base_dir"]
    output_subdir_pattern = config["output_subdir_pattern"]
    seed = config["seed"]
    verbose = config["verbose"]
    
    # Validate base images directory
    base_path = Path(base_images_dir)
    if not base_path.exists():
        print(f"✗ Base images directory not found: {base_images_dir}")
        sys.exit(1)
    
    # Create output base directory
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n🔧 Configuration:")
        print(f"   Base images: {base_images_dir}")
        print(f"   Output base: {output_base_dir}")
        print(f"   Seed: {seed if seed else 'random'}")
        print(f"   Total pairs: {sum(noise_modes.values())}")
        print(f"\n📊 Noise mode breakdown:")
        for mode, count in noise_modes.items():
            print(f"   - {mode.upper()}: {count} pairs")
    
    # Generate datasets for each noise mode
    results = []
    total_pairs = 0
    successful_pairs = 0
    
    for noise_mode, num_pairs in noise_modes.items():
        # Construct output directory
        subdir_name = output_subdir_pattern.format(noise_mode=noise_mode)
        output_dir = output_base_path / subdir_name
        
        # Generate dataset with noise parameters
        result = generate_dataset_for_mode(
            base_images_dir=base_images_dir,
            noise_mode=noise_mode,
            num_pairs=num_pairs,
            output_dir=str(output_dir),
            noise_params=noise_parameters,
            seed=seed,
            verbose=verbose,
        )
        
        results.append(result)
        total_pairs += num_pairs
        
        if result["status"] == "success":
            successful_pairs += num_pairs
    
    # Print summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total pairs requested: {total_pairs}")
        print(f"Total pairs generated: {successful_pairs}")
        print(f"\nPer-mode results:")
        for result in results:
            mode = result["noise_mode"].upper()
            status = "✓" if result["status"] == "success" else "✗"
            num = result["num_pairs"]
            if result["status"] == "success":
                print(f"  {status} {mode:8s}: {num} pairs → {result['output_dir']}")
            else:
                print(f"  {status} {mode:8s}: Failed - {result.get('error', 'Unknown error')}")
        
        print(f"\n📁 All datasets saved to: {output_base_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bulk FinFET SEM datasets with configurable noise modes"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CONFIG["seed"],
        help="Random seed for reproducibility (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=CONFIG["output_base_dir"],
        help="Output base directory (default: %(default)s)"
    )
    parser.add_argument(
        "--base-images-dir",
        type=str,
        default=CONFIG["base_images_dir"],
        help="Input base images directory (default: %(default)s)"
    )
    
    args = parser.parse_args()
    
    # Update config with CLI arguments
    config = CONFIG.copy()
    config["seed"] = args.seed
    config["output_base_dir"] = args.output_dir
    config["base_images_dir"] = args.base_images_dir
    
    main(config)
