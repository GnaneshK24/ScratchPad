"""
DriftSense Synthetic Dataset Generator - FinFET SEM Image Pipeline

Generates synthetic image pairs (Reference + Search) for Navigation-Error Recovery training.
Supports both DRAM-style generated patterns and FinFET SEM images from base CAD layouts.

Features:
- FinFET pipeline: reads raw 10k x 10k CAD images, extracts 1000x1000 crops
- Three noise modes: LOW, MEDIUM, HIGH (with independent noise per image)
- Ground truth tracking with coordinate transformation after augmentation
- SEM-accurate edge glow and lithography-aware filters
- Modular architecture for extensibility

Usage:
    # DRAM-style generated patterns
    python dataset_generator.py --architecture dram --num_pairs 30 --output_dir data/train --seed 42
    
    # FinFET SEM images from base directory
    python dataset_generator.py --finfet --output_count 30 --input_dir finfet_base_images --output_dir output --seed 42
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import cv2
from tqdm import tqdm
import csv
from scipy import ndimage

from utils import (
    create_dram_grid, create_finfet_structure, add_edge_brightening,
    add_independent_noise, apply_blur, apply_rotation, apply_scaling,
    compute_center_coordinates
)


class FinFETSEMDatasetGenerator:
    """
    Generate FinFET SEM image pairs from base 10k x 10k CAD layout images.
    
    Pipeline:
    1. Load raw CAD images (10,000 x 10,000 pixels)
    2. Randomly extract 1000 x 1000 regions
    3. Create identical reference and search base copies
    4. Apply SEM filter pipeline (edge glow, lithography-accurate patterns)
    5. Apply independent, categorized noise modes (LOW/MEDIUM/HIGH)
    6. Track ground truth coordinates with transformation
    7. Save outputs with ground_truth.csv
    """
    
    def __init__(self, input_dir: str = 'finfet_base_images', seed: int = None,
                 noise_mode: str = 'random', noise_params: Dict = None):
        """
        Initialize FinFET SEM dataset generator.
        
        Args:
            input_dir: Directory containing 10k x 10k base CAD images
            seed: Random seed for reproducibility
            noise_mode: 'random', 'clean', 'low', 'medium', 'high', or 'standard'
            noise_params: Dictionary with noise parameters for each mode (overrides defaults)
                         Format: {'clean': {...}, 'low': {...}, 'medium': {...}, 'high': {...}}
        """
        self.input_dir = Path(input_dir)
        self.seed = seed
        self.noise_mode = self._normalize_noise_mode(noise_mode)
        self.noise_params = noise_params or {}
        
        if seed is not None:
            np.random.seed(seed)
        
        # Load available base images
        self.base_images = self._load_base_images()
        self.ground_truth_data = []

    def _normalize_noise_mode(self, noise_mode: str) -> str:
        """Normalize user-facing noise mode aliases."""
        if noise_mode is None:
            return 'random'
        value = str(noise_mode).lower()
        aliases = {
            'random': 'random',
            'clean': 'clean',
            'low': 'low',
            'medium': 'medium',
            'standard': 'medium',
            'normal': 'medium',
            'high': 'high'
        }
        return aliases.get(value, 'random')
    
    def _get_noise_param(self, mode: str, param_name: str, default_value):
        """
        Get noise parameter with fallback to default.
        
        Args:
            mode: Noise mode ('clean', 'low', 'medium', 'high')
            param_name: Parameter name
            default_value: Default value if not in noise_params
        
        Returns:
            Parameter value from noise_params or default
        """
        if mode in self.noise_params and param_name in self.noise_params[mode]:
            return self.noise_params[mode][param_name]
        return default_value
    
    def _load_base_images(self) -> List[np.ndarray]:
        """
        Load all 10k x 10k base CAD images from input directory.
        
        Returns:
            List of loaded images (grayscale)
        """
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        base_images = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
        
        image_files = [f for f in self.input_dir.glob('*') 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            raise FileNotFoundError(f"No image files found in {self.input_dir}")
        
        print(f"Loading {len(image_files)} base CAD images from {self.input_dir}...")
        
        for img_path in tqdm(image_files, desc="Loading base layouts", unit="layout", dynamic_ncols=True):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Validate dimensions
                if img.shape[0] < 1000 or img.shape[1] < 1000:
                    print(f"  ⚠ Skipping {img_path.name}: too small ({img.shape}), need ≥1000x1000")
                    continue
                base_images.append(img)
            else:
                print(f"  ✗ Failed to load {img_path.name}")
        
        print(f"✓ Loaded {len(base_images)} valid base images")
        return base_images
    
    def _crop_region(self, base_image: np.ndarray, crop_size: int = 1000) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Randomly extract a crop_size x crop_size region from a larger base image.
        
        Args:
            base_image: Input image (≥ 1000 x 1000)
            crop_size: Size of crop to extract (default: 1000)
        
        Returns:
            Tuple of (cropped_image, (crop_start_y, crop_start_x))
        """
        h, w = base_image.shape[:2]
        
        # Ensure crop is within bounds
        max_y = h - crop_size
        max_x = w - crop_size
        
        if max_y <= 0 or max_x <= 0:
            raise ValueError(f"Base image ({h}x{w}) too small for {crop_size}x{crop_size} crop")
        
        # Random crop location
        crop_y = np.random.randint(0, max_y + 1)
        crop_x = np.random.randint(0, max_x + 1)
        
        # Extract crop
        crop = base_image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
        
        return crop, (crop_y, crop_x)
    
    def _apply_lithography_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lithography-accurate pattern filter to simulate realistic CAD-to-physical transitions.
        
        Physical basis: Lithography masks create slightly rounded feature corners due to
        diffraction and resist-flow during development.
        
        Args:
            image: Input grayscale image
        
        Returns:
            Filtered image with rounded features
        """
        # Slight morphological smoothing to round corners
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        smoothed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Blend with original to preserve detail
        blended = cv2.addWeighted(image, 0.7, smoothed, 0.3, 0)
        
        # Round any remaining hard corners in the bright structure mask while leaving
        # the rest of the image largely untouched.
        threshold = max(10, int(np.percentile(blended, 85)))
        feature_mask = (blended > threshold).astype(np.uint8) * 255
        if feature_mask.mean() > 0:
            radius = max(3, min(blended.shape[:2]) // 150)
            radius = min(radius, 9)
            round_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
            rounded_mask = cv2.morphologyEx(feature_mask, cv2.MORPH_CLOSE, round_kernel, iterations=1)
            rounded_mask = cv2.morphologyEx(rounded_mask, cv2.MORPH_OPEN, round_kernel, iterations=1)
            rounded_mask = cv2.GaussianBlur(rounded_mask.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
            rounded_mask = np.clip(rounded_mask, 0, 255).astype(np.uint8)
            blended = cv2.addWeighted(blended.astype(np.float32), 0.85, rounded_mask.astype(np.float32), 0.15, 0).astype(np.uint8)
        
        return blended.astype(np.uint8)
    
    def _apply_edge_glow_sem(self, image: np.ndarray, strength: float = 0.25) -> np.ndarray:
        """
        Apply SEM edge-glow effect to simulate secondary electron emission at topological boundaries.
        
        Physical basis: In SEM, secondary electrons are preferentially emitted from edges and
        corners due to enhanced electric fields, creating bright halos around features.
        
        Reference: Goldstein et al. "Scanning Electron Microscopy and X-ray Microanalysis" (2017)
        
        Args:
            image: Input grayscale image
            strength: Glow strength factor (0-1, default: 0.25)
        
        Returns:
            Image with edge glow applied
        """
        image_float = image.astype(np.float32) / 255.0
        
        # Compute edge magnitude using Sobel operators
        edges_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
        edges_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
        edges_mag = np.sqrt(edges_x**2 + edges_y**2)
        edges_mag = edges_mag / (edges_mag.max() + 1e-8)
        
        # Apply small Gaussian blur to the edges to create halo effect
        edges_blurred = cv2.GaussianBlur(edges_mag, (5, 5), 1.0)
        
        # Add glow to original image
        glowed = image_float + strength * edges_blurred
        glowed = np.clip(glowed, 0, 1) * 255
        
        return glowed.astype(np.uint8)
    
    def _adjust_contrast_gamma(self, image: np.ndarray, gamma: float = 0.8) -> np.ndarray:
        """
        Adjust contrast and apply gamma correction to match typical SEM dynamic range.
        
        SEM images typically show lower gamma (brightened mid-tones) due to secondary
        electron detection characteristics.
        
        Args:
            image: Input grayscale image
            gamma: Gamma correction factor (default: 0.8, brightens image)
        
        Returns:
            Gamma-corrected image
        """
        image_float = image.astype(np.float32) / 255.0
        corrected = np.power(image_float, gamma)
        corrected = (corrected * 255).astype(np.uint8)
        
        # Enhance contrast slightly
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(corrected)
        
        return enhanced
    
    def _apply_realistic_sem_appearance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a more SEM-realistic appearance inspired by the target processing pipeline:
        - invert + mid-gray mapping
        - subtle low-frequency drift
        - edge glow + roughness
        - directional anisotropy
        - Poisson + Gaussian noise
        - final contrast compression

        This keeps the dataset contract intact while producing more realistic SEM-like images.
        """
        image_float = image.astype(np.float32) / 255.0

        # Invert and map to SEM-like mid-gray range
        image_float = 1.0 - image_float
        image_float = 0.3 + image_float * 0.45

        # Slight base blur to softens lithography edges
        image_float = cv2.GaussianBlur(image_float, (5, 5), 1.0)

        # Low-frequency background drift (SEM scan drift / charging)
        h, w = image_float.shape
        lf_noise = np.random.normal(0, 1, (h, w)).astype(np.float32)
        lf_noise = cv2.GaussianBlur(lf_noise, (0, 0), sigmaX=40, sigmaY=40)
        lf_noise = lf_noise / (np.max(np.abs(lf_noise)) + 1e-8)
        image_float = image_float + 0.05 * lf_noise

        # Edge detection and roughness for realistic SEM glow
        sobelx = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx ** 2 + sobely ** 2)
        edges = edges / (np.max(edges) + 1e-8)

        rough = np.random.normal(0, 1, (h, w)).astype(np.float32)
        rough = cv2.GaussianBlur(rough, (0, 0), sigmaX=2)
        rough = rough / (np.max(np.abs(rough)) + 1e-8)
        edges_rough = np.clip(edges + 0.25 * rough * edges, 0, 1)

        glow = cv2.GaussianBlur(edges_rough, (0, 0), sigmaX=1.5)
        glow += cv2.GaussianBlur(edges_rough, (0, 0), sigmaX=4.0)
        glow = glow / (np.max(glow) + 1e-8)
        image_float = image_float + 0.18 * glow

        # Directional anisotropy to mimic scan direction / astigmatism
        kernel = np.array([[0.05], [0.9], [0.05]], dtype=np.float32)
        image_float = cv2.filter2D(image_float, -1, kernel)

        # Add noise in SEM-like manner: Poisson + Gaussian
        image_float = np.clip(image_float, 0, 1)
        image_float = np.random.poisson(image_float * 255.0) / 255.0
        image_float = image_float + np.random.normal(0, 0.015, image_float.shape).astype(np.float32)

        # Final contrast compression
        image_float = np.clip(image_float, 0, 1)
        image_float = image_float ** 0.8

        return np.clip(image_float * 255.0, 0, 255).astype(np.uint8)

    def _apply_base_sem_filters(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the complete base SEM filter pipeline to both reference and search images.

        This combines the original lithography/edge logic with a more SEM-realistic appearance
        stage to better match the target look of real SEM micrographs.
        """
        # Step 1: Preserve lithography-style smoothing
        image = self._apply_lithography_filter(image)

        # Step 2: Add realistic scanning-emission response
        image = self._apply_realistic_sem_appearance(image)

        # Step 3: Final contrast tuning for consistency
        image = self._adjust_contrast_gamma(image, gamma=0.8)

        return image

    def _apply_charging_effect(self, image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """
        Apply background charging effect (intensity gradients) common in SEM.
        
        Physical basis: Beam charging causes non-uniform contrast, especially visible as
        intensity gradients across the field of view.
        
        Args:
            image: Input grayscale image
            intensity: Strength of charging effect (0-1)
        
        Returns:
            Image with charging effect
        """
        h, w = image.shape[:2]
        
        # Create radial gradient from random center
        cy = np.random.uniform(0.2, 0.8) * h
        cx = np.random.uniform(0.2, 0.8) * w
        
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        dist = dist / dist.max()
        
        # Apply vignetting-like charging gradient
        charging_gradient = 1 - intensity * dist
        
        image_float = image.astype(np.float32)
        charged = image_float * charging_gradient
        charged = np.clip(charged, 0, 255)
        
        return charged.astype(np.uint8)
    
    def _apply_astigmatism_blur(self, image: np.ndarray, direction: str = 'horizontal', 
                               strength: float = 1.0) -> np.ndarray:
        """
        Apply directional astigmatism blur (common SEM aberration).
        
        Physical basis: Astigmatism causes differential magnification along perpendicular axes,
        resulting in directional blur patterns.
        
        Args:
            image: Input grayscale image
            direction: 'horizontal', 'vertical', or 'diagonal'
            strength: Blur strength (default: 1.0)
        
        Returns:
            Blurred image
        """
        kernel_size = int(3 + 2 * strength)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        if direction == 'horizontal':
            # Horizontal motion blur kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, 3))
        elif direction == 'vertical':
            # Vertical motion blur kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_size))
        else:  # diagonal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        kernel = kernel.astype(np.float32) / kernel.sum()
        blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred.astype(np.uint8)
    
    def _apply_charging_streaks(self, image: np.ndarray, direction: str = 'horizontal',
                               num_streaks: int = 3) -> np.ndarray:
        """
        Apply beam charging streaks (horizontal/vertical banding).
        
        Physical basis: Raster scanning artifacts or charging-induced stripe patterns
        that appear as parallel bands in SEM images.
        
        Args:
            image: Input grayscale image
            direction: 'horizontal' or 'vertical'
            num_streaks: Number of streaks to add
        
        Returns:
            Image with streaks
        """
        h, w = image.shape[:2]
        streaked = image.astype(np.float32)
        
        if direction == 'horizontal':
            for _ in range(num_streaks):
                y_pos = np.random.randint(0, h - 10)
                streak_height = np.random.randint(5, 15)
                intensity = np.random.uniform(0.05, 0.15)
                streaked[y_pos:y_pos+streak_height, :] *= (1 - intensity)
        else:  # vertical
            for _ in range(num_streaks):
                x_pos = np.random.randint(0, w - 10)
                streak_width = np.random.randint(5, 15)
                intensity = np.random.uniform(0.05, 0.15)
                streaked[:, x_pos:x_pos+streak_width] *= (1 - intensity)
        
        streaked = np.clip(streaked, 0, 255)
        return streaked.astype(np.uint8)
    
    def _apply_vignetting(self, image: np.ndarray, strength: float = 0.2) -> np.ndarray:
        """
        Apply vignetting effect (darkening at edges).
        
        Physical basis: Beam aberrations and detector geometry cause reduced signal intensity
        at image periphery.
        
        Args:
            image: Input grayscale image
            strength: Vignetting strength (0-1)
        
        Returns:
            Vignette image
        """
        h, w = image.shape[:2]
        
        # Create circular vignette mask
        kernel_x = cv2.getGaussianKernel(w, w/3)
        kernel_y = cv2.getGaussianKernel(h, h/3)
        kernel = kernel_y @ kernel_x.T
        kernel = kernel / kernel.max()
        
        # Blend vignette strength
        vignetted = image.astype(np.float32) * (1 - strength + strength * kernel)
        vignetted = np.clip(vignetted, 0, 255)
        
        return vignetted.astype(np.uint8)
    
    def _round_feature_corners(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply a soft corner-rounding pass to crisp bright structures."""
        if image.ndim != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.shape[-1] == 3 else image

        image_float = image.astype(np.float32)
        threshold = max(10, int(np.percentile(image_float, 82)))
        feature_mask = (image_float > threshold).astype(np.uint8) * 255
        if feature_mask.mean() == 0:
            return image

        radius = max(3, min(image.shape[:2]) // max(120, int(150 / max(strength, 0.1))))
        radius = min(radius, 9)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        rounded = cv2.morphologyEx(feature_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        rounded = cv2.morphologyEx(rounded, cv2.MORPH_OPEN, kernel, iterations=1)
        rounded = cv2.GaussianBlur(rounded.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
        rounded = np.clip(rounded, 0, 255).astype(np.uint8)

        blended = cv2.addWeighted(image_float, 0.82, rounded.astype(np.float32), 0.18 * strength, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_noise_mode_clean(self, reference: np.ndarray, search: np.ndarray,
                               pair_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """Clean mode: minimal, lighter degradation on reference than search."""
        # Get parameters with defaults
        ref_gaussian = self._get_noise_param('clean', 'reference_gaussian', 0.018)
        search_gaussian = self._get_noise_param('clean', 'search_gaussian', 0.035)
        
        reference_noise = add_independent_noise(reference, noise_level=ref_gaussian, noise_type='gaussian')
        search_noise = add_independent_noise(search, noise_level=search_gaussian, noise_type='gaussian')
        return reference_noise, search_noise, 0.0

    def _apply_noise_mode_low(self, reference: np.ndarray, search: np.ndarray, 
                             pair_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        LOW mode:
        - slight Gaussian noise and Poisson noise
        - no rotation
        - reference is lighter than search
        """
        # Get parameters with defaults
        ref_gaussian = self._get_noise_param('low', 'reference_gaussian', 0.025)
        ref_poisson = self._get_noise_param('low', 'reference_poisson', 0.016)
        ref_brightness_scale = self._get_noise_param('low', 'reference_brightness_scale', 1.08)
        ref_brightness_offset = self._get_noise_param('low', 'reference_brightness_offset', 6)
        
        search_gaussian = self._get_noise_param('low', 'search_gaussian', 0.055)
        search_poisson = self._get_noise_param('low', 'search_poisson', 0.035)
        search_brightness_scale = self._get_noise_param('low', 'search_brightness_scale', 1.14)
        search_brightness_offset = self._get_noise_param('low', 'search_brightness_offset', 8)
        
        reference_noise = add_independent_noise(reference, noise_level=ref_gaussian, noise_type='gaussian')
        reference_noise = add_independent_noise(reference_noise, noise_level=ref_poisson, noise_type='poisson')
        reference_noise = np.clip(reference_noise.astype(np.float32) * ref_brightness_scale + ref_brightness_offset, 0, 255).astype(np.uint8)

        search_aug = add_independent_noise(search, noise_level=search_gaussian, noise_type='gaussian')
        search_aug = add_independent_noise(search_aug, noise_level=search_poisson, noise_type='poisson')
        search_aug = np.clip(search_aug.astype(np.float32) * search_brightness_scale + search_brightness_offset, 0, 255).astype(np.uint8)

        return reference_noise, search_aug, 0.0
    
    def _apply_noise_mode_medium(self, reference: np.ndarray, search: np.ndarray,
                                pair_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        MEDIUM mode:
        - blur + slight astigmatism
        - more realism, no rotation
        - reference degradation remains lighter than search
        """
        # Get parameters with defaults
        ref_blur_ksize = self._get_noise_param('medium', 'reference_blur_ksize', 3)
        ref_blur_sigma = self._get_noise_param('medium', 'reference_blur_sigma', 0.8)
        ref_astigmatism_direction = self._get_noise_param('medium', 'reference_astigmatism_direction', 'horizontal')
        ref_astigmatism_strength = self._get_noise_param('medium', 'reference_astigmatism_strength', 0.8)
        ref_gaussian = self._get_noise_param('medium', 'reference_gaussian', 0.06)
        ref_poisson = self._get_noise_param('medium', 'reference_poisson', 0.03)
        
        search_blur_ksize = self._get_noise_param('medium', 'search_blur_ksize', 5)
        search_blur_sigma = self._get_noise_param('medium', 'search_blur_sigma', 1.2)
        search_astigmatism_strength = self._get_noise_param('medium', 'search_astigmatism_strength', 1.2)
        search_gaussian = self._get_noise_param('medium', 'search_gaussian', 0.10)
        search_poisson = self._get_noise_param('medium', 'search_poisson', 0.06)
        
        reference_blur = cv2.GaussianBlur(reference, (ref_blur_ksize, ref_blur_ksize), sigmaX=ref_blur_sigma, sigmaY=ref_blur_sigma)
        reference_aug = self._apply_astigmatism_blur(reference_blur, direction=ref_astigmatism_direction, strength=ref_astigmatism_strength)
        reference_aug = add_independent_noise(reference_aug, noise_level=ref_gaussian, noise_type='gaussian')
        reference_aug = add_independent_noise(reference_aug, noise_level=ref_poisson, noise_type='poisson')

        search_aug = cv2.GaussianBlur(search, (search_blur_ksize, search_blur_ksize), sigmaX=search_blur_sigma, sigmaY=search_blur_sigma)
        blur_direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
        search_aug = self._apply_astigmatism_blur(search_aug, direction=blur_direction, strength=search_astigmatism_strength)
        search_aug = add_independent_noise(search_aug, noise_level=search_gaussian, noise_type='gaussian')
        search_aug = add_independent_noise(search_aug, noise_level=search_poisson, noise_type='poisson')

        return reference_aug, search_aug, 0.0
    
    def _apply_noise_mode_high(self, reference: np.ndarray, search: np.ndarray,
                              pair_idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        HIGH mode:
        - vignetting
        - heavier noise, blur, and astigmatism
        - controlled rotation not exceeding ±3 degrees
        - reference remains milder than search
        """
        # Get parameters with defaults
        ref_blur_ksize = self._get_noise_param('high', 'reference_blur_ksize', 5)
        ref_blur_sigma = self._get_noise_param('high', 'reference_blur_sigma', 1.0)
        ref_astigmatism_direction = self._get_noise_param('high', 'reference_astigmatism_direction', 'horizontal')
        ref_astigmatism_strength = self._get_noise_param('high', 'reference_astigmatism_strength', 1.1)
        ref_gaussian = self._get_noise_param('high', 'reference_gaussian', 0.08)
        ref_poisson = self._get_noise_param('high', 'reference_poisson', 0.05)
        ref_vignetting = self._get_noise_param('high', 'reference_vignetting', 0.08)
        
        search_blur_ksize = self._get_noise_param('high', 'search_blur_ksize', 7)
        search_blur_sigma = self._get_noise_param('high', 'search_blur_sigma', 2.0)
        search_astigmatism_strength = self._get_noise_param('high', 'search_astigmatism_strength', 2.2)
        search_gaussian = self._get_noise_param('high', 'search_gaussian', 0.18)
        search_poisson = self._get_noise_param('high', 'search_poisson', 0.12)
        search_vignetting = self._get_noise_param('high', 'search_vignetting', 0.22)
        rotation_range = self._get_noise_param('high', 'rotation_range', 3.0)
        
        reference_blur = cv2.GaussianBlur(reference, (ref_blur_ksize, ref_blur_ksize), sigmaX=ref_blur_sigma, sigmaY=ref_blur_sigma)
        reference_aug = self._apply_astigmatism_blur(reference_blur, direction=ref_astigmatism_direction, strength=ref_astigmatism_strength)
        reference_aug = add_independent_noise(reference_aug, noise_level=ref_gaussian, noise_type='gaussian')
        reference_aug = add_independent_noise(reference_aug, noise_level=ref_poisson, noise_type='poisson')
        reference_aug = self._apply_vignetting(reference_aug, strength=ref_vignetting)

        search_aug = cv2.GaussianBlur(search, (search_blur_ksize, search_blur_ksize), sigmaX=search_blur_sigma, sigmaY=search_blur_sigma)
        blur_direction = np.random.choice(['horizontal', 'vertical', 'diagonal'])
        search_aug = self._apply_astigmatism_blur(search_aug, direction=blur_direction, strength=search_astigmatism_strength)
        search_aug = add_independent_noise(search_aug, noise_level=search_gaussian, noise_type='gaussian')
        search_aug = add_independent_noise(search_aug, noise_level=search_poisson, noise_type='poisson')
        search_aug = self._apply_vignetting(search_aug, strength=search_vignetting)

        # Controlled rotation cap: +/- rotation_range degrees only on search image.
        rotation_angle = np.random.uniform(-rotation_range, rotation_range)
        if rotation_angle != 0:
            search_aug = apply_rotation_with_crop(search_aug, rotation_angle, crop_size=1000)

        return reference_aug, search_aug, rotation_angle
    
    def _transform_ground_truth(self, center_x: float, center_y: float, 
                               rotation_angle: float, image_size: int = 1000) -> Tuple[float, float]:
        """
        Transform ground truth center coordinates based on applied rotation.
        
        When the search image is rotated, the reference pattern's center location
        changes relative to the image coordinate system.
        
        Args:
            center_x: Original center X (initially 500 for 1000x1000 image)
            center_y: Original center Y (initially 500 for 1000x1000 image)
            rotation_angle: Rotation angle in degrees
            image_size: Image size (default: 1000)
        
        Returns:
            Tuple of transformed (center_x, center_y)
        """
        # Use OpenCV's matrix directly.  Image y increases downwards, so the
        # usual Cartesian sine signs are wrong here.  This must exactly match
        # ``apply_rotation_with_crop`` above.
        matrix = cv2.getRotationMatrix2D((image_size / 2.0, image_size / 2.0), rotation_angle, 1.0)
        point = np.array([[[center_x, center_y]]], dtype=np.float32)
        transformed = cv2.transform(point, matrix)[0, 0]
        return float(transformed[0]), float(transformed[1])

    @staticmethod
    def _resolve_official_center_rule(reference: np.ndarray, search: np.ndarray,
                                      rotation_angle: float = 0.) -> Tuple[float, float, int, float]:
        """Resolve the official FinFET tie rule on the final generated pair.

        The source crop remains the physical provenance of the reference.  If
        the generated pair has several equally valid structural occurrences,
        the challenge defines its label as the occurrence nearest the search
        centre.  This uses the same deterministic fused-map equivalence
        definition as localization diagnostics, without reading annotations or
        any stored ground truth.
        """
        from localization.center_rule import resolve_equivalent_peak
        from localization.classical_matcher import ClassicalSEMLocalizer, _representations, _variant

        matcher = ClassicalSEMLocalizer()
        template = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA) if reference.shape == (1000, 1000) else reference
        if rotation_angle:
            template = _variant(template, rotation_angle, 1.)
        search_repr = _representations(search, matcher.config)
        template_repr = _representations(template, matcher.config)
        fused = matcher._fused(search_repr, template_repr)
        peaks = matcher._peaks(fused, 100, matcher.config.nms_radius)
        equivalent, chosen = resolve_equivalent_peak(
            peaks,
            (search.shape[1] / 2., search.shape[0] / 2.),
            matcher.config.equivalent_score_tolerance,
        )
        return chosen['x'], chosen['y'], len(equivalent), float(peaks[0][2])
    
    def generate_image_pair(self, pair_idx: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a single FinFET SEM image pair with augmentation and ground truth.
        
        Args:
            pair_idx: Index of the pair being generated
        
        Returns:
            Tuple of (reference_image, search_image, metadata_dict)
        """
        # Step 1: Randomly select base image
        base_image = self.base_images[np.random.randint(0, len(self.base_images))]
        
        # Step 2: Reference uses a random 1000x1000 crop from the 10k base image.
        reference_base, (crop_y, crop_x) = self._crop_region(base_image, crop_size=1000)
        
        # Step 3: Search uses a fresh full-base-image pass, then downsampled to 1000x1000
        # before SEM-style filtering. This keeps the search as a lower-resolution whole-scene
        # representation rather than the exact same crop as the reference.
        search_base = cv2.resize(base_image, (1000, 1000), interpolation=cv2.INTER_AREA)
        
        # Step 4: Apply corner rounding before any SEM filtering so the underlying shape
        # geometry is softened before edge glow, noise, and blur are applied.
        reference_base = self._round_feature_corners(reference_base, strength=0.6)
        search_base = self._round_feature_corners(search_base, strength=0.9)

        # Step 5: Apply the SEM filter pipeline to both images.
        reference = self._apply_base_sem_filters(reference_base)
        search = self._apply_base_sem_filters(search_base)
        
        # Step 6: Apply categorized noise mode
        if self.noise_mode == 'random':
            noise_mode = np.random.choice(['low', 'medium', 'high'])
        else:
            noise_mode = self.noise_mode
        
        if noise_mode == 'clean':
            reference, search, rotation_angle = self._apply_noise_mode_clean(reference, search, pair_idx)
        elif noise_mode == 'low':
            reference, search, rotation_angle = self._apply_noise_mode_low(reference, search, pair_idx)
        elif noise_mode == 'medium':
            reference, search, rotation_angle = self._apply_noise_mode_medium(reference, search, pair_idx)
        else:  # high
            reference, search, rotation_angle = self._apply_noise_mode_high(reference, search, pair_idx)
        
        # Step 7: Resolve the official centre tie rule from the final output
        # images.  Independent noise can change a near-equivalence band, so
        # resolving before augmentation would not satisfy the saved-pair
        # contract.  This creates labels only for new pairs.
        official_center_x, official_center_y, equivalent_match_count, best_match_score = self._resolve_official_center_rule(
            reference, search, rotation_angle
        )

        # Step 8: Keep the reference crop provenance for auditing.  It is not
        # necessarily the official label in a periodic layout: the official
        # label is the centre-rule resolution computed above.
        # Calculate the centre of the reference crop in the downsampled search image.
        # Reference was cropped from (crop_y, crop_x) in the 10k base image.
        # The reference patch is 1000x1000, so its center in the base image is:
        # (crop_y + 500, crop_x + 500)
        # When the 10k image is downsampled to 1000x1000 (10:1 ratio), these coordinates map to:
        ref_center_in_base_y = crop_y + 500.0
        ref_center_in_base_x = crop_x + 500.0
        
        # Map from 10k coordinate space to 1000x1000 downsampled space
        # Scaling factor: 1000 / 10000 = 0.1
        initial_center_y = ref_center_in_base_y * 0.1
        initial_center_x = ref_center_in_base_x * 0.1
        
        # The centre-rule choice was made in final search-image coordinates,
        # including any high-noise rotation, so no second coordinate transform
        # is applied here.
        gt_center_x, gt_center_y = official_center_x, official_center_y
        
        # Step 6: Create metadata
        metadata = {
            'pair_id': pair_idx,
            'dataset_type': 'FinFET_SEM',
            'reference_shape': reference.shape,
            'search_shape': search.shape,
            'noise_mode': noise_mode,
            'rotation_angle': float(rotation_angle),
            'ground_truth_center': {
                'x': float(gt_center_x),
                'y': float(gt_center_y)
            },
            'initial_center': {
                'x': float(initial_center_x),
                'y': float(initial_center_y)
            },
            'center_rule_initial_center': {
                'x': float(official_center_x),
                'y': float(official_center_y)
            },
            'center_rule_equivalent_match_count': int(equivalent_match_count),
            'center_rule_best_match_score': float(best_match_score),
            'crop_location_in_base': {
                'y': int(crop_y),
                'x': int(crop_x),
                'base_image_size': 10000
            }
        }
        
        return reference, search, metadata
    
    def generate_dataset(self, output_count: int = 30, 
                        output_dir: str = 'output') -> None:
        """
        Generate complete FinFET SEM dataset of image pairs.
        
        Args:
            output_count: Number of image pairs to generate
            output_dir: Output directory for images and ground truth
        """
        output_path = Path(output_dir)
        reference_dir = output_path / 'reference'
        search_dir = output_path / 'search'
        
        reference_dir.mkdir(parents=True, exist_ok=True)
        search_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {output_count} FinFET SEM image pairs...")
        
        self.ground_truth_data = []
        
        for i in tqdm(range(output_count), desc="Generating FinFET pairs", unit="pair", dynamic_ncols=True):
            # Create image pair
            reference, search, metadata = self.generate_image_pair(i)
            
            # Save reference image
            ref_filename = f"ref_{i:03d}.png"
            ref_path = reference_dir / ref_filename
            cv2.imwrite(str(ref_path), reference)
            
            # Save search image
            search_filename = f"search_{i:03d}.png"
            search_path = search_dir / search_filename
            cv2.imwrite(str(search_path), search)
            
            # Prepare ground truth record
            gt_record = {
                'pair_id': i,
                'reference_file': ref_filename,
                'search_file': search_filename,
                'center_x': metadata['ground_truth_center']['x'],
                'center_y': metadata['ground_truth_center']['y'],
                'noise_mode': metadata['noise_mode'],
                'rotation_angle': metadata['rotation_angle']
            }
            
            self.ground_truth_data.append(gt_record)
        
        # Save ground truth CSV
        gt_csv_path = output_path / 'ground_truth.csv'
        self._save_ground_truth_csv(gt_csv_path)
        
        print(f"\n✓ FinFET SEM dataset generated successfully!")
        print(f"  - {output_count} image pairs")
        print(f"  - Reference images: {reference_dir}")
        print(f"  - Search images: {search_dir}")
        print(f"  - Ground truth: {gt_csv_path}")
        print(f"\nDataset Statistics:")
        self._print_ground_truth_stats()
    
    def _save_ground_truth_csv(self, output_path: Path) -> None:
        """Save ground truth coordinates to CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = ['pair_id', 'reference_file', 'search_file', 'center_x', 'center_y', 
                     'noise_mode', 'rotation_angle']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.ground_truth_data)
        
        print(f"  - Ground truth CSV: {output_path}")
    
    def _print_ground_truth_stats(self) -> None:
        """Print statistics about generated ground truth."""
        if not self.ground_truth_data:
            return
        
        center_xs = [d['center_x'] for d in self.ground_truth_data]
        center_ys = [d['center_y'] for d in self.ground_truth_data]
        rotations = [d['rotation_angle'] for d in self.ground_truth_data]
        
        print(f"  Center X: {np.mean(center_xs):.1f} ± {np.std(center_xs):.1f} (range: [{min(center_xs):.1f}, {max(center_xs):.1f}])")
        print(f"  Center Y: {np.mean(center_ys):.1f} ± {np.std(center_ys):.1f} (range: [{min(center_ys):.1f}, {max(center_ys):.1f}])")
        print(f"  Rotation: {np.mean(rotations):.2f}° ± {np.std(rotations):.2f}° (range: [{min(rotations):.2f}°, {max(rotations):.2f}°])")
        
        # Noise mode distribution
        noise_counts = {}
        for d in self.ground_truth_data:
            mode = d['noise_mode']
            noise_counts[mode] = noise_counts.get(mode, 0) + 1
        
        print(f"  Noise distribution: {noise_counts}")


def apply_rotation_with_crop(image: np.ndarray, angle: float, crop_size: int = 1000) -> np.ndarray:
    """
    Apply rotation and crop/pad to maintain exact output size.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        crop_size: Target output size (default: 1000)
    
    Returns:
        Rotated and cropped/padded image of size (crop_size, crop_size)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Rotate with reflection border
    rotated = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Ensure output is exactly crop_size x crop_size by cropping center
    if rotated.shape[0] >= crop_size and rotated.shape[1] >= crop_size:
        start_y = (rotated.shape[0] - crop_size) // 2
        start_x = (rotated.shape[1] - crop_size) // 2
        cropped = rotated[start_y:start_y+crop_size, start_x:start_x+crop_size]
    else:
        # Pad if necessary
        pad_top = max(0, (crop_size - rotated.shape[0]) // 2)
        pad_left = max(0, (crop_size - rotated.shape[1]) // 2)
        cropped = cv2.copyMakeBorder(rotated, pad_top, crop_size - rotated.shape[0] - pad_top,
                                     pad_left, crop_size - rotated.shape[1] - pad_left,
                                     cv2.BORDER_REFLECT)
    
    return cropped


class SyntheticDatasetGenerator:
    """Generate synthetic image pairs for wafer inspection training."""
    
    def __init__(self, architecture: str = 'dram', seed: int = None):
        """
        Initialize dataset generator.
        
        Args:
            architecture: 'dram' or 'finfet'
            seed: Random seed for reproducibility
        """
        self.architecture = architecture.lower()
        if self.architecture not in ['dram', 'finfet']:
            raise ValueError(f"Architecture must be 'dram' or 'finfet', got {architecture}")
        
        if seed is not None:
            np.random.seed(seed)
        
        self.annotations = {'pairs': []}
    
    def generate_reference_image(self, size: int = 100, **kwargs) -> np.ndarray:
        """
        Generate a reference image.
        
        Args:
            size: Image size (square, default: 100x100)
        
        Returns:
            Reference image (grayscale)
        """
        if self.architecture == 'dram':
            pitch = kwargs.get('pitch', 5)  # Finer pitch for reference
            reference = create_dram_grid(size, size, pitch=pitch)
        else:  # finfet
            fin_width = kwargs.get('fin_width', 2)
            fin_spacing = kwargs.get('fin_spacing', 4)
            reference = create_finfet_structure(size, size, fin_width=fin_width,
                                              fin_spacing=fin_spacing, gate_bars=1)
        
        # Add edge brightening to reference (lower noise level)
        reference = add_edge_brightening(reference, strength=0.4)
        reference = add_independent_noise(reference, noise_level=0.05)
        
        return reference
    
    def generate_search_image(self, size: int = 1000, 
                            downsampling_factor: int = 10,
                            reference_size: int = 100) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Generate a search image with reference pattern embedded at random location.
        
        Args:
            size: Search image size (1000x1000)
            downsampling_factor: Factor by which reference is shrunk in search (10x)
            reference_size: Size of reference image (100x100 by default)
        
        Returns:
            Tuple of (search_image, reference_center_coordinates)
        """
        # Create tiled base structure (larger, then downsample to ~10x reduction)
        base_size = size * downsampling_factor
        
        if self.architecture == 'dram':
            pitch = 50  # Larger pitch for base
            base_image = create_dram_grid(base_size, base_size, pitch=pitch)
        else:  # finfet
            fin_width = 20
            fin_spacing = 40
            base_image = create_finfet_structure(base_size, base_size,
                                               fin_width=fin_width,
                                               fin_spacing=fin_spacing, gate_bars=3)
        
        # Downsample to create 10x smaller version
        downsampled = cv2.resize(base_image, (size, size), interpolation=cv2.INTER_AREA)
        
        # Add degradations to search image (higher noise)
        search = add_edge_brightening(downsampled, strength=0.3)
        search = add_independent_noise(search, noise_level=0.15)  # Higher noise than reference
        
        # Apply realistic degradation variations
        blur_amount = np.random.uniform(0, 2)  # Slight blur
        if blur_amount > 0:
            search = apply_blur(search, kernel_size=3, sigma=blur_amount)
        
        rotation_angle = np.random.uniform(-2, 2)  # Small rotations
        if rotation_angle != 0:
            search = apply_rotation(search, rotation_angle)
        
        scale_var = np.random.uniform(0.98, 1.02)  # Small scaling variation
        if scale_var != 1.0:
            search = apply_scaling(search, scale_var)
        
        # Randomly place reference center within valid region
        ref_half = reference_size // 2
        margin = int(ref_half * 1.2)  # Ensure reference fits with margin
        
        center_x = np.random.uniform(margin, size - margin)
        center_y = np.random.uniform(margin, size - margin)
        
        return search, (center_x, center_y)
    
    def create_image_pair(self, pair_idx: int = 0, 
                         reference_size: int = 100,
                         search_size: int = 1000) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Create a complete image pair (reference + search with ground truth).
        
        Args:
            pair_idx: Pair index for naming
            reference_size: Reference image size (default: 100x100)
            search_size: Search image size (default: 1000x1000)
        
        Returns:
            Tuple of (reference_image, search_image, metadata_dict)
        """
        # Generate reference
        reference = self.generate_reference_image(size=reference_size)
        
        # Generate search with embedded reference
        search, gt_center = self.generate_search_image(
            size=search_size,
            downsampling_factor=10,
            reference_size=reference_size
        )
        
        # Create metadata
        metadata = {
            'pair_id': pair_idx,
            'architecture': self.architecture,
            'reference_shape': reference.shape,
            'search_shape': search.shape,
            'ground_truth_center': {
                'x': float(gt_center[0]),
                'y': float(gt_center[1])
            },
            'reference_size': reference_size,
            'search_size': search_size,
            'downsampling_factor': 10
        }
        
        return reference, search, metadata
    
    def generate_dataset(self, num_pairs: int = 30, 
                        output_dir: str = 'data/train',
                        reference_size: int = 100,
                        search_size: int = 1000) -> None:
        """
        Generate complete dataset of image pairs.
        
        Args:
            num_pairs: Number of image pairs to generate
            output_dir: Output directory for images and annotations
            reference_size: Reference image size
            search_size: Search image size
        """
        output_path = Path(output_dir)
        images_dir = output_path / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating {num_pairs} {self.architecture.upper()}-style image pairs...")
        
        self.annotations = {'pairs': []}
        
        for i in tqdm(range(num_pairs), desc=f"Generating {self.architecture.upper()} pairs", unit="pair", dynamic_ncols=True):
            # Create image pair
            reference, search, metadata = self.create_image_pair(
                pair_idx=i,
                reference_size=reference_size,
                search_size=search_size
            )
            
            # Save reference image
            ref_path = images_dir / f"reference_{i:03d}.png"
            cv2.imwrite(str(ref_path), reference)
            
            # Save search image
            search_path = images_dir / f"search_{i:03d}.png"
            cv2.imwrite(str(search_path), search)
            
            # Update metadata with file paths
            metadata['reference_path'] = str(ref_path.relative_to(output_path))
            metadata['search_path'] = str(search_path.relative_to(output_path))
            
            self.annotations['pairs'].append(metadata)
        
        # Save annotations
        annotations_path = output_path / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        
        print(f"\n✓ Dataset generated successfully!")
        print(f"  - {num_pairs} image pairs")
        print(f"  - Images saved to: {images_dir}")
        print(f"  - Annotations saved to: {annotations_path}")
        print(f"\nSample annotation:")
        print(json.dumps(self.annotations['pairs'][0], indent=2))


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic dataset for DriftSense Navigation-Error Recovery'
    )
    
    # FinFET-specific arguments
    parser.add_argument('--finfet', action='store_true',
                       help='Enable FinFET SEM pipeline (reads from finfet_base_images/)')
    parser.add_argument('--input_dir', type=str, default='finfet_base_images',
                       help='Input directory with 10k x 10k base CAD images (for --finfet mode)')
    parser.add_argument('--output_count', type=int, default=30,
                       help='Number of image pairs to generate (for --finfet mode)')
    
    # DRAM-specific arguments (and general)
    parser.add_argument('--architecture', type=str, default='dram', 
                       choices=['dram', 'finfet'],
                       help='Architecture style (default: dram, ignored with --finfet)')
    parser.add_argument('--num_pairs', type=int, default=30,
                       help='Number of image pairs to generate (for DRAM mode)')
    parser.add_argument('--output_dir', type=str, default='data/train',
                       help='Output directory (default: data/train)')
    parser.add_argument('--reference_size', type=int, default=100,
                       help='Reference image size in pixels (default: 100, DRAM mode)')
    parser.add_argument('--search_size', type=int, default=1000,
                       help='Search image size in pixels (default: 1000, DRAM mode)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (optional)')
    parser.add_argument('--noise_mode', type=str, default='random',
                       choices=['random', 'clean', 'low', 'medium', 'high', 'standard'],
                       help='Noise/augmentation profile: random, clean, low, medium, high, or standard (alias of medium)')
    
    args = parser.parse_args()
    
    # Compatibility rule:
    # If the user supplies a base-image directory with actual CAD/layout PNGs,
    # even the legacy DRAM flag should route to the FinFET SEM pipeline.
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    input_dir = Path(args.input_dir)
    has_base_images = input_dir.exists() and any(
        f.suffix.lower() in image_extensions for f in input_dir.iterdir()
    )
    use_layout_images = args.finfet or (args.architecture == 'dram' and has_base_images)
    
    # Route to appropriate generator
    if use_layout_images:
        # FinFET SEM pipeline (with DRAM compatibility alias support)
        print("="*70)
        print("DriftSense FinFET SEM Dataset Generator")
        if args.architecture == 'dram' and has_base_images:
            print("Compatibility mode: --architecture dram is using provided base layout images")
        print("="*70)
        print(f"Base layouts: {args.input_dir} ({'found' if has_base_images else 'not found'})")
        print(f"Output: {args.output_dir} | Pairs: {args.output_count} | Noise: {args.noise_mode}")
        
        try:
            generator = FinFETSEMDatasetGenerator(
                input_dir=args.input_dir,
                seed=args.seed,
                noise_mode=args.noise_mode
            )
            generator.generate_dataset(
                output_count=args.output_count,
                output_dir=args.output_dir
            )
        except FileNotFoundError as e:
            print(f"\n✗ Error: {e}")
            print(f"\nTo use base-layout generation, you must create the '{args.input_dir}/' directory")
            print("and place 10k x 10k base CAD images (PNG/JPG/TIFF) inside it.")
            return
    else:
        # Traditional DRAM-style generated patterns
        print("="*70)
        print("DriftSense DRAM-Style Dataset Generator")
        print("="*70)
        
        generator = SyntheticDatasetGenerator(
            architecture=args.architecture,
            seed=args.seed
        )
        generator.generate_dataset(
            num_pairs=args.num_pairs,
            output_dir=args.output_dir,
            reference_size=args.reference_size,
            search_size=args.search_size
        )


if __name__ == '__main__':
    main()
