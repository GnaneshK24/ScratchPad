"""
FinFET Dataset Validation & Testing Script

This script validates the generated FinFET SEM dataset and provides
comprehensive testing to ensure all requirements are met.

Usage:
    python validate_finfet_dataset.py --dataset_dir output
"""

import argparse
import csv
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import sys


class FinFETDatasetValidator:
    """Validate FinFET SEM dataset structure and content."""
    
    def __init__(self, dataset_dir: str):
        """
        Initialize validator.
        
        Args:
            dataset_dir: Path to dataset directory (containing reference/, search/, ground_truth.csv)
        """
        self.dataset_dir = Path(dataset_dir)
        self.results = {
            'structure': {},
            'images': {},
            'ground_truth': {},
            'validation': []
        }
    
    def validate_structure(self) -> bool:
        """Validate directory structure."""
        print("\n" + "="*70)
        print("1. STRUCTURE VALIDATION")
        print("="*70)
        
        required_dirs = ['reference', 'search']
        required_files = ['ground_truth.csv']
        
        all_valid = True
        
        # Check directories
        for dir_name in required_dirs:
            dir_path = self.dataset_dir / dir_name
            exists = dir_path.exists() and dir_path.is_dir()
            status = "✓" if exists else "✗"
            print(f"{status} Directory: {dir_name}/")
            self.results['structure'][dir_name] = exists
            all_valid = all_valid and exists
        
        # Check files
        for file_name in required_files:
            file_path = self.dataset_dir / file_name
            exists = file_path.exists() and file_path.is_file()
            status = "✓" if exists else "✗"
            print(f"{status} File: {file_name}")
            self.results['structure'][file_name] = exists
            all_valid = all_valid and exists
        
        return all_valid
    
    def validate_images(self) -> bool:
        """Validate reference and search images."""
        print("\n" + "="*70)
        print("2. IMAGE VALIDATION")
        print("="*70)
        
        all_valid = True
        
        ref_dir = self.dataset_dir / 'reference'
        search_dir = self.dataset_dir / 'search'
        
        # Get all reference images
        ref_files = sorted([f for f in ref_dir.glob('*.png')])
        search_files = sorted([f for f in search_dir.glob('*.png')])
        
        print(f"\nReference images: {len(ref_files)}")
        print(f"Search images: {len(search_files)}")
        
        # Check count matching
        if len(ref_files) != len(search_files):
            print("✗ ERROR: Reference and search image counts don't match!")
            all_valid = False
        else:
            print(f"✓ Count match: {len(ref_files)} pairs")
        
        # Check dimensions for subset (first 5)
        test_count = min(5, len(ref_files))
        print(f"\nTesting first {test_count} image pairs...")
        
        for i in range(test_count):
            # Reference
            ref_img = cv2.imread(str(ref_files[i]), cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                print(f"✗ Failed to load reference {i}: {ref_files[i].name}")
                all_valid = False
                continue
            
            ref_shape = ref_img.shape
            ref_valid = ref_shape == (1000, 1000)
            status = "✓" if ref_valid else "✗"
            print(f"{status} Reference {i}: shape={ref_shape}, dtype={ref_img.dtype}, range=[{ref_img.min()}, {ref_img.max()}]")
            
            if not ref_valid:
                all_valid = False
            
            # Search
            search_img = cv2.imread(str(search_files[i]), cv2.IMREAD_GRAYSCALE)
            if search_img is None:
                print(f"✗ Failed to load search {i}: {search_files[i].name}")
                all_valid = False
                continue
            
            search_shape = search_img.shape
            search_valid = search_shape == (1000, 1000)
            status = "✓" if search_valid else "✗"
            print(f"{status} Search {i}:    shape={search_shape}, dtype={search_img.dtype}, range=[{search_img.min()}, {search_img.max()}]")
            
            if not search_valid:
                all_valid = False
        
        self.results['images']['total_pairs'] = len(ref_files)
        self.results['images']['images_validated'] = test_count
        
        return all_valid
    
    def validate_ground_truth(self) -> bool:
        """Validate ground truth CSV file."""
        print("\n" + "="*70)
        print("3. GROUND TRUTH CSV VALIDATION")
        print("="*70)
        
        csv_path = self.dataset_dir / 'ground_truth.csv'
        
        if not csv_path.exists():
            print("✗ ERROR: ground_truth.csv not found")
            return False
        
        print(f"✓ CSV file exists: {csv_path.name}")
        
        # Read CSV
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            print(f"✗ ERROR reading CSV: {e}")
            return False
        
        print(f"✓ CSV parsed successfully")
        print(f"  Total pairs: {len(rows)}")
        
        # Check columns
        required_columns = ['pair_id', 'reference_file', 'search_file', 'center_x', 'center_y', 'noise_mode', 'rotation_angle']
        columns = rows[0].keys() if rows else []
        columns_valid = all(col in columns for col in required_columns)
        
        if columns_valid:
            print(f"✓ Required columns present: {', '.join(required_columns)}")
        else:
            print(f"✗ ERROR: Missing columns. Found: {list(columns)}")
            return False
        
        # Validate data
        errors = []
        noise_modes = {}
        center_xs = []
        center_ys = []
        rotations = []
        
        for i, row in enumerate(rows):
            try:
                pair_id = int(row['pair_id'])
                ref_file = row['reference_file']
                search_file = row['search_file']
                center_x = float(row['center_x'])
                center_y = float(row['center_y'])
                noise_mode = row['noise_mode']
                rotation_angle = float(row['rotation_angle'])
                
                # Check file existence
                ref_path = self.dataset_dir / 'reference' / ref_file
                search_path = self.dataset_dir / 'search' / search_file
                
                if not ref_path.exists():
                    errors.append(f"Row {i}: Reference file not found: {ref_file}")
                
                if not search_path.exists():
                    errors.append(f"Row {i}: Search file not found: {search_file}")
                
                # Validate ranges
                if not (0 <= center_x <= 1000):
                    errors.append(f"Row {i}: center_x out of range: {center_x}")
                if not (0 <= center_y <= 1000):
                    errors.append(f"Row {i}: center_y out of range: {center_y}")
                
                if noise_mode not in ['low', 'medium', 'high', 'clean', 'standard']:
                    errors.append(f"Row {i}: Invalid noise_mode: {noise_mode}")
                
                # Track statistics
                center_xs.append(center_x)
                center_ys.append(center_y)
                rotations.append(rotation_angle)
                noise_modes[noise_mode] = noise_modes.get(noise_mode, 0) + 1
                
            except Exception as e:
                errors.append(f"Row {i}: Parse error: {e}")
        
        if errors:
            print(f"\n✗ Found {len(errors)} errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
            return False
        else:
            print(f"✓ All rows validated (no errors)")
        
        # Statistics
        print(f"\nGround Truth Statistics:")
        if center_xs:
            print(f"  Center X: min={min(center_xs):.1f}, max={max(center_xs):.1f}, mean={np.mean(center_xs):.1f}, std={np.std(center_xs):.2f}")
            print(f"  Center Y: min={min(center_ys):.1f}, max={max(center_ys):.1f}, mean={np.mean(center_ys):.1f}, std={np.std(center_ys):.2f}")
            print(f"  Rotation: min={min(rotations):.2f}°, max={max(rotations):.2f}°, mean={np.mean(rotations):.2f}°, std={np.std(rotations):.2f}°")
        
        print(f"  Noise mode distribution: {noise_modes}")
        
        self.results['ground_truth']['total_rows'] = len(rows)
        self.results['ground_truth']['noise_distribution'] = noise_modes
        
        return True
    
    def validate_independence(self) -> bool:
        """Check that reference and search images are different."""
        print("\n" + "="*70)
        print("4. IMAGE INDEPENDENCE VALIDATION")
        print("="*70)
        
        ref_dir = self.dataset_dir / 'reference'
        search_dir = self.dataset_dir / 'search'
        
        ref_files = sorted([f for f in ref_dir.glob('*.png')])
        
        test_count = min(3, len(ref_files))
        all_different = True
        
        print(f"Comparing {test_count} reference/search pairs...\n")
        
        for i in range(test_count):
            ref_path = ref_files[i]
            search_name = ref_path.name.replace('ref_', 'search_')
            search_path = search_dir / search_name
            
            ref_img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
            search_img = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
            
            if ref_img is None or search_img is None:
                print(f"✗ Pair {i}: Could not load images")
                all_different = False
                continue
            
            # Calculate difference
            diff = cv2.absdiff(ref_img, search_img)
            diff_sum = diff.sum()
            diff_mean = diff.mean()
            
            is_different = diff_sum > 0
            status = "✓" if is_different else "✗"
            
            print(f"{status} Pair {i}: Difference sum={diff_sum}, mean={diff_mean:.2f}")
            
            if not is_different:
                print(f"  WARNING: Reference and search images are identical!")
                all_different = False
        
        return all_different
    
    def run_full_validation(self) -> bool:
        """Run all validation checks."""
        print("\n" + "#"*70)
        print("# FinFET SEM Dataset Validation Report")
        print("#"*70)
        print(f"Dataset directory: {self.dataset_dir}")
        
        checks = [
            ("Structure", self.validate_structure),
            ("Images", self.validate_images),
            ("Ground Truth", self.validate_ground_truth),
            ("Independence", self.validate_independence),
        ]
        
        results = {}
        for name, check_fn in checks:
            results[name] = check_fn()
        
        # Summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        all_passed = True
        for name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")
            all_passed = all_passed and passed
        
        print("\n" + "#"*70)
        if all_passed:
            print("# ✓ ALL VALIDATIONS PASSED - Dataset is ready for use!")
        else:
            print("# ✗ VALIDATION FAILED - Please check errors above")
        print("#"*70)
        
        return all_passed


def main():
    """Command-line interface for dataset validation."""
    parser = argparse.ArgumentParser(
        description='Validate FinFET SEM dataset structure and content'
    )
    parser.add_argument('--dataset_dir', type=str, default='output',
                       help='Path to dataset directory (default: output)')
    
    args = parser.parse_args()
    
    validator = FinFETDatasetValidator(args.dataset_dir)
    passed = validator.run_full_validation()
    
    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
