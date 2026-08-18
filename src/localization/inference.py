"""Public API for the training-free classical SEM localizer."""
from pathlib import Path
import cv2
from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer

def localize(search_image, reference_image, config=None):
    if isinstance(search_image, (str, Path)): search_image = cv2.imread(str(search_image), cv2.IMREAD_GRAYSCALE)
    if isinstance(reference_image, (str, Path)): reference_image = cv2.imread(str(reference_image), cv2.IMREAD_GRAYSCALE)
    return ClassicalSEMLocalizer(config or ClassicalMatcherConfig()).localize(search_image, reference_image)
