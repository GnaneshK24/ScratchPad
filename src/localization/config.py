"""Shared geometry settings for the training-free classical localizer."""
from dataclasses import dataclass

@dataclass(frozen=True)
class LocalizationConfig:
    search_size: int = 1000
    reference_size: int = 100
