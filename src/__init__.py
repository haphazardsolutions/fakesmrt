# src/__init__.py
"""
fakesmrt: Lightweight Language Model Training Framework
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from .data_pipeline import CorpusStream, TextChunk
from .train import TrainingConfig, TrainingHarness

__all__ = [
    'CorpusStream',
    'TextChunk',
    'TrainingConfig',
    'TrainingHarness',
]

# tests/__init__.py
"""
fakesmrt test suite
"""
# This can be empty, as it's mainly to mark the directory as a Python package
# It allows pytest to properly import test modules

# src/data/__init__.py (if we create this subdirectory)
"""
fakesmrt data processing modules
"""
# Empty is fine - just marks it as a package directory
