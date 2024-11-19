"""
Configuration management utilities for fakesmrt
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    chunk_size: int
    buffer_size: int
    file_extensions: list[str]
    max_chunks: Optional[int]
    corpus_path: str

@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_seq_length: int

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    warmup_steps: int
    max_grad_norm: float
    gradient_accumulation_steps: int
    mixed_precision: bool

@dataclass
class SystemConfig:
    device: str
    num_workers: int
    dtype: str
    seed: int

@dataclass
class PathConfig:
    output_dir: str
    checkpoint_dir: str
    log_dir: str

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    system: SystemConfig
    paths: PathConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            system=SystemConfig(**config_dict['system']),
            paths=PathConfig(**config_dict['paths'])
        )

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file(s).
    
    Args:
        config_path: Optional path to config file. If None, loads default config.
    
    Returns:
        Config object containing all settings
    """
    # Load default config
    default_config_path = Path(__file__).parent.parent.parent / 'configs' / 'default.yaml'
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config not found at {default_config_path}")
    
    with open(default_config_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Override with user config if provided
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            user_config = yaml.safe_load(f)
            
        # Deep merge user config into default config
        deep_update(config_dict, user_config)
    
    return Config.from_dict(config_dict)

def deep_update(base_dict: dict, update_dict: dict) -> None:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
