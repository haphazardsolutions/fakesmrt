"""
Test configuration management functionality
"""
import pytest
from pathlib import Path
import tempfile
import yaml
from src.utils.config import load_config, Config

def test_load_default_config():
    """Test loading default configuration"""
    config = load_config()
    assert isinstance(config, Config)
    assert config.data.chunk_size == 512
    assert config.model.vocab_size == 8192
    assert config.training.batch_size == 32

def test_load_custom_config():
    """Test loading and merging custom configuration"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as tmp:
        # Write custom config
        custom_config = {
            'data': {
                'chunk_size': 256,
                'max_chunks': 1000
            },
            'model': {
                'hidden_size': 128
            }
        }
        yaml.dump(custom_config, tmp)
        tmp.flush()
        
        # Load config with custom overrides
        config = load_config(tmp.name)
        
        # Check overridden values
        assert config.data.chunk_size == 256
        assert config.data.max_chunks == 1000
        assert config.model.hidden_size == 128
        
        # Check non-overridden values remain default
        assert config.data.buffer_size == 1000
        assert config.model.vocab_size == 8192

def test_invalid_config_path():
    """Test handling of invalid config path"""
    with pytest.raises(FileNotFoundError):
        load_config('nonexistent.yaml')

def test_config_types():
    """Test that config values have correct types"""
    config = load_config()
    
    assert isinstance(config.data.chunk_size, int)
    assert isinstance(config.model.hidden_size, int)
    assert isinstance(config.training.learning_rate, float)
    assert isinstance(config.system.device, str)
    assert isinstance(config.training.mixed_precision, bool)

def test_paths_expansion():
    """Test that paths are properly handled"""
    config = load_config()
    
    assert isinstance(config.paths.output_dir, str)
    assert isinstance(config.paths.checkpoint_dir, str)
    assert isinstance(config.paths.log_dir, str)
