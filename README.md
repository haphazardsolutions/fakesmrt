# fakesmrt: Lightweight Language Model Training Framework

A modular framework for training and deploying resource-efficient language models on modest hardware.

### Note

fakesmrt is being ported over from our blobjectdb platform, and at the moment, there's only one volunteer analyst working part-part-part-time on the task. It is likely that the codebase will not be functional for quite some time. We appreciate your patience.

### Nomenclature

*SSM*: Rather than training LLMs, Large Language Models, fakesmrt focuses on SSMs, Small Semantic Models. These one-megaparameter models are trained on English text converted to the bAIsic semantic language, allowing training data containing divergent vocabularies to contribute to compatible micromodels.

*bAIsic* and *MESH*: Each micromodel is attached to a bAIsic (Baseline Artificial Intelligence Semantic Interpretation Codex) MESH (Meaning-Embedded Semantic Hash) containing mappings of semantic roots to context-specific vocabulary. When blending micromodels, we also combine their attached MESH objects, allowing the blended logic to communicate through a larger matrix of conceptualized vocabulary.

*X*: To maintain brevity and clarity, we employ "X" as a parameter unit count. 1KX indicates one *kiloparameter* (a model with 1024 parameters), while 1MX indicates one *megaparameter* (a model with 1024 x 1024 = 1048576 parameters), and so on.

## Overview

fakesmrt enables training of small, efficient language models on consumer hardware through:
- Statistical bootstrapping of micromodels (1MX models)
- Model mixing and upscaling architectures
- Distributed training and model exchange
- Efficient data processing pipelines

### Key Features

- Train models with just 1M parameters (1MX models)
- Mix and evolve models through statistical bootstrapping
- Operate fully on CPU with minimal RAM requirements (4GB)
- Distribute training across peer networks
- Clean and preprocess training data efficiently

## Sub-Projects

### Data Cleanser
- Memory-efficient text corpus processor
- Robust encoding detection and handling
- Streaming architecture for large datasets
- Configurable chunk size and filtering

### 1MX Micromodel Trainer
- Trains tiny (1M parameter) language models
- Statistical bootstrapping for model evolution
- Optimized for consumer CPU hardware
- Mixed precision and gradient accumulation

### Model Mixer
- Combine multiple 1MX models statistically
- Weight analysis and pruning
- Adaptive mixing strategies
- Memory-efficient implementation

### bAIsic / MESH Upscaler
- Baseline Artificial Intelligence Semantic Interpretation Codex
- Meaning-Embedded Semantic Hash
- Context-specific model enhancement
- Lightweight attention mechanisms
- Dynamic model routing

### Distributed Architecture
- Peer-to-peer model exchange
- Distributed training coordination
- Model verification and trust
- Bandwidth-efficient protocols

### Training Database (Coming Soon)
- Relational storage for cleaned text chunks
- On-demand chunk serving
- Metadata and statistics tracking
- Efficient retrieval patterns

## Requirements

- Python 3.9+
- 4GB RAM minimum (8GB recommended)
- 100GB free disk space
- CPU with AVX2 support

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/fakesmrt.git
cd fakesmrt

# Set up environment
./setup.sh

# Run tests
python -m pytest tests/

# Start training a 1MX model
python src/train_1mx.py
```

## Architecture

The system uses a modular architecture where components can be used independently or combined:

```
Data Pipeline → 1MX Training → Model Mixing → Upscaling
       ↑              ↑             ↑            ↑
       └──────────────┴─────────────┴────────────┘
                Distributed Layer
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Status

- [x] Data pipeline
- [x] Basic training harness
- [x] 1MX model implementation
- [ ] Model mixing (in progress)
- [ ] Upscaler implementation
- [ ] Distributed architecture
- [ ] Training database

## License

MIT License. See [LICENSE](LICENSE) for details.
