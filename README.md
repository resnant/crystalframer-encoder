# CrystalFramer Encoder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
<!-- [![PyPI version](https://badge.fury.io/py/crystalframer-encoder.svg)](https://badge.fury.io/py/crystalframer-encoder) -->
[![GitHub](https://img.shields.io/badge/github-crystalframer--encoder-blue)](https://github.com/yourusername/crystalframer-encoder)

Crystal structure encoder based on pre-trained CrystalFramer models for materials property prediction and analysis.

📚 [日本語ドキュメントはこちら (Japanese Documentation)](README_ja.md)

## Table of Contents

- [Quick Start](#quick-start)
- [📦 Installation](#-installation)
- [🚀 Basic Usage](#-basic-usage)
- [💻 Command Line Interface](#-command-line-interface)
- [📚 Examples](#-examples)
- [📖 API Reference](#-api-reference)
- [📋 Requirements](#-requirements)
- [🔧 Troubleshooting](#-troubleshooting)
- [📝 Citation](#-citation)

## Quick Start

```bash
# Install from git
pip install git+https://github.com/YOUR_USERNAME/crystalframer-encoder.git                          
```
<!--todo: PyPI
pip install crystalframer-encoder -->

```python
import crystalframer_encoder as cfe
from pymatgen.core import Structure, Lattice

# Load pre-trained encoder
encoder = cfe.CrystalEncoder.from_pretrained("path/to/model.ckpt")

# Create crystal structure
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# Encode structure
embedding = encoder.encode_single(structure)
print(f"Embedding shape: {embedding.shape}")  # [1, 128]
```

## 📦 Installation

### Install from PyPI (Recommended)

```bash
pip install crystalframer-encoder
```

### Install from GitHub

```bash
pip install git+https://github.com/yourusername/crystalframer-encoder.git
```

### Development Installation

```bash
git clone https://github.com/yourusername/crystalframer-encoder.git
cd crystalframer-encoder
pip install -e .[dev]
```

### Using Docker

```bash
# Build Docker container
docker build -t crystalframer:latest ./docker

# Run Docker container
docker run --rm --gpus 1 -it -v $(pwd):/workspace -w /workspace crystalframer:latest bash

# Install package inside container
pip install git+https://github.com/yourusername/crystalframer-encoder.git
# Or for development
cd /workspace/crystalframer-encoder
pip install -e .
```

**Note**: CrystalFramer dependencies (PyTorch, PyTorch Geometric, etc.) must be properly installed. See [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for details.

## 🚀 Basic Usage

### 1. Using Pre-trained Models

```python
import crystalframer_encoder as cfe

# Method 1: Load from checkpoint file
encoder = cfe.CrystalEncoder.from_pretrained("/path/to/model.ckpt")

# Method 2: Specify hparams.yaml explicitly
encoder = cfe.CrystalEncoder.from_pretrained(
    "/path/to/model.ckpt", 
    hparams_path="/path/to/hparams.yaml"
)

# Method 3: Override embedding dimension
encoder = cfe.CrystalEncoder.from_pretrained(
    "/path/to/model.ckpt", 
    embedding_dim=256
)

# Create crystal structure
from pymatgen.core import Structure, Lattice
lattice = Lattice.cubic(4.0)
structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])

# Encode single structure
embedding = encoder.encode_single(structure)
print(f"Embedding shape: {embedding.shape}")  # [1, embedding_dim]

# Batch encode multiple structures
structures = [structure1, structure2, structure3, ...]
embeddings = encoder.encode(structures, batch_size=32, show_progress=True)
print(f"Embeddings shape: {embeddings.shape}")  # [num_structures, embedding_dim]
```

### 2. Obtaining Pre-trained Models

Pre-trained models can be obtained through:

1. **From CrystalFramer repository**: Use models trained with the original CrystalFramer
2. **Train your own**: Train on custom datasets using CrystalFramer

Example pre-trained models (JARVIS dataset):
- Formation energy prediction model
- Optical bandgap prediction model

**Note**: Large model files (.ckpt) are not included in the Git repository. They need to be downloaded separately or trained using CrystalFramer.

### 3. Hyperparameter Files

CrystalFramer Encoder automatically searches for hyperparameter files in:

- `hparams.yaml` in the same directory as the model file
- `hparams.yaml` in the parent directory of the model file
- `hyper_parameters` within the checkpoint

```python
# Directory structure example:
# model_directory/
# ├── best.ckpt
# └── hparams.yaml  # Automatically detected

encoder = cfe.CrystalEncoder.from_pretrained("/path/to/model_directory/best.ckpt")
```

### 4. Training from Scratch

```python
from crystalframer_encoder.utils.params import Params

# Load configuration from file
params = Params("/path/to/config.json")

# Create encoder
encoder = cfe.CrystalEncoder(params, embedding_dim=256)

# Use as regular PyTorch model
optimizer = torch.optim.Adam(encoder.parameters())
# ... training loop
```

## 💻 Command Line Interface

After installation, the `crystalframer-encode` command becomes available:

```bash
# Encode single structure
crystalframer-encode structure.cif -m model.ckpt -o embeddings.npz

# Encode multiple structures
crystalframer-encode *.cif -m model.ckpt -o embeddings.npz --batch-size 64

# Output results as JSON to stdout
crystalframer-encode structure.cif -m model.ckpt

# Show help
crystalframer-encode --help
```

## 📚 Examples

### Crystal Structure Similarity Analysis

```python
# Compute similarity between structures
similarity_matrix = encoder.compute_similarity(
    structures1, structures2, metric='cosine'
)

# Find most similar structures
query_structure = structures[0]
candidates = structures[1:]
similar_structures = encoder.find_similar_structures(
    query_structure, candidates, top_k=5, metric='cosine'
)

for idx, score in similar_structures:
    print(f"Structure {idx}: similarity = {score:.3f}")
```

### Using as Features for Machine Learning

```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Extract crystal features
crystal_features = encoder.encode(structures)

# Use with other ML models
model = RandomForestRegressor()
model.fit(crystal_features.numpy(), target_values)

# Make predictions
predictions = model.predict(crystal_features.numpy())
```

### Visualizing Crystal Structures

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Encode crystal structures
embeddings = encoder.encode(structures)

# Reduce dimensions with t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings.numpy())

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Crystal Structure Embeddings')
plt.show()
```

### Using with Custom Datasets

```python
from torch.utils.data import Dataset, DataLoader

class CrystalDataset(Dataset):
    def __init__(self, structures, labels, encoder):
        self.structures = structures
        self.labels = labels
        self.encoder = encoder
    
    def __getitem__(self, idx):
        structure = self.structures[idx]
        label = self.labels[idx]
        
        # Extract features with encoder
        embedding = self.encoder.encode_single(structure)
        
        return embedding.squeeze(0), label
    
    def __len__(self):
        return len(self.structures)

# Create dataset and dataloader
dataset = CrystalDataset(structures, labels, encoder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Saving and Loading Encoders

```python
# Save encoder
encoder.save_encoder("my_crystal_encoder.pth")

# Load saved encoder
encoder = cfe.CrystalEncoder.load_encoder("my_crystal_encoder.pth")
```

## 📖 API Reference

### CrystalEncoder

#### Initialization

```python
CrystalEncoder(
    params,                 # Params object
    embedding_dim=None      # Output embedding dimension (uses model_dim if None)
)
```

#### Class Methods

```python
CrystalEncoder.from_pretrained(
    model_path_or_name,     # Model file path or predefined model name
    hparams_path=None,      # Hyperparameter file path
    embedding_dim=None,     # Output embedding dimension
    device='auto',          # Device to use
    **kwargs                # Other parameters
)
```

#### Main Methods

- `encode(structures, batch_size=32, show_progress=False)`: Batch encode multiple structures
- `encode_single(structure)`: Encode single structure
- `compute_similarity(structures1, structures2=None, metric='cosine')`: Compute structure similarity
- `find_similar_structures(query, candidates, top_k=5, metric='cosine')`: Find similar structures
- `get_embedding_dim()`: Get embedding dimension
- `save_encoder(path)`: Save encoder
- `load_encoder(path, device='auto')`: Load encoder

### Params

```python
from crystalframer_encoder.utils.params import Params

# Load from JSON or YAML file
params = Params("/path/to/config.json")
params = Params("/path/to/config.yaml")

# Create from dictionary
params = Params.from_dict(config_dict)

# Get/set parameters
value = params.get('key', default_value)
params.set('key', value)
```

## 📋 Requirements

- Python >= 3.8
- PyTorch >= 1.12.0
- PyTorch Geometric >= 2.0.0
- pymatgen >= 2022.0.0
- PyYAML >= 5.4.0
- Other dependencies are automatically installed

## ⚠️ Important Notes

1. **Docker paths**: Always use paths starting with `/workspace/` in Docker containers
2. **Memory usage**: Be careful with memory usage when processing large structures or batches
3. **Device management**: Ensure proper CUDA environment for GPU usage
4. **Dependencies**: Verify that PyTorch Geometric and pymatgen are correctly installed

## 🔧 Troubleshooting

### Common Issues

1. **Path errors in Docker**
   ```
   FileNotFoundError: Model '/mnt/data/work/...' not found.
   ```
   → Use paths starting with `/workspace/` in Docker environment

2. **Import error**
   ```
   ModuleNotFoundError: No module named 'crystalframer_encoder'
   ```
   → Install the package: `pip install -e .`

3. **CUDA out of memory**
   → Reduce batch size or use CPU:
   ```python
   encoder = cfe.CrystalEncoder.from_pretrained(model_path, device='cpu')
   ```

4. **Hyperparameter file not found**
   ```
   Warning: hparams file not found, using default configuration
   ```
   → Explicitly specify file path with `hparams_path` parameter

5. **PyTorch Geometric import error**
   → Use CrystalFramer Docker image or install PyTorch Geometric correctly

## 🧪 Testing

To verify the package is working correctly:

```bash
# Run in Docker container
cd /workspace/crystalframer-encoder
python test_encoder.py  # Basic test script
python -m pytest tests/  # Unit tests (requires pytest)
```

## 📄 License

This code follows the same license as the CrystalFramer project (MIT License).

## 📝 Citation

If you use CrystalFramer Encoder in your research, please cite the original CrystalFramer paper:

```bibtex
@inproceedings{ito2025crystalframer,
  title     = {Rethinking the role of frames for SE(3)-invariant crystal structure modeling},
  author    = {Yusei Ito and 
               Tatsunori Taniai and
               Ryo Igarashi and
               Yoshitaka Ushiku and
               Kanta Ono},
  booktitle = {The Thirteenth International Conference on Learning Representations (ICLR 2025)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=gzxDjnvBDa}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💬 Support

For issues and questions, create an issue