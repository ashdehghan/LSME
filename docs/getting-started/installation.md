# Installation

## Requirements

- Python 3.9 or higher
- NumPy >= 1.20.0
- NetworkX >= 2.5
- PyTorch >= 2.0.0

## Install from PyPI

```bash
pip install lsme
```

## Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/elmspace/LSME.git
cd LSME
pip install -e .
```

### Using UV (Recommended)

```bash
git clone https://github.com/elmspace/LSME.git
cd LSME
uv pip install -e .
```

## Optional Dependencies

### For Development

Install with test dependencies:

```bash
pip install -e ".[dev]"
```

This includes:

- pytest
- pytest-cov

### For Documentation

Install with documentation dependencies:

```bash
pip install -e ".[docs]"
```

This includes:

- mkdocs
- mkdocs-material
- mkdocstrings

## Verify Installation

```python
import lsme
print(lsme.__version__)  # Should print version number

from lsme import LSME
print(LSME.available_methods())  # ['stochastic', 'deterministic', 'random_walk', 'eigenvalue']
```

## GPU Support

LSME automatically detects and uses CUDA if available for encoder training. To verify GPU support:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

To explicitly use CPU:

```python
from lsme import LSME
lsme = LSME(encoder_kwargs={'device': 'cpu'})
```
