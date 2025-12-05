# LSME: Local Structural Matrix Embeddings

This repository contains the code and manuscript for the LSME (Local Structural Matrix Embeddings) project.

## Repository Structure

```
LSME/
├── code/          # Python framework implementation
│   ├── lsme/      # Core library
│   └── README.md  # Code documentation
│
└── manuscript/    # Paper (ICML 2025 format)
    ├── main.tex   # Main LaTeX document
    └── sections/  # Paper sections
```

## Code

The `code/` directory contains the Python implementation of LSME for computing local structural embeddings on graphs.

See [code/README.md](code/README.md) for installation and usage instructions.

## Manuscript

The `manuscript/` directory contains the LaTeX source for the paper.

### Building the Paper

```bash
cd manuscript
make        # Build PDF
make clean  # Clean build artifacts
make view   # Open PDF (macOS)
```

## License

TBD
