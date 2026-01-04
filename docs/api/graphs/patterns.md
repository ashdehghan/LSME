# Pattern Functions

Standalone functions for creating individual graph patterns.

## Usage

```python
from lsme.graphs import (
    build_random,
    build_barbell,
    build_web_pattern,
    build_star_pattern,
    build_dense_star,
    build_crossed_diamond,
    build_dynamic_star,
)

# Random (Erdos-Renyi)
G = build_random(n_nodes=50, edge_prob=0.1, seed=42)

# Barbell
G = build_barbell(m1=10, m2=10)

# Web pattern
G = build_web_pattern(n_rings=3, spokes=6)

# Star pattern
G = build_star_pattern(n_arms=5, arm_length=3)
```

## API Reference

::: lsme.graphs.build_random
    options:
      show_root_heading: true

::: lsme.graphs.build_barbell
    options:
      show_root_heading: true

::: lsme.graphs.build_web_pattern
    options:
      show_root_heading: true

::: lsme.graphs.build_star_pattern
    options:
      show_root_heading: true

::: lsme.graphs.build_dense_star
    options:
      show_root_heading: true

::: lsme.graphs.build_crossed_diamond
    options:
      show_root_heading: true

::: lsme.graphs.build_dynamic_star
    options:
      show_root_heading: true
