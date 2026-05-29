<div align="center">

# DiffNano

**Differentiable Nanophotonics Design in PyTorch**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Bring PyTorch autograd to electromagnetic simulation — gradient-based inverse design of metasurfaces, metalenses, photonic crystals, and waveguide components **without writing a single adjoint equation by hand**.

</div>

---

## Solvers

| Solver | Type | Best For |
|:-------|:-----|:---------|
| **Differentiable FDTD** | 2D/3D time-domain with CPML | Broadband, transient, arbitrary geometries |
| **Differentiable RCWA** | Fourier-domain, periodic structures | Metasurfaces, gratings, metalenses |
| **Differentiable FDFD** | Frequency-domain, steady-state | CW problems, GPU-native dense solve |
| **Neural Surrogate** | CNN-accelerated RCWA | 10-50x optimization speedup |

All solvers are **PyTorch-native** — run on CPU/GPU/MPS, integrate with Adam, L-BFGS, and any PyTorch optimizer.

---

## Design Capabilities

- **Multiple parameterizations** — density maps, height profiles, B-spline curvilinear masks
- **Fabrication-aware** — lithography modeling (Hopkins), DFM constraints in the autograd graph
- **Robust optimization** — process-variation-aware via differentiable Monte Carlo
- **Multi-objective Pareto** — automated Pareto front discovery
- **Learned representation** — VAE latent space optimization
- **End-to-end** — optical specification to GDSII export

---

## Quick Start

### Metalens Optimization

```python
from diffnano import MetalensDesigner

designer = MetalensDesigner(
    wavelength_nm=532.0,
    numerical_aperture=0.8,
    diameter_um=50.0,
    pixel_size_nm=200.0,
)
height_map, loss_history = designer.optimize(n_steps=500)
designer.export_gds("metalens.gds", height_map)
```

### Photonic Crystal Bandgap

```python
from diffnano import PhCDesigner

phc = PhCDesigner(lattice="hexagonal", n_air=1.0, n_material=3.5)
density, history = phc.maximize_bandgap(n_steps=200)
```

### Broadband Multi-Wavelength

```python
from diffnano import RCWASolver, BroadbandOptimizer

solver = RCWASolver(fourier_orders=10, wavelength_nm=532.0)
optimizer = BroadbandOptimizer(
    solver, wavelengths_nm=[500.0, 532.0, 600.0], grid_shape=(32, 32),
)
density, history = optimizer.optimize(n_steps=200)
```

---

## Installation

```bash
# Core
pip install -e .

# GPU support
pip install -e ".[cuda]"   # CUDA 12+
pip install -e ".[mps]"    # Apple Silicon

# Development
pip install -e ".[dev]"
```

---

## Architecture

```
diffnano/
├── solvers/
│   ├── fdtd2d.py            # 2D FDTD (CPML, checkpointing)
│   ├── fdtd3d.py            # 3D FDTD
│   ├── rcwa.py              # RCWA for periodic structures
│   ├── fdfd2d.py            # Frequency-domain (GPU-native)
│   ├── litho.py             # Hopkins lithography model
│   ├── surrogate.py         # CNN-accelerated RCWA
│   ├── fab_model.py         # Learned fabrication model (U-Net)
│   └── resist.py            # Differentiable resist model
├── design/
│   ├── parameterization.py  # Density, height map, B-spline
│   ├── projection.py        # Heaviside + beta-continuation
│   ├── curvilinear.py       # B-spline SDF rasterization
│   ├── representation_learning.py  # VAE latent optimization
│   ├── constraints_shared/  # Cross-domain DFM primitives
│   └── robustness/          # MC robust optimization
├── workflows/
│   ├── metalens.py          # Metalens inverse design
│   ├── phc.py               # Photonic crystal bandgap
│   ├── waveguide.py         # Waveguide bends / converters
│   ├── broadband.py         # Multi-wavelength optimization
│   ├── multi_objective.py   # Pareto front exploration
│   └── end_to_end.py        # Spec-to-GDSII pipeline
├── benchmark/               # Reference designs & metrics
└── export/
    └── gds.py               # GDS-II export (gdstk)
```

---

## Roadmap

| Version | Scope | Status |
|:--------|:------|:-------|
| v0.1 | RCWA + metalens + DFM + robust MC | Done |
| v0.2 | 2D FDTD + photonic crystal + FDFD | Done |
| v0.3 | 3D FDTD + adaptive robust optimization | Done |
| v0.4 | Neural surrogate + broadband | Done |
| v0.5 | Learned fabrication model + curvilinear masks | Done |
| v0.6 | Multi-objective Pareto + end-to-end + VAE | Done |
| v1.0 | Full benchmark suite + arXiv paper | Planned |

---

## License

Apache License 2.0
