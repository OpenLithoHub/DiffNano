# DiffNano

> **Differentiable Nanophotonics Design in PyTorch**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

**DiffNano** is an open-source framework for inverse design of nanophotonic devices using fully differentiable electromagnetic simulation. It brings PyTorch autograd to the FDTD and RCWA solvers that nanophotonics researchers already use — enabling gradient-based optimization of metasurfaces, metalenses, photonic crystals, and waveguide components without writing a single adjoint equation by hand.

---

## Key Features

- **Differentiable FDTD** — 2D and 3D FDTD with CPML absorbing boundaries and full autograd support
- **Differentiable RCWA** — rigorous coupled-wave analysis for periodic structures (metasurfaces, gratings)
- **Differentiable FDFD** — frequency-domain solver for CW steady-state problems; GPU-native via dense linear algebra
- **PyTorch-native** — runs on CPU/GPU/MPS; integrates with standard optimizers (Adam, L-BFGS)
- **DFM-native design** — fabrication constraints in the autograd graph; lithography-aware optimization
- **Robust optimization** — process-variation-aware design via differentiable Monte Carlo with adaptive axial sampling
- **Neural surrogate** — CNN-accelerated RCWA for 10-50x optimization speedup
- **Curvilinear masks** — B-spline boundary parameterization with differentiable SDF rasterization
- **Multi-objective Pareto** — automated Pareto front discovery across objectives
- **Learned representation** — VAE-based latent space optimization
- **End-to-end pipeline** — optical specification to GDSII export

---

## Installation

```bash
pip install -e .
```

GPU support:
```bash
pip install -e ".[cuda]"   # CUDA 12+
pip install -e ".[mps]"    # Apple Silicon
```

---

## Quickstart

### Metalens Phase Profile Optimization

```python
import torch
from diffnano import MetalensDesigner

# Define target: converging lens at λ=532nm, NA=0.8, diameter=50μm
designer = MetalensDesigner(
    wavelength_nm=532.0,
    numerical_aperture=0.8,
    diameter_um=50.0,
    pixel_size_nm=200.0,
)

# Run optimization
height_map, loss_history = designer.optimize(n_steps=500, lr=1e-3)

# Export to GDS for fabrication
designer.export_gds("metalens_optimized.gds", height_map)
```

### Photonic Crystal Bandgap Optimization

```python
from diffnano import PhCDesigner

# Topology-optimize a photonic crystal for maximum bandgap
phc = PhCDesigner(lattice="hexagonal", n_air=1.0, n_material=3.5)
density, history = phc.maximize_bandgap(n_steps=200)
```

### Broadband Multi-Wavelength Optimization

```python
from diffnano import RCWASolver, BroadbandOptimizer

solver = RCWASolver(fourier_orders=10, wavelength_nm=532.0)
optimizer = BroadbandOptimizer(
    solver, wavelengths_nm=[500.0, 532.0, 600.0], grid_shape=(32, 32),
)
density, history = optimizer.optimize(n_steps=200)
```

---

## Architecture

```
diffnano/
├── solvers/
│   ├── fdtd2d.py        # Differentiable 2D FDTD (CPML, checkpointing)
│   ├── fdtd3d.py        # Differentiable 3D FDTD (CPML, checkpointing)
│   ├── rcwa.py          # Differentiable RCWA for periodic structures
│   ├── fdfd2d.py        # Frequency-domain FD (GPU-native, dense solve)
│   ├── litho.py         # Hopkins lithography model (Gaussian PSF)
│   ├── surrogate.py     # Neural surrogate (CNN-accelerated RCWA)
│   ├── fab_model.py     # Learned fabrication model (U-Net)
│   └── resist.py        # Differentiable resist model
├── design/
│   ├── parameterization.py  # Density, height map, B-spline representations
│   ├── projection.py        # Heaviside projection + beta-continuation
│   ├── curvilinear.py       # Curvilinear mask (fixed SDF rasterization)
│   ├── representation_learning.py  # VAE latent space optimization
│   ├── constraints_shared/  # Cross-domain DFM constraint primitives
│   └── robustness/
│       ├── core.py           # MC robust optimization
│       ├── adaptive.py       # Adaptive axial + curriculum
│       └── subspace.py       # Multi-axis correlated perturbations
├── workflows/
│   ├── metalens.py      # Metalens inverse design
│   ├── dfm_metalens.py  # DFM-native metalens (litho + optical joint)
│   ├── phc.py           # Photonic crystal bandgap optimization
│   ├── waveguide.py     # Waveguide bend / mode converter
│   ├── broadband.py     # Multi-wavelength optimization
│   ├── multi_objective.py  # Pareto front exploration
│   └── end_to_end.py    # Full specification-to-GDSII pipeline
├── benchmark/
│   ├── datasets.py      # Reference designs from literature
│   └── metrics.py       # Transmission, Strehl ratio, bandgap
└── export/
    └── gds.py           # GDS-II export (gdstk)
```

---

## Roadmap

- [x] **v0.1** — RCWA solver + metalens workflow + DFM-metalens + robust MC
- [x] **v0.2** — 2D FDTD + photonic crystal + waveguide + FDFD
- [x] **v0.3** — 3D FDTD + adaptive robust optimization + multi-axis perturbations
- [x] **v0.4** — Neural surrogate + broadband multi-wavelength optimization
- [x] **v0.5** — Learned fabrication model + differentiable resist + curvilinear masks
- [x] **v0.6** — Multi-objective Pareto + end-to-end pipeline + VAE representation
- [ ] **v1.0** — Full benchmark suite + arXiv paper + JOSS submission

---

## License

Apache 2.0
