<p align="center">
  <img src="docs/assets/logo.png" alt="DiffNano" width="240" />
</p>

# DiffNano

> **Differentiable Nanophotonics Design in PyTorch**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)]()

**DiffNano** is an open-source framework for inverse design of nanophotonic devices using fully differentiable electromagnetic simulation. It brings PyTorch autograd to the FDTD and RCWA solvers that nanophotonics researchers already use — enabling gradient-based optimization of metasurfaces, metalenses, photonic crystals, and waveguide components without writing a single adjoint equation by hand.

> **Organization:** [OpenLithoHub](https://github.com/OpenLithoHub) — open-source computational photonics and lithography tools.

---

## Why DiffNano?

The nanophotonics design loop today looks like this:

```
guess geometry → run FDTD (Lumerical / Meep / CST) → evaluate figure of merit
    → manual gradient estimate or evolutionary search → repeat (days to weeks)
```

DiffNano replaces the manual gradient step with automatic differentiation:

```
guess geometry → differentiable FDTD/RCWA → figure of merit → loss.backward() → optimizer.step()
```

**The gap this fills:** Existing open-source solvers (Meep, S4, RETICOLO) are not differentiable. Proprietary tools (Lumerical, CST) are closed. Research frameworks with autograd support (Tidy3D, ceviche) are either cloud-only, incomplete, or unmaintained. DiffNano is the first PyTorch-native, GPU-accelerated, fully open-source differentiable EM solver designed for inverse design workflows.

---

## Key Features

- **Differentiable FDTD** — 2D and 3D FDTD with full autograd support; gradients flow through every time step
- **Differentiable RCWA** — rigorous coupled-wave analysis for periodic structures (metasurfaces, gratings); exact gradients via eigendecomposition differentation
- **Differentiable FDFD** — frequency-domain solver for CW steady-state problems; GPU-native via dense linear algebra
- **PyTorch-native** — runs on CPU/GPU/MPS; integrates with standard optimizers (Adam, L-BFGS) and ML training loops
- **DFM-native design** — fabrication constraints in the autograd graph; lithography-aware optimization (Hopkins model + learned fab model + differentiable resist)
- **Robust optimization** — process-variation-aware design via differentiable Monte Carlo with adaptive axial sampling (C5 + C7)
- **Neural surrogate** — CNN-accelerated RCWA for 10-50x optimization speedup with periodic full-solver correction
- **Curvilinear masks** — B-spline boundary parameterization with differentiable SDF rasterization (no NaN gradients)
- **Multi-objective Pareto** — automated Pareto front discovery across optical, fabrication, and robustness objectives
- **Learned representation** — VAE-based latent space optimization for faster design space exploration
- **End-to-end pipeline** — optical specification to GDSII export in a single differentiable pipeline
- **Benchmark suite** — standardized figures of merit and reference designs for fair comparison across methods

---

## Installation

```bash
pip install diffnano
```

GPU support (recommended):
```bash
pip install diffnano[cuda]   # CUDA 12+
pip install diffnano[mps]    # Apple Silicon
```

---

## Quickstart

### Metalens Phase Profile Optimization

```python
import torch
from diffnano import RCWASolver, MetalensDesigner

# Define target: converging lens at λ=532nm, NA=0.8, diameter=50μm
designer = MetalensDesigner(
    wavelength_nm=532.0,
    numerical_aperture=0.8,
    diameter_um=50.0,
    pixel_size_nm=200.0,   # compatible with DUV lithography
)

# Initialize height map (learnable parameter)
height_map = torch.rand(designer.grid_shape, requires_grad=True)

optimizer = torch.optim.Adam([height_map], lr=1e-3)

for step in range(500):
    loss = designer.phase_matching_loss(height_map)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 50 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")

# Export to GDS for fabrication
designer.export_gds("metalens_optimized.gds", height_map.detach())
```

### 2D Photonic Crystal Bandgap Optimization

```python
from diffnano import FDTDSolver2D, PhCDesigner

solver = FDTDSolver2D(
    grid_resolution=20,   # pixels per wavelength
    pml_layers=10,
    device="cuda",
)

# Topology-optimize a photonic crystal for maximum bandgap
phc = PhCDesigner(solver, lattice="hexagonal", n_air=1.0, n_material=3.5)
optimized_density = phc.maximize_bandgap(n_steps=1000)
phc.visualize(optimized_density)
```

---

## Architecture

```
diffnano/
├── solvers/
│   ├── fdtd2d.py        # Differentiable 2D FDTD
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
│       ├── core.py           # C5: MC robust optimization
│       ├── adaptive.py       # C7: Adaptive axial + curriculum
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

## Benchmarks

| Task | Method | Figure of Merit | Time (A100) |
|---|---|---|---|
| Metalens, NA=0.5, λ=532nm | DiffNano (RCWA) | Strehl ratio 0.87 | 12 min |
| Metalens, NA=0.5, λ=532nm | Adjoint (Lumerical) | Strehl ratio 0.84 | 4.2 h |
| Beam splitter 1×2, λ=1550nm | DiffNano (FDFD) | Efficiency 96.1% | 3 min |
| PhC bandgap maximization | DiffNano (FDTD) | Gap/midgap 42% | 28 min |

*Benchmarks are reproducible; see `notebooks/benchmarks/`.*

---

## Roadmap

- [x] **v0.1** — RCWA solver + metalens workflow + DFM-metalens + robust MC
- [x] **v0.2** — 2D FDTD + photonic crystal + waveguide + FDFD
- [x] **v0.3** — 3D FDTD + C7 adaptive robust optimization + multi-axis perturbations
- [x] **v0.4** — Neural surrogate + broadband multi-wavelength optimization
- [x] **v0.5** — Learned fabrication model + differentiable resist + curvilinear masks
- [x] **v0.6** — Multi-objective Pareto + end-to-end pipeline + VAE representation
- [ ] **v1.0** — Full benchmark suite + arXiv paper + JOSS submission

---

## Relation to OpenLithoHub

DiffNano and [OpenLithoHub](https://github.com/OpenLithoHub/OpenLithoHub) share a common design philosophy: bring differentiable optimization to physical patterning problems. They share:

- **Fabrication constraint primitives** — minimum feature size and curvature penalty functions
- **GDS/OASIS export utilities** — layout export with process-aware design rules
- **PDK definitions** — process node parameters used consistently across both tools

DiffNano focuses on the electromagnetic design problem (what pattern achieves the target optical response). OpenLithoHub focuses on the lithographic transfer problem (how to print that pattern accurately on silicon).

---

## Citation

If you use DiffNano in your research, please cite:

```bibtex
@software{diffnano2026,
  title   = {DiffNano: Differentiable Nanophotonics Design in PyTorch},
  author  = {OpenLithoHub Contributors},
  year    = {2026},
  url     = {https://github.com/OpenLithoHub/DiffNano},
  license = {Apache-2.0}
}
```

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). By contributing you agree to the [Contributor License Agreement](CLA-INDIVIDUAL.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
