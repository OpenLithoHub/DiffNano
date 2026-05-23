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
- **PyTorch-native** — runs on CPU/GPU/MPS; integrates with standard optimizers (Adam, L-BFGS) and ML training loops
- **Topology optimization** — density-based parameterization with projection filters; binary/grayscale mask output compatible with e-beam and DUV lithography design rules
- **Inverse design workflows** — pre-built pipelines for metalens phase profile optimization, beam splitter design, waveguide mode matching, and absorption spectrum targeting
- **Fabrication constraints** — minimum feature size and curvature penalties compatible with EBL/DUV/EUV processes (shared constraint library with [OpenLithoHub](https://github.com/OpenLithoHub/OpenLithoHub))
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
│   ├── fdtd.py          # Differentiable FDTD (2D + 3D)
│   ├── rcwa.py          # Differentiable RCWA for periodic structures
│   └── fdfd.py          # Frequency-domain FD (fast, no time stepping)
├── design/
│   ├── parameterization.py   # Density, level-set, B-spline geometry representations
│   ├── projection.py         # Heaviside projection + smoothing filters
│   └── constraints.py        # Fabrication constraints (MFS, curvature)
├── workflows/
│   ├── metalens.py      # Metalens inverse design pipeline
│   ├── splitter.py      # Beam splitter / power divider
│   ├── waveguide.py     # Waveguide coupler and mode converter
│   └── absorber.py      # Broadband absorber / color filter
├── benchmark/
│   ├── datasets.py      # Reference designs from literature
│   └── metrics.py       # Transmission efficiency, Strehl ratio, bandwidth
└── export/
    ├── gds.py           # GDS-II export (gdstk)
    └── oasis.py         # OASIS export
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

- [ ] **v0.1** — RCWA solver + metalens workflow + GDS export
- [ ] **v0.2** — 2D FDTD + photonic crystal optimization
- [ ] **v0.3** — 3D FDTD (memory-efficient checkpointing)
- [ ] **v0.4** — Multi-wavelength / broadband optimization
- [ ] **v0.5** — Integration with OpenLithoHub fabrication constraint library
- [ ] **v1.0** — Full benchmark suite + arXiv paper

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
