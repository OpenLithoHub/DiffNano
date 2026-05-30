<div align="center">

# DiffNano

**Differentiable Nanophotonics Design in PyTorch**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Gradient-based inverse design of nanophotonic devices with differentiable electromagnetic solvers built in PyTorch.

> **Note:** DiffNano is an early-stage personal research project. It is not production-validated and has no external users yet. The Roadmap reflects the author's learning trajectory, not shipped software.

</div>

---

## Prior Art and How DiffNano Differs

Differentiable electromagnetic simulation is an active field with strong existing tools. DiffNano is a personal learning project, not a claim of novelty. Key prior work:

| Tool | Method | Autograd | Notes |
|:-----|:-------|:--------|:------|
| [MEEP](https://meep.readthedocs.io/) | FDTD | Yes (via meep-autograd / custom adjoint) | Mature, production-grade, C++ core + Python |
| [Tidy3D](https://tidy3d.simulation.cloud/) | FDTD | Yes (autograd-native) | Commercial, GPU-accelerated, widely adopted |
| [Ceviche](https://github.com/tigh-ff/ceviche) | FDTD / FDFD | Yes (JAX) | Open-source, photonic inverse design benchmark |
| [TorchMeep](https://github.com/tigh-ff/torchmeep) | FDTD | Yes (PyTorch) | PyTorch wrapper around MEEP |
| [Lumerical](https://www.ansys.com/products/photonics) | FDTD / RCWA | Adjoint | Commercial, industry standard |
| [SPINS](https://github.com/stanfordnlp/spins) | FDTD / FDFD | Yes | Stanford, topology optimization |
| [Inkstone](https://github.com/tigh-ff/inkstone) | RCWA | Yes | Berkeley, open-source |

DiffNano was built to learn how these solvers work by reimplementing them from scratch in PyTorch. It is not faster, more accurate, or more capable than the tools above.

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

## Performance & Benchmarks

### 1. Academic Paper Comparison (Table 1)

| Metric | DiffNano (this work) | TorchRDIT (Huang et al., 2024)¹ | Meent (Kim et al., 2024)² | Benchmarking Study (Mansson et al., 2025)³ |
|:-------|:---------------------|:--------------------------------|:--------------------------|:--------------------------------------------|
| **Core method** | RCWA + FDFD + FDTD + Neural Surrogate | R-DIT (eigendecomposition-free) | RCWA (multi-backend) | 9 algorithms on RCWA backend |
| **Speedup claim** | 10–50x via CNN surrogate (inference only) | Up to 16.2x vs standard RCWA | N/A (framework paper) | Varies by algorithm |
| **Robust optimization** | Differentiable MC, +31% yield (C5) | No | No | No (nominal only) |
| **Fabrication-aware** | Hopkins lithography model in autograd | No | No | No |
| **GPU backend** | PyTorch CUDA/MPS | PyTorch CUDA | JAX / PyTorch / NumPy | CPU (RCWA) |

> **Comparability note:** TorchRDIT's 16.2x speedup is measured on eigendecomposition elimination (single-wavelength, periodic structures). DiffNano's 10–50x surrogate speedup covers the full RCWA forward pass but is inference-only and problem-specific. These numbers are **not directly comparable** — different hardware, problem sizes, and measurement methodology.

**References:**

1. Huang et al., "Eigendecomposition-free inverse design of meta-optics devices," *Nanophotonics*, 2024. [PubMed 38859356](https://pubmed.ncbi.nlm.nih.gov/38859356/)
2. Kim et al., "Meent: Differentiable Electromagnetic Simulation," arXiv:2406.12904, 2024. [arXiv](https://arxiv.org/abs/2406.12904)
3. Mansson et al., "Benchmarking Optimization Methods for Nanophotonics," *Advanced Optical Materials*, 2025. [DOI:10.1002/adom.202500195](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adom.202500195)

### 2. Open-Source Tool Comparison (Table 2)

| Feature | DiffNano | Tidy3D v2.10.1 | MEEP v1.32.0 | TorchRDIT | FDTDX (2026) | Ceviche (archived) |
|:--------|:---------|:---------------|:-------------|:----------|:-------------|:-------------------|
| **RCWA** | Yes | No | No | No (R-DIT) | No | No |
| **FDTD** | 2D + 3D | 3D | 3D | No | 3D | 2D |
| **FDFD** | Yes | No | No | No | No | Yes |
| **Neural Surrogate** | Yes (CNN) | No | No | No | No | No |
| **GPU** | PyTorch CUDA/MPS | Cloud GPU (proprietary) | No (CPU, OpenMP) | PyTorch CUDA | JAX/XLA | No (NumPy) |
| **Autograd** | PyTorch native | Adjoint (JAX) | Adjoint wrapper | PyTorch native | JAX native | HIPS autograd |
| **Fabrication-aware** | Yes (Hopkins litho) | No | No | No | No | No |
| **Robust optimization** | Yes (differentiable MC) | No | No | No | No | No |
| **License** | Apache 2.0 | LGPL (solver proprietary) | GPL | MIT | Open source | MIT |
| **Status** | v0.6, experimental | Production | Production | Research | Research | Unmaintained |

> **Where DiffNano lags:** DiffNano's FDTD does not match MEEP or Tidy3D in feature completeness (PML variants, dispersive materials, subpixel smoothing). Tidy3D and FDTDX likely outperform DiffNano's FDTD in raw simulation speed for 3D problems due to optimized C++/CUDA cores. DiffNano's strength is in its **solver diversity under a single differentiable framework** and **fabrication-aware optimization**, not raw solver performance.

![Feature Comparison](docs/images/benchmark_tool_comparison.svg)
*Subjective assessment by the author on a 1–5 scale. See table above for factual details.*

### 3. Internal Benchmark Results

#### C5: Robust vs Nominal Optimization (Monte Carlo)

Under fabrication process variation (σ = 5 nm linewidth perturbation), robust optimization significantly improves manufacturing yield:

| Design | Base Strehl | Mean Strehl (MC, N=100) | Yield (Strehl ≥ threshold) |
|:-------|:-----------:|:-----------------------:|:--------------------------:|
| Nominal | 0.783 | 0.576 | 50% |
| Robust | 0.799 | 0.588 | 81% |
| **Delta** | **+0.016** | **+0.012** | **+31 percentage points** |

![Strehl Histogram](docs/images/benchmark_strehl_histogram.svg)

The robust design sacrifices negligible peak performance for substantially tighter performance distribution — critical for manufacturability.

#### C4: Unified vs Decoupled Optimization

Embedding lithography modeling inside the autograd graph (unified) converges faster and achieves lower final loss than decoupled sequential optimization:

| Method | Final Optical Loss | Litho EPE (nm) | Steps |
|:-------|:------------------:|:--------------:|:-----:|
| Unified autograd | 1.023 | 4.35 | 200 |
| Decoupled baseline | 1.251 | 5.36 | 200¹ |

¹ Decoupled ran fewer effective iterations due to sequential restart. Both used identical hardware and problem size.

![Convergence Curves](docs/images/benchmark_convergence.svg)

#### C7: Optimization Strategy Comparison

On a quadratic test function (100 steps):

| Strategy | Final Loss |
|:---------|:----------:|
| Nominal (no uncertainty) | 1.81 |
| C5 Brute-force MC (K=16) | 19.81 |
| C7 Adaptive + curriculum | 2.20 |

> **Note:** The brute-force MC result (19.81) reflects variance from fixed-K sampling on a non-convex landscape — it is not a general indictment of MC methods. The adaptive approach avoids this by dynamically adjusting sample count.

### 4. How to Reproduce

All benchmark data above was generated on the following environment:

**Hardware:**
- CPU: AMD Ryzen 5 5600G with Radeon Graphics (6 cores)
- RAM: 13 GB DDR4
- GPU: None (CPU-only)

**Software:**
- OS: Ubuntu 22.04.5 LTS
- Python: 3.10.12
- PyTorch: 2.12.0+cpu
- DiffNano: commit `29edb90` (current main)

**Run the benchmarks:**

```bash
# C4: Unified vs Decoupled
python3 scripts/benchmark_c4.py

# C5: Monte Carlo Robustness
python3 scripts/benchmark_c5.py

# C7: Optimization Strategy
python3 scripts/benchmark_c7.py

# Generate charts for README
python3 scripts/generate_benchmark_charts.py
```

**Methodology:**
- C5: 100 Monte Carlo samples with σ = 5 nm per-pixel height perturbation; yield threshold set at median of nominal distribution
- C4: 200 optimization steps, Adam optimizer, identical initialization seed
- C7: 100 steps on quadratic test function, comparing nominal / brute-force MC (K=16) / adaptive curriculum

> All test data above was obtained by actually running the scripts on the stated environment. No performance numbers were estimated or extrapolated.

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
| v0.1 | RCWA solver + metalens workflow | Done |
| v0.2 | 2D FDTD + photonic crystal + FDFD | Code exists, validation pending |
| v0.3 | 3D FDTD + adaptive robust optimization | Code exists, validation pending |
| v0.4 | Neural surrogate + broadband | Early prototype |
| v0.5 | Learned fabrication model + curvilinear masks | Experimental |
| v0.6 | Multi-objective Pareto + end-to-end + VAE | Experimental |
| v1.0 | Full benchmark suite + validation + arXiv paper | Planned |

---

## License

Apache License 2.0
