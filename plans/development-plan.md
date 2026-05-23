# DiffNano — Development Plan

**Status:** Pre-implementation planning
**Created:** 2026-05-23
**Patent strategy:** China first-filing before any core algorithm push

---

## Patent Strategy

1. Implement core algorithms locally (do NOT push until CN filing)
2. Submit China invention patent application (locks priority date)
3. Push code to GitHub the same day or next day (open source)
4. File PCT within 12 months using CN filing as priority base

**Do NOT push the following until CN filing:**
- `diffnano/solvers/rcwa.py` — eigendecomposition differentiation (core claim)
- `diffnano/solvers/fdtd.py` — differentiable FDTD time-stepping
- Any code implementing the claims listed in Section 5 below

---

## v0.1 Milestone — RCWA Solver + Metalens Workflow

**Target:** 3-4 months
**Gate for CN patent filing**

### Core deliverables

- [ ] `diffnano/solvers/rcwa.py` — differentiable RCWA
  - S-matrix formulation for periodic multilayer structures
  - Eigendecomposition via `torch.linalg.eig` with custom backward
  - Handles degenerate eigenvalues (critical for high-symmetry structures)
  - Validated against S4 (Lu & White 2012) on grating test cases
- [ ] `diffnano/solvers/fdfd.py` — 2D frequency-domain FD (faster than FDTD for CW)
  - Sparse linear system solve with adjoint-mode gradient
  - PML absorbing boundaries
- [ ] `diffnano/design/parameterization.py`
  - Height map → phase profile (thin-element approximation)
  - Density → permittivity (projection filter, β-continuation)
- [ ] `diffnano/design/constraints.py`
  - Minimum feature size penalty (shared primitive with OpenLithoHub)
  - Binarization penalty (encourage 0/1 density)
- [ ] `diffnano/workflows/metalens.py`
  - Target phase profile generation (converging/diverging lens)
  - Phase matching loss + Strehl ratio evaluation
  - Adam + L-BFGS optimization loop
- [ ] `diffnano/export/gds.py` — GDS-II export (reuse OpenLithoHub gdstk wrapper)
- [ ] Validation: reproduce metalens design from Devlin et al. 2016 (Science)
- [ ] Benchmark: compare Strehl ratio vs Lumerical adjoint (reference from literature)

---

## v0.2 Milestone — 2D FDTD + Photonic Crystal

**Target:** 3-4 months after v0.1

- [ ] `diffnano/solvers/fdtd.py` — differentiable 2D FDTD
  - Yee grid, explicit time stepping, full autograd through all steps
  - Memory-efficient: gradient checkpointing over time steps
  - CPML boundaries
- [ ] `diffnano/workflows/phc.py` — photonic crystal bandgap optimization
  - PWE (plane wave expansion) for band structure
  - Topology optimization: maximize bandgap/midgap ratio
- [ ] Validation: reproduce Molesky et al. 2018 (Nature Photonics) inverse design results

---

## v0.3 Milestone — 3D FDTD

**Target:** 2-3 months after v0.2

- [ ] 3D FDTD with gradient checkpointing (memory: O(√T) instead of O(T))
- [ ] Distributed solve across multiple GPUs
- [ ] 3D metalens validation (NA > 0.9)

---

## v0.4+ — Broadband, Multi-physics, OpenLithoHub Integration

- [ ] Multi-wavelength optimization (broadband metalens)
- [ ] Thermal-optical co-design (interface with DiffCFD heat transfer)
- [ ] Shared fabrication constraint library with OpenLithoHub
- [ ] Leaderboard + benchmark dataset (metalens designs + target specs)

---

## Patent Claims (Draft — Pre-filing, Confidential)

### C1 — Differentiable RCWA with stable eigendecomposition backward
A method for computing exact gradients through rigorous coupled-wave analysis by differentiating the eigendecomposition of the transfer matrix with a numerically stable backward pass that handles near-degenerate eigenvalues, enabling gradient-based optimization of periodic nanophotonic structures without finite-difference approximation.

### C2 — Joint fabrication constraint + electromagnetic optimization
A co-optimization framework that simultaneously minimizes electromagnetic performance loss (phase matching error, Strehl ratio deficit) and fabrication constraint violation (minimum feature size, curvature) using a shared differentiable loss function, such that the optimized design is manufacturable by electron-beam or DUV lithography without post-processing.

### C3 — Differentiable FDTD with O(√T) memory checkpointing
A memory-efficient differentiable FDTD implementation that uses gradient checkpointing with O(√T) memory complexity (vs O(T) for naive autograd), enabling backpropagation through arbitrarily long FDTD simulations on GPU hardware with fixed memory budget.

### C4 — Shared constraint primitives between lithography and nanophotonics
A software architecture in which fabrication constraint functions (minimum critical dimension, curvature radius, corner rounding) are defined as a shared library used identically by computational lithography optimization (ILT/OPC) and nanophotonic inverse design, such that a design optimized with nanophotonic objectives is simultaneously constrained by lithographic process rules.

---

## Key References

- Devlin et al. (2016) — Broadband high-efficiency dielectric metasurfaces. *Science*
- Molesky et al. (2018) — Inverse design in nanophotonics. *Nature Photonics*
- Lu & White (2012) — Exploiting disorder in the design of photonic devices. *Nature Photonics* (S4 RCWA)
- Phan et al. (2020) — High-efficiency, large-area, topology-optimized metasurfaces. *Light: Science & Applications*
- Lin et al. (2019) — Topology-optimized dual-polarization Dirac cones. *Physical Review B*
- Pestourie et al. (2018) — Inverse design of large-area metasurfaces. *Optics Express*

---

## Competitive Landscape

| Tool | Differentiable | Open-source | GPU | Status |
|---|---|---|---|---|
| Lumerical FDTD | No (adjoint only) | No | Yes | Commercial |
| Meep | No | Yes | Partial | Active |
| S4 (RCWA) | No | Yes | No | Unmaintained |
| ceviche | Yes (FDFD only) | Yes | No | Unmaintained |
| Tidy3D | No | Cloud API | Yes | Commercial |
| **DiffNano** | **Yes (full autograd)** | **Yes** | **Yes** | **Building** |
