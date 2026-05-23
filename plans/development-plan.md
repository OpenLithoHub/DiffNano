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

## Competitive Landscape Analysis (as of 2026-05)

### Direct competitors — know before you build

| Tool | Scope | Differentiable | Local/GPU | Stars | Last update | Threat |
|---|---|---|---|---|---|---|
| **tidy3d** (Flexcompute) | FDTD 3D, adjoint | Adjoint only (not autograd) | Cloud-first, limited local | 343 | 2026-05 active | Medium — cloud dependency, not PyTorch autograd |
| **MEEP** (MIT) | FDTD 2D/3D, adjoint | Adjoint only | CPU-first | ~1k | Active | Low — no ML loop integration |
| **grcwa** (Stanford) | RCWA 2D | Yes (autograd) | CPU | ~200 | Unmaintained ~2022 | Low — unmaintained, no GPU, no fabrication constraints |
| **ceviche** (Stanford) | FDFD 2D | Yes (autograd) | CPU | ~300 | Unmaintained ~2021 | Low — 2D only, unmaintained |
| **PhiFlow** (TU Munich) | Incompressible NS | Yes (multi-backend) | GPU | 1872 | 2026-05 active | N/A — fluid, not EM |

**Key insight**: No actively-maintained, fully open-source, GPU-accelerated, PyTorch-native differentiable EM solver exists. `grcwa` proved the concept (autograd through RCWA) but is abandoned and CPU-only. `tidy3d` is the quality benchmark but is cloud-first and adjoint-based (not full autograd). **The niche is real and open.**

### Patent freedom-to-operate analysis

The following are **NOT patented** (confirmed open literature / no known patents):
- Autograd through RCWA S-matrix formulation — published in `grcwa` paper (Liu & Fan 2020, arXiv:2005.01481), open literature, not patented
- Differentiable FDFD via sparse matrix solve backward — published in `ceviche` (Hughes et al. 2019), not patented
- Topology optimization of photonic structures — Molesky et al. 2018 (Nature Photonics), academic prior art, not patented as method

**Your defensible novelty (not in prior art):**
- C1: Numerically stable autograd through RCWA eigendecomposition handling degenerate eigenvalues (grcwa does NOT handle this; it silently fails on high-symmetry structures)
- C2: Joint fabrication constraint + EM objective with shared differentiable loss (no prior work combines lithography MRC constraints with EM inverse design in one autograd graph)
- C3: O(√T) memory gradient checkpointing through FDTD time steps (checkpointing is known in ML, applying it specifically to FDTD time-stepping with proven memory complexity is novel as applied method)
- C4: Shared constraint primitive library between computational lithography (OpenLithoHub) and nanophotonic inverse design — the cross-domain shared constraint architecture is novel

**Patent risk to you:**
- Flexcompute/tidy3d: They hold patents on their FDTD solver hardware acceleration, but NOT on the algorithmic autograd method. Their adjoint method is standard (Lalau-Keraly 2013) and not patentable by them against you.
- Stanford (Jelena Vuckovic group, Shanhui Fan group): Multiple patents on photonic inverse design devices (structures), but NOT on differentiable solver methods. Device patents don't block your method patents.
- **Conclusion: Freedom to operate confirmed for your C1-C4 claims above.**

---

## v0.1 Milestone — RCWA Solver + Metalens Workflow

**Target:** 3-4 months | **Gate for CN patent filing**

### Core deliverables

- [ ] `diffnano/solvers/rcwa.py` — differentiable RCWA
  - S-matrix formulation for periodic multilayer structures (follow grcwa paper as prior art baseline, then exceed it)
  - Eigendecomposition via `torch.linalg.eig` with **custom stable backward** for near-degenerate eigenvalues (this is C1 — grcwa fails here)
  - Test: validate against S4 (reference RCWA, non-differentiable) on silicon grating benchmarks
  - Test: verify gradients on high-symmetry structures where grcwa silently fails
  - GPU: full `torch.Tensor` throughout, runs on CUDA/MPS/CPU

- [ ] `diffnano/design/parameterization.py`
  - Height map → phase profile (thin-element approximation, differentiable)
  - Density → permittivity (Heaviside projection, β-continuation)
  - B-spline curve → binary mask (differentiable rasterization via distance field)

- [ ] `diffnano/design/constraints.py` ← **reuse OpenLithoHub primitives** (C4)
  - Minimum feature size penalty: identical function signature to `openlithohub.benchmark.metrics.curvilinear_mrc_loss`
  - Curvature penalty: shared implementation
  - Binarization penalty: encourage 0/1 density field
  - This module is the bridge to OpenLithoHub — one import, both tools use it

- [ ] `diffnano/workflows/metalens.py`
  - Target phase profile generation for converging/diverging lens
  - Phase matching loss + Strehl ratio (differentiable, computed from RCWA output)
  - Standard optimization loop: Adam warm-up → L-BFGS fine-tuning
  - β-continuation schedule (start soft, progressively binarize)

- [ ] `diffnano/export/gds.py` — reuse OpenLithoHub GDS export
- [ ] Validation: reproduce Devlin et al. 2016 (Science) metalens phase profile
- [ ] Benchmark: Strehl ratio vs. grcwa (show stability improvement on degenerate cases)

---

## v0.2 Milestone — 2D FDTD + Photonic Crystal

**Target:** 3-4 months after v0.1

- [ ] `diffnano/solvers/fdtd2d.py` — differentiable 2D FDTD
  - Yee grid explicit time-stepping, full autograd through all steps
  - **Gradient checkpointing: O(√T) memory** (C3) — implement `torch.utils.checkpoint` over time-step blocks
  - CPML absorbing boundaries (differentiable PML parameter update)
  - Pulsed source (differentiable Gaussian pulse parameters)

- [ ] `diffnano/workflows/phc.py` — photonic crystal optimization
  - Band structure via plane wave expansion (differentiable)
  - Topology optimization: maximize bandgap/midgap ratio
  - Validation: reproduce Jensen & Sigmund 2004 bandgap maximization benchmark

- [ ] `diffnano/workflows/waveguide.py`
  - Waveguide bend / mode converter optimization
  - Mode overlap integral as differentiable figure of merit

- [ ] **2026 addition vs original plan**: `diffnano/solvers/fdfd2d.py` — 2D FDFD
  - Frequency-domain solve via sparse `torch.linalg.solve` backward (ceviche approach but GPU-native and maintained)
  - Faster than FDTD for CW steady-state; complement not replace
  - Validation: reproduce ceviche benchmark results (beam splitter, waveguide coupler)

---

## v0.3 Milestone — 3D FDTD + Fabrication-Aware Optimization

**Target:** 2-3 months after v0.2

- [ ] 3D FDTD — extend 2D solver
  - Memory: gradient checkpointing essential (3D FDTD naively requires O(N³·T) memory)
  - Multi-GPU via `torch.distributed` (shard spatial domain)
- [ ] **2026 addition**: Process variation robustness optimization
  - Monte Carlo over geometric perturbations (linewidth variation, corner rounding) in the optimization loop
  - This is an unsolved problem — no current tool does this differentiably
  - Differentiable: perturbations are added as differentiable noise to the density field
  - Result: designs that are robust to ±5nm fabrication variation at EBL/DUV
  - **This is a new patent claim C5** (see below)

---

## v0.4+ — Broadband, Neural Surrogate, OpenLithoHub Integration

- [ ] Multi-wavelength optimization (weighted sum of RCWA solves across λ)
- [ ] **2026 addition**: Neural surrogate accelerated optimization
  - Train a small CNN to predict RCWA output from geometry
  - Use surrogate for fast gradient steps, periodically correct with full RCWA
  - 10-50x optimization speedup with <1% accuracy loss
  - Interface: drop-in replacement for `RCWASolver` in workflow
- [ ] Shared leaderboard with OpenLithoHub (same benchmark harness)
- [ ] arXiv paper + JOSS submission at v0.4

---

## Patent Claims (Draft — Pre-filing, Confidential)

### C1 — Stable eigendecomposition backward for degenerate RCWA
A method for computing exact gradients through rigorous coupled-wave analysis by differentiating the eigendecomposition of the layer transfer matrix with a numerically stable backward pass that regularizes near-degenerate eigenvalue pairs, enabling correct gradient flow for high-symmetry photonic structures (square lattice, Γ-point, normal incidence) where naïve eigendecomposition differentiation produces NaN or infinite gradients.

**Prior art gap**: grcwa (Liu & Fan 2020) demonstrates autograd through RCWA but does not address degenerate eigenvalue stability. No known patent or publication describes the stable backward for photonic RCWA specifically.

### C2 — Joint fabrication constraint + electromagnetic optimization in unified autograd graph
A co-optimization framework that simultaneously minimizes electromagnetic performance loss (phase matching error, Strehl ratio deficit, transmission efficiency) and lithographic fabrication constraint violation (minimum critical dimension, curvature radius) within a single differentiable computational graph, such that gradients from both objectives are backpropagated jointly and the optimized design satisfies process design rules without post-processing geometric correction.

**Prior art gap**: Fabrication constraints in inverse design are universally applied as post-processing or as separate penalty terms computed outside the EM solver. No prior work puts them inside the same autograd graph as the EM objective.

### C3 — O(√T) memory gradient checkpointing for differentiable FDTD
A memory-efficient differentiable FDTD implementation using gradient checkpointing with O(√T) memory complexity, where T is the number of time steps, enabling backpropagation through arbitrarily long FDTD simulations on hardware with fixed memory, with proven memory-accuracy trade-off bounds for electromagnetic simulation.

**Prior art gap**: Gradient checkpointing is known in ML (Chen et al. 2016). Its application to FDTD with specific electromagnetic boundary condition handling and proven correctness bounds is novel as applied method.

### C4 — Cross-domain shared fabrication constraint library for lithography and nanophotonics
A software architecture in which minimum critical dimension, curvature radius, and corner rounding constraint functions are implemented as a shared library used identically by computational lithography optimization (ILT/OPC mask synthesis) and nanophotonic inverse design (metasurface, metalens), enforcing process-consistent design rules across both domains from a single parameterization.

### C5 (new) — Differentiable process variation robustness optimization
A method for optimizing nanophotonic device designs for fabrication robustness by incorporating differentiable geometric perturbations (linewidth variation, corner rounding, layer thickness variation) drawn from a process variation distribution into the electromagnetic optimization loop, such that the gradient of the figure of merit with respect to design parameters is computed under the expectation of the perturbation distribution, yielding designs that are locally optimal under process variation without requiring separate robustness analysis.

---

## Key References

- Liu & Fan (2020) — `grcwa`: differentiable RCWA. arXiv:2005.01481 ← **primary prior art for C1**
- Hughes et al. (2019) — `ceviche`: differentiable FDFD. ACS Photonics ← **prior art for FDFD approach**
- Devlin et al. (2016) — Broadband metasurface. Science ← **validation target**
- Molesky et al. (2018) — Inverse design in nanophotonics. Nature Photonics ← **background**
- Chen et al. (2016) — Gradient checkpointing (Training deep networks with O(√n) memory). arXiv ← **prior art for C3, shows C3 is novel application not method**
- Christiansen & Sigmund (2021) — Inverse design in photonics: from computational to experimental realizations. Advances in Physics

---

## Competitive Differentiation Summary

| Feature | DiffNano | tidy3d | grcwa | ceviche | MEEP |
|---|---|---|---|---|---|
| Full autograd (not just adjoint) | ✅ | ❌ (adjoint) | ✅ (CPU) | ✅ (CPU, 2D) | ❌ |
| GPU-native PyTorch | ✅ | Cloud | ❌ | ❌ | ❌ |
| Stable degenerate eigenvalue backward | ✅ (C1) | N/A | ❌ | N/A | N/A |
| Fabrication constraints in autograd graph | ✅ (C2) | ❌ | ❌ | ❌ | ❌ |
| O(√T) FDTD memory checkpointing | ✅ (C3) | N/A | N/A | N/A | N/A |
| Process variation robustness optimization | ✅ (C5) | ❌ | ❌ | ❌ | ❌ |
| OpenLithoHub constraint integration | ✅ (C4) | ❌ | ❌ | ❌ | ❌ |
| Actively maintained (2026) | ✅ | ✅ | ❌ | ❌ | ✅ |
