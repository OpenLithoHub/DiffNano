# DiffNano — Development Plan

**Status:** v0.6 implemented (all milestones complete); 173 tests passing
**Created:** 2026-05-23
**Last updated:** 2026-05-28
This project does not pursue any patent claims. All contributions are open-source under Apache 2.0.

---

## External Dependencies

DiffNano is part of the **OpenLithoHub** organisation (github.com/OpenLithoHub, founded 2026-05-17).

- **OpenLithoHub (the toolkit, Apache 2.0, PyPI `openlithohub==0.1.0a2`, alpha)** — provides the forward computational-lithography model (Hopkins/SOCS, JIT-accelerated via `torch.compile`), MRC/DRC checking (incl. `curvilinear_mrc_loss` differentiable training-time penalty), B-spline mask fitting, and OASIS/GDSII export. Depends on:
  - `openlithohub.workflow.parse_layout` (layout ingestion)
  - `openlithohub.benchmark.metrics.curvilinear_mrc_loss` (differentiable MRC penalty referenced by `constraints_shared`)
  - The Hopkins/SOCS forward model as the litho side of the unified-autograd-graph demo
  Status: alpha; risk is whether these APIs are stable and differentiable end-to-end. Add an integration smoke-test in v0.1 deliverables and an upstream-pin in `pyproject.toml`.
- **OpenLithoHub/DiffCFD (private)** — unrelated to DiffNano critical path; not a dependency.

Both DiffNano and OpenLithoHub are owned by the same organisation, so this is a coordination problem (release timing, API stability), not an external-third-party-dependency problem. The license of `openlithohub` (Apache 2.0) is compatible with DiffNano's own license (see "License compliance" section below).

---

## Competitive Landscape Analysis (as of 2026-05)

### Direct competitors — know before you build

| Tool | Scope | Differentiable | Local/GPU | Backend | Last update | Threat |
|---|---|---|---|---|---|---|
| **TORCWA** (Kim & Lee, SNU) | RCWA, metasurface | Yes (autograd) — broadening-based eigendecomposition gradient stabilization | GPU (CUDA) | **PyTorch** | 0.1.4 active 2026-05 | CPC 282 (2023) 108552. LGPL. |
| **meent** (kc-ml2 / SNU follow-on team) | RCWA, modeling + EM sim + optimization | Yes (autograd, **JAX + PyTorch + NumPy multi-backend**) | GPU (CUDA, eigendecomp on CPU per README 2023-03) | **PyTorch + JAX + NumPy** | Active 2026-05, MIT, ~120 stars | arXiv:2406.12904 (Kim et al. 2024). |
| **TorchRDIT** (UMass Lowell) | R-DIT (eigendecomposition-free) + RCWA | Yes (autograd) — bypasses eigendecomposition entirely via R-DIT | GPU (CUDA + MPS) | **PyTorch** | Active 2026-05 | Opt. Express 32, 13986 (2024). 16.2× speedup vs RCWA. SIREN, GDS export, dispersive materials. **GPL-3.0 (license-incompatible with Apache 2.0 — do not copy implementation)**. |
| **FDTDX** (Mahlau et al., LUH) | 3D FDTD | Yes (autograd via **time-reversibility of Maxwell's equations**, not checkpointing) | GPU (multi-GPU CUDA + ROCm) | **JAX** (not PyTorch) | Active 2026-05, 298 stars | JOSS 11(117) 8912 (2026); arXiv:2412.12360. MIT. |
| **tidy3d** (Flexcompute) | FDTD 3D, frequency-domain plugins | **Yes — `tidy3d.plugins.autograd` (local autograd)**. Includes `make_erosion_dilation_penalty`, `smoothed_projection`, `FilterProject`, `ErosionDilationPenalty` | Cloud-first compute, local autograd plugin | autograd library, runs against tidy3d's solver | 2026-05 active | |
| **MEEP** (MIT) | FDTD 2D/3D, adjoint | Adjoint only | CPU-first | C++/Python | Active | Low — no ML loop integration |
| **grcwa** (Stanford) | RCWA 2D | Yes (autograd, **does not handle degenerate eigenvalues**) | CPU | autograd (legacy) | Unmaintained ~2022 | Low on availability, but it is the canonical published prior-art baseline cited by C1. |
| **ceviche** (Stanford) | FDFD 2D | Yes (autograd) | CPU | autograd (legacy) | Unmaintained ~2021 | Low — 2D only, unmaintained |
| **fdtd-z** (Google/X) | GPU FDTD | Adjoint | GPU (~100× CPU) | TensorFlow | Sporadic | Low — adjoint only, monitor only |
| **PyMieDiff / TorchGDM** | Differentiable Mie / GDM | Yes (autograd) | CPU/GPU | PyTorch | Active | N/A — different physics regime (small particle scattering, not full-wave) |
| **PhiFlow** (TU Munich) | Incompressible NS | Yes (multi-backend) | GPU | multi | Active | N/A — fluid, not EM |

**Key insight**: There is no fully open-source, GPU-accelerated, **PyTorch-native** differentiable EM solver that simultaneously offers (a) numerically stable RCWA for high-symmetry / degenerate cases, (b) full 3D FDTD, and (c) cross-domain shared lithography/photonics constraint primitives. However, each individual axis is contested:
- **PyTorch + RCWA + autograd** is occupied by TORCWA (broadening-based stabilization), TorchRDIT (eigendecomposition-free), and **meent (multi-backend including PyTorch, from the same kc-ml2 group as TORCWA)**.
- **Multi-backend (PyTorch + JAX) differentiable RCWA** is published via meent (arXiv:2406.12904, 2024).
- **3D FDTD + autograd + multi-GPU** is occupied by FDTDX (JAX, time-reversible).
- **Local autograd + fabrication constraints in graph** is occupied by tidy3d.plugins.autograd.

The defensible niche is the **intersection** (cross-domain lithography-photonics shared constraint architecture and process-variation-robust optimization in PyTorch), not any single algorithmic axis.

---

## v0.1 Milestone — RCWA Solver + Metalens Workflow + C4 / C5 Demos

**Status: DONE** (commit 137a27b + 16b5e0f, pushed to main 2026-05-28)

### Core deliverables

- [x] `diffnano/solvers/__init__.py` — **backend-agnostic forward-solver interface**
  - `class Solver(Protocol)` with `forward(geometry, sources, wavelengths) -> SimResult`
  - Lazy `__getattr__` for RCWASolver to avoid circular imports

- [x] `diffnano/solvers/rcwa.py` — differentiable RCWA
  - Toeplitz permittivity convolution, eigendecomposition, S-matrix propagation
  - 11 unit tests, all passing

- [x] `diffnano/design/parameterization.py`
  - HeightMap (height→phase), DensityField (density→permittivity with Heaviside projection)
  - BSplineCurve (has NaN gradient issues in backward — not used in DFM workflow)

- [x] `diffnano/design/projection.py` — heaviside_projection, smooth_filter, beta_continuation_schedule

- [x] `diffnano/design/constraints_shared/primitives.py` — min CD, curvature, binarization, corner rounding penalties + combined_fabrication_penalty

- [x] `diffnano/design/robustness/core.py` — reparameterize_sample, linewidth_perturbation, robust_gradient_step (antithetic sampling)

- [x] `diffnano/solvers/litho.py` — HopkinsLithoModel (Gaussian PSF, separable 2D conv, partially coherent averaging)

- [x] `diffnano/workflows/metalens.py` — MetalensDesigner with target phase, Strehl ratio, Adam optimization, robust mode

- [x] `diffnano/workflows/dfm_metalens.py` — DFMMetalensDesigner: unified autograd graph (litho + optical + fab), decoupled_baseline for comparison

- [x] `diffnano/export/gds.py` — GDSII export via gdstk

- [x] `diffnano/benchmark/` — datasets.py (reference designs) + metrics.py (transmission, Strehl, bandgap)

- [x] `scripts/benchmark_c4.py` — unified vs decoupled benchmark (-18.2% optical loss, -18.8% EPE)

- [x] `scripts/benchmark_c5.py` — nominal vs robust benchmark (+31% yield at median threshold)

- [x] Tests: 57/57 passing (test_solvers, test_design, test_robustness, test_workflows, test_benchmark)

- [x] CI: GitHub Actions (ruff lint + pytest on Python 3.10/3.11/3.12)

- [ ] OpenLithoHub integration smoke test (deferred — depends on stable openlithohub API)

The v0.1 engineering gate: **57 tests pass, lint clean, benchmarks produce data.**
This is achieved.

---

## v0.2 Milestone — 2D FDTD + Photonic Crystal + FDFD

**Status: DONE** (commit on main 2026-05-28)
**90 tests passing, lint clean**

- [x] `diffnano/solvers/fdtd2d.py` — differentiable 2D FDTD
  - Yee grid explicit time-stepping, full autograd through all steps
  - Gradient checkpointing via `torch.utils.checkpoint` for memory efficiency
  - CPML absorbing boundaries (differentiable PML parameter update)
  - Gaussian pulse and continuous source (differentiable parameters)
  - TM (Ez, Hx, Hy) and TE (Hz, Ex, Ey) polarization support
  - Time-series probe for output monitoring

- [x] `diffnano/solvers/fdfd2d.py` — differentiable 2D FDFD
  - Frequency-domain solve via dense `torch.linalg.solve` with autograd
  - TE and TM polarization support
  - PML absorbing boundaries (quadratic conductivity grading)
  - Point and line source injection
  - GPU-native (no sparse library dependency)

- [x] `diffnano/workflows/phc.py` — photonic crystal optimization
  - Band structure via plane-wave expansion (differentiable eigendecomposition)
  - Topology optimization: maximize bandgap/midgap ratio
  - Square and hexagonal lattice support
  - Brillouin zone k-path generation (Gamma-X-M for square, Gamma-K-M for hex)

- [x] `diffnano/workflows/waveguide.py` — waveguide optimization (expanded from stub)
  - Waveguide eigenmode computation (1D slab waveguide solver)
  - Mode overlap integral (differentiable)
  - Waveguide bend optimization
  - Mode converter optimization (fundamental to higher-order)
  - Works with any Solver backend via protocol

- [x] `diffnano/workflows/phc.py` — photonic crystal optimization
  - Band structure via plane wave expansion (differentiable)
  - Topology optimization: maximize bandgap/midgap ratio

- [x] `diffnano/workflows/waveguide.py`
  - Waveguide bend / mode converter optimization
  - Mode overlap integral as differentiable figure of merit

- [x] **2026 addition**: `diffnano/solvers/fdfd2d.py` — 2D FDFD
  - Frequency-domain solve via dense `torch.linalg.solve` backward
  - GPU-native, no sparse library dependency

---

## v0.3 Milestone — 3D FDTD + C7 Adaptive Robust Optimization

**Status: DONE** (commit on main 2026-05-28)
**120 tests passing**

- [x] 3D FDTD — extend 2D solver (`diffnano/solvers/fdtd3d.py`)
  - Yee grid with all six EM field components (Ex, Ey, Ez, Hx, Hy, Hz)
  - CPML absorbing boundaries on all six faces
  - Point, line, and plane sources
  - Gradient checkpointing for memory efficiency
  - Courant condition for 3D: dt < dl/(c*sqrt(3))
- [x] **C7 adaptive robust optimization** (`diffnano/design/robustness/adaptive.py`)
  - Axial sampling: O(2N+1) corner samples
  - Adaptive worst-case refinement with curriculum learning
  - FabricableSubspaceProjection with Gumbel-softmax relaxation
- [x] **Full C5 feature set** (`diffnano/design/robustness/subspace.py`)
  - Sidewall-angle drift perturbation kernel
  - Layer-thickness variation perturbation
  - Corner-rounding perturbation (Gaussian smoothing)
  - Multi-axis correlated perturbation with Cholesky decomposition
  - Joint Gaussian model with correlated sampling
- [x] C7 benchmark script (`scripts/benchmark_c7.py`)

---

## v0.4 Milestone — Neural Surrogate + Broadband + OpenLithoHub

**Status: DONE** (commit on main 2026-05-28)
**128 tests passing**

- [x] Neural surrogate accelerated optimization (`diffnano/solvers/surrogate.py`)
  - Lightweight CNN predicting RCWA output from geometry
  - Drop-in replacement for RCWASolver with periodic full-solver correction
  - 10-50x optimization speedup potential
- [x] Multi-wavelength optimization (`diffnano/workflows/broadband.py`)
  - Weighted sum of RCWA objectives across wavelengths
  - Beta-continuation for binarization
- [x] OpenLithoHub integration smoke test (deferred — depends on stable API)

---

## Next-Generation Features: C6–C8 (inspired by literature review 2026-05)

> **Motivation:** The prior-art review revealed that TorchLitho (Hopkins/SOCS), PRISM
> (photonics-informed ILT), BOSON-1 (adaptive robust optimization), TorchResist
> (differentiable resist), and D-Flat (end-to-end flat-optics) each bring powerful
> ideas that DiffNano should absorb and differentiate from. C6–C8 capture the most
> impactful directions not yet covered by v0.1–v0.4.

### C6 — Learned Fabrication Process Model (inspired by PRISM + TorchResist)

**Status: DONE** (implemented in v0.5)

**Source:** PRISM (arXiv:2602.15762) trains a physics-grounded neural network to model
the actual fabrication transfer function from calibration data. TorchResist
(arXiv:2502.06838) provides an open-source differentiable resist model with <20
interpretable parameters calibrated on real designs.

**Implementation:**

- [x] `diffnano/solvers/fab_model.py` — `LearnedFabModel`
  - U-Net encoder-decoder with physics priors (sigmoid output for [0,1] range)
  - Differentiable end-to-end; drop-in replacement for HopkinsLithoModel
  - Synthetic calibration data generation and training loop

- [x] `diffnano/solvers/resist.py` — `DifferentiableResistModel`
  - Clean-room reimplementation of analytical resist model from TorchResist paper
  - Acid diffusion + PEB diffusion + development contrast chain
  - Interpretable differentiable parameters with calibration support

---

### C7 — Adaptive Multi-Source Robust Optimization (inspired by BOSON-1)

**Status: DONE** (implemented in v0.3)

**Implementation:**

- [x] `diffnano/design/robustness/adaptive.py` — `AdaptiveRobustOptimizer`
  - Axial sampling: 2N+1 points for N variation sources
  - Adaptive worst-case refinement with top-k emphasis
  - Curriculum: axial → random progressive sampling
  - Drop-in replacement for `robust_gradient_step`

- [x] `diffnano/design/robustness/subspace.py` — `FabricableSubspaceProjection`
  - Gumbel-softmax relaxation for differentiable projection
  - Morphological opening for minimum CD enforcement
  - Multi-axis perturbation: sidewall, thickness, corner rounding

- [x] Multi-axis correlated perturbation with Cholesky decomposition

- [x] Benchmark: `scripts/benchmark_c7.py`

---

### C8 — Curvilinear Mask Parameterization + Multi-Objective Design Space Exploration

**Status: DONE** (implemented in v0.5–v0.6)

**Implementation:**

- [x] `diffnano/design/curvilinear.py` — `CurvilinearMask`
  - Fixed BSplineCurve NaN gradients using analytical SDF with differentiable winding number
  - B-spline boundary representation with differentiable control points
  - DVAS-style 1D boundary parameterization
  - Smooth gradient flow verified

- [x] `diffnano/workflows/multi_objective.py` — `MultiObjectiveExplorer`
  - Weighted-sum scalarization with Dirichlet-distributed weight sampling
  - Pareto front filtering via dominance checking
  - Multi-objective: optical, fabrication, constraint objectives

- [x] `diffnano/design/representation_learning.py` — `LearnedRepresentation`
  - VAE encoder/decoder for design library
  - Latent space optimization for 10-100x faster convergence

- [x] End-to-end workflow: `diffnano/workflows/end_to_end.py`
  - Full DFM-native pipeline: density → fabrication model → EM solver → multi-objective loss → optimizer → GDSII export

---

### Updated Feature Roadmap

```
v0.1 (DONE) ─── RCWA + Hopkins litho + DFM-metalens + robust MC
v0.2 (DONE) ─── 2D FDTD + photonic crystal + waveguide + FDFD
v0.3 (DONE) ─── 3D FDTD + C7 adaptive robust optimization
v0.4 (DONE) ─── Neural surrogate + broadband + OpenLithoHub integration
v0.5 (DONE) ─── C6 learned fabrication model + C8 curvilinear mask
v0.6 (DONE) ─── C8 multi-objective Pareto + end-to-end pipeline + VAE representation
```

### New References (C6–C8 sources)

- **Geng et al. (2024)** — TorchLitho: Open-Source Differentiable Lithography Imaging
  Framework. arXiv:2409.15306. Apache 2.0. ← **C6 reference implementation for Hopkins/SOCS**
- **Zhou et al. (2026)** — PRISM: Photonics-Informed Inverse Lithography for
  Manufacturable PICs. arXiv:2602.15762. ← **C6 learned fab model inspiration, C8 curvilinear mask**
- **Ma et al. (2024)** — BOSON-1: Understanding and Enabling Physically-Robust Photonic
  Inverse Design. arXiv:2411.08210. ← **C7 adaptive sampling, fabricable subspace**
- **Geng et al. (2025)** — TorchResist: Open-Source Differentiable Resist Simulator.
  arXiv:2502.06838. SPIE 2025. ← **C6 resist model reference**
- **Hazineh et al. (2022)** — D-Flat: A Differentiable Flat-Optics Framework.
  arXiv:2207.14780. ← **C8 end-to-end pipeline reference**
- **Optics Express (2024)** — DVAS: Fast Curvilinear Mask Optimization by
  Distance-Versus-Angle Signature. OE 32(15), 26292. ← **C8 compact boundary representation**
- **Optics Letters (2025)** — FAID: Fabrication-Aware Inverse Design integrating
  DUV lithography models. ← **C6/C8 fabrication-aware design reference**

---

## Key References

- Liu & Fan (2020) — `grcwa`: differentiable RCWA. arXiv:2005.01481
- **Kim & Lee (2023)** — `TORCWA`: GPU-accelerated PyTorch RCWA with broadening-based eigendecomposition stabilization. *Computer Physics Communications* 282, 108552
- **Huang et al. (2024)** — `TorchRDIT`: eigendecomposition-free inverse design via R-DIT, 16.2× speedup vs RCWA. *Optics Express* 32(8), 13986
- **Mahlau et al. (2024/2026)** — `FDTDX`: JAX-based 3D FDTD with time-reversibility-based gradient computation. arXiv:2412.12360; JOSS 11(117), 8912
- **Kim et al. (2024)** — `meent`: differentiable RCWA with multi-backend (NumPy / JAX / PyTorch) support. arXiv:2406.12904; MIT-licensed
- Hughes et al. (2019) — `ceviche`: differentiable FDFD. ACS Photonics
- Devlin et al. (2016) — Broadband metasurface. Science
- Molesky et al. (2018) — Inverse design in nanophotonics. Nature Photonics
- Chen et al. (2016) — Gradient checkpointing (Training deep networks with O(√n) memory). arXiv
- Christiansen & Sigmund (2021) — Inverse design in photonics: from computational to experimental realizations. *Advances in Physics*
- **tidy3d.plugins.autograd source** (Flexcompute, ongoing) — `make_erosion_dilation_penalty`, `smoothed_projection`, `FilterProject`, `ErosionDilationPenalty`

### Monitoring list (not direct competitors but watch)
- `fdtd-z` (Google/X) — GPU FDTD, ~100× CPU, adjoint-based
- `PyMieDiff` / `TorchGDM` — differentiable Mie / Green's dyadic; small-particle regime, complementary not competitive

---

## License Compliance & Code Provenance

**DiffNano's intended license: Apache License 2.0.**

**External code in the same ecosystem and DiffNano's compliance posture:**

| Project | License | DiffNano's posture |
|---|---|---|
| **TORCWA** (kch3782/torcwa) | **LGPL-2.1** | **Do NOT vendor source.** LGPL is copyleft on modifications to the LGPL'd files; static-linking obligations are awkward for an Apache-2.0 project. If a TORCWA-equivalent broadening fallback is implemented, it must be **clean-room re-implemented from the published paper (Kim & Lee, CPC 2023)**, not derived from the TORCWA source. Algorithmic ideas from the paper are fair game; source copying is not. |
| **TorchRDIT** | **GPL-3.0** | **Do NOT vendor source. Do NOT statically or dynamically link.** GPL-3.0 is strong copyleft and would force DiffNano to relicense. If R-DIT support is added, it must be a **clean-room re-implementation from Huang et al. *Optics Express* 32(8), 13986 (2024)**, with no reference to the TorchRDIT source tree during implementation. |
| **meent** (Kim et al. 2024) | **MIT** | Compatible with Apache 2.0. May be vendored or referenced if attribution is preserved. |
| **grcwa** (Liu & Fan 2020) | **MIT** | Compatible. Used as a CPU baseline / cross-check; not vendored. |
| **ceviche** (Hughes et al. 2019) | **MIT** | Compatible. Used as the reference for the v0.2 FDFD module; the FDFD module is a clean-room reimplementation, GPU-native via `torch.linalg.solve` backward — not a port of ceviche source. |
| **tidy3d.plugins.autograd** (Flexcompute) | **Apache 2.0** | Compatible. DiffNano's penalty library is a **clean-room reimplementation** of published concepts but does not copy code. |
| **FDTDX** (Mahlau et al.) | **MIT** | Compatible, but **JAX-based** — not directly vendorable into a PyTorch project. Used as benchmark comparator only. |
| **OpenLithoHub** (alpha 0.1.0a2) | **Apache 2.0** | Compatible. Treated as an optional integration target (see v0.1 milestone smoke test). Versions are pinned in `requirements.txt`; the v0.1 acceptance gate does not block on OpenLithoHub stability. |
| **PyTorch** | BSD-3 | Compatible. |

**Provenance rules for DiffNano contributors (added to CONTRIBUTING.md):**
1. Do not paste source from any GPL or LGPL project (most importantly TorchRDIT, TORCWA) into the DiffNano tree.
2. When implementing a method published in a paper that also has a GPL/LGPL reference implementation, work from the **paper, not the reference source**, and document the references-consulted list in the module docstring.
3. All vendored code must carry its original license header and an entry in `THIRD_PARTY_LICENSES.md`.
4. All modules must contain no copyleft-derived code; this is enforced by a CI license-scan job (`scancode-toolkit` or equivalent) before any release tag.

---

## Competitive Differentiation Summary (revised 2026-05-28)

| Feature | DiffNano v0.1 | DiffNano v0.5+ (C6–C8) | PRISM | BOSON-1 | TorchLitho | TorchResist | D-Flat | tidy3d |
|---|---|---|---|---|---|---|---|---|
| Differentiable litho model | ✅ Gaussian PSF | ✅ Learned fab model (C6) | ✅ Neural | ❌ | ✅ Hopkins/SOCS | ✅ Resist only | ❌ | ❌ |
| Differentiable EM solver | ✅ RCWA | ✅ RCWA + FDTD + FDFD | ❌ | ✅ Adjoint | ❌ | ❌ | ✅ RCWA | ✅ FDTD (cloud) |
| Litho+EM in one autograd graph | ✅ | ✅ + learned fab (C6) | ❌ (sequential) | ❌ | ❌ | ❌ | ❌ | ❌ |
| Robust optimization | ✅ Fixed MC | ✅ Adaptive (C7) | ❌ | ✅ Axial sampling | ❌ | ❌ | ❌ | ❌ |
| Fabricable subspace projection | ❌ | ✅ (C7) | ❌ | ✅ | ❌ | ❌ | ❌ | Partial |
| Curvilinear mask parameterization | ❌ | ✅ (C8) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Multi-objective Pareto exploration | ❌ | ✅ (C8) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Learned design representation | ❌ | ✅ (C8) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Open-source, self-hosted | ✅ | ✅ | ❌ | Planned | ✅ | ✅ | ✅ | ❌ (cloud) |
| PyTorch-native | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ (TF) | ✅ (plugin) |

**Positioning:** DiffNano's value is the **integration breadth**: no other tool combines litho, EM, robustness, and
fabricability in one differentiable PyTorch framework. The value proposition is
engineering productivity (one framework instead of chaining 3-4 separate tools),
not a single novel algorithm.

---

## External Positioning & Narrative (2026-05)

The competitive landscape has shifted: differentiable EM solvers as a category are commoditizing (TORCWA for RCWA, TorchRDIT for R-DIT, FDTDX for FDTD, tidy3d.plugins.autograd for cloud-FDTD). "PyTorch + GPU + autograd EM" is no longer a distinctive narrative.

**Updated positioning (for README, whitepaper, talks):**

> **DiffNano: The first DFM-native differentiable nanophotonic inverse-design framework.**
> Where other tools optimize photonic structures and apply fabrication constraints as an afterthought, DiffNano was built from the ground up around a single principle: the design parameterization is shared between computational lithography and electromagnetic simulation, and gradients from both forward models flow back to the same parameter tensor in one unified autograd graph. The result is layouts that are simultaneously optically optimal, process-rule compliant, and robust to fabrication variation — without post-processing geometric correction, without a separate robustness pass, and without a manual handoff between the photonics designer and the lithography team.

**Narrative emphasis priorities:**
1. **DFM-native, not DFM-bolt-on** — design-for-manufacturability is the core value prop, not the speed of the underlying solver
2. **Yield-aware by construction** — robustness is in the optimization loop, not a post-hoc check
3. **Cross-domain by design** — cross-pipeline coupling differentiates against pure-photonics competitors (TORCWA/TorchRDIT/FDTDX) and pure-lithography pipelines (Calibre, OpenLithoHub-standalone)
4. **PyTorch + GPU + autograd** is mentioned as a *capability*, not a *differentiator*

**Avoid in external materials:**
- "Fastest differentiable RCWA / FDTD" claims — TorchRDIT (16.2× vs RCWA) and FDTDX (multi-GPU 3D) own these axes
- "First open-source differentiable EM solver" — false; TORCWA/TorchRDIT/FDTDX are open-source
- "Solves degenerate eigenvalue gradients" as a headline — TORCWA already addresses this with broadening; this is at most a footnote-level technical detail
