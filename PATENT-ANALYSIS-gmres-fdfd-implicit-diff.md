# Patent Analysis: Matrix-Free GMRES + FDFD Implicit Differentiation

> **Status**: NOT RECOMMENDED — direct prior art found (Mao & Fan 2025)

## Conclusion

**Do NOT file.** External expert found direct prior art that anticipates all claimed novel elements.

## Key Prior Art (from expert review)

1. **Mao & Fan (Stanford, 2025, arXiv:2509.03622)** — "Accurate and scalable deep Maxwell solvers using multilevel iterative methods":
   - Explicitly uses implicit differentiation over autodiff for memory reasons
   - Uses F-GMRES + FDFD (Maxwell frequency-domain)
   - Applies to nanophotonic inverse design adjoint gradient computation
   - Custom JVP for implicit differentiation
   - **This is a direct match to all claimed novel elements**

2. **Implicit differentiation through iterative solvers is a mature method**:
   - Blondel et al. 2022 (JAXopt)
   - Nonconvex.jl: matrix-free GMRES for implicit differentiation (documented feature)
   - Bolte & Pauwels 2023 "One-step differentiation of iterative algorithms"
   - PRDP (ICLR 2025)

3. **FDFD adjoint memory savings are already commercialized** — Ceviche (Stanford, 2019), Tidy3D autograd (2024+), Lumerical adjoint solver

4. **Complex GMRES + Givens rotations** — standard complex extension of Saad & Schultz 1986, textbook material

5. **O(restart × N_grid) memory** — definitional property of restarted GMRES, not a novel result

6. **Gradient accuracy 2.33e-08** — expected precision of implicit differentiation, not a patentable technical effect

## Decision

All files associated with this innovation are cleared for push.
