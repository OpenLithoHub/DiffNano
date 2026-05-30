"""End-to-end DFM-native pipeline (C8 completion).

Full pipeline from optical specification to GDSII export:
  specification → representation → curvilinear mask → learned fab model
  → EM solver → multi-objective loss → optimizer → GDSII

This workflow combines all DiffNano capabilities into a single
differentiable pipeline.
"""

from __future__ import annotations

import torch

__all__ = ["EndToEndPipeline"]


class EndToEndPipeline:
    """End-to-end DFM-native nanophotonic design pipeline.

    Combines learned representation, curvilinear mask, fabrication model,
    EM solver, and multi-objective optimization into a single differentiable
    pipeline from specification to GDSII export.

    Parameters
    ----------
    solver
        EM solver (RCWA, FDTD, or FDFD).
    fab_model
        Fabrication model (HopkinsLithoModel or LearnedFabModel).
    grid_shape : tuple[int, int]
    wavelengths_nm : list of float
    device : str or torch.device
    """

    def __init__(
        self,
        solver=None,
        fab_model=None,
        grid_shape: tuple[int, int] = (32, 32),
        wavelengths_nm: list[float] | None = None,
        eps_low: float = 1.0,
        eps_high: float = 12.0,
        device: str | torch.device = "cpu",
    ):
        self.solver = solver
        self.fab_model = fab_model
        self.grid_shape = grid_shape
        self.wavelengths_nm = wavelengths_nm or [532.0]
        self.eps_low = eps_low
        self.eps_high = eps_high
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def forward_pass(
        self,
        density: torch.Tensor,
        optical_loss_fn: callable | None = None,
        fab_weight: float = 0.1,
        constraint_weight: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Run one forward pass through the entire pipeline.

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
            Design density field.
        optical_loss_fn : callable, optional
            ``fn(solver_output) -> loss``. Default: transmission maximization.
        fab_weight : float
            Weight for fabrication loss.
        constraint_weight : float
            Weight for fabrication constraint penalties.

        Returns
        -------
        results : dict
            ``{"total_loss": ..., "optical_loss": ..., "fab_loss": ...,
            "constraint_loss": ..., "field": ...}``
        """
        from diffnano.design.constraints_shared import combined_fabrication_penalty

        # Caller is responsible for projection
        binary = density

        # Fabrication model forward
        if self.fab_model is not None:
            fab_result = self.fab_model.forward(binary)
            # Handle both dict (HopkinsLithoModel) and SimResult (LearnedFabModel)
            if isinstance(fab_result, dict):
                printed = fab_result["printed_contour"]
                fab_loss = fab_result["epe"]
            else:
                printed = fab_result.field.squeeze(0)
                fab_loss = ((binary - printed) ** 2).mean()
        else:
            printed = binary
            fab_loss = torch.tensor(0.0, dtype=torch.float64, device=self._device)

        # EM solver forward
        optical_loss = torch.tensor(0.0, dtype=torch.float64, device=self._device)
        field = None

        if self.solver is not None:
            # Convert density to permittivity layers
            H, W = density.shape
            n_layers = min(5, H)
            layer_h = H // n_layers
            remainder = H % n_layers
            layers = []
            y0 = 0
            for i in range(n_layers):
                extra = 1 if i < remainder else 0
                y1 = min(y0 + layer_h + extra, H)
                avg = printed[y0:y1, :].mean(dim=0)
                layers.append(self.eps_low + (self.eps_high - self.eps_low) * avg)
                y0 = y1

            geometry = torch.stack(layers)
            result = self.solver.forward(geometry, wavelengths=self.wavelengths_nm)
            field = result.field

            if optical_loss_fn is not None:
                optical_loss = optical_loss_fn(result)
            else:
                # Default: maximize transmission (negative sum)
                optical_loss = -result.field.sum()

        # Fabrication constraints
        constraint_loss = combined_fabrication_penalty(binary)

        # Total loss
        total = optical_loss + fab_weight * fab_loss + constraint_weight * constraint_loss

        return {
            "total_loss": total,
            "optical_loss": optical_loss,
            "fab_loss": fab_loss,
            "constraint_loss": constraint_loss,
            "field": field,
            "printed": printed,
        }

    def optimize(
        self,
        n_steps: int = 200,
        lr: float = 0.01,
        optical_loss_fn: callable | None = None,
        fab_weight: float = 0.1,
        constraint_weight: float = 0.05,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        """Run end-to-end optimization.

        Parameters
        ----------
        n_steps : int
        lr : float
        optical_loss_fn : callable, optional
        fab_weight : float
        constraint_weight : float
        verbose : bool

        Returns
        -------
        density : Tensor, shape ``(H, W)``
        history : dict of lists
        """
        from diffnano.design.projection import beta_continuation_schedule, heaviside_projection

        density = torch.rand(
            *self.grid_shape,
            device=self._device,
            dtype=torch.float64,
        )
        density = density.detach().requires_grad_(True)

        opt = torch.optim.Adam([density], lr=lr)

        history = {
            "total": [],
            "optical": [],
            "fab": [],
            "constraint": [],
        }

        for step in range(n_steps):
            # Beta-continuation on the raw density
            beta = beta_continuation_schedule(step, n_steps, beta_start=1.0, beta_end=32.0)
            projected = heaviside_projection(density, beta=beta)

            results = self.forward_pass(
                projected,
                optical_loss_fn=optical_loss_fn,
                fab_weight=fab_weight,
                constraint_weight=constraint_weight,
            )

            loss = results["total_loss"]

            opt.zero_grad()
            loss.backward()

            if density.grad is not None and torch.isnan(density.grad).any():
                if verbose:
                    print(f"Step {step}: NaN gradient, stopping.")
                break

            opt.step()
            with torch.no_grad():
                density.clamp_(0.0, 1.0)

            history["total"].append(results["total_loss"].item())
            history["optical"].append(results["optical_loss"].item())
            history["fab"].append(results["fab_loss"].item())
            history["constraint"].append(results["constraint_loss"].item())

            if verbose and step % 20 == 0:
                print(
                    f"Step {step:4d}: total={results['total_loss'].item():.6f} "
                    f"opt={results['optical_loss'].item():.6f} "
                    f"fab={results['fab_loss'].item():.6f}"
                )

        return density.detach(), history

    def export_gds(
        self,
        density: torch.Tensor,
        filepath: str,
        threshold: float = 0.5,
        pixel_size_nm: float = 5.0,
    ) -> None:
        """Export optimized density to GDSII.

        Parameters
        ----------
        density : Tensor, shape ``(H, W)``
        filepath : str
        threshold : float
        pixel_size_nm : float
        """
        from diffnano.export.gds import export_density_to_gds

        binary = (density > threshold).float()
        export_density_to_gds(binary, filepath, pixel_size_nm=pixel_size_nm)
