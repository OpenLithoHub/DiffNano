"""DiffNano: Differentiable Nanophotonics Design in PyTorch."""

__version__ = "0.6.0"

__all__ = [
    "__version__",
    "RCWASolver",
    "FDFDSolver2D",
    "FDTDSolver2D",
    "FDTDSolver3D",
    "NeuralSurrogate",
    "HopkinsLithoModel",
    "LearnedFabModel",
    "DifferentiableResistModel",
    "MetalensDesigner",
    "DFMMetalensDesigner",
    "PhCDesigner",
    "WaveguideDesigner",
    "BroadbandOptimizer",
    "MultiObjectiveExplorer",
    "EndToEndPipeline",
]


def __getattr__(name):
    if name in (
        "RCWASolver",
        "FDFDSolver2D",
        "FDTDSolver2D",
        "FDTDSolver3D",
        "NeuralSurrogate",
        "HopkinsLithoModel",
        "LearnedFabModel",
        "DifferentiableResistModel",
    ):
        from diffnano import solvers

        return getattr(solvers, name)
    if name in (
        "MetalensDesigner",
        "DFMMetalensDesigner",
        "PhCDesigner",
        "WaveguideDesigner",
        "BroadbandOptimizer",
        "MultiObjectiveExplorer",
        "EndToEndPipeline",
    ):
        from diffnano import workflows

        return getattr(workflows, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
