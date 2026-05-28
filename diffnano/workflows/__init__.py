"""Pre-built inverse design workflows."""

__all__ = [
    "MetalensDesigner", "DFMMetalensDesigner", "PhCDesigner", "WaveguideDesigner",
    "BroadbandOptimizer", "MultiObjectiveExplorer", "EndToEndPipeline",
]


def __getattr__(name):
    if name == "MetalensDesigner":
        from diffnano.workflows.metalens import MetalensDesigner
        return MetalensDesigner
    if name == "DFMMetalensDesigner":
        from diffnano.workflows.dfm_metalens import DFMMetalensDesigner
        return DFMMetalensDesigner
    if name == "PhCDesigner":
        from diffnano.workflows.phc import PhCDesigner
        return PhCDesigner
    if name == "WaveguideDesigner":
        from diffnano.workflows.waveguide import WaveguideDesigner
        return WaveguideDesigner
    if name == "BroadbandOptimizer":
        from diffnano.workflows.broadband import BroadbandOptimizer
        return BroadbandOptimizer
    if name == "MultiObjectiveExplorer":
        from diffnano.workflows.multi_objective import MultiObjectiveExplorer
        return MultiObjectiveExplorer
    if name == "EndToEndPipeline":
        from diffnano.workflows.end_to_end import EndToEndPipeline
        return EndToEndPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
