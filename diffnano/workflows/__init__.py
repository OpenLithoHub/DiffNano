"""Pre-built inverse design workflows."""

__all__ = ["MetalensDesigner", "DFMMetalensDesigner", "PhCDesigner", "WaveguideDesigner"]


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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
