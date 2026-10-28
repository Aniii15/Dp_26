"""Live (on-the-fly) stitching for blade micro-inspection.

Public entry points::

    from live import LiveStitcher, LiveConfig
    stitcher = LiveStitcher(LiveConfig(), out_prefix="runs/blade")
    for frame in capture():
        stitcher.submit(frame, tag="col01_Ydown_1")   # non-blocking
    stitcher.finalize()
"""

from .config import LiveConfig
from .mosaic import IncrementalMosaic
from .pipeline import Decision, LiveStitcher
from .quality import assess_intrinsic, focus_ratio, sharpness
from .registration import Registrar, Registration

__all__ = [
    "LiveConfig", "LiveStitcher", "Decision", "IncrementalMosaic",
    "Registrar", "Registration", "assess_intrinsic", "focus_ratio", "sharpness",
]
