"""Tuning knobs for the live (on-the-fly) blade stitcher.

Every threshold here is either measured from the production scans in
``image_stitching/from_scratch/test_images`` or has a comment explaining what
it guards against.  Values are deliberately conservative: it is far cheaper to
reject one extra frame than to blend a bad transform into a defect mosaic.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LiveConfig:
    # ── Feature registration ────────────────────────────────────────────────
    work_width: int = 1000
    """Width (px) images are scaled to for feature extraction/registration.

    1920 px frames register ~4x slower with no accuracy benefit; the final
    blend still uses the full-resolution frame.
    """
    feature: str = "orb"
    """``orb`` (fast, ~40 ms/frame) or ``sift`` (slower, more robust to scale)."""
    n_features: int = 4000
    polish_width: int = 600
    """Resolution for the optional ECC sub-pixel polish (0 disables).  ECC is
    only ever accepted when it improves the measured NCC, because on these
    frames the illumination vignette pulls it off by a whole hole pitch."""
    ratio_test: float = 0.75
    ransac_px: float = 2.5
    min_inliers: int = 15
    """Minimum RANSAC inliers.  Good adjacent pairs give 55-1500 here, so 15 is
    permissive; the NCC gate below is what actually rejects wrong models."""
    prior_tol_px: float = 60.0
    """Base tolerance of the expected-step band.  Matches further than this
    from the predicted displacement are treated as outliers - this is what
    kills the false locks on the periodic cooling-hole pattern."""
    prior_tol_rel: float = 0.30
    """Extra tolerance proportional to the predicted step size.  When frames
    were rejected the predicted step is a *sum* of several moves and is much
    less certain, so the band has to widen with it."""
    loose_ratio_retry: float = 0.12
    """If the strict ratio test produces no usable model, retry with
    ``ratio_test + this`` (capped at 0.9) and half the inlier floor.

    Measured need: a pair spanning two motor steps (after one frame was
    rejected) yielded only 8 strict matches, but 31 with ratio 0.85 and the
    correct shift at NCC 0.95.  Without the retry the whole rest of the column
    came out as 'unregistered'."""
    retry_without_prior: bool = True
    """If the bounded attempt fails, retry unbounded and let the NCC gate
    decide.  Needed after a burst of rejected frames, where the predicted step
    can be off by more than the band."""
    polish_below_ncc: float = 0.92
    """Run the (slow) ECC sub-pixel polish only when the ORB result is already
    mediocre.  Measured: on good pairs the polish changes nothing."""

    # ── Acceptance / rejection ──────────────────────────────────────────────
    park_disconnected: bool = True
    """When a frame cannot be registered against anything already stitched,
    start a new band with it and keep stitching from there.

    The alternative (rejecting forever) throws away the rest of the column: on
    the recorded col02 a single 49-keypoint frame broke the chain and the
    following ten good frames were lost.  A parked frame never gets a made-up
    transform - it becomes the origin of its own band, 64 px to the right of the
    existing content - and the report lists it, so the gap is visible.  Bounded
    by ``max_disconnected_bands``."""
    max_disconnected_bands: int = 8
    """If parking is enabled, stop after this many bands."""

    soft_unregisterable_ratio: float = 0.65
    """If a frame fails to register *and* is this much softer than the last
    accepted frame, it is a capture fault: drop it instead of starting a new
    mosaic band for it."""
    min_overlap: float = 0.18
    """Fraction of the frame that must overlap already-accepted content."""
    min_ncc: float = 0.55
    """Photometric correlation of the warped frame with its neighbour.

    Correct adjacent pairs measure 0.74-0.99 here (0.74 at ~13% overlap), while
    wrong models land near 0.3.  0.55 keeps the tightest legitimate overlaps and
    rejects the rest."""
    min_new_area: float = 0.10
    """A frame covering <10% uncovered area is a duplicate (stalled motor)."""
    max_step_frac: float = 0.85
    """Reject transforms that imply a jump larger than this fraction of the
    frame size - physically impossible for one motor step."""

    # ── Blur / exposure gate ────────────────────────────────────────────────
    blur_ratio: float = 0.72
    """Keep the frame only if its (exposure-normalised) Tenengrad sharpness is
    at least this fraction of the previously accepted frame, measured on the
    *overlap region only*.  Same-pixels comparison removes content bias."""
    min_sharpness: float = 1.5e-3
    """Absolute floor on normalised Tenengrad; below this nothing is in focus.

    Only catches gross cases (lens cap, total defocus): across one scan the
    absolute value legitimately varies 10x with illumination, which is why the
    relative gates below exist.
    """
    pre_blur_ratio: float = 0.30
    """Coarse pre-gate against the previously accepted frame's absolute
    sharpness.  Deliberately lax.

    Measured frame-to-frame ratios over the recorded scans: median 1.01,
    p5 = 0.58, p1 = 0.40 - the low tail is normal illumination drift, so a
    tighter threshold would throw away good frames.  0.30 only catches gross
    defocus, which matters because such a frame yields too few ORB matches to
    register at all and used to be misfiled as a disconnected band."""
    max_clipped_low: float = 0.55
    max_clipped_high: float = 0.30
    """Reject frames that are mostly black (shutter/lighting fault) or blown."""
    min_contrast: float = 4.0
    """Standard deviation of luma; flat frames carry no stitching information."""

    max_cached_anchors: int = 400
    """Anchors whose features stay in memory for the fallback search.

    Each cached anchor costs ~0.6 MB (1000 px grey + 4000 ORB descriptors), so
    400 anchors ~ 240 MB.  Keeping the whole scan searchable is what lets a new
    column be linked to the previous one instead of being dumped in a separate
    band.
    """

    # ── Rejected-frame policy ───────────────────────────────────────────────
    blur_policy: str = "fill"
    """What to do with a frame that registered correctly but is blurrier than
    its neighbour:

    ``reject`` - drop it.  Leaves a hole; only use if the capture loop
                 guarantees a retake.
    ``fill``   - blend it *only* into canvas pixels that are still empty, so
                 it never degrades good imagery, and record those pixels in a
                 low-quality mask for QC.  Default: an empty hole in a defect
                 mosaic is worse than a soft patch that is flagged.
    """
    fill_min_new_area: float = 0.02
    """A blurry frame that adds less than this much new coverage is dropped."""

    # ── Drift control ───────────────────────────────────────────────────────
    anchors: int = 2
    """Register each new frame against this many previously accepted frames
    (the most recent one plus spatially overlapping older ones) and average
    the implied global poses.  Bounds accumulated drift."""
    anchor_overlap: float = 0.30
    fallback_anchors: int = 8
    """When the temporal predecessor cannot be matched (column change, skipped
    frames, motor jump), retry against the nearest frames already placed,
    nearest first.  Without this a column change cascades into dozens of
    'disconnected' bands."""

    # ── Blending ────────────────────────────────────────────────────────────
    feather_px: int = 64
    feather_gamma: float = 2.0
    """Exponent of the smoothstep blend ramp.  1 is the old linear profile, whose
    gradient jumps at the feather limit and shows up as a seam on low-contrast
    surfaces; 2-3 fades overlapping frames into each other instead."""

    dead_luma: int = 10
    """Luma at or below this is "not image" (clipped sensor border, dead corner).

    Deliberately strict: rusty blade surface reads 30-90, an HDMI/sensor border
    reads 0-8.  These pixels are excluded from blending and from the exposure
    fit; including them is what produced the pale blocks in the delivered
    mosaics (a near-black strip lifted by the 2x gain clip)."""

    max_dead_frac: float = 0.55
    """Above this much dead area the frame is a capture fault, not a border."""

    axis_prior_strict: bool = True
    """Reject a registration whose motion contradicts the commanded axis.

    A serpentine sweep is never purely horizontal or vertical (rig skew, actuator
    angle), so the prior is *learned* from the scan rather than assumed: each
    axis's observed mean displacement and spread become the expected motion and
    the tolerance.  A frame that matches somewhere else - the next periodic hole,
    a reflection - is refused instead of blended in permanently."""

    axis_min_samples: int = 3
    """Observations of an axis before the prior may veto anything.  Until then it
    only nudges the registration prior, which is always safe."""

    axis_prior_rel_tol: float = 0.35
    """Tolerance as a fraction of the learned step, on top of 3 sigma."""

    scalar_gain: bool = False
    """Ablation switch: correct exposure with a single scalar and no validity
    mask, i.e. the behaviour that produced the pale blocks.  Used by the
    regression reproducer; leave False in production."""

    flatten_illumination: bool = True
    """Divide out each frame's own smooth illumination profile before blending.

    The light falls off across the field of view, so every frame carries the same
    low-frequency brightness surface.  Being common to all frames, it cannot be
    removed by matching frames to each other: a frame overlaps its neighbour over
    part of its area, so matching fits the surface from that strip and
    extrapolates the rest - and the extrapolation error is what showed up as
    brighter rectangles in the delivered mosaics.  Measured per frame instead,
    it is a measurement rather than a guess, and afterwards the only remaining
    difference between frames is the exposure level, which the scalar gain
    cancels exactly.  Set False for the old (rectangle-prone) behaviour."""

    flatten_order: int = 4
    """How much structure counts as illumination (polynomial order).

    4 follows a light falloff and a vignette without chasing a scratch or a hole
    row; 0 would disable the normalisation.  Raise it only if the corners of the
    mosaic are still visibly dimmer than the middle."""

    flatten_clip: float = 1.6
    """Largest illumination correction (1.6 = +/-60 %).  A frame that would need
    more than that is badly exposed, not lit unevenly."""

    exposure_gain_clip: tuple[float, float] = (0.75, 1.35)
    output_scale: float = 1.0
    """Scale of the saved mosaic (0.5 halves memory 4x for live previews)."""
    max_canvas_px: float = 60e6
    """Safety valve: refuse to allocate a canvas bigger than this.

    Peak memory is roughly ``max_canvas_px * 16`` bytes (float32 RGB
    accumulator + float32 weights + the uint8 render), i.e. ~1 GB at 60 Mpx,
    plus the same amount again while the canvas grows.  Exceeding the limit is
    not recoverable by catching an exception on a small machine - the OOM killer
    takes the process - so the default is deliberately conservative.

    A full three-column scan of the recorded part needs ~95 Mpx at full
    resolution: run it with ``accum_scale=0.6`` (~34 Mpx, ~0.5 GB) or raise this
    deliberately on a workstation with the memory to match.
    """
    accum_scale: float = 1.0
    """Resolution the mosaic is accumulated at (1.0 = capture resolution).

    Registration always runs on full-resolution frames; only the blend is
    scaled.  0.5 cuts canvas memory 4x for very large parts."""

    # ── Runtime ─────────────────────────────────────────────────────────────
    flush_every: int = 10
    """Write a partial mosaic every N accepted frames so a crash/cancel never
    loses the whole scan."""
    queue_size: int = 8
    save_kept: bool = True
    """Copy accepted frames next to the mosaic (audit trail)."""
    save_rejected: bool = True
    """Write rejected frames to a folder named after the rejection reason, so a
    bad threshold is obvious from the filenames instead of being invisible."""
