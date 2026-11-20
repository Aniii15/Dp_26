#!/usr/bin/env python3
"""Stitch a blade scan as the frames arrive.

Two sources are supported:

* ``--replay DIR``  - replay an existing scan folder in acquisition order.
  Use this to validate tuning on recorded data without touching the machine.
* ``--camera N``    - the HDMI capture card.  Frames are only submitted once the
  image has stopped changing (motor settled), which removes motion-blurred
  captures at the source and cuts the number of useless frames dramatically.

Examples::

    # replay a recorded scan, keeping only usable frames
    python run_live.py --replay ../../main_code/blade_images/test_images --out runs/col01

    # live camera, 1 s settle time, partial mosaic every 5 frames
    python run_live.py --camera 1 --out runs/live --settle 1.0 --flush-every 5

Outputs (``--out`` prefix): ``_mosaic.jpg/.png``, ``_coverage.png``,
``_frames.csv`` (per-frame decision + metrics), ``_report.txt`` and,
optionally, ``_kept/`` and ``_rejected/<reason>__<name>.jpg``.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):                      # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from live.config import LiveConfig
    from live.pipeline import LiveStitcher
else:
    from .config import LiveConfig
    from .pipeline import LiveStitcher

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def acquisition_key(path: Path):
    """(column, seconds-of-day, step-index) so replay matches scan order.

    The sweep walks Y down/up inside a column and then advances X, so a column
    is a contiguous run of frames.  Sorting by the trailing HHMMSS alone breaks
    when a scan crosses midnight and ignores the column boundary.
    """
    name = path.name
    col = re.search(r"col(\d+)", name, re.I)
    stamp = re.search(r"_(\d{6})(?:\.[A-Za-z0-9]+)$", name)
    step = re.search(r"_(\d+)_\d{6}\.", name)
    t = 0
    if stamp:
        s = stamp.group(1)
        t = int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])
    return (int(col.group(1)) if col else 0, t,
            int(step.group(1)) if step else 0, name)


def list_scan(folder: Path, pattern: str | None, include_af: bool) -> list[Path]:
    files = [p for p in sorted(folder.glob(pattern or "*")) if p.suffix.lower() in IMG_EXT]
    if not include_af and pattern is None:
        # *_AF_* frames are autofocus references: duplicates of their neighbour
        # taken while the Z axis is still settling, so they never help a stitch.
        files = [p for p in files if "_AF_" not in p.name]
    return sorted(files, key=acquisition_key)


def replay(files: list[Path], stitcher: LiveStitcher, delay: float,
           preview: bool) -> None:
    for i, path in enumerate(files, 1):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[LIVE] unreadable frame skipped: {path.name}")
            continue
        # Replay is throttled to processing speed: dropping frames here would
        # bias the benchmark and leave holes in the mosaic.
        stitcher.submit(frame, path.name)
        stitcher.wait_idle()
        if preview:
            show_preview(stitcher, frame, path.name)
        if delay:
            time.sleep(delay)
        if i % 25 == 0:
            print(f"[LIVE] ... {i}/{len(files)} submitted")


def show_preview(stitcher: LiveStitcher, frame: np.ndarray, tag: str) -> None:
    small = cv2.resize(frame, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA)
    d = stitcher.decisions[-1] if stitcher.decisions else None
    if d is not None:
        color = (0, 200, 0) if d.ok else (0, 0, 230)
        cv2.putText(small, f"{'KEEP' if d.ok else 'DROP'} {d.reason}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imshow("live capture", small)
    mosaic, _ = stitcher.mosaic.render()
    if mosaic.size > 3:
        h, w = mosaic.shape[:2]
        s = min(1.0, 400 / max(h, 1))
        cv2.imshow("live mosaic", cv2.resize(mosaic, (max(1, int(w * s)), max(1, int(h * s)))))
    cv2.waitKey(1)


def camera_loop(cam_index: int, stitcher: LiveStitcher, settle: float,
                still_diff: float, preview: bool, max_frames: int) -> None:
    """Grab from the capture card, but only submit frames after motion settles."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"cannot open camera index {cam_index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    prev_small = None
    stable_since = None
    submitted = 0
    last_tag = ""
    print("[LIVE] camera open - Ctrl+C to finish. Capturing only when the "
          f"scene is still for {settle:.2f}s.")
    try:
        while max_frames <= 0 or submitted < max_frames:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
            moving = prev_small is not None and \
                float(np.abs(small.astype(np.int16) - prev_small).mean()) > still_diff
            prev_small = small.astype(np.int16)
            now = time.time()
            if moving:
                stable_since = None
            elif stable_since is None:
                stable_since = now
            elif now - stable_since >= settle:
                # Scene has been static long enough -> a sharp capture.
                tag = time.strftime("col_live_%H%M%S")
                if tag != last_tag:
                    stitcher.submit(frame, tag)
                    submitted += 1
                    last_tag = tag
                stable_since = now
            if preview:
                show_preview(stitcher, frame, last_tag)
    except KeyboardInterrupt:
        print("\n[LIVE] stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", type=Path, help="folder of captured frames")
    src.add_argument("--camera", type=int, help="capture-card index, e.g. 1")
    ap.add_argument("--pattern", default=None, help="glob within --replay")
    ap.add_argument("--include-af", action="store_true",
                    help="also replay *_AF_* autofocus reference frames")
    ap.add_argument("--out", default="runs/live", help="output path prefix")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="replay: seconds to sleep between frames")
    ap.add_argument("--settle", type=float, default=0.8,
                    help="camera: seconds of stillness before a capture counts")
    ap.add_argument("--still-diff", type=float, default=1.5,
                    help="camera: mean abs frame-diff below which the scene is still")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--preview", action="store_true", help="show windows (needs a display)")
    ap.add_argument("--flush-every", type=int, default=10,
                    help="write a partial mosaic every N accepted frames")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale of the saved mosaic image")
    ap.add_argument("--accum-scale", type=float, default=LiveConfig.accum_scale,
                    help="resolution the mosaic is accumulated at (0.6 for whole-part scans)")
    ap.add_argument("--anchors", type=int, default=2,
                    help="frames to register against for drift control")
    ap.add_argument("--blur-ratio", type=float, default=LiveConfig.blur_ratio)
    ap.add_argument("--min-overlap", type=float, default=LiveConfig.min_overlap)
    ap.add_argument("--min-new-area", type=float, default=LiveConfig.min_new_area)
    ap.add_argument("--work-width", type=int, default=LiveConfig.work_width)
    ap.add_argument("--no-keep", action="store_true", help="do not copy accepted frames")
    ap.add_argument("--no-rejects", action="store_true",
                    help="do not write rejected frames (default keeps them for audit)")
    args = ap.parse_args()

    cfg = LiveConfig(flush_every=args.flush_every, output_scale=args.scale,
                     accum_scale=args.accum_scale,
                     anchors=args.anchors, blur_ratio=args.blur_ratio,
                     min_overlap=args.min_overlap,
                     min_new_area=args.min_new_area,
                     work_width=args.work_width)
    out = Path(args.out)
    stitcher = LiveStitcher(
        cfg, out_prefix=str(out),
        save_kept_dir=None if args.no_keep else str(out) + "_kept",
        save_rejected_dir=None if args.no_rejects else str(out) + "_rejected")

    if args.replay:
        files = list_scan(args.replay, args.pattern, args.include_af)
        if not files:
            sys.exit(f"no images found in {args.replay}")
        print(f"[LIVE] replaying {len(files)} frames from {args.replay}")
        replay(files[:args.max_frames or None], stitcher, args.delay, args.preview)
    else:
        camera_loop(args.camera, stitcher, args.settle, args.still_diff,
                    args.preview, args.max_frames)

    t0 = time.time()
    paths = stitcher.finalize()
    s = stitcher.summary()
    print("\n" + "=" * 62)
    print(f"frames {s['frames']} | kept {s['accepted']} | dropped {s['rejected']} "
          f"| mean {s['mean_ms']:.0f} ms/frame | finalize {time.time()-t0:.1f}s")
    print(f"mosaic {paths.get('shape')} -> {paths.get('png')}")
    if s["reject_reasons"]:
        print("dropped: " + ", ".join(f"{k}={v}" for k, v in s["reject_reasons"].items()))


if __name__ == "__main__":
    main()
