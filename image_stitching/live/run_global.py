#!/usr/bin/env python3
"""Re-stitch a finished scan folder with the global stitcher.

Uses every frame in the folder (nothing is rejected), solves all poses at once
and blends without blocks.  See ``global_stitch.py`` for how.

Examples::

    # from image_stitching/
    python live/run_global.py ../main_code/blade_images/scan_20261105_200702

    # explicit output prefix, half-resolution output for a quick look
    python live/run_global.py SCAN_DIR --out runs/blade --scale 0.5

Outputs (prefix): ``_mosaic.jpg/.png``, ``_coverage.png``, ``_sources.png``
(which frame each pixel came from, seams in white, motion-model-only frames in
red), ``_low_detail.png``, ``_poses.csv`` and ``_report.txt``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                      # allow direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from live.global_stitch import GlobalConfig, stitch_folder
else:
    from .global_stitch import GlobalConfig, stitch_folder


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="scan folder (e.g. main_code/blade_images/scan_...)")
    ap.add_argument("--out", help="output prefix (default: <folder>/../../blade_mosaic/"
                                  "global_<scan name>)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="output scale (1.0 = capture resolution)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-rotation", action="store_true",
                    help="translation-only poses (old behaviour)")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"not a folder: {folder}")
        return 2
    out = args.out
    if not out:
        mosaic_dir = folder.resolve().parent.parent / "blade_mosaic"
        out = str(mosaic_dir / f"global_{folder.name}")
    cfg = GlobalConfig(output_scale=args.scale, threads=args.threads,
                       solve_rotation=not args.no_rotation)
    t0 = time.time()
    paths = stitch_folder(folder, out, cfg)
    print(f"done in {time.time() - t0:.0f}s -> {paths['jpg']}")
    print(f"report: {paths['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
