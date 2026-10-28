#!/usr/bin/env python3
"""Find (and optionally delete) frames that cannot help - or actively hurt - a stitch.

Dry run by default: it prints a plan and writes a CSV, and changes nothing.

How a frame is judged
---------------------
1. **Hard gates** (no reference needed): unreadable, underexposed,
   overexposed, no contrast, grossly defocused.
2. **Duplicate**: near-identical to the previous kept frame in the same column
   (stalled motor / repeated capture).
3. **Unmatchable**: cannot be registered against either neighbour with a sane
   NCC.  On the recorded scans this catches the defocused stretches exactly -
   a soft frame yields too few ORB matches, and any model it does produce
   correlates poorly with its neighbour.
4. **Soft**: registers, but the sharpness measured *on the overlapping pixels*
   is below ``--blur-ratio`` of its neighbour's.  Comparing the same surface is
   the only reliable way here: absolute sharpness varies ~10x within one pass
   because of illumination drift, and the objective is soft at the field edges
   in *every* frame, so a per-frame absolute threshold either deletes half the
   scan or nothing at all.

Usage
-----
    python purge.py SCAN_DIR                       # report only (safe)
    python purge.py SCAN_DIR --move-to ../trash    # move flagged frames out
    python purge.py SCAN_DIR --delete              # delete (asks first)
    python purge.py SCAN_DIR --drop-af             # also flag *_AF_* frames
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from live.config import LiveConfig
    from live.quality import assess_intrinsic, focus_ratio
    from live.registration import Registrar
    from live.run_live import list_scan
else:
    from .config import LiveConfig
    from .quality import assess_intrinsic, focus_ratio
    from .registration import Registrar
    from .run_live import list_scan

DUP_NCC = 0.995


@dataclass
class Verdict:
    name: str
    column: str
    keep: bool = True
    reason: str = ""
    sharpness: float = 0.0
    overlap_sharp_ratio: float = 1.0
    ncc_prev: float = 0.0
    ncc_next: float = 0.0
    contrast: float = 0.0
    mean_luma: float = 0.0
    notes: list[str] = field(default_factory=list)


def small_gray(frame: np.ndarray, size=(320, 180)) -> np.ndarray:
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size,
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def near_duplicate(a: np.ndarray, b: np.ndarray) -> bool:
    """Same picture within a couple of pixels of actuator wobble."""
    if a.std() < 1e-3 or b.std() < 1e-3:
        return False
    best = -1.0
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(b, m, (b.shape[1], b.shape[0]),
                                     borderMode=cv2.BORDER_REPLICATE)
            best = max(best, float(np.corrcoef(a.ravel(), shifted.ravel())[0, 1]))
    return best >= DUP_NCC


def overlap_ratio(reg, registrar, prev_token: str, cur_token: str) -> float:
    prev_work = registrar.work_gray(prev_token)
    cur_work = registrar.work_gray(cur_token)
    if prev_work is None or cur_work is None:
        return 1.0
    h, w = prev_work.shape[:2]
    warped = cv2.warpAffine(cur_work, reg.affine, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones(cur_work.shape, np.uint8), reg.affine,
                          (w, h), flags=cv2.INTER_NEAREST) > 0
    return focus_ratio(warped, prev_work, mask)


def analyse(paths: list[Path], cfg: LiveConfig, blur_ratio: float,
            drop_af: bool) -> list[Verdict]:
    reg = Registrar(cfg)
    verdicts: list[Verdict] = []
    frames: dict[str, np.ndarray] = {}

    # pass 1: decode, hard gates, features
    for p in paths:
        column = p.name.split("_")[0]
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            verdicts.append(Verdict(p.name, column, False, "unreadable"))
            continue
        frames[p.name] = frame
        rep = assess_intrinsic(frame, cfg)
        v = Verdict(p.name, column, True, "", rep.sharpness,
                    contrast=rep.contrast, mean_luma=rep.mean_luma)
        if not rep.ok:
            v.keep, v.reason = False, ",".join(rep.reasons)
        reg.prepare(p.name, frame)
        verdicts.append(v)

    # pass 2: duplicate + registration-based checks
    usable = [v for v in verdicts if v.reason != "unreadable"]
    for i, v in enumerate(usable):
        if not v.keep:
            continue
        cur = frames.get(v.name)
        if cur is None:
            continue
        prev = next((u for u in reversed(usable[:i]) if u.keep and u.column == v.column), None)
        nxt = next((u for u in usable[i + 1:] if u.column == v.column and u.reason != "unreadable"), None)

        if prev is not None and near_duplicate(small_gray(frames[prev.name]), small_gray(cur)):
            v.keep, v.reason = False, f"duplicate_of:{prev.name}"
            continue
        if drop_af and "_AF_" in v.name:
            v.keep, v.reason = False, "autofocus_reference"
            continue

        ratios, nccs = [], []
        for other in (prev, nxt):
            if other is None or other.name not in frames:
                continue
            r = reg.register(other.name, v.name, None)
            if r is not None:
                ratios.append(overlap_ratio(r, reg, other.name, v.name))
                nccs.append(r.ncc)
        if prev is not None and nccs:
            v.ncc_prev = round(nccs[0], 3)
        if nxt is not None and len(nccs) > 1:
            v.ncc_next = round(nccs[1], 3)

        if not ratios:
            v.keep, v.reason = False, "unmatchable"
            continue
        best = max(ratios)          # one good neighbour is enough
        v.overlap_sharp_ratio = round(best, 3)
        if best < blur_ratio:
            v.keep, v.reason = False, f"soft(ratio={best:.2f})"
    return verdicts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--pattern", default=None)
    ap.add_argument("--blur-ratio", type=float, default=0.72,
                    help="flag when overlap sharpness < this ratio of neighbour")
    ap.add_argument("--drop-af", action="store_true",
                    help="also flag *_AF_* autofocus reference frames")
    ap.add_argument("--move-to", type=Path, default=None)
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip confirmation")
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()

    paths = list_scan(args.folder, args.pattern, include_af=True)
    if not paths:
        sys.exit(f"no images in {args.folder}")
    verdicts = analyse(paths, LiveConfig(), args.blur_ratio, args.drop_af)

    flagged = [v for v in verdicts if not v.keep]
    counts: dict[str, int] = {}
    for v in flagged:
        key = v.reason.split("(")[0].split(":")[0]
        counts[key] = counts.get(key, 0) + 1

    total = len(verdicts)
    print(f"{total} frames in {args.folder}")
    print(f"  keep    : {total - len(flagged)}")
    print(f"  flagged : {len(flagged)}")
    for k, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:24s} {n}")
    if flagged:
        print("\nfirst 20 flagged:")
        for v in flagged[:20]:
            print(f"  {v.name:44s} {v.reason}")

    csv_path = args.csv_out or (args.folder / "purge_report.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(asdict(verdicts[0])))
        wr.writeheader()
        wr.writerows(asdict(v) for v in verdicts)
    print(f"\nper-frame CSV: {csv_path}")

    if not flagged:
        return
    if args.delete and args.move_to:
        sys.exit("choose either --delete or --move-to")
    if not args.delete and not args.move_to:
        print("\nDry run - nothing changed. Re-run with --move-to DIR or "
              "--delete when the list looks right.")
        return
    if not args.yes:
        action = "DELETE" if args.delete else f"move to {args.move_to}"
        if input(f"\n{action} {len(flagged)} files? [y/N] ").strip().lower() not in ("y", "yes"):
            print("aborted")
            return
    if args.move_to:
        args.move_to.mkdir(parents=True, exist_ok=True)
        for v in flagged:
            shutil.move(str(args.folder / v.name), str(args.move_to / v.name))
        print(f"moved {len(flagged)} files to {args.move_to}")
    else:
        for v in flagged:
            (args.folder / v.name).unlink()
        print(f"deleted {len(flagged)} files")


if __name__ == "__main__":
    main()
