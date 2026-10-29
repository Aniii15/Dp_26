#!/usr/bin/env python3
"""Planar close-range blade mosaic from a motorized raster scan.

NOTE (offline/batch version): this script stitches a folder *after* a scan.  For
on-the-fly stitching with frame rejection, see ../live/README.md.  Measured on
the recorded scans, this implementation's phase-correlation seeding is defeated
by the periodic cooling-hole pattern and its ECC score is not a geometry check,
so treat the diagnostics file as informational only.

Uses phase-correlation for translation initialization, ECC affine refinement,
ORB+RANSAC fallback, exposure compensation and distance-transform feathering.
It deliberately avoids panorama/spherical projection: these are near-planar
macro views of a fixed object, not a rotating camera panorama.

Install: pip install opencv-python numpy
Run: python blade_stitch.py /path/to/scan_20261021_214947 --out blade_mosaic
Optional: --pattern 'col01_Ydown_*.jpg' (default selects sequential Ydown/Yup/Xr
images and excludes *_AF focus-reference frames).
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
import cv2
import numpy as np


def scan_order(p: Path):
    # Filename sequence is the primary deterministic order; image matching
    # still verifies/refines each transition.
    m = re.search(r'_(\d{6})\.(?:jpe?g|png|tif{1,2})$', p.name, re.I)
    return int(m.group(1)) if m else p.name


def gray_small(im, max_dim=1400):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, max_dim/max(g.shape))
    if scale < 1:
        g = cv2.resize(g, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    return g, scale


def estimate(prev, cur):
    """Warp cur into prev coordinates; return 2x3 affine, in full-res pixels."""
    a, sa = gray_small(prev)
    b, sb = gray_small(cur)
    # For nominally fixed camera orientation, phase correlation gives a good
    # translation seed. ECC can then absorb tiny rotation/scale/shear.
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        shift, response = cv2.phaseCorrelate(np.float32(a), np.float32(b))
        warp[0, 2], warp[1, 2] = shift
        cc, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 250, 1e-7), None, 5)
        if cc < .12:
            raise RuntimeError(f'low ECC correlation {cc:.3f}')
        # IMPORTANT: OpenCV ECC returns the sampling transform from template
        # coordinates into input coordinates. For composing image-to-mosaic
        # poses, we need the opposite (cur -> prev), so invert it here.
        W = np.vstack([warp, [0, 0, 1]]).astype(np.float64)
        warp = np.linalg.inv(W)[:2].astype(np.float32)
        # Convert from the independently resized coordinate systems back to
        # full-resolution pixels: x_full_prev = S_prev^-1 W S_cur x_full_cur.
        # The current scan uses equal-size frames, but keep the general scaling.
        warp[0, 2] /= sa
        warp[1, 2] /= sa
        warp[0, 0] *= sb/sa
        warp[0, 1] *= sb/sa
        warp[1, 0] *= sb/sa
        warp[1, 1] *= sb/sa
        return warp, float(cc), 'ECC-inverted'
    except (cv2.error, RuntimeError):
        # ORB homography RANSAC is useful on textured blade/fixture features;
        # enforce near-affine transform to prevent projective runaway.
        orb = cv2.ORB_create(5000)
        k1, d1 = orb.detectAndCompute(a, None)
        k2, d2 = orb.detectAndCompute(b, None)
        if d1 is None or d2 is None:
            raise RuntimeError('not enough features for registration')
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
        good = [m for m, n in pairs if m.distance < .72*n.distance]
        if len(good) < 12:
            raise RuntimeError(f'only {len(good)} ORB matches')
        src = np.float32([k2[m.queryIdx].pt for m in good])
        dst = np.float32([k1[m.trainIdx].pt for m in good])
        H, inl = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000, confidence=.999)
        if H is None or inl.sum() < 10:
            raise RuntimeError('RANSAC registration failed')
        H = H.astype(np.float32)
        H[:, 2] /= sb
        H[:, :2] *= sa/sb
        return H, float(inl.mean()), 'ORB/RANSAC'


def compose(images, names, outbase):
    # Chain adjacent transforms in common canvas coordinates; adjacent
    # sequencing is determined by timestamps/names (scan acquisition order).
    transforms = [np.eye(3, dtype=np.float64)]
    reports = []
    for i in range(1, len(images)):
        A, score, method = estimate(images[i-1], images[i])
        T = np.vstack([A, [0, 0, 1]]).astype(np.float64)
        transforms.append(transforms[-1] @ T)
        reports.append((names[i-1], names[i], method, score))
    corners = []
    for im, T in zip(images, transforms):
        h, w = im.shape[:2]
        c = np.float64([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]) @ T.T
        corners.extend(c[:, :2])
    corners = np.asarray(corners)
    lo = np.floor(corners.min(0)).astype(int)
    hi = np.ceil(corners.max(0)).astype(int)
    W, H = int(hi[0]-lo[0]), int(hi[1]-lo[1])
    if W*H > 1_000_000_000:
        raise RuntimeError(
            'Mosaic canvas exceeds safety limit; inspect registration drift.')
    offset = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], float)
    accum = np.zeros((H, W, 3), np.float32)
    weights = np.zeros((H, W), np.float32)
    # Distance-to-edge feathering minimizes visible seams. Per-frame gain is
    # estimated against already accumulated overlap (bounded to avoid artifacts).
    for idx, (im, T) in enumerate(zip(images, transforms)):
        M = (offset@T)[:2].astype(np.float32)
        h, w = im.shape[:2]
        warped = cv2.warpAffine(
            im, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        mask = cv2.warpAffine(np.ones((h, w), np.uint8),
                              M, (W, H), flags=cv2.INTER_NEAREST) > 0
        dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        wt = np.minimum(dist, 80).astype(np.float32)*mask
        # Robust RGB-channel-independent luminance gain on existing overlap.
        overlap = mask & (weights > 0)
        if overlap.sum() > 1000:
            old = accum[overlap]/weights[overlap, None]
            new = warped[overlap].astype(np.float32)
            old_y = cv2.cvtColor(
                old.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
            new_y = cv2.cvtColor(
                new.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(float)
            valid = new_y > 8
            if valid.sum() > 100:
                gain = float(
                    np.clip(np.median(old_y[valid]/new_y[valid]), .75, 1.33))
                warped = np.clip(warped.astype(float)*gain,
                                 0, 255).astype(np.uint8)
        accum += warped.astype(np.float32)*wt[:, :, None]
        weights += wt
    result = np.zeros_like(accum, dtype=np.uint8)
    valid = weights > 0
    result[valid] = np.clip(
        accum[valid]/weights[valid, None], 0, 255).astype(np.uint8)
    # Crop to supported bounds and save transparent coverage mask for QC.
    ys, xs = np.where(valid)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    result = result[y0:y1, x0:x1]
    coverage = (valid[y0:y1, x0:x1]*255).astype(np.uint8)
    cv2.imwrite(str(outbase)+'_mosaic.jpg', result,
                [cv2.IMWRITE_JPEG_QUALITY, 97])
    # Also write lossless output so inspection/measurement is not degraded by
    # another JPEG encoding pass.
    cv2.imwrite(str(outbase)+'_mosaic.png', result,
                [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(outbase)+'_coverage.png', coverage)
    with open(str(outbase)+'_registration.txt', 'w') as f:
        f.write(
            'Adjacent-frame registration diagnostics (low scores, gaps, or drift need review)\n')
        for a, b, m, s in reports:
            f.write(f'{a} -> {b}: {m}, score={s:.4f}\n')
    print(
        f'Saved {outbase}_mosaic.jpg ({result.shape[1]}x{result.shape[0]}), coverage mask, diagnostics')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('folder', type=Path)
    ap.add_argument('--out', default='blade')
    ap.add_argument('--pattern', default=None,
                    help='glob; defaults to sequential primary scan frames, excluding AF references')
    ap.add_argument('--max-images', type=int, default=0,
                    help='debug limit; 0=all')
    args = ap.parse_args()
    files = sorted(args.folder.glob(args.pattern or '*'), key=scan_order)
    files = [p for p in files if p.suffix.lower() in (
        '.jpg', '.jpeg', '.png', '.tif', '.tiff')]
    if args.pattern is None:
        files = [p for p in files if '_AF_' not in p.name and re.search(
            r'_(Ydown|Yup|Xr)', p.name, re.I)]
    if args.max_images:
        files = files[:args.max_images]
    if len(files) < 2:
        sys.exit('Need at least two images; check folder/pattern.')
    images = []
    kept = []
    for p in files:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is not None:
            images.append(im)
            kept.append(p.name)
    print(
        f'Registering {len(images)} images in filename/time order. First: {kept[0]} / last: {kept[-1]}')
    compose(images, kept, args.out)


if __name__ == '__main__':
    main()
