"""Online (on-the-fly) stitching pipeline.

Flow per captured frame::

    intrinsic gate  ->  bounded registration  ->  acceptance gate  ->  blend
    (black/flat/blur)   (ORB + prior + NCC)      (overlap/new-area)   (feathered)

Design decisions that matter for a live capture:

* **Nothing is written until a frame has been judged.**  A blurry, redundant or
  unregisterable frame is never blended into the mosaic and never lands in the
  "kept" folder, so only good material reaches the inspection stage.
* **A rejected frame still propagates its pose.**  If a frame is soft but
  registers fine, its transform keeps the motion prior current, so the *next*
  frame is compared against the right place instead of against a stale one.
* **Drift control by anchors.**  Each frame can be registered against the last
  accepted frame *and* older overlapping ones; the implied global poses are
  averaged.  Chaining purely pairwise (what ``blade_stitch.py`` does) lets
  every small error accumulate - measured 1% loop-closure error per column,
  i.e. ~120 px of shear over a 10 000 px travel.
* **Partial output never stops.**  Every ``flush_every`` accepted frames the
  mosaic is written to disk, so a cancel or crash leaves a usable result.
* **Units.**  All poses are 3x3, mapping *full-resolution frame pixels* into
  shared mosaic pixels.
"""
from __future__ import annotations

import csv
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import LiveConfig
from .mosaic import AddResult, IncrementalMosaic
from .quality import QualityReport, assess_intrinsic, focus_ratio
from .registration import Registrar, Registration


@dataclass
class Node:
    """An accepted frame that can serve as a registration anchor."""

    tag: str
    token: str
    pose: np.ndarray          # 3x3, full-res frame px -> mosaic px
    shape: tuple[int, int]    # (h, w) of the full-resolution frame
    blended: bool = True      # False for duplicates that only serve as anchors


@dataclass
class Decision:
    tag: str
    index: int
    status: str = "rejected"          # accepted | rejected | dropped
    reason: str = ""
    dx: float = 0.0
    dy: float = 0.0
    ncc: float = 0.0
    overlap: float = 0.0
    new_area: float = 0.0
    sharp_ratio: float = 1.0
    gain: float = 1.0
    inliers: int = 0
    method: str = ""
    ms: float = 0.0
    pose: np.ndarray | None = None
    quality: QualityReport | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("accepted", "filled")

    @property
    def flagged(self) -> bool:
        """Stitched, but only as a hole filler - worth re-capturing."""
        return self.status == "filled"

    def row(self) -> dict:
        q = self.quality
        pose = (round(float(self.pose[1, 2]), 1) if self.pose is not None else "",
                round(float(self.pose[0, 2]), 1) if self.pose is not None else "")
        return {
            "index": self.index,
            "tag": self.tag,
            "status": self.status,
            "reason": self.reason,
            "dx_px": round(self.dx, 2),
            "dy_px": round(self.dy, 2),
            "ncc": round(self.ncc, 4),
            "overlap": round(self.overlap, 4),
            "new_area": round(self.new_area, 4),
            "sharp_ratio": round(self.sharp_ratio, 3),
            "sharpness": round(q.sharpness, 5) if q else "",
            "mean_luma": round(q.mean_luma, 1) if q else "",
            "gain": round(self.gain, 4),
            "inliers": self.inliers,
            "method": self.method,
            "pose_y": pose[0],
            "pose_x": pose[1],
            "ms": round(self.ms, 1),
        }


class LiveStitcher:
    """Sequential, thread-safe on-the-fly stitcher.

    ``submit()`` is non-blocking; work happens on a background thread so a
    camera or sweep loop is never stalled by registration.
    """

    def __init__(self, cfg: LiveConfig | None = None, out_prefix: str | Path = "live",
                 on_decision=None, save_rejected_dir: str | Path | None = None,
                 save_kept_dir: str | Path | None = None, log=print):
        self.cfg = cfg or LiveConfig()
        self.prefix = str(out_prefix)
        self.log = log
        self.on_decision = on_decision
        self.mosaic = IncrementalMosaic(self.cfg)
        self.registrar = Registrar(self.cfg)
        self.nodes: list[Node] = []
        self.decisions: list[Decision] = []
        self._pending_step: np.ndarray | None = None   # full-res px, in mosaic coords
        self._counter = 0
        self._last_flush = 0
        self._finalized = False
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, self.cfg.queue_size))
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending = 0
        self._canvas_warned = False
        self._last_abs_sharpness = 0.0
        self._kept_dir = Path(save_kept_dir) if save_kept_dir else None
        self._rej_dir = Path(save_rejected_dir) if save_rejected_dir else None
        Path(self.prefix).parent.mkdir(parents=True, exist_ok=True)
        for d in (self._kept_dir, self._rej_dir):
            if d:
                d.mkdir(parents=True, exist_ok=True)
        self._start_worker()

    # ── public API ──────────────────────────────────────────────────────────
    def submit(self, frame: np.ndarray, tag: str | None = None,
               meta: dict | None = None,
               weight: np.ndarray | None = None) -> bool:
        """Queue a captured frame.  Returns False if backpressure dropped it."""
        self._counter += 1
        tag = tag or f"frame_{self._counter:04d}"
        # Count the frame *before* it becomes visible to the worker, otherwise
        # wait_idle() can observe the queue draining and a pending count of 0
        # while a frame is still in flight.
        with self._lock:
            self._pending += 1
        try:
            w = None if weight is None else np.asarray(weight, np.float32)
            self._queue.put((self._counter, tag, frame.copy(), meta or {}, w),
                            timeout=2.0)
            return True
        except queue.Full:
            with self._lock:
                self._pending = max(0, self._pending - 1)
            self._record(Decision(tag=tag, index=self._counter,
                                  status="dropped", reason="backpressure"))
            return False

    def process(self, frame: np.ndarray, tag: str, meta: dict | None = None,
                index: int | None = None, record: bool = True,
                weight: np.ndarray | None = None) -> Decision:
        """Process one frame synchronously and account for it.

        Recording happens here rather than in the worker so that a caller using
        this method directly (an embedded GUI, a test) gets the same
        bookkeeping as the queue: ``summary()`` and the CSV stay consistent with
        what was actually stitched.
        """
        d = self._process(frame, tag, meta, index, weight=weight)
        if record:
            self._record(d)
        return d

    def _process(self, frame: np.ndarray, tag: str, meta: dict | None = None,
                 index: int | None = None,
                 weight: np.ndarray | None = None) -> Decision:
        """The actual decision logic for one frame."""
        t0 = time.perf_counter()
        cfg = self.cfg
        d = Decision(tag=tag, index=self._counter if index is None else index)
        d.extra["frame"] = frame

        # ── 1. cheap intrinsic gate ─────────────────────────────────────────
        rep = assess_intrinsic(frame, cfg)
        d.quality = rep
        if not rep.ok:
            d.reason = ",".join(rep.reasons)
            d.ms = (time.perf_counter() - t0) * 1000
            return d

        # Coarse "is this frame wildly out of focus?" check against the recent
        # baseline.  Cheap (no features) and it stops defocused frames from
        # failing registration and being misfiled as disconnected bands.
        abs_ratio = 1.0
        if self._last_abs_sharpness:
            abs_ratio = rep.sharpness / self._last_abs_sharpness
            d.extra["abs_sharp_ratio"] = round(abs_ratio, 3)
            if abs_ratio < cfg.pre_blur_ratio:
                d.reason = f"out_of_focus(x{abs_ratio:.2f})"
                d.sharp_ratio = abs_ratio
                d.ms = (time.perf_counter() - t0) * 1000
                return d

        token = tag
        self.registrar.prepare(token, frame)

        # ── 2. first frame becomes the origin ───────────────────────────────
        if not self.nodes:
            pose = np.eye(3)
            res = self._safe_add(frame, pose, weight=weight)
            if res is None:
                d.reason = "canvas_full"
                d.ms = (time.perf_counter() - t0) * 1000
                return d
            self._accept(d, token, pose, frame.shape[:2], res)
            d.method = "origin"
            d.pose = pose
            d.ms = (time.perf_counter() - t0) * 1000
            return d

        # ── 3. register against anchors ─────────────────────────────────────
        prior = self._predicted_step()
        results: list[tuple[Node, Registration]] = []
        for node in self._anchor_candidates(prior):
            scale = self.registrar.scale(node.token)
            r = self.registrar.register(node.token, token,
                                        None if prior is None else prior * scale)
            if r is None or r.ncc < cfg.min_ncc or r.overlap < cfg.min_overlap:
                continue
            results.append((node, r))

        if not results:
            # Column change / skipped frames / motor jump: the temporal
            # predecessor is not the right anchor any more.  Search the frames
            # already placed for the nearest ones and retry.
            for node in self._nearest_nodes(prior):
                r = self.registrar.register(node.token, token, None)
                if r is not None and r.ncc >= cfg.min_ncc and r.overlap >= cfg.min_overlap:
                    results.append((node, r))
                if results:
                    break

        if not results and abs_ratio < cfg.soft_unregisterable_ratio:
            # Soft frame that also failed to match: this is a capture fault,
            # not a new region of the part.  Do not start a band for it.
            d.reason = f"unregisterable_soft(x{abs_ratio:.2f})"
            d.sharp_ratio = abs_ratio
            d.ms = (time.perf_counter() - t0) * 1000
            return d

        if not results:
            # No geometric link to anything already stitched: a motor jump, a
            # column with no overlap, or a frame too soft to match.  Never
            # invent a transform for it.
            if not cfg.park_disconnected or \
                    self.mosaic.disconnected >= cfg.max_disconnected_bands:
                d.reason = "unregistered"
                d.method = "none"
                d.ms = (time.perf_counter() - t0) * 1000
                return d
            try:
                res, pose = self.mosaic.add_disconnected(frame)
            except MemoryError as exc:
                self._canvas_full(exc)
                d.reason = "canvas_full"
                d.ms = (time.perf_counter() - t0) * 1000
                return d
            self._accept(d, token, pose, frame.shape[:2], res)
            d.reason = "unregistered-parked"
            d.method = "parked"
            d.ms = (time.perf_counter() - t0) * 1000
            return d

        best_node, best = max(results, key=lambda nr: nr[1].score)
        d.ncc, d.overlap, d.inliers, d.method = (
            best.ncc, best.overlap, best.inliers, best.method)
        d.dx, d.dy = (float(v) for v in best.shift)

        # ── 4. fuse the poses implied by every successful anchor ────────────
        pose = self._fuse_poses(results)
        d.pose = pose

        def reject(reason: str, keep_as_anchor: bool = True) -> Decision:
            d.reason = reason
            if keep_as_anchor:
                # The pose is trustworthy even though the frame is not worth
                # blending - most often a duplicate.  Keep it in the anchor
                # chain: otherwise the next frame is matched across two motor
                # steps, the overlap halves, matching collapses and the scan
                # cascades into "disconnected" bands (observed: 104 bands).
                self._anchor(d, token, pose, frame.shape[:2])
            d.ms = (time.perf_counter() - t0) * 1000
            return d

        # ── 5. gates that need the transform ────────────────────────────────
        if "/implausible-step" in best.method:
            return reject("implausible_step", keep_as_anchor=False)

        sharp = self._sharpness_ratio(best_node, token, best)
        d.sharp_ratio = sharp
        if sharp < cfg.blur_ratio:
            if cfg.blur_policy != "fill":
                return reject(f"blurrier_than_accepted({sharp:.2f})")
            # Keep the frame only as a hole filler: it may fill canvas pixels
            # that are still empty, never overwrite imagery that is already
            # there, and every pixel it contributes is flagged in the QC mask.
            new_area = self.mosaic.probe_new_area(frame, pose)
            d.new_area = new_area
            if new_area < cfg.fill_min_new_area:
                return reject(f"soft_and_redundant({sharp:.2f})")
            res = self._safe_add(frame, pose, empty_only=True, low_quality=True)
            if res is None:
                return reject("canvas_full")
            self._accept(d, token, pose, frame.shape[:2], res)
            d.status = "filled"
            d.reason = f"soft_fill({sharp:.2f})"
            d.ms = (time.perf_counter() - t0) * 1000
            self._maybe_flush()
            return d

        new_area = self.mosaic.probe_new_area(frame, pose)
        d.new_area = new_area
        if new_area < cfg.min_new_area:
            return reject(f"duplicate(new_area={new_area:.2f})")

        # ── 6. accept and blend ─────────────────────────────────────────────
        res = self._safe_add(frame, pose, weight=weight)
        if res is None:
            return reject("canvas_full")
        self._accept(d, token, pose, frame.shape[:2], res)
        d.ms = (time.perf_counter() - t0) * 1000
        self._maybe_flush()
        return d

    # ── live preview (for a GUI) ────────────────────────────────────────────
    def preview(self, max_width: int = 480, timeout: float = 2.0
                ) -> tuple[np.ndarray | None, dict]:
        """Thread-safe snapshot of the mosaic so far, as RGB, for a live view.

        Returns ``(rgb_or_None, summary)``.  Blending happens on the worker
        thread while holding ``self._lock``, so the snapshot takes the same lock
        and can never observe a half-blended canvas (torn image) or a canvas
        being grown underneath it.

        ``max_width`` keeps the per-second cost sane: rendering the full canvas
        is ~180 ms at part scale, and the GUI polls about once a second.
        """
        if not self._lock.acquire(timeout=timeout):
            return None, {}          # worker is busy; the GUI shows the old frame
        try:
            summary = self.summary()
            if self.mosaic.count == 0:
                return None, summary
            mosaic, coverage, lowq = self.mosaic.render()
            covered = int(np.count_nonzero(coverage))
            soft = int(np.count_nonzero(lowq))
            summary["low_quality_pct"] = (100.0 * soft / covered) if covered else 0.0
        finally:
            self._lock.release()

        h, w = mosaic.shape[:2]
        if max_width and w > max_width:
            k = max_width / float(w)
            mosaic = cv2.resize(mosaic, (int(max_width), max(1, int(round(h * k)))),
                                interpolation=cv2.INTER_AREA)
        summary["preview_w"], summary["preview_h"] = mosaic.shape[1], mosaic.shape[0]
        return cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB), summary

    def flush(self, prefix: str | None = None) -> dict:
        """Write the mosaic as it stands now (manual snapshot from the GUI)."""
        with self._lock:
            return self.mosaic.save(prefix or (self.prefix + "_partial"))

    def finalize(self) -> dict:
        """Stop the worker, flush the mosaic and write the audit trail."""
        if self._finalized:
            return self._last_paths
        self._finalized = True
        self.stop()
        with self._lock:
            self._last_paths = self.mosaic.save(self.prefix)
            self._write_csv()
            self._write_report()
        return self._last_paths

    _last_paths: dict = {}

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Block until every submitted frame has been processed."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pending <= 0 and self._queue.empty():
                    return True
            time.sleep(0.01)
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=120)
            self._worker = None

    # ── stats ───────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        acc = [d for d in self.decisions if d.ok]
        rej = [d for d in self.decisions if not d.ok]
        filled = [d for d in acc if d.flagged]
        reasons: dict[str, int] = {}
        for d in rej:
            key = (d.reason or "unknown").split("(")[0]
            reasons[key] = reasons.get(key, 0) + 1
        return {
            "frames": len(self.decisions),
            "accepted": len(acc) - len(filled),
            "filled": len(filled),
            "rejected": len(rej),
            "reject_reasons": reasons,
            "disconnected": self.mosaic.disconnected,
            "mean_ncc": float(np.mean([d.ncc for d in acc])) if acc else 0.0,
            "mean_ms": float(np.mean([d.ms for d in self.decisions])) if self.decisions else 0.0,
            "canvas": self.mosaic.bbox(),
            "columns": self._column_stats(),
        }

    @staticmethod
    def _column_of(tag: str) -> str:
        """Column label from a frame tag such as ``col02_Ydown_14_154952``."""
        head = tag.split("_")[0]
        return head if head.lower().startswith("col") else "scan"

    def _column_stats(self) -> dict:
        stats: dict[str, dict] = {}
        for d in self.decisions:
            col = self._column_of(d.tag)
            c = stats.setdefault(col, {"frames": 0, "stitched": 0, "dropped": 0,
                                      "reasons": {}, "parked": 0})
            c["frames"] += 1
            if d.ok:
                c["stitched"] += 1
            else:
                c["dropped"] += 1
                key = (d.reason or "unknown").split("(")[0]
                c["reasons"][key] = c["reasons"].get(key, 0) + 1
            if d.method == "parked":
                c["parked"] += 1
        return stats

    # ── internals ───────────────────────────────────────────────────────────
    def _start_worker(self) -> None:
        def run():
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    idx, tag, frame, meta, wt = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                with self._lock:
                    try:
                        d = self.process(frame, tag, meta, index=idx,
                                         weight=wt)
                    except Exception as exc:
                        # One bad frame (or one exhausted canvas limit) must
                        # never kill the pipeline mid-scan.
                        import traceback
                        d = Decision(tag=tag, index=idx, status="rejected",
                                     reason=f"error:{type(exc).__name__}",
                                     extra={"detail": traceback.format_exc()})
                        self._record(d)
                    finally:
                        self._pending = max(0, self._pending - 1)
                if self.on_decision:
                    try:
                        self.on_decision(d)
                    except Exception as exc:      # never let a callback kill us
                        self.log(f"[LIVE] decision callback failed: {exc}")

        self._worker = threading.Thread(target=run, daemon=True,
                                        name="live-stitcher")
        self._worker.start()

    def _record(self, d: Decision) -> None:
        self.decisions.append(d)
        label = "FILL" if d.flagged else ("KEEP" if d.ok else "DROP")
        self.log(f"[LIVE] {label:4s} {d.tag:34s} "
                 f"{(d.reason or 'ok'):34s} ncc={d.ncc:.2f} ov={d.overlap:.2f} "
                 f"sharp={d.sharp_ratio:.2f} {d.ms:.0f}ms")
        self._save_frame_files(d)

    def _save_frame_files(self, d: Decision) -> None:
        frame = d.extra.pop("frame", None)
        if frame is None:
            return
        if d.ok and self._kept_dir is not None:
            cv2.imwrite(str(self._kept_dir / f"{d.tag}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif not d.ok and self._rej_dir is not None:
            reason = (d.reason or "rejected").split("(")[0]
            cv2.imwrite(str(self._rej_dir / f"{reason}__{d.tag}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    def _accept(self, d: Decision, token: str, pose: np.ndarray,
                shape: tuple[int, int], res: AddResult) -> None:
        d.status = "accepted"
        d.new_area = res.new_area
        d.gain = res.gain
        d.pose = pose
        if d.quality is not None:
            self._last_abs_sharpness = max(d.quality.sharpness, 1e-9)
        self.nodes.append(Node(tag=d.tag, token=token, pose=pose, shape=shape))
        self._pending_step = None
        self._trim_cache()

    def _canvas_full(self, exc: MemoryError) -> None:
        if not self._canvas_warned:
            self._canvas_warned = True
            self.log(f"[LIVE] STITCHING STOPPED: {exc}")
            self.log("[LIVE] Frames are still being assessed but no longer blended. "
                     "Lower accum_scale or raise max_canvas_px to continue.")

    def _safe_add(self, frame: np.ndarray, pose: np.ndarray,
                  weight: np.ndarray | None = None,
                  **kw) -> AddResult | None:
        """Blend, or give up cleanly if the canvas safety limit is reached.

        ``weight`` (per-pixel focus confidence) is only honoured for pixels that
        are *allowed* to overwrite, i.e. not for hole-filling soft frames.
        """
        if weight is not None and not kw.get("empty_only"):
            kw["weight"] = weight
        try:
            return self.mosaic.add(frame, pose, **kw)
        except MemoryError as exc:
            self._canvas_full(exc)
            return None

    def _anchor(self, d: Decision, token: str, pose: np.ndarray,
                shape: tuple[int, int]) -> None:
        """Register a frame as a pose anchor without blending it."""
        d.pose = pose
        self.nodes.append(Node(tag=d.tag, token=token, pose=pose, shape=shape,
                               blended=False))
        self._pending_step = pose[:2, 2] - self.nodes[-2].pose[:2, 2] \
            if len(self.nodes) > 1 else None
        self._trim_cache()

    def _predicted_step(self) -> np.ndarray | None:
        """Expected step in full-res mosaic pixels, or None if unknown."""
        if self._pending_step is not None:
            return self._pending_step
        if len(self.nodes) < 2:
            return None
        return self.nodes[-1].pose[:2, 2] - self.nodes[-2].pose[:2, 2]

    def _nearest_nodes(self, prior: np.ndarray | None) -> list[Node]:
        """Frames whose placed footprint is closest to where this one should be."""
        if len(self.nodes) < 2:
            return []
        last = self.nodes[-1]
        h, w = last.shape
        hint = last.pose[:2, 2] + (prior if prior is not None else np.zeros(2))
        scored = []
        for node in self.nodes[:-1]:
            d = node.pose[:2, 2] - hint
            # distance in frame units, so non-square frames score sensibly
            norm = float(np.hypot(d[0] / w, d[1] / h))
            if norm <= 2.0:
                scored.append((norm, node))
        scored.sort(key=lambda t: t[0])
        return [n for _, n in scored[:self.cfg.fallback_anchors]]

    def _anchor_candidates(self, prior: np.ndarray | None) -> list[Node]:
        """Most recent accepted node plus older nodes predicted to overlap."""
        out = [self.nodes[-1]]
        if self.cfg.anchors <= 1:
            return out
        last = self.nodes[-1]
        pred = last.pose[:2, 2] + (prior if prior is not None else np.zeros(2))
        h, w = last.shape
        for node in reversed(self.nodes[:-1]):
            x0 = max(pred[0], node.pose[0, 2])
            y0 = max(pred[1], node.pose[1, 2])
            x1 = min(pred[0] + w, node.pose[0, 2] + node.shape[1])
            y1 = min(pred[1] + h, node.pose[1, 2] + node.shape[0])
            inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if inter / float(h * w) >= self.cfg.anchor_overlap:
                out.append(node)
            if len(out) >= self.cfg.anchors:
                break
        return out

    def _fuse_poses(self, results: list[tuple[Node, Registration]]) -> np.ndarray:
        """Average the global poses implied by all successful registrations."""
        best_node, best = max(results, key=lambda nr: nr[1].score)
        poses = [n.pose @ np.vstack([r.full_res(), [0, 0, 1]]) for n, r in results]
        weights = np.array([max(r.score, 1e-3) for _, r in results])
        weights /= weights.sum()
        fused = np.eye(3)
        fused[:2, :2] = (best_node.pose @ np.vstack([best.full_res(), [0, 0, 1]]))[:2, :2]
        fused[:2, 2] = sum(w * p[:2, 2] for w, p in zip(weights, poses))
        return fused

    def _sharpness_ratio(self, node: Node, token: str, reg: Registration) -> float:
        prev_work = self.registrar.work_gray(node.token)
        cur_work = self.registrar.work_gray(token)
        if prev_work is None or cur_work is None:
            return 1.0
        h, w = prev_work.shape[:2]
        warped = cv2.warpAffine(cur_work, reg.affine, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(np.ones(cur_work.shape, np.uint8), reg.affine,
                              (w, h), flags=cv2.INTER_NEAREST) > 0
        return focus_ratio(warped, prev_work, mask)

    def _trim_cache(self) -> None:
        # Keep enough history for the fallback anchor search, which may reach
        # back several frames after a column change.
        keep = {n.token for n in self.nodes[-self.cfg.max_cached_anchors:]}
        self.registrar.keep_only(keep)

    def _maybe_flush(self) -> None:
        if self.cfg.flush_every <= 0:
            return
        if self.mosaic.count - self._last_flush < self.cfg.flush_every:
            return
        self._last_flush = self.mosaic.count
        try:
            self.mosaic.save(self.prefix + "_partial")
            self.log(f"[LIVE] partial mosaic flushed ({self.mosaic.count} frames)")
        except Exception as exc:
            self.log(f"[LIVE] flush failed: {exc}")

    def _write_csv(self) -> None:
        rows = [d.row() for d in self.decisions]
        if not rows:
            return
        path = Path(self.prefix + "_frames.csv")
        with open(path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        self.log(f"[LIVE] per-frame log: {path}")

    def _write_report(self) -> None:
        s = self.summary()
        lines = [
            "Live stitch report",
            "=" * 62,
            f"frames submitted  : {s['frames']}",
            f"accepted (stitched): {s['accepted']}",
            f"soft fill-ins     : {s['filled']}",
            f"rejected          : {s['rejected']}",
            f"disconnected bands: {s['disconnected']}",
            f"mean NCC (kept)   : {s['mean_ncc']:.3f}",
            f"mean time/frame   : {s['mean_ms']:.0f} ms",
            f"canvas bbox       : {s['canvas']}",
            "",
            "Rejections by reason:",
        ]
        for k, v in sorted(s["reject_reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:36s} {v}")
        cols = s["columns"]
        if len(cols) > 1 or "scan" not in cols:
            lines += ["", "Per column:",
                      f"  {'column':10s} {'frames':>6s} {'stitched':>8s} "
                      f"{'dropped':>7s} {'bands':>5s}  verdict"]
            for name, c in cols.items():
                lost = (c["reasons"].get("unregistered", 0)
                        + c["reasons"].get("unregisterable_soft", 0)
                        + c["reasons"].get("canvas_full", 0))
                verdict = "ok" if lost <= 0.15 * max(c["frames"], 1) and not c["parked"] \
                    else "CHECK"
                lines.append(f"  {name:10s} {c['frames']:6d} {c['stitched']:8d} "
                             f"{c['dropped']:7d} {c['parked']:5d}  {verdict}")
            for name, c in cols.items():
                if c["parked"]:
                    lines.append(f"  {name}: stitched in {c['parked']} disconnected band(s) - "
                                 "part of that travel had no overlap")
                lost = c["reasons"].get("unregistered", 0) + \
                    c["reasons"].get("unregisterable_soft", 0)
                if lost > 0.15 * max(c["frames"], 1):
                    lines.append(f"  {name}: {lost} frames could not be placed - too soft or "
                                 "low-texture to register; re-scan this column")

        errs = [d for d in self.decisions if d.reason.startswith("error:")]
        if errs:
            lines += ["", "Errors (pipeline kept running):"]
            for d in errs[:20]:
                lines.append(f"  {d.tag:44s} {d.reason}")
                detail = (d.extra.get("detail") or "").strip().splitlines()
                lines += [f"      {ln}" for ln in detail[-4:]]
        faults = {"out_of_focus", "unregisterable_soft", "underexposed",
                  "overexposed", "no_contrast", "unregistered", "canvas_full"}
        bad = [d for d in self.decisions if not d.ok]
        redundant = [d for d in bad
                     if (d.reason or "").split("(")[0] not in faults]
        missing = [d for d in bad if (d.reason or "").split("(")[0] in faults]
        if redundant:
            lines += ["", f"Redundant frames - safe to delete ({len(redundant)}):"]
            for d in redundant:
                lines.append(f"  {d.tag:44s} {d.reason}")
        if missing:
            lines += ["", f"Capture faults - re-capture these ({len(missing)}):"]
            for d in missing:
                lines.append(f"  {d.tag:44s} {d.reason}")
        soft = [d for d in self.decisions if d.flagged]
        if soft:
            lines += ["", "Stitched as hole fillers - re-capture these for inspection:"]
            for d in soft:
                lines.append(f"  {d.tag:44s} {d.reason}")
        lines += ["", f"mosaic: {self.prefix}_mosaic.png"]
        Path(self.prefix + "_report.txt").write_text("\n".join(lines) + "\n")
        self.log("[LIVE] report:\n" + "\n".join(lines[:14]))
