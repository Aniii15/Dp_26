#!/usr/bin/env python3
"""Headless smoke test for the Tk interface.

There is no display (and often no tkinter) on a build machine or in CI, so the
whole UI is constructed against a *stub* tkinter that records what the layout
code asks for.  That catches the failure modes that actually break this file:

* a widget used before it is created,
* a bad keyword argument, wrong parent type,
* a callback that explodes on the first invocation,
* formatting bugs in the mosaic/status text (e.g. a renamed summary key),
* a log-tag rule that returns a tag that was never configured.

It does **not** validate real geometry - that still needs one look on a machine
with a display.  Run:  python test_ui_headless.py
"""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WIDGET_CALLS: list[tuple[str, dict]] = []
CALLBACK_ERRORS: list[str] = []


class _Var:
    """Mimics tk.StringVar / IntVar / BooleanVar."""

    def __init__(self, master=None, value=None, **kw):
        self._value = value
        self._traces = []

    def get(self):
        return self._value

    def set(self, value):
        self._value = value

    def trace_add(self, mode, callback):
        self._traces.append((mode, callback))

    def trace(self, mode, callback):
        self._traces.append((mode, callback))


class _Widget:
    """Anything with pack/grid/bind/configure; records constructor kwargs."""

    _kind = "Widget"

    def __init__(self, master=None, cnf=None, **kw):
        WIDGET_CALLS.append((self._kind, dict(kw)))
        self._kw = dict(kw)
        self._children = []
        if master is not None and hasattr(master, "_children"):
            master._children.append(self)

    # geometry managers ------------------------------------------------
    def pack(self, *a, **kw):
        return self

    def grid(self, *a, **kw):
        return self

    def place(self, *a, **kw):
        return self

    def pack_propagate(self, *a, **kw):
        return self

    # widget protocol --------------------------------------------------
    def configure(self, cnf=None, **kw):
        self._kw.update(kw)
        return self

    config = configure

    def cget(self, key):
        return self._kw.get(key, "")

    def bind(self, sequence=None, func=None, add=None):
        return ""

    def unbind(self, sequence=None):
        return None

    def bind_all(self, *a, **kw):
        return ""

    def unbind_all(self, *a, **kw):
        return None

    def after(self, ms, func=None, *args):
        return "after#1"

    def after_cancel(self, *a, **kw):
        return None

    def destroy(self):
        return None

    def update_idletasks(self):
        return None

    def winfo_width(self):
        return 640

    def winfo_height(self):
        return 420

    def winfo_screenwidth(self):
        return 1920

    def winfo_screenheight(self):
        return 1080

    def winfo_children(self):
        return list(self._children)

    def winfo_exists(self):
        return 1

    def focus_set(self):
        return None

    def columnconfigure(self, *a, **kw):
        return None

    def rowconfigure(self, *a, **kw):
        return None

    def tag_configure(self, *a, **kw):
        return None

    def insert(self, *a, **kw):
        return None

    def delete(self, *a, **kw):
        return None

    def see(self, *a, **kw):
        return None

    def index(self, *a, **kw):
        return "1.0"

    def yview(self, *a, **kw):
        return None

    def set(self, *a, **kw):
        return None

    def get(self, *a, **kw):
        return 1.0

    def moveto(self, *a, **kw):
        return None

    def select(self, *a, **kw):
        return None

    def focus(self, *a, **kw):
        return None

    def add_tag(self, *a, **kw):
        return None

    def tag_add(self, *a, **kw):
        return None

    def yview_scroll(self, *a, **kw):
        return None

    def xview(self, *a, **kw):
        return None

    def itemconfigure(self, *a, **kw):
        return None

    def create_window(self, *a, **kw):
        return 1

    def bbox(self, *a, **kw):
        return (0, 0, 100, 100)

    def mainloop(self, *a, **kw):
        return None

    def protocol(self, *a, **kw):
        return None

    def title(self, *a, **kw):
        return None

    def geometry(self, *a, **kw):
        return "1280x900+0+0" if not a else None

    def minsize(self, *a, **kw):
        return None

    def maxsize(self, *a, **kw):
        return None

    def resizable(self, *a, **kw):
        return None

    def iconify(self, *a, **kw):
        return None

    def attributes(self, *a, **kw):
        return None

    def wm_attributes(self, *a, **kw):
        return None

    def tk_setPalette(self, *a, **kw):          # noqa: N802
        return None

    def option_add(self, *a, **kw):
        return None

    def winfo_rootx(self):
        return 0

    def winfo_rooty(self):
        return 0

    def state(self, *a, **kw):
        return None

    def add(self, *a, **kw):
        return None

    def sash_place(self, *a, **kw):
        return None


def _widget_class(kind):
    return type(kind, (_Widget,), {"_kind": kind})


class _Style(_Widget):
    """ttk.Style stub."""

    _kind = "ttk.Style"

    def theme_use(self, name=None):
        return "clam" if name is None else None

    def theme_names(self):
        return ("clam", "default")

    def map(self, *a, **kw):
        return None

    def layout(self, *a, **kw):
        return None

    def element_create(self, *a, **kw):
        return None


def _install_stub_tk():
    tk = types.ModuleType("tkinter")
    for kind in ("Frame", "Label", "Button", "Entry", "Radiobutton",
                 "Checkbutton", "Scale", "Canvas", "Scrollbar", "PanedWindow",
                 "Notebook", "Style", "Toplevel", "Text", "LabelFrame",
                 "Separator", "Spinbox", "Listbox"):
        setattr(tk, kind, _widget_class(kind))
    tk.Tk = _widget_class("Tk")
    for var in ("StringVar", "IntVar", "DoubleVar", "BooleanVar"):
        setattr(tk, var, type(var, (_Var,), {}))
    tk.TclError = type("TclError", (Exception,), {})

    ttk = types.ModuleType("tkinter.ttk")
    for kind in ("Notebook", "Scrollbar", "Frame", "Label", "Button",
                 "Entry", "Separator", "Combobox", "Treeview", "Progressbar"):
        setattr(ttk, kind, _widget_class("ttk." + kind))
    ttk.Style = _Style
    when = types.ModuleType("tkinter.ttk.when")
    tk.ttk = ttk

    st = types.ModuleType("tkinter.scrolledtext")
    st.ScrolledText = _widget_class("ScrolledText")

    mb = types.ModuleType("tkinter.messagebox")
    mb.showerror = lambda *a, **k: None
    mb.showinfo = lambda *a, **k: None
    mb.showwarning = lambda *a, **k: None
    mb.askyesno = lambda *a, **k: True

    const = types.ModuleType("tkinter.constants")
    for name in ("END", "LEFT", "RIGHT", "TOP", "BOTTOM", "X", "Y", "BOTH",
                 "N", "S", "E", "W", "NW", "NSEW", "HORIZONTAL", "VERTICAL",
                 "DISABLED", "NORMAL", "WORD", "CENTER", "FLAT", "RAISED",
                 "SUNKEN", "GROOVE", "RIDGE"):
        setattr(const, name, name.lower())

    sys.modules.update({
        "tkinter": tk,
        "tkinter.ttk": ttk,
        "tkinter.scrolledtext": st,
        "tkinter.messagebox": mb,
        "tkinter.constants": const,
    })
    return tk


def _install_stub_net():
    """requests is only used for HTTP calls; keep the import cheap."""
    if "requests" not in sys.modules:
        req = types.ModuleType("requests")
        req.get = lambda *a, **k: (_ for _ in ()).throw(OSError("no network"))
        req.exceptions = types.SimpleNamespace(RequestException=OSError)
        sys.modules["requests"] = req


def main() -> int:
    _install_stub_tk()
    _install_stub_net()

    import main3

    failures: list[str] = []

    def check(name, fn):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            import traceback
            tb = traceback.extract_tb(sys.exc_info()[2])[-1]
            where = f" (line {tb.lineno}: {tb.line})" if tb else ""
            failures.append(f"{name}: {type(exc).__name__}: {exc}{where}")
            print(f"FAIL {name}: {type(exc).__name__}: {exc}{where}")

    # 1. the whole window builds -------------------------------------------
    app = None

    def build():
        nonlocal app
        app = main3.MotorControlApp()

    check("window builds (all tabs, panels, status bar)", build)
    if app is None:
        return 1

    # 2. every tab page was registered -------------------------------------
    def tabs():
        kinds = [k for k, _ in WIDGET_CALLS]
        assert "ttk.Notebook" in kinds, "no notebook created"
        # five pages, one per tab
        assert len(app._notebook._children) == 5, app._notebook._children
    check("5 tab pages attached to the notebook", tabs)

    # 3. widgets referenced by the periodic ticks exist --------------------
    def tick_targets():
        for attr in ("_feed_label", "_feed_zoom_var", "_feed_status_var",
                     "_md_status_var", "_bd_status_var", "_autosweep_status",
                     "_autosweep_timer_var", "_frame_a_status", "_log_text",
                     "_mosaic_label", "_mosaic_stat_var", "_sweep_stitch_var",
                     "_last_stitch_label", "btn_autosweep", "btn_stop_autosweep",
                     "btn_af", "btn_go_start", "_sb_conn_var", "_sb_motion_var",
                     "_sb_feed_var", "_sb_col_var", "_sb_stitch_var"):
            assert hasattr(app, attr), f"missing widget {attr}"
    check("all widgets used by ticks/updates exist", tick_targets)

    # 4. the mosaic renderer survives a realistic summary ------------------
    def mosaic_render():
        import numpy as np
        rgb = np.zeros((60, 40, 3), np.uint8)
        summary = {"accepted": 43, "filled": 8, "rejected": 31,
                   "disconnected": 0, "mean_ncc": 0.955,
                   "canvas": (-4, 0, 1200, 2738), "low_quality_pct": 5.9}
        app._mosaic_render(rgb, summary)
        text = app._mosaic_stat_var.get()
        assert "43 kept" in text and "NCC 0.95" in text, text
        assert "43 accepted" in app._sweep_stitch_var.get()
        assert "stitched 43" in app._sb_stitch_var.get()
        # empty / errored summary must not raise
        app._mosaic_render(None, {})
        app._mosaic_render(None, {"error": "boom"})
        app._mosaic_render(None, {"accepted": 0})
    check("mosaic renderer + counters", mosaic_render)

    # 5. stitch decisions colour the label ---------------------------------
    def decisions():
        class D:
            def __init__(self, ok, flagged, reason, tag="x"):
                self.ok, self.flagged, self.reason, self.tag = ok, flagged, reason, tag

        for ok, flagged, expected in ((True, False, "KEEP"),
                                      (True, True, "FILL"),
                                      (False, False, "DROP")):
            app._stitch_decision_ui(D(ok, flagged, "reason"))
            assert app._last_stitch_var.get().startswith(expected), \
                app._last_stitch_var.get()
    check("stitch decision label (KEEP/FILL/DROP)", decisions)

    # 6. log tagging returns only configured tags -------------------------
    def log_tags():
        configured = {"warn", "error", "stitch", "ok", ""}
        samples = ["[SWEEP] ✓ boundary detected", "[LIVE] DROP frame ncc=0.10",
                   "[STITCH] mosaic saved", "[AF] Timeout waiting for motor",
                   "⛔ EMERGENCY STOP SENT", "plain line",
                   "[AI] Failed to load model"]
        for line in samples:
            tag = app._log_tag(line)
            assert tag in configured, (line, tag)
        assert app._log_tag("[LIVE] DROP x") == "error"
        assert app._log_tag("⛔ EMERGENCY STOP SENT") == "error"
    check("log severity tags", log_tags)

    # 7. resize handler only reacts to the toplevel -----------------------
    def resize():
        class E:
            def __init__(self, widget, height):
                self.widget, self.height = widget, height

        child = main3.tk.Frame()
        app._on_window_resize(E(child, 1200))       # ignored
        app._on_window_resize(E(app, 900))
        assert 230 <= app._vis_h <= 900, app._vis_h
        app._on_window_resize(E(app, 600))
    check("window resize keeps the visual row in bounds", resize)

    # 8. expensive network paths are safe without a device ----------------
    def no_camera():
        # _start_feed opens a camera; make sure a failure is reported, not raised
        import cv2
        real = cv2.VideoCapture

        class Failing:
            def isOpened(self):
                return False

        cv2.VideoCapture = lambda *a, **k: Failing()
        try:
            messages = []
            main3.messagebox.showerror = lambda *a, **k: messages.append(a)
            app._start_feed()
            assert messages, "no error surfaced when the camera is missing"
        finally:
            cv2.VideoCapture = real
    check("missing camera is reported, not fatal", no_camera)

    # 9. a real stitcher drives the real mosaic panel ----------------------
    def end_to_end():
        try:
            import numpy as np
            sys.path.insert(0, str(HERE.parent / "image_stitching"))
            from live import LiveConfig, LiveStitcher
        except Exception as exc:                       # noqa: BLE001
            print(f"SKIP end-to-end mosaic (deps missing: {exc})")
            return

        import tempfile
        rng = np.random.default_rng(7)
        tex = rng.integers(60, 200, (900, 900, 3), dtype=np.uint8)
        tex = __import__("cv2").blur(tex, (5, 5))
        frame = lambda dy: tex[dy:dy + 600, 100:800].copy()

        import os
        with tempfile.TemporaryDirectory() as td:
            app._stitcher = LiveStitcher(
                LiveConfig(flush_every=0), out_prefix=os.path.join(td, "run"),
                on_decision=app._on_stitch_decision, log=lambda m: None)
            app._stitcher_dir = td
            for i, dy in enumerate((0, 150, 300)):
                app._stitcher.submit(frame(dy), f"f{i}")
            app._stitcher.wait_idle(60)
            rgb, summary = app._stitcher.preview(400)
            app._mosaic_render(rgb, summary)
            app._stitch_finalize()
            assert summary["accepted"] >= 3, summary
            assert app._mosaic_stat_var.get().count("kept") == 1
            assert app._sweep_stitch_var.get().startswith("3 accepted"), \
                app._sweep_stitch_var.get()
    check("real stitcher -> mosaic panel", end_to_end)

    # 10. background results reach Tk only through the queue ---------------
    def ui_queue():
        seen = []
        app._post(lambda: seen.append("a"))
        app._post(lambda: 1 / 0)              # a raising callback is contained
        app._post(lambda: seen.append("b"))
        app._poll_ui_queue()
        assert seen == ["a", "b"], seen
        assert not app._ui_queue.qsize()
    check("_post/_poll_ui_queue hand-off (and error containment)", ui_queue)

    def no_cross_thread_after():
        src = (HERE / "main3.py").read_text()
        # Tk's after() must never be called from a worker thread
        assert "after(0," not in src, "cross-thread self.after(0, ...) found"
    check("no cross-thread after() left in main3.py", no_cross_thread_after)

    # 11. the GUI's own autofocus path, driven by a fake motor + camera ------
    def gui_autofocus():
        if main3._focus_engine is None:
            print("SKIP GUI autofocus (focus_engine missing)")
            return
        import focus_tests as ft

        rig = ft.Rig(focus_z=900.0, tilt=600.0)
        saved_move_z = main3.move_z          # restored before returning
        main3.Z_STEPS_PER_MM = 1600
        app._get_frame = rig.capture           # camera
        moves = []

        def fake_move_z(steps, direction, log_fn=None):
            # move_z(delta, direction): unsigned count + direction flag
            d = int(steps) * (1 if direction else -1)
            assert int(steps) > 0, f"move_z got a signed step count: {steps}"
            moves.append(d)
            rig.move(d)
            return True, "ok"

        main3.move_z = fake_move_z
        app._wait_motion_stop = lambda timeout_s=10.0: True
        app._af_mode_var.set("Region-wise (recommended)")
        app.af_range_var.set("4000")
        app.af_pred_var.set("600")
        app.af_limit_var.set("16000")
        app.af_y_c_var.set("200")
        app.af_y_f_var.set("50")
        app.af_y_d_var.set("0")
        app.af_backlash_var.set("0")

        res = app._autofocus()
        app._poll_ui_queue()          # the real app drains this 20x/second
        assert res is not None, "region engine returned nothing"
        assert abs(res.z_peak - 900) <= 120, res.z_peak
        assert res.bracketed, res.curve_variation
        assert app._af_result_var.get() != "not focused yet"
        assert app._af_detail_var.get().startswith("•")
        assert len(moves) > 10, len(moves)

        # a bare button press has no scan position, so the trend exists but is
        # empty; points arrive from sweep stops, which know their (column, row)
        assert app._focus_trend is not None and len(app._focus_trend) == 0
        # second stop, further down the column: the trend predicts a shift and
        # the engine searches a narrow window around it
        rig2 = ft.Rig(focus_z=700.0, tilt=600.0)

        def move_z2(steps, direction, log_fn=None):
            d = int(steps) * (1 if direction else -1)
            rig2.move(d)
            return True, "ok"

        main3.move_z = move_z2
        app._get_frame = rig2.capture
        for v, u in ((0.0, 900.0), (1.0, 800.0), (2.0, 700.0)):
            app._focus_trend.add(1.0, v, u)
        shift = app._focus_trend.predict(1.0, 3.0)
        assert shift is not None, "trend did not predict"
        hinted = app._autofocus(hint=shift, u=1.0, v=3.0)
        app._poll_ui_queue()
        assert hinted is not None, "hinted scan returned nothing"
        assert abs(hinted.z_peak - 700) <= 150, hinted.z_peak
        assert hinted.bracketed, (hinted.z_peak, hinted.curve_variation)
        assert len(app._focus_trend) == 4, len(app._focus_trend)

        # the result panel reports the Z footprint, so "it searches more range
        # than needed" is measurable instead of a guess
        detail = app._af_detail_var.get()
        assert "searched" in detail and "travelled" in detail, detail
        assert "cap" in detail, detail

        # and the weights handed to the stitcher are a real 0..1 map
        w = app._focus_weight(rig.capture())
        assert w is not None and w.shape == (ft.H, ft.W), None if w is None else w.shape
        assert 0.0 <= float(w.min()) and float(w.max()) <= 1.0
        main3.move_z = saved_move_z
        main3.Z_STEPS_PER_MM = 1600
    check("GUI autofocus path (region engine, hint, weights)", gui_autofocus)

    # 12. the emergency stop overrides whatever is running -----------------
    def safety_latch():
        import requests
        main3._SAFETY.rearm()
        sent = []

        def sender():
            sent.append(1)
            return (len(sent) >= 2), "STOP"
        main3._SAFETY.sender = sender

        assert not main3._SAFETY.halted
        main3._SAFETY.trip("test")
        assert main3._SAFETY.halted
        for _ in range(50):                 # the hardware stop runs in a thread
            if main3._SAFETY.acknowledged:
                break
            time.sleep(0.02)
        assert main3._SAFETY.acknowledged, "stop was never acknowledged"
        assert len(sent) >= 2, "the stop was not retried until it landed"

        # motion is refused while latched, and nothing reaches the network
        calls = []
        real_get = requests.get
        requests.get = lambda *a, **k: calls.append(a) or type(
            "R", (), {"text": "OK"})()
        try:
            for fn, args in ((main3.move_z, (10, 1)), (main3.move_x, (10, 1)),
                             (main3.move_r, (10, 1)),
                             (main3.move_actuator, ("down", 500, 70)),
                             (main3.set_speed, (800,))):
                try:
                    ok, msg = fn(*args)
                except main3.Halted:
                    ok, msg = False, "Halted"
                assert not ok, f"{fn.__name__} returned {ok!r}/{msg!r}"
            assert not calls, f"latched stop still sent {calls}"
        finally:
            requests.get = real_get

        # re-arming is the only way out, and it re-enables motion
        main3._SAFETY.rearm()
        assert not main3._SAFETY.halted
        requests.get = lambda *a, **k: calls.append(a) or type(
            "R", (), {"text": "OK"})()
        try:
            ok, _ = main3.move_z(10, 1)
            assert ok and calls, "motion did not resume after re-arm"
        finally:
            requests.get = real_get
    check("estop: latches, refuses motion, retries /stop, re-arms", safety_latch)

    def safety_interrupts_a_wait():
        """A stop wakes a sleeping scan immediately - not after the delay."""
        main3._SAFETY.rearm()
        main3._SAFETY.sender = lambda: (True, "STOP")
        outcome = {}

        def sleeper():
            started = time.time()
            try:
                main3.haltable_sleep(30.0)
                outcome["result"] = "slept"
            except main3.Halted:
                outcome["result"] = "aborted"
            outcome["delay"] = time.time() - started

        t = threading.Thread(target=sleeper, daemon=True)
        t.start()
        time.sleep(0.1)
        main3._SAFETY.trip("test")
        t.join(5)
        assert outcome.get("result") == "aborted", outcome
        assert outcome["delay"] < 2.0, f"took {outcome['delay']:.1f}s to abort"
        main3._SAFETY.rearm()
    check("estop: a 30 s wait aborts in well under a second", safety_interrupts_a_wait)

    def safety_aborts_a_running_sweep():
        """The sweep unwinds mid-move instead of finishing the column."""
        main3._SAFETY.rearm()
        main3._SAFETY.sender = lambda: (True, "STOP")
        finished, moves = [], []

        def fake_move(direction, duration_ms, speed, log_fn=None):
            moves.append(direction)
            if len(moves) == 2:
                main3._SAFETY.trip("operator")
            return True, "OK"

        saved = (main3.move_actuator, app._get_frame, app._save_sweep_image,
                 main3.set_speed, app._wait_motion_stop, app._boundary_pct)
        main3.move_actuator = fake_move
        app._get_frame = lambda: np.zeros((48, 64, 3), np.uint8)
        app._save_sweep_image = lambda *a, **k: None
        app._boundary_pct = lambda *a, **k: {"left": 0.0, "right": 0.0,
                                             "upper": 0.0, "lower": 0.0}
        app._wait_motion_stop = lambda *a, **k: True
        main3.set_speed = lambda *a, **k: (True, "OK")
        try:
            app._autosweep_running = True
            app._sweep_worker()
            app._poll_ui_queue()
            finished.append(True)
        finally:
            (main3.move_actuator, app._get_frame, app._save_sweep_image,
             main3.set_speed, app._wait_motion_stop, app._boundary_pct) = saved
        assert finished, "the sweep worker did not return"
        assert len(moves) <= 3, f"sweep kept moving after the stop: {moves}"
        assert not app._autosweep_running, "sweep still marked running"
        assert "HALTED" in app._autosweep_status.get(), app._autosweep_status.get()
        main3._SAFETY.rearm()
    check("estop: a running sweep unwinds mid-move", safety_aborts_a_running_sweep)

    def safety_ui():
        main3._SAFETY.rearm()
        main3._SAFETY.sender = lambda: (True, "STOP")
        app._emergency_stop()            # trip
        app._poll_ui_queue()
        assert main3._SAFETY.halted
        assert "RE-ARM" in app.btn_estop.cget("text").upper()
        assert str(app.btn_autosweep.cget("state")) == "disabled"
        assert str(app.btn_af.cget("state")) == "disabled", \
            "a motion control stayed live while latched"
        # a latched machine refuses to start a new scan
        app._autosweep_running = False
        app._start_autosweep()
        assert not app._autosweep_running, "sweep started while latched"
        # pressing the same button re-arms and restores the controls
        app._emergency_stop()
        app._poll_ui_queue()
        assert not main3._SAFETY.halted
        assert "EMERGENCY STOP" in app.btn_estop.cget("text").upper()
        assert str(app.btn_af.cget("state")) != "disabled", \
            "controls were not restored on re-arm"
    check("estop: button latches, disables motion, re-arms", safety_ui)

    # 16. scan geometry: step sizes are measured, not guessed --------------
    def geometry_units():
        import scan_geometry as sg
        g = sg.ScanGeometry()
        # the recorded col01: same 1200 ms command, wildly different travel
        for dy in (412, 310, 315, 243, 427, 264, 421, 218, 434, 242):
            g.observe("Y", 0, dy, (1080, 1920))
        assert g.y_samples == 10
        assert 280 < g.measured_y_px < 360, g.measured_y_px
        assert 0.6 < g.y_overlap(1080) < 0.75, g.y_overlap(1080)

        # one column advance: 0.4 mm of stepper -> ~363 px of image
        g.observe("X", 363, -15, (1080, 1920), moved_mm=0.4)
        assert g.px_per_mm and 850 < g.px_per_mm < 980, g.px_per_mm
        assert 0.75 < g.x_overlap(1920) < 0.85, g.x_overlap(1920)

        # a 0.4 mm column step re-images 4/5 of the previous column: wasteful
        wide = g.columns_needed(8000, 1920, 0.81)
        assert wide >= 15, wide
        assert g.columns_needed(8000, 1920, 0.50) < wide / 2, "no saving found"

        # the suggestions have to be *stricter* about the target, not random
        x = g.suggested_x_step_mm(1920)
        assert x and abs(1920 * (1 - g.target_x_overlap) - x * g.px_per_mm) < 20, x

        # one bad registration must not double the step
        before = g.y_time_s
        g.observe("Y", 0, 1070, (1080, 1920))    # nonsense: a whole frame
        g.observe("Y", 0, 5, (1080, 1920))       # duplicate
        assert abs(g.suggested_y_time(1080) - before) < 0.35, g.suggested_y_time(1080)

        # auto-tune off means the command never moves on its own
        g.auto_tune = False
        assert g.apply((1080, 1920)) == {}
    check("scan geometry: px/mm, overlap, column count, damping", geometry_units)

    def geometry_panel():
        geo = app._scan_geo
        assert geo is not None, "no geometry controller"
        app._frame_shape = (1080, 1920)
        app._scan_geo_observe("Y", 0.0, 320.0)
        app._last_x_cmd_mm = 0.4
        app._scan_geo_observe("X", 363.0, -10.0)
        app._scan_geo_refresh()
        text = app._scan_geo_var.get()
        assert "overlap" in text and "px/mm" in text, text
        # entries round-trip into the controller
        app.sg_x_step.set("1.20")
        app.sg_y_overlap.set("50")
        app._scan_geo_sync()
        assert abs(app._scan_geo.x_step_mm - 1.20) < 1e-6
        assert abs(app._scan_geo.target_y_overlap - 0.50) < 1e-6
        # out-of-range typing is clamped, not fatal
        app.sg_y_time.set("nonsense")
        app.sg_x_step.set("999")
        app._scan_geo_sync()
        assert app._scan_geo.x_step_mm <= 20.0
        app._scan_geo_recommend()
        assert float(app.sg_x_step.get()) > 0.4, "recommendation was not applied"
    check("scan geometry panel: readout, entries, clamping, recommendation", geometry_panel)

    def geometry_drives_the_sweep():
        """The sweep must use the panel's numbers, not hardcoded ones."""
        main3._SAFETY.rearm()
        main3._SAFETY.sender = lambda: (True, "STOP")
        app.sg_y_time.set("2.00")
        app.sg_y_speed.set("90")
        app.sg_x_step.set("1.50")
        app.sg_auto.set(False)
        app._scan_geo_sync()
        seen = []
        saved = (main3.move_actuator, main3.move_x, app._get_frame,
                 app._save_sweep_image, main3.set_speed,
                 app._wait_motion_stop, app._boundary_pct, app._autofocus)
        moves = [0]

        def rec_act(direction, duration_ms, speed, log_fn=None):
            seen.append(("Y", duration_ms, speed))
            moves[0] += 1
            return True, "OK"

        def rec_x(steps, direction, log_fn=None):
            seen.append(("X", steps, direction))
            app._autosweep_stop_evt.set()      # one column is enough to check
            return True, "OK"

        main3.move_actuator = rec_act
        main3.move_x = rec_x
        app._get_frame = lambda: np.zeros((48, 64, 3), np.uint8)
        app._save_sweep_image = lambda *a, **k: None
        app._wait_motion_stop = lambda *a, **k: True
        main3.set_speed = lambda *a, **k: (True, "OK")
        app._autofocus = lambda **k: None
        # the boundary is "found" as soon as we have moved twice
        app._boundary_pct = lambda *a, **k: {
            "left": 0.0, "right": 0.0,
            "upper": 50.0 if moves[0] >= 2 else 0.0,
            "lower": 50.0 if moves[0] >= 2 else 0.0}
        try:
            app._autosweep_running = True
            app._autosweep_stop_evt.clear()
            app._sweep_worker()
            app._poll_ui_queue()
        finally:
            (main3.move_actuator, main3.move_x, app._get_frame,
             app._save_sweep_image, main3.set_speed,
             app._wait_motion_stop, app._boundary_pct, app._autofocus) = saved
        ys = [x for x in seen if x[0] == "Y"]
        assert len(ys) >= 3, ys
        # the sweep moves use the panel's pulse width ...
        assert all(abs(d - 2000) <= 1 for _, d, _ in ys[:2]), ys
        # ... the corrective move is the fixed 1.25x of it, still derived
        assert abs(ys[2][1] - 2500) <= 1, ys
        assert all(s == 90 for _, _, s in ys), ys
        assert abs(app._last_x_cmd_mm - 1.50) < 1e-6, app._last_x_cmd_mm
    check("scan geometry: the sweep obeys the panel (Y ms, PWM, X mm)",
          geometry_drives_the_sweep)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S)")
        for f in failures:
            print("  -", f)
        return 1
    print(f"all checks passed ({len(WIDGET_CALLS)} widgets constructed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
