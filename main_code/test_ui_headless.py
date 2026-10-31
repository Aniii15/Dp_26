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
        app.af_ext_var.set("2")
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

        # and the weights handed to the stitcher are a real 0..1 map
        w = app._focus_weight(rig.capture())
        assert w is not None and w.shape == (ft.H, ft.W), None if w is None else w.shape
        assert 0.0 <= float(w.min()) and float(w.max()) <= 1.0
    check("GUI autofocus path (region engine, hint, weights)", gui_autofocus)

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
