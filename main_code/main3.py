"""
ESP32 Wireless Motor Control — Desktop GUI  (run: python main3.py)

Motors: X, Y, Z = MKS SERVO42D closed-loop steppers on 4 mm-pitch lead
screws (3200 pulses/rev -> 800 pulses = 1 mm), R rotation on TB6600
(1600 steps/rev).  Firmware: esp32_motors - a move is queued and the HTTP reply
comes back at once; /status reports the pulses still queued per axis.
Talks to the ESP32 over WiFi via HTTP GET.

Features: live HDMI camera feed, visual motion detection (closed-loop
feedback), blade boundary detection, Siamese-CNN focus scoring with
in-app training, coarse-to-fine autofocus, and an autonomous
boundary-guided sweep that saves images per column.
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, filedialog
import threading
import csv
import time
import queue
import os
import sys
import subprocess

import requests
import numpy as np
import cv2

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import scan_geometry as _scan_geometry
except ImportError:            # optional: the sweep falls back to its constants
    _scan_geometry = None

try:
    import focus_engine as _focus_engine
except ImportError:              # optional: only needed for region-wise AF
    _focus_engine = None

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ──────────────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────────────
ESP32_IP = "192.168.4.1"
ESP32_PORT = 80
BASE_URL = f"http://{ESP32_IP}:{ESP32_PORT}"
TIMEOUT_S = 5

# X/Y/Z mechanics: 3200 pulses per motor revolution, 4 mm lead-screw pitch.
PULSES_PER_REV = 3200
SCREW_PITCH_MM = 4.0
STEPS_PER_MM = PULSES_PER_REV / SCREW_PITCH_MM   # 800 pulses = 1 mm (1.25 um/pulse)
Z_STEPS_PER_MM = STEPS_PER_MM
X_STEPS_PER_MM = STEPS_PER_MM
Y_STEPS_PER_MM = STEPS_PER_MM

# Y was a timed linear actuator (1.2 s @ PWM 70 ~= 0.35 mm); it is now a
# stepper, so the sweep commands it in mm (ScanGeometry.y_step_mm).
Y_CORRECTION_RATIO = 1.25   # corrective / back-off Y move vs. a sweep move
# DIR pin level for each Y direction.  VERIFY ON THE RIG; swap if inverted.
Y_DIR = {"down": 0, "up": 1}
MOVE_TIMEOUT_S = 30.0       # longest wait for a queued move to finish
SWEEP_SPEED_US_DEFAULT = 1000   # sweep step half-period: 1000 us = 500 Hz
R_STEPS_PER_REV = 1600
DEGREES_PER_STEP = 360 / R_STEPS_PER_REV

CAMERA_INDEX = 0               # OpenCV capture index of the HDMI card

# Motion detection tuning (live-adjustable from the UI)
MOTION_THRESHOLD = 20000   # changed-pixel count that counts as motion
STILLNESS_FRAMES = 5       # consecutive still frames = motor stopped
FRAME_DIFF_THRESHOLD = 20      # per-pixel diff that counts as "changed"

# Boundary detection tuning
BOUNDARY_THRESHOLD = 30  # gray level separating blade from background
UPPER_BOUNDARY_THRESHOLD = 50  # stricter threshold for upper-boundary checks

# Live (on-the-fly) mosaicking.  When enabled, every frame the sweep saves is
# also pushed to the streaming stitcher in image_stitching/live, which blends it
# into a growing mosaic and rejects blurry/duplicate/unregisterable frames as
# they arrive.  Set to False to keep the original save-only behaviour.
# Autofocus engine.  "region" = region-wise wide-range scan (focus_engine.py):
# scores a tile grid, searches the whole Z travel, extends the window when the
# peak sits on an edge, and can bracket+stack.  "cnn" = the original Siamese
# hill climb, kept because it is trained in-house and still useful for small
# corrections near focus.
FOCUS_MODE = "region"
FOCUS_CONFIG = {
    # Hard cap on the Z search, either side of where the scan starts (0 = as far
    # as z_travel_limit).  This is a bound, not a distance: a scan stops as soon
    # as the peak is bracketed, and a prediction narrows it further.  Set it to
    # the blade's real depth envelope + margin once you know it - every capture
    # outside that envelope is pure cost.
    "range_steps": 3000,     # +-0.94 mm cap
    "coarse_steps": 100,      # 0.125 mm - resolution of the bracketing walk
    "fine_steps": 25,         # 31 um - refinement pass around the peak
    "z_travel_limit": 12000,  # 15 mm either way: refuse to drive further
    "settle_s": 0.20,         # wait after each Z move before grabbing a frame
    "backlash_steps": 0,      # set to ~2x the measured backlash if Z is sloppy
    "in_focus_ratio": 0.90,   # tile counts as focused at 90 % of its own best
    "max_samples": 60,        # hard cap on captures per focus scan
    "predictive_window": 300,  # window confirmed around a predicted focus (0.375 mm)
    # Focus on the LEFT 40 % of the view (x0, y0, x1, y1 as fractions).  The
    # blade curves away from the lens towards the right, so no single Z makes
    # the whole frame sharp; the right side is seen again - by the sharp left
    # part of the next strip - one X step later, and the global stitcher picks
    # the sharpest view of every spot.  (1.0, 1.0) corner = whole frame.
    "roi": (0.0, 0.0, 0.40, 1.0),
}
# Bracket + fuse a Z-stack whenever a single Z cannot make the whole frame sharp.
FOCUS_STACKING = True
# 3 exposures (peak, +-span/2) fuse as well as 5 and cost 2 extra captures
# instead of 4; raise it if the blade curves so steeply that 3 leaves soft tiles.
FOCUS_STACK_SAMPLES = 3
# Feed a per-pixel focus weight to the stitcher, so a blurred region loses the
# blend to a sharp neighbour instead of being baked into the mosaic.
STITCH_FOCUS_WEIGHTS = True

LIVE_STITCH = True
LIVE_STITCH_CONFIG = {
    "flush_every": 10,      # write a partial mosaic every N accepted frames
    "output_scale": 1.0,    # 0.5 halves mosaic memory 4x for very large parts
    "anchors": 2,           # frames registered against, for drift control
    # The live mosaic is a preview: the deliverable is the global stitch made
    # from every saved frame after the sweep.  Half resolution keeps the
    # preview fast and well inside memory (no more "canvas_full").
    "accum_scale": 0.5,
}

# After every sweep, re-stitch the whole scan folder with the global stitcher
# (image_stitching/live/global_stitch.py): every frame is used, poses are
# solved over all overlaps at once, and the blend has no blocks.  Runs in the
# background; the result replaces the live preview in the mosaic panel.
# Where the learned sweep geometry (px/mm, achieved steps) is remembered between
# runs.  A calibration is only valid for one rig + lens + resolution, so it is
# re-learned from every scan and never shipped as a constant.
GEOMETRY_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "blade_mosaic", "scan_geometry.json")

GLOBAL_STITCH = True
GLOBAL_STITCH_CONFIG = {
    "output_scale": 1.0,    # 0.5 for a quick look at very large parts
}


# ──────────────────────────────────────────────────────────────────────────────
#  Motion Detection (closed-loop motor feedback)
# ──────────────────────────────────────────────────────────────────────────────

class MotionDetector:
    """Detects motor motion by frame differencing."""

    def __init__(self):
        self._last_gray = None
        self._moving = False
        self._still_count = 0
        self._last_motion = 0.0
        self._lock = threading.Lock()

    def process_frame(self, frame_bgr):
        if frame_bgr is None:
            return
        with self._lock:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if self._last_gray is None:
                self._last_gray = gray
                return
            diff = cv2.absdiff(gray, self._last_gray)
            motion_pixels = (diff > FRAME_DIFF_THRESHOLD).sum()
            if motion_pixels > MOTION_THRESHOLD:
                self._still_count = 0
                self._moving = True
                self._last_motion = time.time()
            else:
                self._still_count += 1
                if self._still_count >= STILLNESS_FRAMES:
                    self._moving = False
            self._last_gray = gray

    def is_moving(self) -> bool:
        with self._lock:
            return self._moving

    def is_still(self) -> bool:
        with self._lock:
            return not self._moving and self._still_count >= STILLNESS_FRAMES

    def motion_since(self, t: float) -> bool:
        """Has any motion been seen after time ``t``?"""
        with self._lock:
            return self._last_motion >= t


# ──────────────────────────────────────────────────────────────────────────────
#  HTTP Communication Layer
# ──────────────────────────────────────────────────────────────────────────────

def http_get(endpoint: str, params: dict = None, log_fn=None) -> tuple[bool, str]:
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT_S)
        text = resp.text.strip()
        if log_fn:
            log_fn(f"GET {endpoint} {params or ''} → {text}")
        return True, text
    except requests.exceptions.ConnectionError:
        msg = f"Connection error: Cannot reach {ESP32_IP}"
    except requests.exceptions.Timeout:
        msg = f"Timeout: ESP32 did not respond within {TIMEOUT_S}s"
    except Exception as e:
        msg = f"Unexpected error: {e}"
    if log_fn:
        log_fn(f"[ERROR] {msg}")
    return False, msg


def move_z(steps: int, direction: int, log_fn=None) -> tuple[bool, str]:
    return http_get("/z", {"steps": steps, "dir": direction}, log_fn)


def move_x(steps: int, direction: int, log_fn=None) -> tuple[bool, str]:
    return http_get("/x", {"steps": steps, "dir": direction}, log_fn)


def move_r(steps: int, direction: int, log_fn=None) -> tuple[bool, str]:
    return http_get("/r", {"steps": steps, "dir": direction}, log_fn)


def move_y(steps: int, direction: int, log_fn=None) -> tuple[bool, str]:
    return http_get("/y", {"steps": steps, "dir": direction}, log_fn)


def move_y_dir(direction: str, mm: float, log_fn=None) -> tuple[bool, str]:
    """Move Y by ``mm`` towards "up" / "down" (replaces the timed actuator)."""
    return move_y(mm_to_y_steps(mm), Y_DIR[direction], log_fn)


def axis_remaining(axis: str) -> int | None:
    """Pulses still queued on ``axis`` ("X"/"Y"/"Z"/"R"), None if unreadable.

    The firmware's /status reply looks like ``"X=0 Y=120 Z=0 R=0 "``."""
    ok, resp = http_get("/status")
    if not ok:
        return None
    for tok in resp.split():
        k, _, v = tok.partition("=")
        if k == axis and v.isdigit():
            return int(v)
    return None


def wait_idle(axis: str, timeout_s: float = MOVE_TIMEOUT_S) -> bool:
    """Block until the firmware has emitted every queued pulse on ``axis``.

    esp32_motors answers "OK" *before* the move happens, so without this a
    frame grabbed right after a move command is taken mid-move."""
    end = time.time() + timeout_s
    while time.time() < end:
        rem = axis_remaining(axis)
        if rem is None:
            return False
        if rem == 0:
            return True
        time.sleep(0.02)
    return False


def emergency_stop(log_fn=None) -> tuple[bool, str]:
    return http_get("/stop", log_fn=log_fn)


def set_speed(us: int, log_fn=None) -> tuple[bool, str]:
    return http_get("/speed", {"us": us}, log_fn)


def ping_esp32(log_fn=None) -> bool:
    ok, resp = http_get("/ping", log_fn=log_fn)
    return ok and "PONG" in resp


# ──────────────────────────────────────────────────────────────────────────────
#  Unit Helpers
# ──────────────────────────────────────────────────────────────────────────────

def mm_to_z_steps(mm: float) -> int:
    return max(1, int(round(mm * Z_STEPS_PER_MM)))


def mm_to_x_steps(mm: float) -> int:
    return max(1, int(round(mm * X_STEPS_PER_MM)))


def mm_to_y_steps(mm: float) -> int:
    return max(1, int(round(mm * Y_STEPS_PER_MM)))


def degrees_to_r_steps(deg: float) -> int:
    return max(1, int(round(deg / DEGREES_PER_STEP)))


# ──────────────────────────────────────────────────────────────────────────────
#  Siamese Focus Model
# ──────────────────────────────────────────────────────────────────────────────

if HAS_TORCH:
    class CropMiddleThird:
        def __call__(self, img):
            w, h = img.size
            new_w = w // 3
            return TF.crop(img, top=0, left=new_w, height=h, width=new_w)

    class SiameseFocusCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )
            self.classifier = nn.Sequential(
                nn.Linear(16384 * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

        def forward(self, img_a, img_b):
            feat_a = self.extractor(img_a)
            feat_b = self.extractor(img_b)
            return self.classifier(torch.cat((feat_a, feat_b), dim=1))


# Absolute so the model loads whether the GUI is launched from main_code/,
# the repo root, or an IDE (a bare filename is resolved against the cwd).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "siamese_focus_model.pth")
AI_MODEL = AI_TRANSFORM = AI_DEVICE = AI_OPTIMIZER = AI_CRITERION = None
_ai_load_attempted = False


def load_siamese_model():
    """Returns (model, transform, device, optimizer, criterion) or all Nones."""
    none = (None,) * 5
    if not HAS_TORCH:
        print("[WARNING] PyTorch not installed. Siamese disabled.")
        return none
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file {MODEL_PATH} not found.")
        return none
    try:
        print("[AI] Loading SiameseFocusCNN...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseFocusCNN().to(device)
        model.load_state_dict(torch.load(
            MODEL_PATH, map_location=device, weights_only=True))
        model.eval()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        transform = transforms.Compose([
            CropMiddleThird(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        print(f"[AI] Successfully loaded {MODEL_PATH}!")
        return model, transform, device, optimizer, criterion
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return none


def init_ai():
    global AI_MODEL, AI_TRANSFORM, AI_DEVICE, AI_OPTIMIZER, AI_CRITERION, _ai_load_attempted
    if AI_MODEL is None and HAS_TORCH and not _ai_load_attempted:
        _ai_load_attempted = True
        AI_MODEL, AI_TRANSFORM, AI_DEVICE, AI_OPTIMIZER, AI_CRITERION = load_siamese_model()


def _frame_to_tensor(frame_bgr):
    gray = Image.fromarray(cv2.cvtColor(
        frame_bgr, cv2.COLOR_BGR2RGB)).convert('L')
    return AI_TRANSFORM(gray).unsqueeze(0).to(AI_DEVICE)


def predict_focus_change(frame_a_bgr, frame_b_bgr) -> float:
    """Probability that frame B is sharper than frame A (0.5 = unknown)."""
    init_ai()
    if AI_MODEL is None or frame_a_bgr is None or frame_b_bgr is None:
        return 0.5
    try:
        with torch.no_grad():
            output = AI_MODEL(_frame_to_tensor(frame_a_bgr),
                              _frame_to_tensor(frame_b_bgr))
            return torch.sigmoid(output).item()
    except Exception as e:
        print(f"[ERROR] predict_focus_change failed: {e}")
        return 0.5


def train_siamese_pair(frame_a_bgr, frame_b_bgr, label: float, lr: float):
    init_ai()
    if AI_MODEL is None:
        return False, "Model not loaded"
    try:
        for param_group in AI_OPTIMIZER.param_groups:
            param_group['lr'] = lr

        tensor_a = _frame_to_tensor(frame_a_bgr)
        tensor_b = _frame_to_tensor(frame_b_bgr)
        target = torch.tensor([[label]], dtype=torch.float32, device=AI_DEVICE)

        AI_MODEL.train()
        AI_OPTIMIZER.zero_grad()
        loss = AI_CRITERION(AI_MODEL(tensor_a, tensor_b), target)
        loss.backward()
        AI_OPTIMIZER.step()
        AI_MODEL.eval()

        torch.save(AI_MODEL.state_dict(), MODEL_PATH)
        return True, f"Loss: {loss.item():.4f}"
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────────────────────────────────────
#  Boundary Detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_boundaries(frame_bgr, threshold=None) -> tuple[int, int, int, int] | None:
    """Bounding box (left, right, upper, lower) of the blade, or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        threshold = BOUNDARY_THRESHOLD
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return x, x + w, y, y + h


def draw_focus_roi(frame, roi) -> np.ndarray:
    """Outline the autofocus region (fractions x0, y0, x1, y1) in yellow."""
    if not roi or tuple(roi) == (0.0, 0.0, 1.0, 1.0):
        return frame
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = roi
    cv2.rectangle(frame, (int(x0 * w) + 2, int(y0 * h) + 2),
                  (int(x1 * w) - 2, int(y1 * h) - 2), (0, 215, 255), 2)
    return frame


def draw_boundaries(frame, boundaries) -> np.ndarray:
    """Draw green boundary lines on frame."""
    if boundaries is None:
        return frame
    left, right, upper, lower = boundaries
    h, w = frame.shape[:2]
    green, thickness = (0, 255, 0), 2
    cv2.line(frame, (left, 0), (left, h), green, thickness)
    cv2.line(frame, (right, 0), (right, h), green, thickness)
    cv2.line(frame, (0, upper), (w, upper), green, thickness)
    cv2.line(frame, (0, lower), (w, lower), green, thickness)
    return frame


# ──────────────────────────────────────────────────────────────────────────────
#  Color Palette (industrial dark theme)
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0f1117",
    "surface":   "#1a1d27",
    "panel":     "#21253a",
    "border":    "#2e3450",
    "accent":    "#4f8ef7",
    "accent2":   "#f7954f",
    "success":   "#4fca8a",
    "danger":    "#f74f4f",
    "warn":      "#f7c94f",
    "text":      "#d8dff0",
    "muted":     "#636b8a",
    "btn_bg":    "#2b3155",
    "btn_hover": "#3b4575",
}

FONT_MONO = ("Courier New", 10)
FONT_LABEL = ("Consolas", 10)
FONT_TITLE = ("Consolas", 12, "bold")
FONT_SMALL = ("Consolas", 9)


# ──────────────────────────────────────────────────────────────────────────────
#  Main Application Window
# ──────────────────────────────────────────────────────────────────────────────

class MotorControlApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ESP32 Motor Control System — Blade Inspector")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        # ~25 menu bar + ~80 dock
        w, h = min(1280, sw - 40), min(900, sh - 110)
        self.geometry(f"{w}x{h}+{max(0, (sw-w)//2)}+40")
        self.minsize(900, 560)
        # self.geometry("1280x860")
        # self.minsize(1100, 800)
        self.configure(bg=C["bg"])

        self._log_queue = queue.Queue()
        self._motion_detector = MotionDetector()
        self._connected = False

        self._feed_running = False
        self._feed_stop_evt = threading.Event()
        self._autosweep_running = False
        self._autosweep_stop_evt = threading.Event()
        self._autosweep_start_time = 0.0
        self._autosweep_column_start_time = 0.0
        self._autosweep_column_num = 0

        self._cap: cv2.VideoCapture | None = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None        # annotated RGB frame for the UI
        self._latest_frame_raw = None    # raw BGR frame for inference/saving
        self._latest_boundaries = None
        self._latest_pred_str = "Waiting..."
        self._frame_a_bgr = None
        self._feed_fps = 0.0

        # Streaming mosaic (optional; enabled by LIVE_STITCH)
        self._stitcher = None
        self._stitcher_dir = None
        self._capture_log = None         # (path, seq) of the running sweep
        self._global_running = False
        self._mosaic_pending = False     # a preview fetch is in flight
        self._focus_trend = None         # FocusTrend, created with the engine
        self._focus_result = None        # last FocusResult, for the Focus tab
        self._focus_ref = 0.0            # running sharpness reference for weights
        self._ui_queue = queue.Queue()   # work for the Tk thread (thread-safe)
        self._mosaic_photo = None        # keeps the Tk image alive
        self._vis_h = 300                # height of the feed|mosaic row

        self._build_ui()
        self._poll_log_queue()
        self._poll_ui_queue()
        self._update_connection_indicator()
        self._feed_ui_tick()
        self._status_ui_tick()
        self._autosweep_timer_tick()
        self._mosaic_ui_tick()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- Scrollable Element ---------------------------------------------------
    def _build_scrollable(self, parent, builder):
        """Wrap a panel in a vertical scroll viewport so it can never be clipped."""
        canvas = tk.Canvas(parent, bg=C["bg"], highlightthickness=0, bd=0)
        bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=C["bg"])
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(
            win, width=e.width))
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _wheel(event):
            if getattr(event, "num", None) == 4:
                step = -1
            elif getattr(event, "num", None) == 5:
                step = 1
            else:
                step = -1 if event.delta > 0 else 1
            canvas.yview_scroll(step * 2, "units")

        def _bind(_):
            canvas.bind_all("<MouseWheel>", _wheel)
            canvas.bind_all("<Button-4>", _wheel)
            canvas.bind_all("<Button-5>", _wheel)

        def _unbind(_):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                canvas.unbind_all(seq)

        canvas.bind("<Enter>", _bind)
        canvas.bind("<Leave>", _unbind)
        builder(inner)
        return canvas

    # ── UI Construction ───────────────────────────────────────────────────────

    def _setup_style(self):
        """Dark ttk theme for the widgets ttk is used for (tabs, scrollbars).

        Everything else is a tk widget with explicit colours from C; this only
        has to make ttk's chrome match, so no global theme surgery.
        """
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TScrollbar", background=C["btn_bg"], troughcolor=C["bg"],
                        bordercolor=C["bg"], arrowcolor=C["muted"], relief="flat")
        style.map("TScrollbar", background=[("active", C["btn_hover"])])
        style.configure("TNotebook", background=C["bg"], borderwidth=0,
                        tabmargins=(6, 6, 6, 0))
        style.configure("TNotebook.Tab", background=C["surface"],
                        foreground=C["muted"], padding=(16, 7), borderwidth=0,
                        font=FONT_LABEL)
        style.map("TNotebook.Tab",
                  background=[("selected", C["panel"]),
                              ("active", C["btn_bg"])],
                  foreground=[("selected", C["accent"]), ("active", C["text"])])

    def _section(self, parent, text, color=None, pady=(10, 4)):
        """Section heading with a rule to its right — replaces bare '▸ TITLE'."""
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill="x", pady=pady)
        tk.Label(row, text=text, bg=C["panel"], fg=color or C["accent"],
                 font=FONT_TITLE).pack(side="left")
        tk.Frame(row, bg=C["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(10, 0), pady=(7, 0))
        return row

    def _card(self, parent, title, color=None, subtitle="", pady=(8, 8)):
        """Full-width bordered card inside a tab."""
        outer = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        outer.pack(fill="x", pady=pady, padx=4)
        inner = tk.Frame(outer, bg=C["panel"], padx=12, pady=10)
        inner.pack(fill="both", expand=True)
        self._section(inner, title, color, pady=(0, 6))
        if subtitle:
            tk.Label(inner, text=subtitle, bg=C["panel"], fg=C["muted"],
                     font=FONT_SMALL).pack(anchor="w", pady=(0, 6))
        return inner

    def _build_ui(self):
        self._setup_style()
        self._build_header()
        self._build_statusbar()     # packed to the bottom, so it can never clip
        self._build_visuals()       # live feed | live mosaic
        self._build_tabs()
        self.update_idletasks()
        self.bind("<Configure>", self._on_window_resize)

    def _build_header(self):
        top = tk.Frame(self, bg=C["surface"], pady=8, padx=16)
        top.pack(fill="x", side="top")

        tk.Label(top, text="⚙  ESP32 MOTOR CONTROL", bg=C["surface"],
                 fg=C["accent"], font=("Consolas", 15, "bold")).pack(side="left")
        tk.Label(top, text="blade inspection console", bg=C["surface"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=(10, 0))

        conn = tk.Frame(top, bg=C["surface"])
        conn.pack(side="right", padx=10)
        self.conn_dot = tk.Label(conn, text="●", bg=C["surface"],
                                 fg=C["danger"], font=("Consolas", 14))
        self.conn_dot.pack(side="left")
        self.conn_label = tk.Label(conn, text="DISCONNECTED", bg=C["surface"],
                                   fg=C["muted"], font=FONT_LABEL)
        self.conn_label.pack(side="left", padx=4)
        tk.Button(conn, text="PING", command=self._ping,
                  bg=C["btn_bg"], fg=C["text"], font=FONT_SMALL,
                  relief="flat", padx=8, cursor="hand2").pack(side="left", padx=6)

        tk.Label(top, text="ESP32 IP:", bg=C["surface"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="right", padx=(0, 2))
        self.ip_var = tk.StringVar(value=ESP32_IP)
        ip_entry = tk.Entry(top, textvariable=self.ip_var, width=14,
                            bg=C["panel"], fg=C["text"],
                            insertbackground=C["text"], relief="flat",
                            font=FONT_SMALL)
        ip_entry.pack(side="right")
        ip_entry.bind("<Return>", self._apply_ip)

        tk.Button(self, text="⛔  EMERGENCY STOP", command=self._emergency_stop,
                  bg=C["danger"], fg="white", font=("Consolas", 13, "bold"),
                  relief="flat", padx=20, pady=6, cursor="hand2"
                  ).pack(fill="x", padx=16, pady=(8, 4))

    def _build_visuals(self):
        """Live feed beside the growing mosaic — both visible while sweeping."""
        wrap = tk.Frame(self, bg=C["bg"])
        wrap.pack(fill="x", padx=16, pady=(2, 2))
        paned = tk.PanedWindow(wrap, orient="horizontal", bg=C["border"],
                               sashwidth=6, sashrelief="flat", bd=0,
                               handlesize=0, height=self._vis_h)
        paned.pack(fill="both", expand=True)
        self._visuals = paned

        feed_cell = tk.Frame(paned, bg=C["bg"])
        mosaic_cell = tk.Frame(paned, bg=C["bg"])
        paned.add(feed_cell, stretch="always", minsize=260)
        paned.add(mosaic_cell, stretch="always", minsize=200)
        self.after(150, lambda: paned.sash_place(
            0, int(max(300, self.winfo_width() * 0.52)), 0))

        self._build_feed_panel(feed_cell)
        self._build_mosaic_panel(mosaic_cell)

    def _on_window_resize(self, event):
        """Keep the visual row at ~40 % of the window so tabs always fit."""
        if event.widget is not self:
            return
        # The tabs carry the controls, so the visual row only needs ~1/3 of
        # the window; the notebook takes the rest and scrolls if it must.
        target = int(max(220, min(0.34 * event.height, event.height - 320)))
        if abs(target - self._vis_h) >= 8:
            self._vis_h = target
            try:
                self._visuals.configure(height=target)
            except tk.TclError:
                pass

    # ── Live mosaic panel ─────────────────────────────────────────────────────

    def _build_mosaic_panel(self, parent):
        outer = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        outer.pack(fill="both", expand=True, pady=4, padx=(4, 0))
        inner = tk.Frame(outer, bg=C["panel"], padx=10, pady=8)
        inner.pack(fill="both", expand=True)

        hdr = tk.Frame(inner, bg=C["panel"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="▸ LIVE MOSAIC", bg=C["panel"], fg=C["accent2"],
                 font=FONT_TITLE).pack(side="left")
        self._make_btn(hdr, "Folder", self._open_mosaic_folder).pack(
            side="right", padx=2)
        self._make_btn(hdr, "Save now", self._stitch_snapshot).pack(
            side="right", padx=2)
        self._make_btn(hdr, "Re-stitch scan…", self._choose_and_restitch).pack(
            side="right", padx=2)

        tk.Frame(inner, bg=C["border"], height=1).pack(fill="x", pady=4)

        self._mosaic_label = tk.Label(
            inner, bg=C["bg"], fg=C["muted"], font=FONT_SMALL,
            text="No mosaic yet\nstart a sweep", anchor="center")
        self._mosaic_label.pack(fill="both", expand=True)

        self._mosaic_stat_var = tk.StringVar(value="no frames stitched")
        tk.Label(inner, textvariable=self._mosaic_stat_var, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL, anchor="w",
                 justify="left").pack(fill="x", pady=(6, 0))

    def _post(self, fn):
        """Queue a callable to run on the Tk thread.

        `after()` is not safe to call from a worker thread, so every background
        thread hands its result over through this queue instead; the Tk thread
        drains it in `_poll_ui_queue`.
        """
        self._ui_queue.put(fn)

    def _poll_ui_queue(self):
        for _ in range(20):                 # bounded, so one flood can't stall Tk
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as exc:        # a bad callback must not kill the loop
                try:
                    self._log(f"[UI] callback failed: {exc}")
                except Exception:
                    pass
        self.after(50, self._poll_ui_queue)

    def _mosaic_ui_tick(self):
        """1 s loop — pull a snapshot of the live mosaic without blocking Tk.

        Rendering the canvas takes ~180 ms and blending runs on the stitcher's
        worker thread, so the pull happens off-thread and the result is handed
        back through ``_post`` for the Tk thread to paint.
        """
        stitcher = self._stitcher
        if stitcher is not None and not self._mosaic_pending:
            self._mosaic_pending = True

            def fetch(st=stitcher):
                try:
                    rgb, summary = st.preview(520)
                except Exception as exc:          # never kill the UI loop
                    rgb, summary = None, {"error": str(exc)}
                self._post(lambda: self._mosaic_render(rgb, summary))

            threading.Thread(target=fetch, daemon=True,
                             name="mosaic-preview").start()
        self.after(1000, self._mosaic_ui_tick)

    def _mosaic_render(self, rgb, summary):
        self._mosaic_pending = False
        if not summary:
            return
        if "error" in summary:
            self._mosaic_stat_var.set(f"preview error: {summary['error']}")
            return

        if rgb is not None and HAS_PIL:
            ph, pw = rgb.shape[:2]
            box_w = self._mosaic_label.winfo_width()
            box_h = self._mosaic_label.winfo_height()
            box_w = box_w if box_w > 40 else 420
            box_h = box_h if box_h > 40 else 300
            k = min(box_w / pw, box_h / ph)
            size = (max(1, int(pw * k)), max(1, int(ph * k)))
            img = Image.fromarray(rgb).resize(size, Image.LANCZOS)
            self._mosaic_photo = ImageTk.PhotoImage(image=img)
            self._mosaic_label.configure(image=self._mosaic_photo, text="")
        elif rgb is not None:
            self._mosaic_label.configure(
                text="pip install Pillow to preview the mosaic")

        canvas = summary.get("canvas") or (0, 0, 0, 0)
        npct = summary.get("low_quality_pct")
        lines = [
            f"{summary.get('accepted', 0)} kept · {summary.get('filled', 0)} fill"
            f" · {summary.get('rejected', 0)} dropped"
            f" · {summary.get('disconnected', 0)} band(s)",
            f"mean NCC {summary.get('mean_ncc', 0.0):.2f}"
            f" · {canvas[2] - canvas[0]}×{canvas[3] - canvas[1]} px"
            f" · {time.strftime('%H:%M:%S')}",
        ]
        if npct:
            lines.append(f"soft fill-in covers {npct:.1f}% of the mosaic")
        self._mosaic_stat_var.set("\n".join(lines))
        if hasattr(self, "_sweep_stitch_var"):
            self._sweep_stitch_var.set(
                f"{summary.get('accepted', 0)} accepted · "
                f"{summary.get('filled', 0)} soft fill-ins · "
                f"{summary.get('rejected', 0)} rejected · "
                f"NCC {summary.get('mean_ncc', 0.0):.2f}")
        if hasattr(self, "_sb_stitch_var"):
            self._sb_stitch_var.set(
                f"stitched {summary.get('accepted', 0)}"
                f"  ·  dropped {summary.get('rejected', 0)}")

    def _stitch_snapshot(self):
        stitcher = self._stitcher
        if stitcher is None:
            self._log("[STITCH] nothing to save yet — start a sweep first")
            return

        def work():
            try:
                paths = stitcher.flush()
                self._log(f"[STITCH] snapshot saved: {paths.get('png')}")
            except Exception as exc:
                self._log(f"[STITCH] snapshot failed: {exc}")

        threading.Thread(target=work, daemon=True).start()

    def _open_mosaic_folder(self):
        path = self._stitcher_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "blade_mosaic")
        try:
            os.makedirs(path, exist_ok=True)
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)                     # noqa: S606 (Windows)
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            self._log(f"[STITCH] cannot open {path}: {exc}")

    def _on_stitch_decision(self, decision):
        """Called on the stitcher thread — hand the verdict to the Tk thread.

        Note the callback fires while the stitcher holds its own lock, so this
        must stay cheap: queue the decision and return immediately.
        """
        # Feed the measured step back into SCAN GEOMETRY first: this is the
        # coarse, live loop that keeps the overlap right *during* the scan.
        # ``_scan_geo_observe`` gates on single_step() so AF frames and dropped
        # frames never enter the sample, and swallows every exception - a
        # stitcher callback may never stop a sweep.
        if getattr(decision, "status", None) == "accepted":
            try:
                d = decision
                self._scan_geo_observe(d.tag, float(d.dx), float(d.dy))
            except Exception:                 # noqa: BLE001
                pass
        self._post(lambda: self._stitch_decision_ui(decision))

    def _stitch_decision_ui(self, decision):
        if not hasattr(self, "_last_stitch_var"):
            return
        flagged = getattr(decision, "flagged", False)
        label = "FILL" if flagged else ("KEEP" if decision.ok else "DROP")
        colour = C["warn"] if flagged else (
            C["success"] if decision.ok else C["danger"])
        self._last_stitch_var.set(
            f"{label}  {decision.tag}  ·  {decision.reason or 'ok'}")
        self._last_stitch_label.configure(fg=colour)

    # ── Tabs ──────────────────────────────────────────────────────────────────

    def _build_tabs(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=16, pady=(2, 8))
        self._notebook = nb

        pages = [
            ("  Sweep  ", self._build_tab_sweep, True),
            ("  Focus  ", self._build_tab_focus, True),
            ("  Manual  ", self._build_tab_manual, True),
            ("  Calibration  ", self._build_tab_calibration, True),
            ("  Log  ", self._build_tab_log, False),
        ]
        for title, builder, scroll in pages:
            page = tk.Frame(nb, bg=C["bg"])
            nb.add(page, text=title)
            if scroll:
                self._build_scrollable(page, builder)
            else:
                builder(page)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=C["surface"], padx=16, pady=5)
        bar.pack(side="bottom", fill="x")

        self._sb_conn_var = tk.StringVar(value="● OFFLINE")
        self._sb_conn_label = tk.Label(bar, textvariable=self._sb_conn_var,
                                       bg=C["surface"], fg=C["danger"],
                                       font=FONT_SMALL)
        self._sb_conn_label.pack(side="left")

        self._sb_motion_var = tk.StringVar(value="● STILL")
        self._sb_motion_label = tk.Label(bar, textvariable=self._sb_motion_var,
                                         bg=C["surface"], fg=C["success"],
                                         font=FONT_SMALL)
        self._sb_motion_label.pack(side="left", padx=(16, 0))

        self._sb_feed_var = tk.StringVar(value="feed off")
        tk.Label(bar, textvariable=self._sb_feed_var, bg=C["surface"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=(16, 0))

        self._sb_col_var = tk.StringVar(value="column —")
        tk.Label(bar, textvariable=self._sb_col_var, bg=C["surface"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=(16, 0))

        self._sb_stitch_var = tk.StringVar(value="stitched 0  ·  dropped 0")
        tk.Label(bar, textvariable=self._sb_stitch_var, bg=C["surface"],
                 fg=C["accent2"], font=FONT_SMALL).pack(side="right")

        self._sb_clock_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._sb_clock_var, bg=C["surface"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="right", padx=(0, 16))

        self._sb_tick()

    def _sb_tick(self):
        self._sb_clock_var.set(time.strftime("%H:%M:%S"))
        self.after(1000, self._sb_tick)

    # ── Generic Axis Panel (Z / X / R) ────────────────────────────────────────

    def _build_axis_panel(self, parent, title, *, tag, modes, default_mode,
                          default_val, to_steps, describe, mover, buttons):
        f = self._make_panel(parent, title)

        mode_var = tk.StringVar(value=default_mode)
        val_var = tk.StringVar(value=default_val)
        calc_var = tk.StringVar()

        def update_calc(*_):
            try:
                calc_var.set(describe(float(val_var.get()), mode_var.get()))
            except ValueError:
                calc_var.set("— invalid input")

        row = tk.Frame(f, bg=C["panel"])
        row.pack(fill="x", pady=1)
        tk.Label(row, text="Input:", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")
        for m in modes:
            tk.Radiobutton(row, text=m, variable=mode_var, value=m,
                           command=update_calc, bg=C["panel"], fg=C["text"],
                           selectcolor=C["surface"], font=FONT_SMALL,
                           activebackground=C["panel"]).pack(side="left", padx=2)

        entry_row = tk.Frame(f, bg=C["panel"])
        entry_row.pack(fill="x", pady=2)
        val_var.trace_add("write", update_calc)
        self._make_entry(entry_row, val_var, w=7).pack(side="left", padx=2)

        tk.Label(f, textvariable=calc_var, bg=C["panel"],
                 fg=C["accent"], font=FONT_SMALL).pack(anchor="w", pady=1)

        def move(direction, dir_label):
            try:
                steps = to_steps(float(val_var.get()), mode_var.get())
            except ValueError:
                self._log(f"[{tag}] Invalid input value")
                return
            self._log(f"{tag} → {dir_label} {steps} steps")
            self._threaded(mover, steps, direction, self._log)

        btn_row = tk.Frame(f, bg=C["panel"])
        btn_row.pack(fill="x", pady=2)
        for text, direction, dir_label in buttons:
            self._make_btn(btn_row, text,
                           lambda d=direction, l=dir_label: move(d, l)
                           ).pack(side="left", expand=True, fill="x", padx=1)

        update_calc()

    # ── Global Speed Panel ────────────────────────────────────────────────────

    def _build_speed_panel(self, parent):
        f = self._make_panel(parent, "STEPPER SPEED",
                             "Pulse delay µs (lower = faster)")

        self.speed_us = tk.StringVar(value="1000")
        tk.Scale(f, from_=50, to=2000, orient="horizontal", variable=self.speed_us,
                 bg=C["panel"], fg=C["text"], troughcolor=C["surface"],
                 highlightthickness=0, font=FONT_SMALL, label="µs per pulse"
                 ).pack(fill="x", pady=2)
        self._make_btn(f, "Apply Speed", self._apply_speed).pack(
            fill="x", pady=4)

    def _apply_speed(self):
        try:
            us = int(self.speed_us.get())
        except ValueError:
            self._log("[SPEED] Invalid value")
            return
        self._log(f"[SPEED] Setting pulse delay to {us} µs")
        self._threaded(set_speed, us, self._log)

    # ── Motion Detector Panel ─────────────────────────────────────────────────

    def _build_motion_detector_panel(self, parent):
        f = self._make_panel(parent, "MOTION DETECTOR",
                             "X/Y motors only (closed-loop)")

        rows = [
            ("Motion Threshold:", str(MOTION_THRESHOLD), 8, "pixels"),
            ("Stillness Frames:", str(STILLNESS_FRAMES), 4, ""),
            ("Frame Diff Threshold:", str(FRAME_DIFF_THRESHOLD), 4, ""),
        ]
        variables = []
        for label, default, width, suffix in rows:
            row = tk.Frame(f, bg=C["panel"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=C["panel"], fg=C["muted"],
                     font=FONT_SMALL).pack(side="left")
            var = tk.StringVar(value=default)
            var.trace_add("write", lambda *_: self._update_motion_params())
            self._make_entry(row, var, w=width).pack(side="left", padx=4)
            if suffix:
                tk.Label(row, text=suffix, bg=C["panel"], fg=C["muted"],
                         font=FONT_SMALL).pack(side="left")
            variables.append(var)
        self.md_motion_thresh, self.md_still_frames, self.md_frame_diff = variables

        self._md_status_var = tk.StringVar(value="● STILL")
        tk.Label(f, textvariable=self._md_status_var, bg=C["panel"],
                 fg=C["success"], font=("Consolas", 10, "bold")).pack(anchor="w", pady=4)

    def _update_motion_params(self):
        global MOTION_THRESHOLD, STILLNESS_FRAMES, FRAME_DIFF_THRESHOLD
        try:
            MOTION_THRESHOLD = int(self.md_motion_thresh.get())
        except ValueError:
            pass
        try:
            STILLNESS_FRAMES = int(self.md_still_frames.get())
        except ValueError:
            pass
        try:
            FRAME_DIFF_THRESHOLD = int(self.md_frame_diff.get())
        except ValueError:
            pass

    # ── Boundary Detector Panel ───────────────────────────────────────────────

    def _build_boundary_detector_panel(self, parent):
        f = self._make_panel(parent, "BOUNDARY DETECTOR",
                             "Blade edge detection")

        row = tk.Frame(f, bg=C["panel"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Black Threshold:", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")
        self.bd_threshold = tk.StringVar(value=str(BOUNDARY_THRESHOLD))
        self.bd_threshold.trace_add(
            "write", lambda *_: self._update_boundary_params())
        self._make_entry(row, self.bd_threshold, w=4).pack(side="left", padx=4)
        tk.Label(row, text="(0-255)", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")

        self._bd_status_var = tk.StringVar(value="No boundary detected")
        tk.Label(f, textvariable=self._bd_status_var, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(anchor="w", pady=4)

    def _update_boundary_params(self):
        global BOUNDARY_THRESHOLD
        try:
            BOUNDARY_THRESHOLD = int(self.bd_threshold.get())
        except ValueError:
            pass

    def _status_ui_tick(self):
        """Refresh motion and boundary status indicators (100 ms loop)."""
        moving = self._motion_detector.is_moving()
        if moving:
            self._md_status_var.set("● MOVING")
            self._feed_label.configure(fg=C["danger"])
        else:
            self._md_status_var.set("● STILL")
            self._feed_label.configure(fg=C["success"])
        if hasattr(self, "_sb_motion_var"):
            self._sb_motion_var.set("● MOVING" if moving else "● STILL")
            self._sb_motion_label.configure(
                fg=C["danger"] if moving else C["success"])

        with self._frame_lock:
            b = self._latest_boundaries
        if b is None:
            self._bd_status_var.set("No boundary detected")
        else:
            left, right, upper, lower = b
            self._bd_status_var.set(
                f"L:{left} R:{right} U:{upper} D:{lower} px")

        self.after(100, self._status_ui_tick)

    def _autosweep_timer_tick(self):
        """1 s loop — live total/column elapsed time while a sweep is running."""
        if self._autosweep_running:
            total = int(time.time() - self._autosweep_start_time)
            col = int(time.time() - self._autosweep_column_start_time)
            self._autosweep_timer_var.set(
                f"⏱ Total: {total // 60}m {total % 60}s"
                f"   Column {self._autosweep_column_num}: {col // 60}m {col % 60}s")
            if hasattr(self, "_sb_col_var"):
                self._sb_col_var.set(
                    f"col {self._autosweep_column_num} · {total // 60}m {total % 60}s")
        self.after(1000, self._autosweep_timer_tick)

    # ── Embedded DSLR Feed Panel ──────────────────────────────────────────────

    def _build_feed_panel(self, parent):
        outer = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        outer.pack(fill="both", expand=True, pady=4)
        inner = tk.Frame(outer, bg=C["panel"], padx=8, pady=8)
        inner.pack(fill="both", expand=True)

        hdr = tk.Frame(inner, bg=C["panel"])
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="▸ DSLR LIVE FEED", bg=C["panel"],
                 fg=C["accent"], font=FONT_TITLE).pack(side="left")

        self.btn_start_feed = tk.Button(
            hdr, text="▶ Start Feed", command=self._start_feed,
            bg=C["success"], fg="#0f1117", font=("Consolas", 10, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
            activebackground="#3aaa6a", activeforeground="white")
        self.btn_start_feed.pack(side="left", padx=10)

        self.btn_stop_feed = tk.Button(
            hdr, text="■ Stop Feed", command=self._stop_feed,
            bg=C["danger"], fg="white", font=("Consolas", 10, "bold"),
            relief="flat", padx=8, pady=2, cursor="hand2",
            activebackground="#c03030", activeforeground="white", state="disabled")
        self.btn_stop_feed.pack(side="left", padx=4)

        tk.Frame(inner, bg=C["border"], height=1).pack(fill="x", pady=4)

        # Zoom control (percent of panel width)
        size_row = tk.Frame(inner, bg=C["panel"])
        size_row.pack(fill="x", pady=(0, 4))
        tk.Label(size_row, text="Zoom:", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")
        self._feed_zoom_var = tk.IntVar(value=100)
        tk.Scale(size_row, from_=10, to=100, resolution=5, orient="horizontal",
                 variable=self._feed_zoom_var, bg=C["panel"], fg=C["text"],
                 troughcolor=C["surface"], highlightthickness=0,
                 font=FONT_SMALL, length=140, showvalue=True).pack(side="left", padx=4)
        tk.Label(size_row, text="%  of panel width", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")

        tk.Label(inner, text="■ yellow box = autofocus region "
                             f"(FOCUS_CONFIG['roi'] = {FOCUS_CONFIG.get('roi')})",
                 bg=C["panel"], fg=C["warn"], font=FONT_SMALL).pack(anchor="w", pady=(0, 4))

        self._feed_label = tk.Label(inner, bg=C["bg"], anchor="nw")
        self._feed_label.pack(fill="both", expand=True, anchor="nw")

        self._feed_status_var = tk.StringVar(
            value="No feed — press Start Feed")
        tk.Label(inner, textvariable=self._feed_status_var,
                 bg=C["panel"], fg=C["muted"], font=FONT_SMALL).pack(anchor="w", pady=4)

    # ── Tab: Sweep ────────────────────────────────────────────────────────────

    # ── Scan geometry (how far the machine steps) ────────────────────────────

    def _scan_geo_init(self):
        """Load the last learned calibration, or start from the defaults."""
        geo = None
        if _scan_geometry is not None:
            geo = _scan_geometry.ScanGeometry.load(GEOMETRY_CAL_PATH)
        if geo is None and _scan_geometry is not None:
            geo = _scan_geometry.ScanGeometry()
        self._scan_geo = geo
        self._frame_shape = None          # set from the first saved frame
        self._last_x_cmd_mm = geo.x_step_mm if geo else 0.4
        self._prev_capture_tag = None
        self._last_sweep_scan_dir = None
        return geo

    def _build_scan_geometry_panel(self, parent):
        """Step sizes for both axes - and what they actually achieve.

        Neither axis reports its position, so the only honest way to choose a
        step is to ask the camera: the stitcher measures the real pixel
        displacement of the frames it registers.  Those measurements are shown
        here, and (with Auto-tune on) fed back into the commands.
        """
        geo = self._scan_geo
        box = self._card(parent, "SCAN GEOMETRY", C["accent2"],
                         "how far the machine steps between frames, and the "
                         "overlap that actually results")

        def row(label, var, unit, width=7):
            r = tk.Frame(box, bg=C["panel"])
            r.pack(fill="x", pady=1)
            tk.Label(r, text=label, bg=C["panel"], fg=C["muted"],
                     font=FONT_SMALL, width=24, anchor="w").pack(side="left")
            self._make_entry(r, var, w=width).pack(side="left")
            tk.Label(r, text=unit, bg=C["panel"], fg=C["muted"],
                     font=FONT_SMALL).pack(side="left", padx=4)

        self.sg_y_step = tk.StringVar(value=f"{geo.y_step_mm:.2f}" if geo else "0.35")
        self.sg_x_step = tk.StringVar(value=f"{geo.x_step_mm:.2f}" if geo else "0.40")
        self.sg_y_overlap = tk.StringVar(
            value=f"{int(round(100 * geo.target_y_overlap))}" if geo else "65")
        self.sg_x_overlap = tk.StringVar(
            value=f"{int(round(100 * geo.target_x_overlap))}" if geo else "50")
        self.sg_auto = tk.BooleanVar(value=bool(geo.auto_tune) if geo else True)

        row("Y step", self.sg_y_step, "mm  per Y move")
        row("X step", self.sg_x_step, "mm  per column")
        row("Target Y overlap", self.sg_y_overlap, "% between frames")
        row("Target X overlap", self.sg_x_overlap, "% between columns")

        ctl = tk.Frame(box, bg=C["panel"])
        ctl.pack(fill="x", pady=(4, 2))
        self.sg_auto_chk = tk.Checkbutton(
            ctl, text="Auto-tune from the measured step", variable=self.sg_auto,
            command=self._scan_geo_sync, bg=C["panel"], fg=C["text"],
            selectcolor=C["surface"], activebackground=C["panel"],
            activeforeground=C["text"], font=FONT_SMALL)
        self.sg_auto_chk.pack(side="left")
        self._make_btn(ctl, "Use recommendation",
                       self._scan_geo_recommend).pack(side="left", padx=6)
        self._make_btn(ctl, "Recalibrate",
                       self._scan_geo_reset).pack(side="left")

        for v in (self.sg_y_step, self.sg_x_step,
                  self.sg_y_overlap, self.sg_x_overlap):
            v.trace_add("write", lambda *_: self._scan_geo_sync())

        self._scan_geo_var = tk.StringVar(
            value="no measurement yet - the pixel scale comes from the first "
                  "column advance")
        tk.Label(box, textvariable=self._scan_geo_var, bg=C["panel"],
                 fg=C["text"], font=FONT_SMALL, anchor="w", justify="left",
                 wraplength=780).pack(fill="x", pady=(4, 0))

    def _scan_geo_sync(self):
        """Push the entries into the controller (called on every edit)."""
        geo = getattr(self, "_scan_geo", None)
        if geo is None:
            return

        def num(var, default, lo=None, hi=None):
            try:
                v = float(var.get())
            except (TypeError, ValueError, AttributeError):
                return default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        geo.y_step_mm = num(self.sg_y_step, geo.y_step_mm, 0.02, 20.0)
        geo.x_step_mm = num(self.sg_x_step, geo.x_step_mm, 0.02, 20.0)
        geo.target_y_overlap = num(self.sg_y_overlap,
                                  100 * geo.target_y_overlap, 5.0, 90.0) / 100.0
        geo.target_x_overlap = num(self.sg_x_overlap,
                                  100 * geo.target_x_overlap, 5.0, 90.0) / 100.0
        geo.auto_tune = bool(self.sg_auto.get())
        self._scan_geo_refresh()

    def _scan_geo_refresh(self):
        geo = getattr(self, "_scan_geo", None)
        if geo is None or not hasattr(self, "_scan_geo_var"):
            return
        self._scan_geo_var.set(geo.advice(self._frame_shape))

    def _scan_geo_recommend(self):
        """Adopt the measurement's recommendation (needs a measurement)."""
        geo = getattr(self, "_scan_geo", None)
        if geo is None or not self._frame_shape:
            self._log("[SWEEP] geometry: no measurement yet, nothing to apply")
            return
        t = geo.suggested_y_step_mm(self._frame_shape[0])
        x = geo.suggested_x_step_mm(self._frame_shape[1])
        if t is not None and hasattr(self, "sg_y_step"):
            self.sg_y_step.set(f"{t:.2f}")
        if x is not None and hasattr(self, "sg_x_step"):
            self.sg_x_step.set(f"{x:.2f}")
        self._log(f"[SWEEP] geometry recommendation applied: {geo.describe()}")
        self._scan_geo_sync()

    def _scan_geo_reset(self):
        """Throw the calibration away and start from the defaults."""
        if _scan_geometry is None:
            return
        self._scan_geo = _scan_geometry.ScanGeometry()
        for var, val in ((self.sg_y_step, "0.35"),
                         (self.sg_x_step, "0.40"), (self.sg_y_overlap, "65"),
                         (self.sg_x_overlap, "50")):
            var.set(val)
        self.sg_auto.set(True)
        self._log("[SWEEP] geometry calibration cleared - defaults restored")
        self._scan_geo_sync()

    def _scan_geo_observe(self, tag: str, dx: float, dy: float):
        """Record one measured step from the live stitcher (stitcher thread)."""
        geo = getattr(self, "_scan_geo", None)
        if geo is None or _scan_geometry is None or not self._frame_shape:
            return
        axis = _scan_geometry.single_step(getattr(self, "_prev_capture_tag", None),
                                          tag)
        self._prev_capture_tag = tag
        if axis is None:
            return
        try:
            geo.observe(axis, dx, dy, tuple(self._frame_shape),
                        moved_mm=self._last_x_cmd_mm if axis == "X" else None)
        except Exception:                 # noqa: BLE001 - never break a scan
            pass

    def _scan_geo_after_scan(self, images_dir: str, phase: str = "end") -> None:
        """Record what this scan used, so its pixel scale is attributable.

        Called twice: once when the scan starts (what it intended) and once when
        it ends (what it actually used - auto-tune may have moved both steps).
        Without the first one, a mid-scan correction would leave the early
        columns unattributable, which is the exact ambiguity that made px/mm
        look inconsistent between two scans of the same rig.
        """
        geo = getattr(self, "_scan_geo", None)
        if geo is None or _scan_geometry is None or not images_dir:
            return
        if phase == "start":
            self._scan_geo_baseline = geo.describe()
        path = _scan_geometry.write_scan_geometry(
            images_dir, geo,
            extra={"phase": phase,
                   "commands_at_start": getattr(self, "_scan_geo_baseline", None),
                   "x_steps_per_mm": X_STEPS_PER_MM,
                   "x_steps": mm_to_x_steps(geo.x_step_mm),
                   "frame_hw": list(self._frame_shape) if self._frame_shape else None,
                   "targets": {"y": geo.target_y_overlap, "x": geo.target_x_overlap}})
        if path:
            self._log(f"[SWEEP] geometry recorded: {os.path.basename(path)}")

    def _geometry_x_cmd(self, images_dir: str) -> float | None:
        """The X step a scan folder was captured with, in mm.

        Re-stitching an old folder must calibrate against *that* folder's
        commands, not against whatever the panel says now - a stale command
        applied to a fresh measurement is how px/mm drifted by 37 % between two
        scans of this rig.
        """
        if _scan_geometry is not None and images_dir:
            try:
                rec = _scan_geometry.read_scan_geometry(images_dir)
            except Exception:                 # noqa: BLE001
                rec = None
            if rec is not None and rec.x_step_mm:
                return float(rec.x_step_mm)
        geo = getattr(self, "_scan_geo", None)
        return geo.x_step_mm if geo else None

    def _scan_geo_from_motion(self, motion: dict, x_cmd_mm: float | None) -> None:
        """Calibrate from the global stitcher's learned motion models.

        This is the authoritative measurement of a finished scan (every frame, a
        globally consistent solve), so it is what the *next* scan should use.
        """
        geo = getattr(self, "_scan_geo", None)
        if geo is None or _scan_geometry is None or not motion or not self._frame_shape:
            return
        got = geo.absorb_motion(motion, tuple(self._frame_shape), x_cmd_mm)
        if not got:
            return
        self._log(f"[GLOBAL] geometry from the solve: {got}")
        for line in _scan_geometry.geometry_lines(geo, motion,
                                                  tuple(self._frame_shape),
                                                  x_cmd_mm):
            self._log(f"[GLOBAL] {line}" if line.strip() else "[GLOBAL]")
        geo.save(GEOMETRY_CAL_PATH)
        self._post(self._scan_geo_refresh)

    def _build_tab_sweep(self, parent):
        self._scan_geo_init()
        self._build_scan_geometry_panel(parent)

        box = self._card(parent, "AUTO SWEEP", C["warn"],
                         "Boundary-guided serpentine sweep.  The step sizes live "
                         "in SCAN GEOMETRY above; the overlap they produce is "
                         "measured while the scan runs and corrected.")

        btn_row = tk.Frame(box, bg=C["panel"])
        btn_row.pack(fill="x", pady=(0, 6))
        self.btn_autosweep = tk.Button(
            btn_row, text="▶ Start sweep", command=self._start_autosweep,
            bg=C["success"], fg="#0f1117", font=("Consolas", 10, "bold"),
            relief="flat", padx=12, pady=5, cursor="hand2",
            activebackground="#3aaa6a", activeforeground="white")
        self.btn_autosweep.pack(side="left", padx=(0, 4))
        self.btn_stop_autosweep = tk.Button(
            btn_row, text="■ Stop", command=self._stop_autosweep,
            bg=C["danger"], fg="white", font=("Consolas", 10, "bold"),
            relief="flat", padx=12, pady=5, cursor="hand2", state="disabled",
            activebackground="#c03030", activeforeground="white")
        self.btn_stop_autosweep.pack(side="left", padx=4)

        self._autosweep_status = tk.StringVar(value="Ready")
        tk.Label(btn_row, textvariable=self._autosweep_status, bg=C["panel"],
                 fg=C["text"], font=FONT_SMALL).pack(side="left", padx=12)

        self._autosweep_timer_var = tk.StringVar(value="")
        tk.Label(box, textvariable=self._autosweep_timer_var, bg=C["panel"],
                 fg=C["accent"], font=FONT_LABEL).pack(anchor="w")

        self._build_sweep_speed_panel(parent)

        # ── live stitching feedback (mirrors the mosaic panel) ───────────────
        stitch = self._card(parent, "LIVE STITCH", C["accent2"],
                            "every saved frame is stitched as it arrives; "
                            "blurry / duplicate frames are rejected")
        self._sweep_stitch_var = tk.StringVar(
            value="waiting for the first stitched frame")
        tk.Label(stitch, textvariable=self._sweep_stitch_var, bg=C["panel"],
                 fg=C["text"], font=FONT_LABEL).pack(anchor="w", pady=(0, 4))
        self._last_stitch_var = tk.StringVar(value="—")
        self._last_stitch_label = tk.Label(
            stitch, textvariable=self._last_stitch_var, bg=C["panel"],
            fg=C["muted"], font=FONT_SMALL, anchor="w")
        self._last_stitch_label.pack(fill="x", pady=(0, 6))

        actions = tk.Frame(stitch, bg=C["panel"])
        actions.pack(fill="x")
        self._make_btn(actions, "Save snapshot now",
                       self._stitch_snapshot).pack(side="left", padx=(0, 4))
        self._make_btn(actions, "Open output folder",
                       self._open_mosaic_folder).pack(side="left")

        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", padx=2)
        self._build_motion_detector_panel(row)
        self._build_boundary_detector_panel(row)

    # ── Sweep motor speed ─────────────────────────────────────────────────────

    def _build_sweep_speed_panel(self, parent):
        """Pulse rate used by every X/Y/Z move of the sweep (AF Z moves too).

        The value is the firmware's step *half*-period: 1000 µs = 500 Hz.
        It is sent when a sweep starts, and "Apply now" changes it mid-sweep;
        the move already in progress finishes at the old rate."""
        card = self._card(parent, "SWEEP MOTOR SPEED", C["accent"],
                          "half-period per pulse, lower = faster; applies to "
                          "X, Y and Z (autofocus) moves")
        self._sweep_speed_us = tk.IntVar(value=SWEEP_SPEED_US_DEFAULT)
        self._sweep_speed_info = tk.StringVar()
        self._sweep_speed_applied = tk.StringVar(value="")

        tk.Scale(card, from_=50, to=2000, resolution=50, orient="horizontal",
                 variable=self._sweep_speed_us, bg=C["panel"], fg=C["text"],
                 troughcolor=C["surface"], highlightthickness=0,
                 font=FONT_SMALL, label="µs half-period",
                 command=lambda _v: self._sweep_speed_refresh()
                 ).pack(fill="x", pady=2)
        tk.Label(card, textvariable=self._sweep_speed_info, bg=C["panel"],
                 fg=C["text"], font=FONT_LABEL).pack(anchor="w")

        row = tk.Frame(card, bg=C["panel"])
        row.pack(fill="x", pady=(4, 0))
        self._make_btn(row, "Apply now", self._apply_sweep_speed).pack(
            side="left", padx=(0, 8))
        tk.Label(row, textvariable=self._sweep_speed_applied, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left")
        self._sweep_speed_refresh()

    def _sweep_speed_value(self) -> int:
        """Selected half-period in µs, snapped to the firmware's 50 µs tick."""
        try:
            us = int(self._sweep_speed_us.get())
        except (tk.TclError, ValueError):
            us = SWEEP_SPEED_US_DEFAULT
        return int(min(2000, max(50, round(us / 50) * 50)))

    def _sweep_speed_refresh(self):
        us = self._sweep_speed_value()
        hz = 1e6 / (2 * us)
        mm_s = hz / STEPS_PER_MM
        rpm = hz / PULSES_PER_REV * 60
        warn = "   ⚠ fast, no accel ramp" if hz > 2000 else ""
        self._sweep_speed_info.set(
            f"{hz:,.0f} Hz · {mm_s:.3f} mm/s · {rpm:.1f} RPM{warn}")

    def _apply_sweep_speed(self):
        us = self._sweep_speed_value()
        state = "live, next move" if self._autosweep_running else "now"
        self._log(f"[SPEED] sweep speed -> {us} µs ({state})")
        self._sweep_speed_applied.set(f"sent {us} µs")
        self._threaded(set_speed, us, self._log)

    # ── Tab: Focus ────────────────────────────────────────────────────────────

    def _build_tab_focus(self, parent):
        box = self._card(parent, "SIAMESE RLHF — FOCUS TRAINING", C["warn"],
                         "set a baseline frame, nudge Z, then label which frame "
                         "looks sharper")

        r1 = tk.Frame(box, bg=C["panel"])
        r1.pack(fill="x", pady=2)
        self._make_btn(r1, "1. Set Frame A (Baseline)",
                       self._set_frame_a).pack(side="left")
        self._frame_a_status = tk.StringVar(value=" Frame A: Not Set")
        tk.Label(r1, textvariable=self._frame_a_status, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=4)
        tk.Label(r1, text="  Auto (ms):", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")
        self.auto_a_ms_var = tk.StringVar(value="0")
        tk.Entry(r1, textvariable=self.auto_a_ms_var, width=5,
                 bg=C["surface"], fg=C["text"], insertbackground=C["text"],
                 relief="flat", font=FONT_SMALL).pack(side="left")

        r2 = tk.Frame(box, bg=C["panel"])
        r2.pack(fill="x", pady=(6, 2))
        tk.Label(r2, text="2. Move Z, then train:", bg=C["panel"],
                 fg=C["text"], font=FONT_SMALL).pack(side="left", padx=(0, 6))
        self._make_btn(r2, "B is Sharper (+)", lambda: self._train_siamese(1.0),
                       bg=C["success"]).pack(side="left", padx=2)
        self._make_btn(r2, "B is Blurrier (−)", lambda: self._train_siamese(0.0),
                       bg=C["danger"]).pack(side="left", padx=2)
        tk.Label(r2, text=" LR:", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left", padx=(6, 2))
        self.ds_lr_var = tk.StringVar(value="0.001")
        tk.Entry(r2, textvariable=self.ds_lr_var, width=6, bg=C["surface"],
                 fg=C["text"], insertbackground=C["text"], relief="flat",
                 font=FONT_SMALL).pack(side="left")

        af = self._card(parent, "AUTOFOCUS", C["accent"],
                        "region-wise wide-range scan: scores a tile grid, "
                        "searches the whole Z travel and extends the window "
                        "instead of assuming the peak is inside it")

        # engine selection
        mode_row = tk.Frame(af, bg=C["panel"])
        mode_row.pack(fill="x", pady=(0, 6))
        tk.Label(mode_row, text="Engine:", bg=C["panel"], fg=C["text"],
                 font=FONT_SMALL).pack(side="left", padx=(0, 6))
        self._af_mode_var = tk.StringVar(
            value="Region-wise (recommended)" if FOCUS_MODE == "region"
            else "CNN hill climb (legacy)")
        for label in ("Region-wise (recommended)", "CNN hill climb (legacy)"):
            tk.Radiobutton(mode_row, text=label, value=label,
                           variable=self._af_mode_var, bg=C["panel"],
                           fg=C["text"], selectcolor=C["surface"],
                           activebackground=C["panel"],
                           activeforeground=C["accent"], font=FONT_SMALL,
                           highlightthickness=0).pack(side="left", padx=4)

        # region-wise parameters
        rw = tk.Frame(af, bg=C["panel"])
        rw.pack(fill="x", pady=2)
        self.af_range_var = tk.StringVar(
            value=str(FOCUS_CONFIG["range_steps"]))
        self.af_pred_var = tk.StringVar(
            value=str(FOCUS_CONFIG.get("predictive_window", 300)))
        self.af_limit_var = tk.StringVar(
            value=str(FOCUS_CONFIG["z_travel_limit"]))
        self.af_backlash_var = tk.StringVar(
            value=str(FOCUS_CONFIG.get("backlash_steps", 0)))
        for label, var, width, suffix in (
                ("Range cap:", self.af_range_var, 6, "steps"),
                ("Predict window:", self.af_pred_var, 5, "steps"),
                ("Travel limit:", self.af_limit_var, 6, "steps"),
                ("Backlash:", self.af_backlash_var, 4, "steps")):
            tk.Label(rw, text=f" {label}", bg=C["panel"], fg=C["muted"],
                     font=FONT_SMALL).pack(side="left")
            tk.Entry(rw, textvariable=var, width=width, bg=C["surface"],
                     fg=C["text"], insertbackground=C["text"], relief="flat",
                     font=FONT_SMALL).pack(side="left", padx=1)
            if suffix:
                tk.Label(rw, text=suffix, bg=C["panel"], fg=C["muted"],
                         font=FONT_SMALL).pack(side="left", padx=(0, 6))

        # per-axis step sizes (C = coarse search step, F = fine step, D = settle)
        self.af_y_c_var, self.af_y_m_var, self.af_y_f_var, self.af_y_d_var = \
            (tk.StringVar(value=v) for v in ("100", "50", "25", "0.25"))
        self.af_x_c_var, self.af_x_m_var, self.af_x_f_var, self.af_x_d_var = \
            (tk.StringVar(value=v) for v in ("100", "50", "25", "0.25"))
        self._build_af_row(af, "AF Y-Shifts:", C["warn"],
                           (self.af_y_c_var, self.af_y_m_var,
                            self.af_y_f_var, self.af_y_d_var))
        self._build_af_row(af, "AF X-Shifts:", C["accent2"],
                           (self.af_x_c_var, self.af_x_m_var,
                            self.af_x_f_var, self.af_x_d_var))
        tk.Label(af, text="C = coarse step, M = medium, F = fine step, "
                          "D = settle delay (s).  Range cap 0 = whole travel; "
                          "set it to the blade's depth envelope to stop the "
                          "search wandering outside it.",
                 bg=C["panel"], fg=C["muted"], font=FONT_SMALL).pack(anchor="w")

        preset_row = tk.Frame(af, bg=C["panel"])
        preset_row.pack(fill="x", pady=(6, 0))
        tk.Label(preset_row, text="Preset:", bg=C["panel"], fg=C["text"],
                 font=FONT_SMALL).pack(side="left", padx=(0, 6))
        for name, flag in (("Fast (sweeps)", "fast"),
                           ("Balanced", "balanced"),
                           ("Accurate (single shot)", "accurate")):
            self._make_btn(preset_row, name,
                           lambda f=flag: self._apply_focus_preset(f)).pack(
                side="left", padx=2)
        self._af_preset_var = tk.StringVar(
            value="these values are also saved in FOCUS_CONFIG")
        tk.Label(af, textvariable=self._af_preset_var, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(anchor="w", pady=(4, 0))

        btn_row = tk.Frame(af, bg=C["panel"])
        btn_row.pack(fill="x", pady=(8, 0))
        self.btn_af = tk.Button(
            btn_row, text="🎯 Auto-Focus Z", command=self._start_autofocus,
            bg=C["accent"], fg="#0f1117", font=("Consolas", 10, "bold"),
            relief="flat", padx=12, pady=5, cursor="hand2")
        self.btn_af.pack(side="left")
        self.btn_af_diag = tk.Button(
            btn_row, text="📏 Measure depth of field",
            command=self._diagnose_focus, bg=C["btn_bg"], fg=C["text"],
            font=FONT_SMALL, relief="flat", padx=10, pady=5, cursor="hand2")
        self.btn_af_diag.pack(side="left", padx=6)

        self._af_result_var = tk.StringVar(value="not focused yet")
        self._af_result_label = tk.Label(
            af, textvariable=self._af_result_var, bg=C["panel"],
            fg=C["muted"], font=FONT_LABEL, anchor="w", justify="left")
        self._af_result_label.pack(fill="x", pady=(8, 2))
        self._af_detail_var = tk.StringVar(value="")
        tk.Label(af, textvariable=self._af_detail_var, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL, anchor="w", justify="left",
                 wraplength=760).pack(fill="x")

        model = self._card(parent, "FOCUS MODEL", C["success"])
        self._model_status_var = tk.StringVar(value=self._model_status_text())
        tk.Label(model, textvariable=self._model_status_var, bg=C["panel"],
                 fg=C["text"], font=FONT_SMALL, anchor="w",
                 justify="left").pack(fill="x", pady=(0, 4))
        tk.Label(model, text=f"file: {os.path.basename(MODEL_PATH)}",
                 bg=C["panel"], fg=C["muted"], font=FONT_SMALL).pack(anchor="w")
        self._make_btn(model, "Reload model", self._reload_model).pack(
            anchor="w", pady=(6, 0))

    def _model_status_text(self) -> str:
        """Human-readable state of the Siamese focus model."""
        if not HAS_TORCH:
            return "PyTorch not installed — focus scoring disabled"
        if AI_MODEL is None:
            exists = os.path.exists(MODEL_PATH)
            return ("not loaded — file missing" if not exists
                    else "not loaded yet (loads on first use)")
        try:
            params = sum(p.numel() for p in AI_MODEL.parameters())
            return f"loaded · {params:,} params · device {AI_DEVICE}"
        except Exception:
            return "loaded"

    def _reload_model(self):
        global _ai_load_attempted
        _ai_load_attempted = False
        self._log("[AI] reloading focus model…")
        init_ai()
        self._model_status_var.set(self._model_status_text())
        self._log(f"[AI] {self._model_status_text()}")

    # ── Tab: Manual ───────────────────────────────────────────────────────────

    def _build_tab_manual(self, parent):
        row1 = tk.Frame(parent, bg=C["bg"])
        row1.pack(fill="x", padx=2, pady=(6, 0))

        self._build_axis_panel(
            row1, "M1 — Z AXIS", tag="Z",
            modes=("mm", "steps"), default_mode="mm", default_val="1.0",
            to_steps=lambda v, m: mm_to_z_steps(v) if m == "mm" else int(v),
            describe=lambda v, m: (f"= {mm_to_z_steps(v):,} pulses" if m == "mm"
                                   else f"= {int(v):,} pulses (direct)"),
            mover=move_z, buttons=[("▲ +", 1, "+"), ("▼ −", 0, "-")])

        self._build_axis_panel(
            row1, "M3 — X AXIS", tag="X",
            modes=("mm", "steps"), default_mode="mm", default_val="1.0",
            to_steps=lambda v, m: mm_to_x_steps(v) if m == "mm" else int(v),
            describe=lambda v, m: (f"= {mm_to_x_steps(v):,} pulses" if m == "mm"
                                   else f"= {int(v):,} pulses (direct)"),
            mover=move_x, buttons=[("◀ −", 0, "-"), ("+ ▶", 1, "+")])

        self._build_axis_panel(
            row1, "M2 — ROTATION", tag="R",
            modes=("steps", "degrees"), default_mode="steps", default_val="1600",
            to_steps=lambda v, m: (int(v) if m == "steps"
                                   else degrees_to_r_steps(v)),
            describe=lambda v, m: (
                f"= {int(v):,} pulses = {v * DEGREES_PER_STEP:.2f}°"
                if m == "steps" else
                f"= {degrees_to_r_steps(v):,} pulses ({v:.2f}°)"),
            mover=move_r, buttons=[("↻ CW", 1, "CW"), ("↺ CCW", 0, "CCW")])

        row2 = tk.Frame(parent, bg=C["bg"])
        row2.pack(fill="x", padx=2)
        self._build_axis_panel(
            row2, "Y AXIS", tag="Y",
            modes=("mm", "steps"), default_mode="mm", default_val="0.35",
            to_steps=lambda v, m: mm_to_y_steps(v) if m == "mm" else int(v),
            describe=lambda v, m: (f"= {mm_to_y_steps(v):,} pulses" if m == "mm"
                                   else f"= {int(v):,} pulses (direct)"),
            mover=move_y, buttons=[("▲ UP", Y_DIR["up"], "up"),
                                   ("▼ DOWN", Y_DIR["down"], "down")])
        self._build_speed_panel(row2)

        self._build_cmd_bar(parent)

    # ── Tab: Calibration ──────────────────────────────────────────────────────

    def _build_tab_calibration(self, parent):
        box = self._card(parent, "CALIBRATION", C["accent2"],
                         "drive X left and Y up to the blade's top-left corner "
                         "before a sweep")
        self.btn_go_start = tk.Button(
            box, text="↖ Go to Starting Point",
            command=self._start_go_to_starting_point,
            bg=C["accent2"], fg="#0f1117", font=("Consolas", 10, "bold"),
            relief="flat", padx=12, pady=5, cursor="hand2")
        self.btn_go_start.pack(anchor="w", pady=(0, 6))
        self._gsp_status = tk.StringVar(value="Ready")
        tk.Label(box, textvariable=self._gsp_status, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(anchor="w")

    # ── Tab: Log ──────────────────────────────────────────────────────────────

    def _build_tab_log(self, parent):
        holder = tk.Frame(parent, bg=C["bg"])
        holder.pack(fill="both", expand=True, pady=6, padx=4)

        hdr = tk.Frame(holder, bg=C["panel"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="▸ LOG", bg=C["panel"], fg=C["accent"],
                 font=FONT_TITLE).pack(side="left", padx=6, pady=4)
        self._make_btn(hdr, "Clear", self._clear_log).pack(
            side="right", padx=4)

        wrap = tk.Frame(holder, bg=C["border"], padx=1, pady=1)
        wrap.pack(fill="both", expand=True, pady=(2, 0))
        self._log_text = scrolledtext.ScrolledText(
            wrap, height=10, bg=C["surface"], fg=C["text"],
            insertbackground=C["text"], relief="flat", font=FONT_SMALL,
            state="disabled", wrap="word", padx=8, pady=6)
        self._log_text.pack(fill="both", expand=True)
        self._log_text.tag_configure("warn", foreground=C["warn"])
        self._log_text.tag_configure("error", foreground=C["danger"])
        self._log_text.tag_configure("stitch", foreground=C["accent2"])
        self._log_text.tag_configure("ok", foreground=C["success"])

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _build_af_row(self, parent, label, fg, variables):
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=C["panel"],
                 fg=fg, font=FONT_SMALL).pack(side="left")
        for name, var in zip(("C", "M", "F", "D"), variables):
            tk.Label(row, text=f" {name}:", bg=C["panel"], fg=C["text"],
                     font=FONT_SMALL).pack(side="left")
            tk.Entry(row, textvariable=var, width=4 if name != "D" else 3,
                     bg=C["surface"], fg=C["text"], relief="flat").pack(side="left", padx=1)

    # ── RLHF Training ─────────────────────────────────────────────────────────

    def _set_frame_a(self):
        frame = self._get_frame()
        if frame is None:
            self._log("[RLHF] No feed available.")
            return
        with self._frame_lock:
            self._frame_a_bgr = frame
        self._frame_a_status.set(" Frame A: Captured!")

    def _train_siamese(self, label: float):
        if self._frame_a_bgr is None:
            self._log("[RLHF] Set Frame A first!")
            return
        frame_b = self._get_frame()
        if frame_b is None:
            return
        try:
            lr = float(self.ds_lr_var.get())
        except ValueError:
            lr = 0.001

        ok, msg = train_siamese_pair(self._frame_a_bgr, frame_b, label, lr)
        if ok:
            self._log(
                f"[AI] Trained {'Sharper' if label == 1.0 else 'Blurrier'} | {msg}")
            self._frame_a_status.set(" Frame A: Used for training.")
        else:
            self._log(f"[AI] Training failed: {msg}")

    # ── AutoFocus ─────────────────────────────────────────────────────────────

    def _parse_af_params(self, c_var, m_var, f_var, d_var, tag):
        try:
            return [int(c_var.get()), int(m_var.get()), int(f_var.get())], float(d_var.get())
        except ValueError:
            self._log(f"[AF] Invalid {tag} AF parameters, using defaults")
            return [800, 400, 200], 2.0

    def _start_autofocus(self):
        try:
            steps = [int(v.get()) for v in (self.af_y_c_var,
                                            self.af_y_m_var, self.af_y_f_var)]
            delay = float(self.af_y_d_var.get())
        except ValueError:
            self._log("[AF] Invalid auto-focus parameters.")
            return

        def worker():
            self._post(lambda: self.btn_af.configure(
                state="disabled", text="⏳ Focusing..."))
            try:
                self._autofocus(steps, delay)
            finally:
                self._post(lambda: self.btn_af.configure(
                    state="normal", text="🎯 Auto-Focus Z"))

        threading.Thread(target=worker, daemon=True).start()

    def _autofocus(self, steps: list | None = None, delay: float | None = None,
                   hint: float | None = None, u: float | None = None,
                   v: float | None = None, stack: bool | None = None):
        """Bring the blade into focus, using the engine selected in the UI.

        Returns a ``FocusResult`` for the region-wise engine, ``None`` for the
        CNN climb.  Kept as one entry point so the sweep, the Auto-Focus button
        and the diagnostics all behave identically.
        """
        if self._af_mode_var.get().startswith("Region"):
            return self._autofocus_regions(hint=hint, u=u, v=v, stack=stack)
        return self._autofocus_cnn(steps or [800, 400, 200], delay or 2.0)

    # ── Region-wise engine (the fix for curved blades) ────────────────────────

    def _focus_cfg(self):
        """Build a FocusConfig from the current UI values."""
        if _focus_engine is None:
            return None

        def num(var, fallback):
            try:
                return float(var.get())
            except (ValueError, AttributeError):
                return fallback
        base = dict(FOCUS_CONFIG)
        base["range_steps"] = int(num(self.af_range_var, base["range_steps"]))
        base["coarse_steps"] = int(num(self.af_y_c_var, base["coarse_steps"]))
        base["fine_steps"] = int(num(self.af_y_f_var, base["fine_steps"]))
        base["predictive_window"] = int(num(self.af_pred_var,
                                            base["predictive_window"]))
        base["z_travel_limit"] = int(
            num(self.af_limit_var, base["z_travel_limit"]))
        base["settle_s"] = num(self.af_y_d_var, base["settle_s"])
        base["backlash_steps"] = int(num(self.af_backlash_var, 0))
        return _focus_engine.FocusConfig(**base)

    def _focus_runner(self, cfg, start_z: float = 0.0):
        """An AutoFocus wired to this app's Z motor and camera feed."""
        def move(delta: int):
            # move_z takes *unsigned* steps plus a direction flag; handing it a
            # signed value and dir=0 sends a negative step count to the ESP32.
            move_z(abs(int(round(delta))), 1 if delta > 0 else 0)

        def wait():
            self._wait_axis("Z")

        return _focus_engine.AutoFocus(
            move, self._get_frame, cfg, wait=wait, log=self._log,
            sleep=time.sleep, steps_per_mm=Z_STEPS_PER_MM, origin=start_z)

    def _autofocus_regions(self, hint=None, u=None, v=None,
                           stack=None) -> "object | None":
        """Scan a wide Z range, score it per region, and say what it found."""
        if _focus_engine is None:
            self._log(
                "[AF] focus_engine.py is missing — falling back to the CNN")
            return self._autofocus_cnn([800, 400, 200], 2.0)
        if self._get_frame() is None:
            self._log("[AF] No feed available, aborting autofocus")
            return None

        cfg = self._focus_cfg()
        if cfg is None:
            return None
        runner = self._focus_runner(cfg)
        try:
            result = runner.run(z_hint=hint)
        except _focus_engine.FocusError as exc:
            self._log(f"[AF] Focus scan could not conclude: {exc}")
            self._set_focus_status("no focus found", "danger")
            return None
        except Exception as exc:                       # noqa: BLE001
            self._log(f"[AF] Error: {exc}")
            return None

        self._focus_result = result
        self._last_focus_runner = runner
        self._post(lambda r=result: self._show_focus_result(r))
        if result.range_limited:
            self._log("[AF] WARNING: the best sample was at the edge of the "
                      "travel — the true focus may be outside the reachable range")
        if self._focus_trend is None:       # stays set across sweeps/buttons
            self._focus_trend = _focus_engine.FocusTrend()
        if u is not None and v is not None:
            self._focus_trend.add(u, v, result.z_peak)

        do_stack = FOCUS_STACKING if stack is None else stack
        if do_stack and result.needs_stacking:
            self._log("[AF] depth spread exceeds depth of field — bracketing Z and "
                      "stacking so the whole frame is sharp")
            self._last_focus_stack = self._capture_focus_stack(runner, result)
        else:
            self._last_focus_stack = None
        return result

    def _capture_focus_stack(self, runner, result):
        """Bracket around the focus and fuse it; returns (frame, weight) or None."""
        try:
            frames = runner.bracket(runner.position,
                                    samples=FOCUS_STACK_SAMPLES)
            if len(frames) < 2:
                return None
            comp, conf = _focus_engine.compose_focus_stack(
                [f for _, f in frames], runner.cfg)
            self._log(f"[AF] stacked {len(frames)} exposures "
                      f"(span {runner.stack.depth_spread():.0f} steps)")
            return comp, conf
        except Exception as exc:                       # noqa: BLE001
            self._log(f"[AF] stacking failed, using the single frame: {exc}")
            return None

    def _show_focus_result(self, result):
        """Paint a FocusResult into the Focus tab."""
        if not hasattr(self, "_af_result_var"):
            return
        self._af_result_var.set(result.summary(Z_STEPS_PER_MM))
        lines = [f"• {line}" for line in result.recommendation(Z_STEPS_PER_MM)]
        runner = getattr(self, "_last_focus_runner", None)
        if runner is not None and getattr(runner, "stack", None) is not None:
            zs = getattr(runner.stack, "z", [])
            if zs:
                span = (max(zs) - min(zs)) / Z_STEPS_PER_MM
                lines.append(
                    f"• searched ±{span / 2:.2f} mm in Z, stage travelled "
                    f"{runner.travel_steps / Z_STEPS_PER_MM:.2f} mm over "
                    f"{result.samples} captures "
                    f"(cap ±{self._focus_cfg().range_steps / 2 / Z_STEPS_PER_MM:.2f} mm)")
        self._af_detail_var.set("\n".join(lines))
        colour = C["success"] if (result.bracketed
                                  and result.in_focus_area >= 0.6) else C["warn"]
        if not result.bracketed:
            colour = C["danger"]
        self._af_result_label.configure(fg=colour)

    def _set_focus_status(self, text, colour="muted"):
        if hasattr(self, "_af_result_var"):
            self._af_result_var.set(text)
            self._af_result_label.configure(fg=C[colour])

    # Presets: the two things that decide how long a scan takes are how far it
    # walks (range/coarse/fine) and how long it waits at each stop (D).  A sweep
    # wants speed on hundreds of stops; a single-shot focus before a capture
    # wants accuracy and can afford a finer walk.
    _FOCUS_PRESETS = {
        "fast": {"range": "1500", "coarse": "150", "fine": "50", "settle": "0.12",
                 "note": "Fast: ±0.94 mm cap, 0.19 mm walk, 0.12 s settle"},
        "balanced": {"range": "3000", "coarse": "100", "fine": "25", "settle": "0.20",
                     "note": "Balanced: ±1.9 mm cap, 0.125 mm walk, 0.20 s settle"},
        "accurate": {"range": "0", "coarse": "60", "fine": "20", "settle": "0.30",
                     "note": "Accurate: whole travel, 75 um walk, 0.30 s settle"},
    }

    def _apply_focus_preset(self, name: str):
        """One click for the range/coarse/fine/settle combination."""
        p = self._FOCUS_PRESETS.get(name)
        if p is None:
            return
        self.af_range_var.set(p["range"])
        for c_var in (self.af_y_c_var, self.af_x_c_var):
            c_var.set(p["coarse"])
        for f_var in (self.af_y_f_var, self.af_x_f_var):
            f_var.set(p["fine"])
        for d_var in (self.af_y_d_var, self.af_x_d_var):
            d_var.set(p["settle"])
        self._af_preset_var.set(p["note"])
        self._log(f"[AF] preset '{name}': {p['note']}")

    def _diagnose_focus(self):
        """Scan Z and report the blade's depth profile — no scanning of the blade."""
        def worker():
            self._post(lambda: self.btn_af_diag.configure(
                state="disabled", text="⏳ Measuring…"))
            try:
                if _focus_engine is None:
                    self._log("[AF] focus_engine.py is missing")
                    return
                cfg = self._focus_cfg()
                if cfg is None or self._get_frame() is None:
                    self._log("[AF] no feed — cannot measure depth of field")
                    return
                runner = self._focus_runner(cfg)
                result = runner.run()
                self._focus_result = result
                self._post(lambda r=result: self._show_focus_result(r))
                self._log(f"[AF] depth of field {result.dof_steps:.0f} steps "
                          f"({result.dof_steps / Z_STEPS_PER_MM:.3f} mm), "
                          f"surface depth spread {result.depth_spread:.0f} steps "
                          f"({result.depth_spread / Z_STEPS_PER_MM:.3f} mm)")
            except Exception as exc:                   # noqa: BLE001
                self._log(f"[AF] measurement failed: {exc}")
            finally:
                self._post(lambda: self.btn_af_diag.configure(
                    state="normal", text="📏 Measure depth of field"))

        threading.Thread(target=worker, daemon=True).start()

    # ── Legacy CNN hill climb ────────────────────────────────────────────────

    def _autofocus_cnn(self, steps: list, delay: float):
        """Coarse-to-fine hill climb on Z using the Siamese focus model.

        Kept for small corrections near focus, where a pairwise preference is
        enough.  It cannot find focus from far away: it has no notion of a range,
        of regions, or of whether the peak was inside what it searched, which is
        why the region-wise engine is the default.
        """
        init_ai()
        if AI_MODEL is None:
            self._log("[AF] Focus model not loaded — autofocus skipped")
            return
        if self._get_frame() is None:
            self._log("[AF] No feed available, aborting autofocus")
            return

        self._log(f"[AF] Start: steps={steps}, delay={delay}s")
        direction = 1

        def step_and_get(dir_mult, size):
            move_z(size, 1 if dir_mult > 0 else 0)
            wait_idle("Z")      # firmware replies before the move is done
            time.sleep(delay)
            return self._get_frame()

        try:
            for step_size in steps:
                self._log(f"[AF] Sweeping with step={step_size}")
                img_current = self._get_frame()
                if img_current is None:
                    continue

                img_next = step_and_get(direction, step_size)
                prob = predict_focus_change(img_current, img_next)
                self._log(f"[AF] Focus probability: {prob:.3f}")

                if prob < 0.4:
                    # Got worse — step back and try the opposite direction
                    img_current = step_and_get(-direction, step_size)
                    direction *= -1
                    img_next = step_and_get(direction, step_size)
                    prob = predict_focus_change(img_current, img_next)
                    self._log(
                        f"[AF] Opposite direction probability: {prob:.3f}")
                    if prob < 0.4:
                        # Worse both ways: already on the peak at this resolution
                        self._log("[AF] Peak at current position")
                        step_and_get(-direction, step_size)
                        continue

                # Climb until focus degrades (capped so a bad model can't run away)
                img_current = img_next
                for i in range(20):
                    img_next = step_and_get(direction, step_size)
                    prob = predict_focus_change(img_current, img_next)
                    self._log(f"[AF] Climb step {i + 1}: prob={prob:.3f}")
                    if prob < 0.4:
                        # Passed the peak — back up; next (finer) pass searches backwards
                        self._log(
                            f"[AF] Passed peak at step={step_size}. Reversing.")
                        step_and_get(-direction, step_size)
                        direction *= -1
                        break
                    img_current = img_next

            self._log("[AF] Peak focus locked")
        except Exception as e:
            self._log(f"[AF] Error: {e}")

    # ── Auto Sweep ────────────────────────────────────────────────────────────

    def _save_sweep_image(self, images_dir: str, column: int, phase: str,
                          frame=None, weight=None):
        """Save the current frame (or a supplied stacked one) and stitch it."""
        if frame is None:
            frame = self._get_frame()
            weight = self._focus_weight(frame) if frame is not None else None
        if frame is None:
            return
        if self._frame_shape is None:
            # The pixel scale needs the frame size, and the frame size is only
            # known once a frame exists.  Taken before the first save so the
            # very first measured step is already usable.
            self._frame_shape = frame.shape[:2]
        filename = f"col{column:02d}_{phase}_{time.strftime('%H%M%S')}.jpg"
        try:
            if cv2.imwrite(os.path.join(images_dir, filename), frame,
                           [cv2.IMWRITE_JPEG_QUALITY, 95]):
                self._log(f"[SWEEP] Image saved: {filename}")
                self._log_capture(images_dir, filename, column, phase)
            else:
                self._log(f"[SWEEP] Failed to save image: {filename}")
        except Exception as e:
            self._log(f"[SWEEP] Error saving image: {e}")
        self._stitch_submit(frame, filename, weight)

    def _log_capture(self, images_dir: str, filename: str, column: int,
                     phase: str):
        """Append to <scan>/capture_log.csv: the exact capture order.

        File names only carry HH:MM:SS, so frames saved in the same second (an
        AF frame and the next step) and scans across midnight are ambiguous.
        The global stitcher reads this log when it exists.  Every saved frame is
        listed - nothing the live stitcher rejects is ever removed."""
        try:
            path = os.path.join(images_dir, "capture_log.csv")
            if self._capture_log is None or self._capture_log[0] != path:
                self._capture_log = [path, 0]
            self._capture_log[1] += 1
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                wr = csv.writer(f)
                if new:
                    wr.writerow(["seq", "file", "column", "phase", "t"])
                wr.writerow([self._capture_log[1], filename, column, phase,
                             f"{time.time():.3f}"])
        except Exception as exc:                    # never break a scan
            self._log(f"[SWEEP] capture log not written: {exc}")

    def _focus_weight(self, frame):
        """Per-pixel focus confidence for the stitcher's blend (or None).

        Without this the stitcher weights frames only by distance from their
        edge, so an out-of-focus region of an otherwise acceptable frame is
        blended at full strength wherever it lands on empty canvas - the blurred
        band the user saw.  With it, the sharp neighbour wins the overlap.
        """
        if not STITCH_FOCUS_WEIGHTS or _focus_engine is None or frame is None:
            return None
        try:
            cfg = self._focus_cfg()
            w, self._focus_ref = _focus_engine.tile_weight_map(
                frame, cfg, reference=self._focus_ref)
            return w
        except Exception as exc:                    # noqa: BLE001
            self._log(f"[STITCH] focus weights unavailable: {exc}")
            return None

    # ── Live mosaic ───────────────────────────────────────────────────────────

    def _stitch_submit(self, frame, filename: str, weight=None):
        """Hand a freshly captured frame to the streaming stitcher.

        Non-blocking: the stitcher runs its own worker thread, so motor timing
        and autofocus are never delayed by registration.  Frames it rejects
        (blurry, duplicate, unregisterable) are logged but otherwise ignored.
        """
        if not LIVE_STITCH or frame is None:
            return
        try:
            if self._stitcher is None:
                # Imported lazily: the GUI must still start on a machine
                # without the stitching pipeline present.
                sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "image_stitching"))
                from live import LiveConfig, LiveStitcher
                out_dir = os.path.join(
                    os.path.dirname(__file__), "blade_mosaic")
                os.makedirs(out_dir, exist_ok=True)
                self._stitcher_dir = out_dir
                self._stitcher = LiveStitcher(
                    LiveConfig(**LIVE_STITCH_CONFIG),
                    out_prefix=os.path.join(
                        out_dir, f"live_{time.strftime('%Y%m%d_%H%M%S')}"),
                    on_decision=self._on_stitch_decision,
                    log=lambda m: self._log(m))
            meta = {"file": filename}
            self._stitcher.submit(frame, filename.rsplit(".", 1)[0], meta,
                                  weight=weight)
        except Exception as e:      # stitching must never break a scan
            self._log(f"[STITCH] disabled after error: {e}")
            self._stitcher = None

    def _stitch_finalize(self):
        if self._stitcher is None:
            return
        stitcher, self._stitcher = self._stitcher, None
        try:
            paths = stitcher.finalize()
            s = stitcher.summary()
            self._log(
                f"[STITCH] mosaic {paths.get('shape')} from {s['accepted']} "
                f"frames, {s['filled']} soft fill-ins, {s['rejected']} dropped "
                f"-> {paths.get('png')}")
        except Exception as e:
            self._log(f"[STITCH] finalize failed: {e}")

    # ── Global (after-the-scan) stitch ───────────────────────────────────────

    def _choose_and_restitch(self):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "blade_images")
        folder = filedialog.askdirectory(
            title="Scan folder to re-stitch",
            initialdir=base if os.path.isdir(base) else None)
        if folder:
            self._start_global_stitch(folder)

    def _start_global_stitch(self, images_dir: str):
        """Re-stitch a whole scan folder in the background."""
        if self._global_running:
            self._log("[GLOBAL] a re-stitch is already running")
            return
        if not images_dir or not os.path.isdir(images_dir):
            self._log(f"[GLOBAL] no such folder: {images_dir}")
            return
        self._global_running = True

        def work():
            try:
                sys.path.insert(0, os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", "image_stitching"))
                from live.global_stitch import GlobalConfig, stitch_folder
                out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "blade_mosaic")
                os.makedirs(out_dir, exist_ok=True)
                name = os.path.basename(os.path.normpath(images_dir))
                prefix = os.path.join(out_dir, f"global_{name}")
                self._log(f"[GLOBAL] re-stitching every frame of {images_dir} "
                          "(this takes a few minutes)")
                paths = stitch_folder(images_dir, prefix,
                                      GlobalConfig(**GLOBAL_STITCH_CONFIG),
                                      log=lambda m: self._log(m))
                self._log(f"[GLOBAL] mosaic {paths.get('shape')} -> {paths.get('jpg')}")
                # The global solve registered every frame against every
                # neighbour, so its per-move motion models are the honest
                # measurement of what this scan's commands did.  Persisted for
                # the next scan (see SWEEP_GEOMETRY.md).
                if paths.get("frame_shape") and self._frame_shape is None:
                    self._frame_shape = tuple(paths["frame_shape"])
                self._scan_geo_from_motion(paths.get("motion"),
                                           self._geometry_x_cmd(images_dir))
                self._show_mosaic_file(paths.get("jpg"))
            except Exception as exc:            # never take the GUI down
                self._log(f"[GLOBAL] re-stitch failed: {exc}")
            finally:
                self._global_running = False

        threading.Thread(target=work, daemon=True, name="global-stitch").start()

    def _show_mosaic_file(self, path):
        """Show a finished mosaic file in the mosaic panel (worker thread)."""
        if not path or not os.path.exists(path):
            return
        img = cv2.imread(path)
        if img is None:
            return
        k = 900.0 / max(img.shape[0], 1)
        if k < 1:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        summary = {"accepted": 0, "filled": 0, "rejected": 0, "disconnected": 0,
                   "mean_ncc": 0.0, "canvas": (0, 0, img.shape[1], img.shape[0])}

        def paint():
            self._mosaic_render(rgb, summary)
            self._mosaic_stat_var.set(f"global stitch: {os.path.basename(path)}")
        self._post(paint)

    def _boundary_pct(self, detect_threshold=None) -> dict | None:
        """Blade boundary distances from frame edges in percent, or None if no frame."""
        frame = self._get_frame()
        if frame is None:
            return None
        b = detect_boundaries(frame, detect_threshold)
        if b is None:
            return {"left": 0.0, "right": 100.0, "upper": 0.0, "lower": 100.0}
        left, right, upper, lower = b
        h, w = frame.shape[:2]
        return {"left": left / w * 100, "right": (w - right) / w * 100,
                "upper": upper / h * 100, "lower": (h - lower) / h * 100}

    def _stop_autosweep(self):
        self._autosweep_stop_evt.set()
        self._log("[SWEEP] Stop requested")

    def _start_autosweep(self):
        if self._autosweep_running:
            return
        self._log("[SWEEP] Starting boundary-based autonomous scan")
        if self._stitcher is not None:
            self._stitch_finalize()      # don't mix two runs in one mosaic
        self._sweep_stitch_var.set("waiting for the first stitched frame")
        self._last_stitch_var.set("—")
        self.btn_autosweep.configure(state="disabled")
        self.btn_stop_autosweep.configure(state="normal")
        self._autosweep_running = True
        self._autosweep_stop_evt.clear()
        # Take the step sizes from the panel *here*, on the Tk thread: the
        # worker must never touch a StringVar (that is a cross-thread Tk call).
        self._scan_geo_sync()
        self._sweep_start_speed_us = self._sweep_speed_value()
        self._sweep_speed_applied.set(f"sent {self._sweep_start_speed_us} µs")
        self._prev_capture_tag = None
        threading.Thread(target=self._sweep_worker, daemon=True).start()

    def _sweep_worker(self):
        """Serpentine scan: Y down to lower edge, X right, Y up to upper edge,
        X right, ... until the right edge of the blade is reached."""
        base_dir = os.path.join(os.path.dirname(__file__), "blade_images")
        images_dir = os.path.join(
            base_dir, f"scan_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(images_dir, exist_ok=True)
        self._log(f"[SWEEP] Image save directory: {images_dir}")

        start_time = time.time()
        self._autosweep_start_time = start_time

        # Step sizes come from SCAN GEOMETRY, which is corrected from the
        # *measured* pixel displacement of the frames the stitcher registers.
        # Nothing here is hardcoded any more.
        geo = getattr(self, "_scan_geo", None)
        if geo is not None and not geo.frame_hw and self._frame_shape:
            geo.frame_hw = tuple(self._frame_shape)
        x_steps = mm_to_x_steps(geo.x_step_mm if geo else 0.4)
        max_y_moves = 100
        self._last_sweep_scan_dir = images_dir
        self._log(f"[SWEEP] geometry: {geo.describe() if geo else 'defaults'}")
        self._scan_geo_after_scan(images_dir, phase="start")

        def y_mm() -> float:
            """Current Y step in mm.

            Read per move rather than captured once, so a correction derived
            from frames already stitched takes effect in the same column.
            """
            return geo.y_step_mm if geo else 0.35

        def y_corr_mm() -> float:
            return Y_CORRECTION_RATIO * y_mm()

        def tune():
            """Adopt measured corrections.  Never raises, never stops a scan."""
            if geo is None or not self._frame_shape:
                return
            try:
                changes = geo.apply(tuple(self._frame_shape))
            except Exception:                 # noqa: BLE001
                return
            for k, (old, new) in changes.items():
                self._log(f"[SWEEP] auto-tune {k}: {old} -> {new}  "
                          f"({geo.advice(self._frame_shape)})")
            if changes:
                self._post(self._scan_geo_refresh)
        LOWER_PCT = 10.0    # lower-boundary trigger
        UPPER_PCT = 20.0    # upper-boundary trigger
        RIGHT_PCT = 10.0    # right-boundary trigger → scan done

        af_y_steps, af_y_delay = self._parse_af_params(
            self.af_y_c_var, self.af_y_m_var, self.af_y_f_var, self.af_y_d_var, "Y")
        af_x_steps, af_x_delay = self._parse_af_params(
            self.af_x_c_var, self.af_x_m_var, self.af_x_f_var, self.af_x_d_var, "X")

        speed_us = getattr(self, "_sweep_start_speed_us", SWEEP_SPEED_US_DEFAULT)
        self._log(f"[SWEEP] motor speed {speed_us} µs half-period "
                  f"({1e6 / (2 * speed_us):,.0f} Hz)")
        set_speed(speed_us, self._log)
        time.sleep(0.2)

        column = 1
        column_times = []
        stop = self._autosweep_stop_evt

        def save(tag, frame=None, weight=None):
            self._save_sweep_image(images_dir, column, tag, frame=frame,
                                   weight=weight)

        # Focus trend along the scan: lets each stop search a narrow window and
        # still escalate when the prediction is wrong.  Without it every stop
        # pays for the full travel, which would double the scan time.
        if _focus_engine is not None:
            self._focus_trend = _focus_engine.FocusTrend()
        y_index = [0]

        def autofocus_y(tag):
            """Focus, save what the focus produced (possibly a Z-stack)."""
            hint = None
            if self._focus_trend is not None:
                pred = self._focus_trend.predict(
                    float(column), float(y_index[0]))
                if pred is None and self._focus_result is not None:
                    # too few points to fit a trend: the previous stop's answer is
                    # still a far better guess than "search the whole travel"
                    pred = 0.0
                if pred is not None:
                    # Every scan leaves the stage *on* the focus, so what the
                    # trend can measure per stop is how far focus moved since the
                    # previous stop - which is exactly what a hint means here.
                    hint = float(np.clip(pred, -3000, 3000))
                    self._log(f"[AF] predicted shift {hint:+.0f} steps from the "
                              f"last stop ({len(self._focus_trend)} points)")
            self._autofocus(hint=hint, u=float(column), v=float(y_index[0]))
            y_index[0] += 1
            time.sleep(0.5)
            stacked = getattr(self, "_last_focus_stack", None)
            if stacked is not None:
                frame, weight = stacked
                save(tag, frame, weight)
            else:
                save(tag)

        def y_sweep(direction, edge, edge_pct, detect_thr, prefix):
            """Step Y until the given boundary edge exceeds edge_pct."""
            for i in range(1, max_y_moves + 1):
                if stop.is_set():
                    return False
                self._log(f"[SWEEP] Moving Y {direction} (attempt {i})")
                t_cmd = time.time()
                move_y_dir(direction, y_mm(), self._log)
                self._wait_move_done(t_cmd, axis="Y")
                time.sleep(0.5)
                save(f"{prefix}_{i}")
                if i % 4 == 0:
                    self._log("[SWEEP] Running AutoFocus after 4 Y moves")
                    autofocus_y(f"{prefix}{i}_AF")
                    tune()
                pct = self._boundary_pct(detect_thr)
                if pct is not None:
                    self._log(
                        f"[SWEEP] {edge.title()} boundary: {pct[edge]:.1f}%")
                    self._post(lambda p=pct[edge], e=edge:
                               self._autosweep_status.set(f"{e.title()}: {p:.1f}%"))
                    if pct[edge] > edge_pct:
                        self._log(f"[SWEEP] ✓ {edge.title()} boundary detected "
                                  f"({pct[edge]:.1f}% > {edge_pct}%)")
                        return True
            self._log("[SWEEP] Max Y moves reached, stopping")
            return False

        def y_correct(direction, tag):
            self._log(
                f"[SWEEP] Corrective Y {direction} shift ({y_corr_mm():.2f} mm)")
            t_cmd = time.time()
            move_y_dir(direction, y_corr_mm(), self._log)
            self._wait_move_done(t_cmd, axis="Y")
            time.sleep(0.5)
            save(tag)

        def x_advance(tag):
            """Move X right + autofocus. True when the right boundary is reached."""
            nonlocal x_steps
            x_steps = mm_to_x_steps(geo.x_step_mm if geo else 0.4)
            # Remember the command: the measured displacement of the frames that
            # follow is what calibrates px/mm, so it has to be attributable.
            self._last_x_cmd_mm = geo.x_step_mm if geo else 0.4
            self._log(f"[SWEEP] Moving X right ({x_steps} steps = "
                      f"{self._last_x_cmd_mm:.2f} mm)")
            t_cmd = time.time()
            move_x(x_steps, 1, self._log)
            self._wait_move_done(t_cmd, axis="X")
            time.sleep(0.5)
            save(f"{tag}_move")
            self._log("[SWEEP] Running AutoFocus after X move")
            # An X advance is 0.4 mm of sideways travel; the surface depth
            # changes by far less than one coarse step, so the focus shift is 0.
            # Passing no hint here made every column turn pay for a full search of
            # the whole Z travel - the single biggest cost in a sweep.
            self._autofocus(hint=0.0, u=float(column), v=float(y_index[0]))
            time.sleep(0.5)
            stacked = getattr(self, "_last_focus_stack", None)
            if stacked is not None:
                save(f"{tag}_AF", stacked[0], stacked[1])
            else:
                save(f"{tag}_AF")
            pct = self._boundary_pct()
            if pct is not None:
                self._log(f"[SWEEP] Right boundary: {pct['right']:.1f}%")
                if pct["right"] > RIGHT_PCT:
                    self._log(
                        "[SWEEP] ✓ Right boundary reached — scan complete")
                    self._post(lambda: self._autosweep_status.set(
                        "Right boundary reached - Done"))
                    return True
            return False

        while not stop.is_set():
            elapsed = int(time.time() - start_time)
            column_start = time.time()
            self._autosweep_column_start_time = column_start
            self._autosweep_column_num = column
            self._log(f"\n[SWEEP] === Column {column} ===")
            self._post(lambda c=column, e=elapsed:
                       self._autosweep_status.set(f"Column {c} | {e // 60}m {e % 60}s"))

            if not y_sweep("down", "lower", LOWER_PCT, None, "Ydown"):
                break
            y_correct("up", "Ycor_up")
            if stop.is_set():
                break
            autofocus_y("Ycor_AF")
            if stop.is_set():
                break
            if x_advance("Xr"):
                break

            if not y_sweep("up", "upper", UPPER_PCT, UPPER_BOUNDARY_THRESHOLD, "Yup"):
                break
            y_correct("down", "Ycor_down")
            if stop.is_set():
                break
            autofocus_y("Yup_AF")
            if stop.is_set():
                break
            if x_advance("Xr2"):
                break

            tune()
            col_elapsed = int(time.time() - column_start)
            column_times.append(col_elapsed)
            self._log(
                f"[SWEEP] Column {column} done in {col_elapsed // 60}m {col_elapsed % 60}s")
            column += 1

        elapsed = int(time.time() - start_time)
        self._log(
            f"[SWEEP] Scan complete or stopped. Total time: {elapsed // 60}m {elapsed % 60}s")
        self._log("[STITCH] writing final mosaic...")
        # Rewrite the record with what the scan *ended up* using: the auto-tune
        # may have changed both steps mid-scan, and the folder must describe the
        # frames in it, not the intent at second zero.
        self._scan_geo_after_scan(images_dir, phase="end")
        self._stitch_finalize()
        if GLOBAL_STITCH:
            self._start_global_stitch(images_dir)
        for n, t in enumerate(column_times, 1):
            self._log(f"[SWEEP]   Column {n}: {t // 60}m {t % 60}s")
        if column_times:
            avg = sum(column_times) // len(column_times)
            self._log(
                f"[SWEEP]   Average per column: {avg // 60}m {avg % 60}s")
        self._autosweep_running = False
        self._post(lambda e=elapsed: self._autosweep_status.set(
            f"Complete | {e // 60}m {e % 60}s"))
        self._post(lambda e=elapsed: self._autosweep_timer_var.set(
            f"⏱ Final total: {e // 60}m {e % 60}s"))
        self._post(lambda: self.btn_autosweep.configure(state="normal"))
        self._post(lambda: self.btn_stop_autosweep.configure(
            state="disabled"))

    # ── Calibration: Go to Starting Point ─────────────────────────────────────

    def _start_go_to_starting_point(self):
        self._log("[GSP] Starting Go to Starting Point routine")
        self.btn_go_start.configure(state="disabled", text="⏳ Calibrating...")
        self._gsp_status.set("Calibrating...")
        threading.Thread(target=self._gsp_worker, daemon=True).start()

    def _gsp_worker(self):
        """Drive X left then Y up to the blade's top-left corner."""
        def set_status(text):
            self._post(lambda: self._gsp_status.set(text))

        try:
            geo = getattr(self, "_scan_geo", None)
            # Only the seek *speed* depends on this; the corner itself is found
            # from the boundary estimate, so a coarse step is safe and faster.
            x_steps = mm_to_x_steps(geo.x_step_mm if geo else 0.4)
            y_mm = geo.y_step_mm if geo else 0.35
            set_speed(1000, self._log)
            time.sleep(0.2)

            # Step 1: X left until left boundary > 20%
            set_status("Moving left...")
            for i in range(1, 101):
                self._log(f"[GSP] Moving X left (attempt {i})")
                t_cmd = time.time()
                move_x(x_steps, 0, self._log)
                self._wait_move_done(t_cmd, axis="X")
                time.sleep(0.5)
                pct = self._boundary_pct()
                if pct is not None:
                    self._log(f"[GSP] Left boundary: {pct['left']:.1f}%")
                    set_status(f"Left: {pct['left']:.1f}%")
                    if pct["left"] > 20.0:
                        self._log(
                            f"[GSP] ✓ Left boundary reached ({pct['left']:.1f}% > 20%)")
                        break
            else:
                self._log("[GSP] Max left moves reached")

            # Step 2: one X step back right
            set_status("Moving right...")
            self._log("[GSP] Step 2: Moving X right one time")
            t_cmd = time.time()
            move_x(x_steps, 1, self._log)
            self._wait_move_done(t_cmd, axis="X")
            time.sleep(0.5)

            # Step 3: Y up until upper boundary > 20%
            set_status("Moving up...")
            for i in range(1, 101):
                self._log(f"[GSP] Moving Y up (attempt {i})")
                t_cmd = time.time()
                move_y_dir("up", y_mm, self._log)
                self._wait_move_done(t_cmd, axis="Y")
                time.sleep(0.5)
                pct = self._boundary_pct(UPPER_BOUNDARY_THRESHOLD)
                if pct is not None:
                    self._log(f"[GSP] Upper boundary: {pct['upper']:.1f}%")
                    set_status(f"Upper: {pct['upper']:.1f}%")
                    if pct["upper"] > 20.0:
                        self._log(
                            f"[GSP] ✓ Upper boundary reached ({pct['upper']:.1f}% > 20%)")
                        break
            else:
                self._log("[GSP] Max up moves reached")

            # Step 4: one Y step back down
            set_status("Moving down...")
            self._log("[GSP] Step 4: Moving Y down one time")
            t_cmd = time.time()
            move_y_dir("down", Y_CORRECTION_RATIO * y_mm, self._log)
            self._wait_move_done(t_cmd, axis="Y")
            time.sleep(0.5)

            self._log("[GSP] ✓ Starting point reached!")
            set_status("✓ Complete")
        except Exception as e:
            self._log(f"[GSP] Error: {e}")
            set_status(f"Error: {e}")
        finally:
            self._post(lambda: self.btn_go_start.configure(
                state="normal", text="↖ Go to Starting Point"))

    # ── Camera Feed ───────────────────────────────────────────────────────────

    def _start_feed(self):
        if self._feed_running:
            return
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            self._log(f"[FEED] Cannot open camera index {CAMERA_INDEX}")
            messagebox.showerror("Camera Error",
                                 f"Cannot open camera index {CAMERA_INDEX}.\n"
                                 "Ensure the HDMI capture card is connected.")
            return

        self._cap = cap
        self._feed_running = True
        self._feed_stop_evt.clear()
        self.btn_start_feed.configure(state="disabled")
        self.btn_stop_feed.configure(state="normal")
        self._log("[FEED] Live feed started")
        threading.Thread(target=self._feed_worker, daemon=True).start()

    def _stop_feed(self):
        if not self._feed_running:
            return
        # Worker owns the capture and releases it when the event fires
        self._feed_stop_evt.set()
        self._feed_running = False
        self.btn_start_feed.configure(state="normal")
        self.btn_stop_feed.configure(state="disabled")
        self._feed_status_var.set("Feed inactive — press Start Feed")

    def _feed_worker(self):
        """Reads frames, runs motion/boundary detection and Siamese inference."""
        cap = self._cap
        last_auto_time = time.time()
        fps_count = 0
        fps_start = time.time()

        while not self._feed_stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Measure achieved capture FPS over ~1 s windows
            fps_count += 1
            fps_now = time.time()
            if fps_now - fps_start >= 1.0:
                self._feed_fps = fps_count / (fps_now - fps_start)
                fps_count = 0
                fps_start = fps_now

            # Auto-refresh Frame A on a user-set interval
            try:
                auto_ms = int(self.auto_a_ms_var.get())
            except ValueError:
                auto_ms = 0
            now = time.time()
            if auto_ms > 0 and (now - last_auto_time) * 1000 >= auto_ms:
                with self._frame_lock:
                    self._frame_a_bgr = frame.copy()
                self._post(lambda ms=auto_ms: self._frame_a_status.set(
                    f" Frame A: Auto ({ms}ms)"))
                last_auto_time = now

            with self._frame_lock:
                self._latest_frame_raw = frame.copy()
                frame_a = self._frame_a_bgr.copy() if self._frame_a_bgr is not None else None

            self._motion_detector.process_frame(frame)

            display = frame.copy()
            boundaries = detect_boundaries(display)
            draw_boundaries(display, boundaries)
            draw_focus_roi(display, FOCUS_CONFIG.get("roi"))

            if frame_a is not None:
                prob = predict_focus_change(frame_a, frame)
                pred_str = ("BETTER (+)" if prob > 0.6 else
                            "WORSE (-)" if prob < 0.4 else "SAME (=)")
            else:
                pred_str = "Set Frame A to begin"

            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            with self._frame_lock:
                self._latest_frame = rgb
                self._latest_pred_str = pred_str
                self._latest_boundaries = boundaries

        cap.release()
        self._cap = None
        self._feed_fps = 0.0
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_raw = None
            self._latest_boundaries = None

    def _get_frame(self):
        """Thread-safe copy of the latest raw frame, or None."""
        with self._frame_lock:
            return self._latest_frame_raw.copy() if self._latest_frame_raw is not None else None

    def _wait_axis(self, axis: str) -> bool:
        """Wait for a stepper move via the firmware's /status pulse counter;
        fall back to visual stillness if /status cannot be read."""
        if wait_idle(axis):
            return True
        self._log(f"[MOTION] {axis}: /status unavailable or timed out - "
                  "falling back to visual feedback")
        return self._wait_motion_stop()

    def _wait_move_done(self, t_cmd: float, expected_s: float = 0.0,
                        start_timeout: float = 0.8, timeout_s: float = 10.0,
                        axis: str | None = None) -> bool:
        """Wait until a commanded move has actually happened *and* stopped.

        With ``axis`` the firmware's /status pulse counter is authoritative
        (the stepper firmware knows exactly when the last pulse went out); the
        visual path below is only the fallback when /status cannot be read.

        ``_wait_motion_stop`` alone returns at once if the detector is still
        "still" from before the move - i.e. before the actuator has even
        started.  The recorded scans show exactly that: each autofocus frame
        sits ~265 px further along the Y move than the frame saved just before
        it, ``Ycor_up`` frames continue ~210 px in the *previous* direction, and
        Y-move frames carry 10-17 % less vertical detail than the (still) AF
        frames - they were grabbed mid-move, motion-blurred.

        So: wait out the commanded duration (whether or not the HTTP call
        blocked), give the detector a moment to *see* motion if it has not yet,
        then wait for stillness.
        """
        if axis is not None:
            if wait_idle(axis):
                return True
            self._log(f"[MOTION] {axis}: /status unavailable or timed out - "
                      "falling back to visual feedback")
        remaining = expected_s - (time.time() - t_cmd)
        if remaining > 0:
            time.sleep(remaining)
        t0 = time.time()
        while (not self._motion_detector.motion_since(t_cmd)
               and time.time() - t0 < start_timeout):
            time.sleep(0.03)
        return self._wait_motion_stop(timeout_s)

    def _wait_motion_stop(self, timeout_s: float = 10.0) -> bool:
        """Block until the motion detector reports stillness, or timeout."""
        start = time.time()
        while time.time() - start < timeout_s:
            if self._motion_detector.is_still():
                self._log("[MOTION] Motor stopped (visual feedback)")
                return True
            time.sleep(0.05)
        self._log(
            f"[MOTION] Timeout waiting for motor to stop (>{timeout_s}s)")
        return False

    def _feed_ui_tick(self):
        """80 ms loop — renders the latest frame into the feed label."""
        with self._frame_lock:
            frame_rgb = self._latest_frame
            pred_str = self._latest_pred_str

        if frame_rgb is not None:
            if HAS_PIL:
                src_h, src_w = frame_rgb.shape[:2]
                panel_w = self._feed_label.winfo_width()
                if panel_w < 10:
                    panel_w = 400
                zoom = self._feed_zoom_var.get() / 100.0
                target_w = max(10, int(panel_w * zoom))
                target_h = max(1, int(src_h * target_w / max(src_w, 1)))

                img = Image.fromarray(frame_rgb).resize(
                    (target_w, target_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self._feed_label.configure(image=photo)
                self._feed_label._photo_ref = photo   # prevent GC of the image
                self._feed_status_var.set(
                    f"Live  ·  {target_w}×{target_h}  ·  {self._feed_fps:.1f} FPS"
                    f"  ·  Siamese: {pred_str}")
                if hasattr(self, "_sb_feed_var"):
                    self._sb_feed_var.set(f"feed {self._feed_fps:.0f} fps")
            else:
                self._feed_status_var.set(
                    "pip install Pillow to enable embedded feed")

        self.after(80, self._feed_ui_tick)

    # ── Manual Command Bar ────────────────────────────────────────────────────

    def _build_cmd_bar(self, parent):
        bar = tk.Frame(parent, bg=C["surface"], padx=10, pady=4)
        bar.pack(fill="x", pady=(0, 4))

        tk.Label(bar, text="CMD:", bg=C["surface"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")
        self.cmd_var = tk.StringVar()
        cmd_entry = tk.Entry(bar, textvariable=self.cmd_var, width=32,
                             bg=C["panel"], fg=C["text"], insertbackground=C["text"],
                             relief="flat", font=FONT_MONO)
        cmd_entry.pack(side="left", padx=(4, 4))
        cmd_entry.bind("<Return>", lambda e: self._send_manual_cmd())
        self._make_btn(bar, "Send", self._send_manual_cmd).pack(side="left")

        tk.Label(bar, text="e.g. /z?steps=800&dir=1  (800 pulses = 1 mm)",
                 bg=C["surface"], fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=8)

        self._cmd_resp_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._cmd_resp_var,
                 bg=C["surface"], fg=C["success"], font=FONT_SMALL).pack(side="left", padx=4)

    def _send_manual_cmd(self):
        cmd = self.cmd_var.get().strip()
        if not cmd:
            return
        if not cmd.startswith("/"):
            cmd = "/" + cmd
        self.cmd_var.set("")

        def _do():
            ok, resp = http_get(cmd, log_fn=self._log)
            self._post(lambda: self._cmd_resp_var.set(f"→ {resp[:60]}"))

        threading.Thread(target=_do, daemon=True).start()

    # ── Widget Helpers ────────────────────────────────────────────────────────

    def _make_panel(self, parent, title: str, subtitle: str = "") -> tk.Frame:
        outer = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        outer.pack(side="left", fill="both", expand=True, padx=4)
        inner = tk.Frame(outer, bg=C["panel"], padx=10, pady=8)
        inner.pack(fill="both", expand=True)

        tk.Label(inner, text=title, bg=C["panel"], fg=C["text"],
                 font=FONT_TITLE).pack(anchor="w")
        if subtitle:
            tk.Label(inner, text=subtitle, bg=C["panel"], fg=C["muted"],
                     font=FONT_SMALL).pack(anchor="w")
        tk.Frame(inner, bg=C["border"], height=1).pack(fill="x", pady=4)
        return inner

    def _make_entry(self, parent, textvariable, w=10):
        return tk.Entry(parent, textvariable=textvariable, width=w,
                        bg=C["surface"], fg=C["text"], insertbackground=C["text"],
                        relief="flat", font=FONT_LABEL)

    def _make_btn(self, parent, text: str, command, bg=None):
        return tk.Button(parent, text=text, command=command,
                         bg=bg or C["btn_bg"], fg=C["text"], font=FONT_SMALL,
                         relief="flat", padx=6, pady=4, cursor="hand2",
                         activebackground=C["btn_hover"], activeforeground="white")

    def _threaded(self, fn, *args):
        threading.Thread(target=fn, args=args, daemon=True).start()

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, message: str):
        self._log_queue.put(message)

    @staticmethod
    def _log_tag(line: str) -> str:
        """Severity tag for a log line — colours the log without extra plumbing."""
        low = line.lower()
        if ("⛔" in line or "error" in low or "failed" in low
                or "[live] drop" in low):
            return "error"
        if "[live]" in low or "[stitch]" in low:
            return "stitch"
        if "[sweep] ✓" in low or "complete" in low or "[ai] successfully" in low:
            return "ok"
        if "warn" in low or "timeout" in low or "[af] passed" in low:
            return "warn"
        return ""

    def _poll_log_queue(self):
        """200 ms loop — moves queued messages into the log panel."""
        drained = False
        self._log_text.configure(state="normal")
        try:
            while True:
                line = self._log_queue.get_nowait()
                self._log_text.insert("end", line + "\n", self._log_tag(line))
                drained = True
        except queue.Empty:
            pass
        if drained:
            # Keep the log bounded to the last ~1000 lines
            lines = int(self._log_text.index("end-1c").split(".")[0])
            if lines > 1000:
                self._log_text.delete("1.0", f"{lines - 1000}.0")
            self._log_text.see("end")
        self._log_text.configure(state="disabled")
        self.after(200, self._poll_log_queue)

    # ── Connection ────────────────────────────────────────────────────────────

    def _ping(self):
        def _do_ping():
            self._log("[PING] Sending ping…")
            self._connected = ping_esp32(self._log)
            self._post(self._update_connection_indicator)
        threading.Thread(target=_do_ping, daemon=True).start()

    def _apply_ip(self, event=None):
        global BASE_URL, ESP32_IP
        ESP32_IP = self.ip_var.get().strip()
        BASE_URL = f"http://{ESP32_IP}:{ESP32_PORT}"
        self._log(f"[CONFIG] IP changed to {ESP32_IP}")

    def _update_connection_indicator(self):
        if self._connected:
            self.conn_dot.configure(fg=C["success"])
            self.conn_label.configure(text="CONNECTED", fg=C["success"])
        else:
            self.conn_dot.configure(fg=C["danger"])
            self.conn_label.configure(text="DISCONNECTED", fg=C["muted"])
        if hasattr(self, "_sb_conn_var"):
            self._sb_conn_var.set(
                "● ONLINE" if self._connected else "● OFFLINE")
            self._sb_conn_label.configure(
                fg=C["success"] if self._connected else C["danger"])

    def _emergency_stop(self):
        self._log("⛔ EMERGENCY STOP SENT")
        self._autosweep_stop_evt.set()
        self._threaded(emergency_stop, self._log)

    def _on_close(self):
        self._feed_stop_evt.set()
        self._autosweep_stop_evt.set()
        # Finalise the mosaic in the background: stopping the stitcher joins its
        # worker, which must not freeze the window or be skipped entirely (that
        # would silently drop everything stitched so far).  _stitch_finalize
        # only touches the log queue, not widgets, so it is thread-safe here.
        if self._stitcher is not None:
            threading.Thread(target=self._stitch_finalize, daemon=True).start()
        self.destroy()


if __name__ == "__main__":
    app = MotorControlApp()
    app.mainloop()
