"""
ESP32 Wireless Motor Control — Desktop GUI  (run: python main3.py)

Motors: M1 Z-axis (1600 steps = 1 mm), M2 rotation (1600 steps/rev),
M3 X-axis (800 steps = 1 mm), linear actuator Y (BTS7960, PWM).
Talks to the ESP32 over WiFi via HTTP GET.

Features: live HDMI camera feed, visual motion detection (closed-loop
feedback), blade boundary detection, Siamese-CNN focus scoring with
in-app training, coarse-to-fine autofocus, and an autonomous
boundary-guided sweep that saves images per column.
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import time
import queue
import os
import sys

import requests
import numpy as np
import cv2

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

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

Z_STEPS_PER_MM = 1600
X_STEPS_PER_MM = 800
R_STEPS_PER_REV = 1600
DEGREES_PER_STEP = 360 / R_STEPS_PER_REV

CAMERA_INDEX = 1               # OpenCV capture index of the HDMI card

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
LIVE_STITCH = True
LIVE_STITCH_CONFIG = {
    "flush_every": 10,      # write a partial mosaic every N accepted frames
    "output_scale": 1.0,    # 0.5 halves mosaic memory 4x for very large parts
    "anchors": 2,           # frames registered against, for drift control
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


def move_actuator(direction: str, duration_ms: int, speed: int, log_fn=None) -> tuple[bool, str]:
    return http_get("/actuator", {"dir": direction, "time": duration_ms, "speed": speed}, log_fn)


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


MODEL_PATH = "siamese_focus_model.pth"
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
        self.title("ESP32 Motor Control System")
        self.geometry("1280x860")
        self.minsize(1100, 800)
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

        self._build_ui()
        self._poll_log_queue()
        self._update_connection_indicator()
        self._feed_ui_tick()
        self._status_ui_tick()
        self._autosweep_timer_tick()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=C["surface"], pady=8, padx=16)
        top.pack(fill="x", side="top")

        tk.Label(top, text="⚙  ESP32 MOTOR CONTROL", bg=C["surface"],
                 fg=C["accent"], font=("Consolas", 15, "bold")).pack(side="left")

        # Connection indicator
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

        # IP entry
        tk.Label(top, text="ESP32 IP:", bg=C["surface"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="right", padx=(0, 2))
        self.ip_var = tk.StringVar(value=ESP32_IP)
        ip_entry = tk.Entry(top, textvariable=self.ip_var, width=14,
                            bg=C["panel"], fg=C["text"], insertbackground=C["text"],
                            relief="flat", font=FONT_SMALL)
        ip_entry.pack(side="right")
        ip_entry.bind("<Return>", self._apply_ip)

        # E-STOP
        tk.Button(self, text="⛔  EMERGENCY STOP", command=self._emergency_stop,
                  bg=C["danger"], fg="white", font=("Consolas", 13, "bold"),
                  relief="flat", padx=20, pady=6, cursor="hand2"
                  ).pack(fill="x", padx=16, pady=(8, 4))

        # Main area
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=16, pady=4)

        # Motor rows
        top_motors = tk.Frame(main, bg=C["bg"])
        top_motors.pack(fill="x")
        row1 = tk.Frame(top_motors, bg=C["bg"])
        row1.pack(fill="x", pady=(0, 2))
        row2 = tk.Frame(top_motors, bg=C["bg"])
        row2.pack(fill="x", pady=(0, 2))
        row3 = tk.Frame(top_motors, bg=C["bg"])
        row3.pack(fill="x", pady=(0, 4))

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
            to_steps=lambda v, m: int(
                v) if m == "steps" else degrees_to_r_steps(v),
            describe=lambda v, m: (f"= {int(v):,} pulses = {v * DEGREES_PER_STEP:.2f}°" if m == "steps"
                                   else f"= {degrees_to_r_steps(v):,} pulses ({v:.2f}°)"),
            mover=move_r, buttons=[("↻ CW", 1, "CW"), ("↺ CCW", 0, "CCW")])

        self._build_actuator_panel(row2)
        self._build_speed_panel(row2)

        self._build_motion_detector_panel(row3)
        self._build_boundary_detector_panel(row3)

        self._build_cmd_bar(main)

        # Bottom half: draggable PanedWindow (feed | controls)
        paned = tk.PanedWindow(main, orient="horizontal", bg=C["border"],
                               sashwidth=6, sashrelief="flat", bd=0, handlesize=0)
        paned.pack(fill="both", expand=True, pady=(4, 0))

        feed_cell = tk.Frame(paned, bg=C["bg"])
        controls_cell = tk.Frame(paned, bg=C["bg"])
        paned.add(feed_cell, stretch="always", minsize=200)
        paned.add(controls_cell, stretch="always", minsize=350)
        self.after(100, lambda: paned.sash_place(
            0, int(self.winfo_width() * 0.55), 0))

        self._build_feed_panel(feed_cell)
        self._build_controls_panel(controls_cell)

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

    # ── Linear Actuator Panel ─────────────────────────────────────────────────

    def _build_actuator_panel(self, parent):
        f = self._make_panel(parent, "LINEAR ACTUATOR",
                             "BTS7960 · PWM ramp control")

        param_row = tk.Frame(f, bg=C["panel"])
        param_row.pack(fill="x", pady=4)
        tk.Label(param_row, text="Speed\n(0-255)", bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left")
        self.act_speed = tk.StringVar(value="70")
        self._make_entry(param_row, self.act_speed,
                         w=6).pack(side="left", padx=6)

        tk.Label(param_row, text="Duration\n(ms)", bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left")
        self.act_duration = tk.StringVar(value="1000")
        self._make_entry(param_row, self.act_duration,
                         w=7).pack(side="left", padx=6)

        tk.Scale(f, from_=0, to=255, orient="horizontal", variable=self.act_speed,
                 bg=C["panel"], fg=C["text"], troughcolor=C["surface"],
                 highlightthickness=0, font=FONT_SMALL, label="PWM Speed"
                 ).pack(fill="x", pady=2)

        btn_row = tk.Frame(f, bg=C["panel"])
        btn_row.pack(fill="x", pady=4)
        self._make_btn(btn_row, "▲ UP", lambda: self._actuator_move("up"),
                       bg="#2b4a7a").pack(side="left", expand=True, fill="x", padx=2)
        self._make_btn(btn_row, "▼ DOWN", lambda: self._actuator_move("down"),
                       bg="#2b4a7a").pack(side="left", expand=True, fill="x", padx=2)
        self._make_btn(btn_row, "■ STOP", lambda: self._threaded(emergency_stop, self._log),
                       bg=C["danger"]).pack(side="left", expand=True, fill="x", padx=2)

    def _actuator_move(self, direction: str):
        try:
            speed = int(self.act_speed.get())
            dur = int(self.act_duration.get())
        except ValueError:
            self._log("[ACT] Invalid speed or duration")
            return
        self._log(f"ACT → {direction.upper()} speed={speed} duration={dur}ms")
        self._threaded(move_actuator, direction, dur, speed, self._log)

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
        if self._motion_detector.is_moving():
            self._md_status_var.set("● MOVING")
            self._feed_label.configure(fg=C["danger"])
        else:
            self._md_status_var.set("● STILL")
            self._feed_label.configure(fg=C["success"])

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

        tk.Label(inner, text="■ middle third = sharpness ROI",
                 bg=C["panel"], fg=C["warn"], font=FONT_SMALL).pack(anchor="w", pady=(0, 4))

        self._feed_label = tk.Label(inner, bg=C["bg"], anchor="nw")
        self._feed_label.pack(fill="both", expand=True, anchor="nw")

        self._feed_status_var = tk.StringVar(
            value="No feed — press Start Feed")
        tk.Label(inner, textvariable=self._feed_status_var,
                 bg=C["panel"], fg=C["muted"], font=FONT_SMALL).pack(anchor="w", pady=4)

    # ── Controls Panel (autosweep / RLHF / calibration / log) ─────────────────

    def _build_controls_panel(self, parent):
        outer = tk.Frame(parent, bg=C["border"], padx=1, pady=1)
        outer.pack(fill="both", expand=True, pady=4, padx=(4, 0))
        inner = tk.Frame(outer, bg=C["panel"], padx=8, pady=8)
        inner.pack(fill="both", expand=True)

        # AUTO SWEEP
        tk.Label(inner, text="▸ AUTO SWEEP", bg=C["panel"], fg=C["warn"],
                 font=FONT_TITLE).pack(anchor="w", pady=(0, 4))
        tk.Label(inner, text="X: ±0.4mm (1000µs)  |  Y: ±1.3s (PWM70)  |  Boundary: 10%",
                 bg=C["panel"], fg=C["accent"], font=FONT_SMALL).pack(anchor="w", pady=4)

        btn_row = tk.Frame(inner, bg=C["panel"])
        btn_row.pack(fill="x", pady=(2, 6))
        self.btn_autosweep = tk.Button(
            btn_row, text="▶ Start", command=self._start_autosweep,
            bg=C["success"], fg="#0f1117", font=("Consolas", 10, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2")
        self.btn_autosweep.pack(side="left", padx=2)
        self.btn_stop_autosweep = tk.Button(
            btn_row, text="■ Stop", command=self._stop_autosweep,
            bg=C["danger"], fg="white", font=("Consolas", 10, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2", state="disabled")
        self.btn_stop_autosweep.pack(side="left", padx=2)

        self._autosweep_status = tk.StringVar(value="Ready")
        tk.Label(inner, textvariable=self._autosweep_status, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(anchor="w", pady=(0, 2))

        self._autosweep_timer_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._autosweep_timer_var, bg=C["panel"],
                 fg=C["accent"], font=FONT_SMALL).pack(anchor="w", pady=(0, 10))

        tk.Frame(inner, bg=C["border"], height=1).pack(fill="x", pady=8)

        # SIAMESE RLHF & AUTO-FOCUS
        tk.Label(inner, text="▸ SIAMESE RLHF & AUTO-FOCUS", bg=C["panel"],
                 fg=C["warn"], font=FONT_TITLE).pack(anchor="w", pady=(0, 4))

        r1 = tk.Frame(inner, bg=C["panel"])
        r1.pack(fill="x", pady=2)
        self._make_btn(r1, "1. Set Frame A (Baseline)",
                       self._set_frame_a).pack(side="left")
        self._frame_a_status = tk.StringVar(value=" Frame A: Not Set")
        tk.Label(r1, textvariable=self._frame_a_status, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(side="left", padx=4)
        tk.Label(r1, text="  Auto (ms):", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left")
        self.auto_a_ms_var = tk.StringVar(value="0")
        tk.Entry(r1, textvariable=self.auto_a_ms_var, width=5, bg=C["surface"],
                 fg=C["text"], insertbackground=C["text"], relief="flat",
                 font=FONT_SMALL).pack(side="left")

        r2 = tk.Frame(inner, bg=C["panel"])
        r2.pack(fill="x", pady=2)
        tk.Label(r2, text="2. Move Z, then train: ", bg=C["panel"], fg=C["text"],
                 font=FONT_SMALL).pack(side="left")
        self._make_btn(r2, "B is Sharper (+)", lambda: self._train_siamese(1.0),
                       bg=C["success"]).pack(side="left", padx=2)
        self._make_btn(r2, "B is Blurrier (-)", lambda: self._train_siamese(0.0),
                       bg=C["danger"]).pack(side="left", padx=2)
        tk.Label(r2, text=" LR:", bg=C["panel"], fg=C["muted"],
                 font=FONT_SMALL).pack(side="left", padx=(6, 2))
        self.ds_lr_var = tk.StringVar(value="0.001")
        tk.Entry(r2, textvariable=self.ds_lr_var, width=6, bg=C["surface"],
                 fg=C["text"], insertbackground=C["text"], relief="flat",
                 font=FONT_SMALL).pack(side="left")

        # AF step sizes: C=coarse, M=medium, F=fine (Z steps), D=settle delay (s)
        self.af_y_c_var, self.af_y_m_var, self.af_y_f_var, self.af_y_d_var = \
            (tk.StringVar(value=v) for v in ("400", "200", "100", "0.1"))
        self.af_x_c_var, self.af_x_m_var, self.af_x_f_var, self.af_x_d_var = \
            (tk.StringVar(value=v) for v in ("400", "200", "100", "0.1"))
        self._build_af_row(inner, "AF Y-Shifts:", C["warn"],
                           (self.af_y_c_var, self.af_y_m_var, self.af_y_f_var, self.af_y_d_var))
        self._build_af_row(inner, "AF X-Shifts:", C["accent2"],
                           (self.af_x_c_var, self.af_x_m_var, self.af_x_f_var, self.af_x_d_var))

        self.btn_af = tk.Button(inner, text="🎯 Auto-Focus Z", command=self._start_autofocus,
                                bg=C["accent"], fg="#0f1117", font=("Consolas", 10, "bold"),
                                relief="flat", padx=10, pady=4, cursor="hand2")
        self.btn_af.pack(anchor="w", pady=4)

        tk.Frame(inner, bg=C["border"], height=1).pack(fill="x", pady=8)

        # CALIBRATION
        tk.Label(inner, text="▸ CALIBRATION", bg=C["panel"], fg=C["accent2"],
                 font=FONT_TITLE).pack(anchor="w", pady=(0, 4))
        self.btn_go_start = tk.Button(
            inner, text="↖ Go to Starting Point", command=self._start_go_to_starting_point,
            bg=C["accent2"], fg="#0f1117", font=("Consolas", 10, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2")
        self.btn_go_start.pack(anchor="w", pady=2)
        self._gsp_status = tk.StringVar(value="Ready")
        tk.Label(inner, textvariable=self._gsp_status, bg=C["panel"],
                 fg=C["muted"], font=FONT_SMALL).pack(anchor="w", pady=(0, 10))

        tk.Frame(inner, bg=C["border"], height=1).pack(fill="x", pady=8)

        # LOG
        tk.Label(inner, text="▸ LOG", bg=C["panel"], fg=C["accent"],
                 font=FONT_TITLE).pack(anchor="w")
        self._log_text = scrolledtext.ScrolledText(
            inner, height=8, bg=C["surface"], fg=C["text"],
            insertbackground=C["text"], relief="flat", font=FONT_SMALL,
            state="disabled", wrap="word")
        self._log_text.pack(fill="both", expand=True, pady=4)

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
            return [1600, 800, 400], 2.0

    def _start_autofocus(self):
        try:
            steps = [int(v.get()) for v in (self.af_y_c_var,
                                            self.af_y_m_var, self.af_y_f_var)]
            delay = float(self.af_y_d_var.get())
        except ValueError:
            self._log("[AF] Invalid auto-focus parameters.")
            return

        def worker():
            self.after(0, lambda: self.btn_af.configure(
                state="disabled", text="⏳ Focusing..."))
            try:
                self._autofocus(steps, delay)
            finally:
                self.after(0, lambda: self.btn_af.configure(
                    state="normal", text="🎯 Auto-Focus Z"))

        threading.Thread(target=worker, daemon=True).start()

    def _autofocus(self, steps: list, delay: float):
        """Coarse-to-fine hill climb on Z using the Siamese focus model."""
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

    def _save_sweep_image(self, images_dir: str, column: int, phase: str):
        frame = self._get_frame()
        if frame is None:
            return
        try:
            filename = f"col{column:02d}_{phase}_{time.strftime('%H%M%S')}.jpg"
            if cv2.imwrite(os.path.join(images_dir, filename), frame):
                self._log(f"[SWEEP] Image saved: {filename}")
            else:
                self._log(f"[SWEEP] Failed to save image: {filename}")
        except Exception as e:
            self._log(f"[SWEEP] Error saving image: {e}")
        self._stitch_submit(frame, filename)

    # ── Live mosaic ───────────────────────────────────────────────────────────

    def _stitch_submit(self, frame, filename: str):
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
                out_dir = os.path.join(os.path.dirname(__file__), "blade_mosaic")
                os.makedirs(out_dir, exist_ok=True)
                self._stitcher_dir = out_dir
                self._stitcher = LiveStitcher(
                    LiveConfig(**LIVE_STITCH_CONFIG),
                    out_prefix=os.path.join(
                        out_dir, f"live_{time.strftime('%Y%m%d_%H%M%S')}"),
                    log=lambda m: self._log(m))
            meta = {"file": filename}
            self._stitcher.submit(frame, filename.rsplit(".", 1)[0], meta)
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
        self.btn_autosweep.configure(state="disabled")
        self.btn_stop_autosweep.configure(state="normal")
        self._autosweep_running = True
        self._autosweep_stop_evt.clear()
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

        x_steps = mm_to_x_steps(0.4)
        y_speed = 70
        y_duration_ms = 1200
        y_correction_ms = 1500
        max_y_moves = 100
        LOWER_PCT = 10.0    # lower-boundary trigger
        UPPER_PCT = 20.0    # upper-boundary trigger
        RIGHT_PCT = 10.0    # right-boundary trigger → scan done

        af_y_steps, af_y_delay = self._parse_af_params(
            self.af_y_c_var, self.af_y_m_var, self.af_y_f_var, self.af_y_d_var, "Y")
        af_x_steps, af_x_delay = self._parse_af_params(
            self.af_x_c_var, self.af_x_m_var, self.af_x_f_var, self.af_x_d_var, "X")

        set_speed(1000, self._log)
        time.sleep(0.2)

        column = 1
        column_times = []
        stop = self._autosweep_stop_evt

        def save(tag):
            self._save_sweep_image(images_dir, column, tag)

        def autofocus_y(tag):
            self._autofocus(af_y_steps, af_y_delay)
            time.sleep(0.5)
            save(tag)

        def y_sweep(direction, edge, edge_pct, detect_thr, prefix):
            """Step Y until the given boundary edge exceeds edge_pct."""
            for i in range(1, max_y_moves + 1):
                if stop.is_set():
                    return False
                self._log(f"[SWEEP] Moving Y {direction} (attempt {i})")
                move_actuator(direction, y_duration_ms, y_speed, self._log)
                self._wait_motion_stop()
                time.sleep(0.5)
                save(f"{prefix}_{i}")
                if i % 4 == 0:
                    self._log("[SWEEP] Running AutoFocus after 4 Y moves")
                    autofocus_y(f"{prefix}{i}_AF")
                pct = self._boundary_pct(detect_thr)
                if pct is not None:
                    self._log(
                        f"[SWEEP] {edge.title()} boundary: {pct[edge]:.1f}%")
                    self.after(0, lambda p=pct[edge], e=edge:
                               self._autosweep_status.set(f"{e.title()}: {p:.1f}%"))
                    if pct[edge] > edge_pct:
                        self._log(f"[SWEEP] ✓ {edge.title()} boundary detected "
                                  f"({pct[edge]:.1f}% > {edge_pct}%)")
                        return True
            self._log("[SWEEP] Max Y moves reached, stopping")
            return False

        def y_correct(direction, tag):
            self._log(
                f"[SWEEP] Corrective Y {direction} shift ({y_correction_ms}ms)")
            move_actuator(direction, y_correction_ms, y_speed, self._log)
            self._wait_motion_stop()
            time.sleep(0.5)
            save(tag)

        def x_advance(tag):
            """Move X right + autofocus. True when the right boundary is reached."""
            self._log("[SWEEP] Moving X right")
            move_x(x_steps, 1, self._log)
            self._wait_motion_stop()
            time.sleep(0.5)
            save(f"{tag}_move")
            self._log("[SWEEP] Running AutoFocus after X move")
            self._autofocus(af_x_steps, af_x_delay)
            time.sleep(0.5)
            save(f"{tag}_AF")
            pct = self._boundary_pct()
            if pct is not None:
                self._log(f"[SWEEP] Right boundary: {pct['right']:.1f}%")
                if pct["right"] > RIGHT_PCT:
                    self._log(
                        "[SWEEP] ✓ Right boundary reached — scan complete")
                    self.after(0, lambda: self._autosweep_status.set(
                        "Right boundary reached - Done"))
                    return True
            return False

        while not stop.is_set():
            elapsed = int(time.time() - start_time)
            column_start = time.time()
            self._autosweep_column_start_time = column_start
            self._autosweep_column_num = column
            self._log(f"\n[SWEEP] === Column {column} ===")
            self.after(0, lambda c=column, e=elapsed:
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

            col_elapsed = int(time.time() - column_start)
            column_times.append(col_elapsed)
            self._log(
                f"[SWEEP] Column {column} done in {col_elapsed // 60}m {col_elapsed % 60}s")
            column += 1

        elapsed = int(time.time() - start_time)
        self._log(
            f"[SWEEP] Scan complete or stopped. Total time: {elapsed // 60}m {elapsed % 60}s")
        self._log("[STITCH] writing final mosaic...")
        self._stitch_finalize()
        for n, t in enumerate(column_times, 1):
            self._log(f"[SWEEP]   Column {n}: {t // 60}m {t % 60}s")
        if column_times:
            avg = sum(column_times) // len(column_times)
            self._log(
                f"[SWEEP]   Average per column: {avg // 60}m {avg % 60}s")
        self._autosweep_running = False
        self.after(0, lambda e=elapsed: self._autosweep_status.set(
            f"Complete | {e // 60}m {e % 60}s"))
        self.after(0, lambda e=elapsed: self._autosweep_timer_var.set(
            f"⏱ Final total: {e // 60}m {e % 60}s"))
        self.after(0, lambda: self.btn_autosweep.configure(state="normal"))
        self.after(0, lambda: self.btn_stop_autosweep.configure(
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
            self.after(0, lambda: self._gsp_status.set(text))

        try:
            x_steps = mm_to_x_steps(0.4)
            y_speed = 70
            set_speed(1000, self._log)
            time.sleep(0.2)

            # Step 1: X left until left boundary > 20%
            set_status("Moving left...")
            for i in range(1, 101):
                self._log(f"[GSP] Moving X left (attempt {i})")
                move_x(x_steps, 0, self._log)
                self._wait_motion_stop()
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
            move_x(x_steps, 1, self._log)
            self._wait_motion_stop()
            time.sleep(0.5)

            # Step 3: Y up until upper boundary > 20%
            set_status("Moving up...")
            for i in range(1, 101):
                self._log(f"[GSP] Moving Y up (attempt {i})")
                move_actuator("up", 1200, y_speed, self._log)
                self._wait_motion_stop()
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
            move_actuator("down", 1500, y_speed, self._log)
            self._wait_motion_stop()
            time.sleep(0.5)

            self._log("[GSP] ✓ Starting point reached!")
            set_status("✓ Complete")
        except Exception as e:
            self._log(f"[GSP] Error: {e}")
            set_status(f"Error: {e}")
        finally:
            self.after(0, lambda: self.btn_go_start.configure(
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
                self.after(0, lambda ms=auto_ms: self._frame_a_status.set(
                    f" Frame A: Auto ({ms}ms)"))
                last_auto_time = now

            with self._frame_lock:
                self._latest_frame_raw = frame.copy()
                frame_a = self._frame_a_bgr.copy() if self._frame_a_bgr is not None else None

            self._motion_detector.process_frame(frame)

            display = frame.copy()
            boundaries = detect_boundaries(display)
            draw_boundaries(display, boundaries)

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

        tk.Label(bar, text="e.g. /z?steps=800&dir=1",
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
            self.after(0, lambda: self._cmd_resp_var.set(f"→ {resp[:60]}"))

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

    def _poll_log_queue(self):
        """200 ms loop — moves queued messages into the log panel."""
        drained = False
        self._log_text.configure(state="normal")
        try:
            while True:
                self._log_text.insert(
                    "end", self._log_queue.get_nowait() + "\n")
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
            self.after(0, self._update_connection_indicator)
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

    def _emergency_stop(self):
        self._log("⛔ EMERGENCY STOP SENT")
        self._autosweep_stop_evt.set()
        self._threaded(emergency_stop, self._log)

    def _on_close(self):
        self._feed_stop_evt.set()
        self._autosweep_stop_evt.set()
        self.destroy()


if __name__ == "__main__":
    app = MotorControlApp()
    app.mainloop()
