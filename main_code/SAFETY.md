# Emergency stop

The stop button in `main3.py` is a **latch**, not a request. Pressing it halts
the machine and keeps it halted; nothing resumes until the operator presses the
same button again (it turns into `🔒 HALTED — click to RE-ARM`).

## Why it changed

The old handler set the autosweep flag and fired one `/stop` over HTTP. That is
not a stop:

* the sweep could be **inside** a 1.5 s actuator move, an autofocus bracket, a
  settle delay (`time.sleep(0.5)`, 14 of them) or a `_wait_motion_stop` poll —
  every one of those ran to completion first, so the machine kept moving for
  seconds after the press;
* a command already queued or in flight could still reach the firmware, i.e.
  motion *after* the stop;
* one HTTP GET can be lost while the firmware is busy stepping, so the stop was
  only ever *attempted*;
* once the sweep flag was cleared the machine would happily start again.

## What it does now

`SafetyStop` (`main3.py`) has three jobs:

1. **Latches.** `trip()` bumps a generation counter and sets the halted event.
2. **Interrupts everything that is waiting.** All scan waits go through
   `haltable_sleep()` / `SafetyStop.wait()`, which return the instant the latch
   is set and raise `Halted` — an exception, not a flag, so a sweep's nested
   loops and callbacks unwind instead of finishing the step they were on. The
   autofocus engine is handed `haltable_sleep` as its `sleep`, so a Z bracket
   stops mid-bracket. `_wait_motion_stop` aborts in ≤50 ms.
3. **Refuses motion.** Every motion wrapper (`move_x`, `move_z`, `move_r`,
   `move_actuator`, `set_speed`) raises `Halted` while latched, and `http_get`
   refuses `/z /x /r /actuator /speed` outright — so even a caller that ignores
   exceptions cannot move the stage. `/stop` and `/ping` are never refused.
4. **Kills the hardware.** `/stop` is retried four times, 250 ms apart, in its
   own thread (the window never freezes), and the log says
   `hardware acknowledged stop` — or warns loudly that it was **not**
   acknowledged, because a stage that ignores the stop is a real hazard.

The Tk side disables every widget that can move the machine (found by walking
the widget tree, so a control added later cannot be forgotten), refuses to start
a new sweep, and drops frames captured while latched from the mosaic.

## Re-arming

Pressing the button again re-arms: the latch clears, the controls come back with
their previous states, and the log records it. There is no automatic re-arm and
no path that clears the latch except an operator press.

## Tests

`test_ui_headless.py` covers it end to end (no hardware, no display):

* the latch refuses all five motion wrappers without touching the network;
* the hardware stop is retried until acknowledged;
* a 30 s wait aborts in well under a second;
* a running sweep unwinds mid-move and is marked `HALTED`;
* the button latches, disables motion controls, refuses to start a sweep, and
  restores everything on re-arm.

```
python test_ui_headless.py     # → estop: ... PASS ×4
```
