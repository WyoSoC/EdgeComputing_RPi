# Radar-Gated ByteTrack Orchestrator (RD-03E + RPi 5 + Hailo YOLOv8)

This documentation covers a **radar-gated vision pipeline** on a Raspberry Pi 5. An **AI-Thinker RD-03E mmWave radar** continuously runs and outputs presence (`present=True/False`). When presence is detected, the system **starts** a ByteTrack + YOLOv8 object detection pipeline. When presence is absent for a configurable delay, the system **stops** the vision pipeline (including camera processes) to save power/compute.


## Prerequisites (Required External Guides)

Complete these guides first (in order):

1) **RD-03E Radar Setup (Wiring + UART + script behavior)**  
https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/AI-Thinker%20RD-03E%20Radar.md

2) **YOLOv8 + ByteTrack + Hailo Object Detection Setup**  
https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/YOLOv8_ByteTrack_ObjectDetection_Guide.md


## Hailo Version Lock (Critical)

All Hailo libraries MUST match the same version.

**Required version: `4.20`**  
Ensure **all** Hailo components installed on the device are aligned to 4.20 (runtime, Python wheel(s), CLI tools, and any related system packages).


## Hardware

- Raspberry Pi 5
- Hailo AI Kit (Hailo-8 / Hailo-8L depending on kit)
- libcamera-compatible camera (e.g., Arducam AF 16MP or Pi Camera module)
- AI-Thinker RD-03E mmWave radar (UART via USB-serial adapter / bridge; typically `/dev/ttyUSB0`)


## What This Orchestrator Does

- Starts **RADAR** process and reads its stdout line-by-line.
- Parses lines for `present=True` / `present=False`.
- When presence becomes active (within `PRESENCE_HOLD_S`):
  - Starts **VISION/BYTETRACK** inside your ByteTrack venv.
- When presence is stale for `OFF_DELAY_S`:
  - Stops **VISION/BYTETRACK** by signaling its **process group** (so rpicam child processes stop too).
- If VISION crashes:
  - Optionally restarts VISION if presence is still active (`RESTART_ON_CRASH=True`).


## Repository / File Layout (Typical)

/home/<user>/
gated_orchestrator.py
rd03e_live_fast.py
bytetrack_env/
bytetrack_venv/
bin/activate
ByteTrack/
bytetrack_detect.py


## Installation Steps

### Step 0 — Validate independent components first
1. Radar works alone:
   - Confirm `rd03e_live_fast.py` prints `present=True/False` continuously.
2. Vision works alone:
   - Confirm ByteTrack script runs and shows detections with the Hailo pipeline.

Only then proceed to gating.


### Step 1 — Place the orchestrator in home
Copy `gated_orchestrator.py` to:
- `/home/<user>/gated_orchestrator.py`

Example:
- `/home/abalhas1/gated_orchestrator.py`


### Step 2 — Confirm paths inside `gated_orchestrator.py`

In the CONFIG section, update if needed:

- `RADAR_SCRIPT`
- `RADAR_DEVICE`
- `RADAR_ARGS`
- `VENV_ACTIVATE`
- `BYTETRACK_SCRIPT`
- `BYTETRACK_CWD`

Typical:
- `RADAR_SCRIPT = /home/<user>/rd03e_live_fast.py`
- `RADAR_DEVICE = /dev/ttyUSB0`
- `VENV_ACTIVATE = /home/<user>/bytetrack_env/bytetrack_venv/bin/activate`
- `BYTETRACK_SCRIPT = /home/<user>/bytetrack_env/ByteTrack/bytetrack_detect.py`
- `BYTETRACK_CWD = /home/<user>/bytetrack_env/ByteTrack`


## gated_orchestrator.py (Copy/Paste)


```python
#!/usr/bin/env python3
"""
gated_orchestrator.py

Radar-gated vision pipeline:
- Always run RD-03E radar script (rd03e_live_fast.py) and parse its stdout for presence.
- When presence detected -> start bytetrack_detect.py inside the bytetrack venv.
- When presence absent for OFF_DELAY_S -> stop bytetrack (and its rpicam child processes).

Notes:
- Radar output must include lines containing: present=True / present=False
- Vision is launched in its own process group so we can stop child processes (rpicam-hello) reliably.
"""

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# =========================
# CONFIG (EDIT THESE)
# =========================

# --- Radar ---
RADAR_SCRIPT = str(Path.home() / "rd03e_live_fast.py")   # <-- EDIT if different
RADAR_DEVICE = "/dev/ttyUSB0"
RADAR_ARGS   = ["--low-latency", "--print-hz", "10"]     # match what you used successfully

# --- Vision / ByteTrack ---
VENV_ACTIVATE = str(Path.home() / "bytetrack_env" / "bytetrack_venv" / "bin" / "activate")
BYTETRACK_SCRIPT = str(Path.home() / "bytetrack_env" / "ByteTrack" / "bytetrack_detect.py")
BYTETRACK_CWD = str(Path.home() / "bytetrack_env" / "ByteTrack")

# --- Gating behavior ---
PRESENCE_HOLD_S   = 1.5   # "recent true" window
OFF_DELAY_S       = 8.0   # stop after this long without true
COOLDOWN_S        = 2.0
RESTART_ON_CRASH  = True

# Optional: set True if you want to see when presence is actually parsed
DEBUG_PRESENCE_PARSE = False


# =========================
# INTERNALS
# =========================

@dataclass
class ProcHandle:
    popen: subprocess.Popen
    name: str


def now() -> float:
    return time.monotonic()


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


# ---- Parse "present=True/False" ----
PRESENT_FIELD_RX = re.compile(
    r"\bpresent\b\s*[:=]\s*(true|false|1|0|yes|no|on|off)\b",
    re.IGNORECASE
)

# Fallback heuristics (in case radar script format changes)
PRESENCE_REGEXES = [
    re.compile(r"\bpresence\b\s*[:=]\s*(1|true|on|yes)\b", re.IGNORECASE),
    re.compile(r"\boccupied\b\s*[:=]\s*(1|true|on|yes)\b", re.IGNORECASE),
    re.compile(r"\bmotion\b\s*[:=]\s*(1|true|on|yes)\b", re.IGNORECASE),
    re.compile(r"\btarget(s)?\b.*\b([1-9]\d*)\b", re.IGNORECASE),
    re.compile(r"\bdetect(ed|ion)?\b.*\b(1|true|on|yes)\b", re.IGNORECASE),
]

TRUE_WORDS  = {"1", "true", "yes", "on"}
FALSE_WORDS = {"0", "false", "no", "off"}


def parse_presence(line: str) -> Optional[bool]:
    """
    Returns:
      True  -> explicit presence detected
      False -> explicit absence detected
      None  -> line doesn't contain a clear presence signal
    """
    s = line.strip()
    if not s:
        return None

    # 1) Exact field match (preferred): "present=True/False"
    m = PRESENT_FIELD_RX.search(s)
    if m:
        v = m.group(1).lower()
        if v in TRUE_WORDS:
            return True
        if v in FALSE_WORDS:
            return False

    # 2) Fallback heuristics
    for rx in PRESENCE_REGEXES:
        m = rx.search(s)
        if not m:
            continue
        for g in m.groups():
            if g is None:
                continue
            gg = g.strip().lower()
            if gg.isdigit():
                return int(gg) > 0
            if gg in TRUE_WORDS:
                return True
        return True

    return None


def spawn_process(cmd, name: str, cwd: Optional[str] = None) -> ProcHandle:
    log(f"START {name}: {cmd}")

    # Show VISION output in terminal; keep RADAR piped so we can parse it.
    if name.startswith("VISION"):
        out = None
    else:
        out = subprocess.PIPE

    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=out,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid  # new process group for clean stop
    )
    return ProcHandle(popen=p, name=name)


def stop_process_tree(ph: ProcHandle, sig=signal.SIGTERM, timeout_s: float = 3.0) -> None:
    p = ph.popen
    if p.poll() is not None:
        return

    try:
        pgid = os.getpgid(p.pid)
    except ProcessLookupError:
        return

    log(f"STOP {ph.name}: send {sig.name} to pgid {pgid}")
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return

    t0 = now()
    while (now() - t0) < timeout_s:
        if p.poll() is not None:
            log(f"STOP {ph.name}: exited with code {p.returncode}")
            return
        time.sleep(0.05)

    log(f"STOP {ph.name}: escalate SIGKILL to pgid {pgid}")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def build_bytetrack_command() -> list:
    bash_cmd = (
        f"source {shlex.quote(VENV_ACTIVATE)} && "
        f"exec python3 {shlex.quote(BYTETRACK_SCRIPT)}"
    )
    return ["bash", "-lc", bash_cmd]


def build_radar_command() -> list:
    return ["python3", RADAR_SCRIPT, RADAR_DEVICE, *RADAR_ARGS]


class Orchestrator:
    def __init__(self):
        self.radar: Optional[ProcHandle] = None
        self.vision: Optional[ProcHandle] = None

        self.last_presence_t: float = 0.0
        self.last_action_t: float = 0.0
        self.running = True

        self._install_signal_handlers()

    def _install_signal_handlers(self):
        def _handler(signum, _frame):
            sig = signal.Signals(signum).name
            log(f"Received {sig} -> shutting down")
            self.running = False

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)

    def start_radar(self):
        ensure_exists(RADAR_SCRIPT, "Radar script")
        self.radar = spawn_process(build_radar_command(), "RADAR")

    def start_vision(self):
        ensure_exists(VENV_ACTIVATE, "Venv activate")
        ensure_exists(BYTETRACK_SCRIPT, "ByteTrack script")

        # cooldown to prevent flapping
        if (now() - self.last_action_t) < COOLDOWN_S:
            return

        if self.vision and self.vision.popen.poll() is None:
            return  # already running

        self.last_action_t = now()
        self.vision = spawn_process(build_bytetrack_command(), "VISION/BYTETRACK", cwd=BYTETRACK_CWD)
        log("VISION state: ON")

    def stop_vision(self):
        # cooldown to prevent flapping
        if (now() - self.last_action_t) < COOLDOWN_S:
            return

        if not self.vision:
            return
        if self.vision.popen.poll() is not None:
            self.vision = None
            return

        self.last_action_t = now()

        # Try SIGINT first (lets ByteTrack save logs on Ctrl+C behavior)
        stop_process_tree(self.vision, sig=signal.SIGINT, timeout_s=2.0)

        # If still alive, try SIGTERM
        if self.vision.popen.poll() is None:
            stop_process_tree(self.vision, sig=signal.SIGTERM, timeout_s=2.0)

        self.vision = None
        log("VISION state: OFF")

    def radar_loop(self):
        assert self.radar and self.radar.popen.stdout

        for line in self.radar.popen.stdout:
            if not self.running:
                break

            line = line.rstrip("\n")
            if line:
                log(f"RADAR> {line}")

            p = parse_presence(line)
            if p is True:
                self.last_presence_t = now()
                if DEBUG_PRESENCE_PARSE:
                    log("PARSE presence=True -> updating last_presence_t")
            elif p is False:
                if DEBUG_PRESENCE_PARSE:
                    log("PARSE presence=False")

            self._gate_step()

        rc = self.radar.popen.poll()
        log(f"RADAR exited (code={rc}).")
        self.running = False

    def _presence_is_active(self) -> bool:
        return self.last_presence_t > 0 and (now() - self.last_presence_t) <= PRESENCE_HOLD_S

    def _presence_is_stale(self) -> bool:
        return self.last_presence_t <= 0 or (now() - self.last_presence_t) >= OFF_DELAY_S

    def _gate_step(self):
        # If vision crashed, optionally restart if presence is still active
        if self.vision and self.vision.popen.poll() is not None:
            log(f"VISION crashed (code={self.vision.popen.returncode})")
            self.vision = None
            if RESTART_ON_CRASH and self._presence_is_active():
                log("Presence still active -> restarting VISION")
                self.start_vision()

        if self._presence_is_active():
            self.start_vision()
        elif self._presence_is_stale():
            self.stop_vision()

    def shutdown(self):
        if self.vision:
            self.stop_vision()
        if self.radar:
            stop_process_tree(self.radar, sig=signal.SIGTERM, timeout_s=2.0)
            self.radar = None

    def run(self):
        ensure_exists(RADAR_SCRIPT, "Radar script")
        ensure_exists(VENV_ACTIVATE, "Venv activate")
        ensure_exists(BYTETRACK_SCRIPT, "ByteTrack script")

        self.start_radar()

        try:
            self.radar_loop()
        finally:
            self.shutdown()
            log("Shutdown complete.")


def main():
    Orchestrator().run()


if __name__ == "__main__":
    main()
