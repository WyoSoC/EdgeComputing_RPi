# Two-Step Gated Vision Pipeline  
## RD-03E Radar -> MLX90640 Thermal Verification -> Hailo YOLOv8 + ByteTrack Vision

This document explains the two-step gated multimodal sensing pipeline used on the Raspberry Pi 5. In this design, the AI-Thinker RD-03E radar continuously monitors for presence. When the radar reports `present=True`, the system does not immediately start the camera and object detection pipeline. Instead, it first runs a headless MLX90640 thermal verification stage. Only if the thermal stage confirms likely living heat does the system start the Hailo YOLOv8 + ByteTrack vision pipeline.

This makes the system more selective than simple radar-gated vision, because radar alone is not enough to activate the expensive camera + AI stage.


## Required Supporting Documentation

Complete or review these first:

1. **RD-03E Radar Setup**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/AI-Thinker%20RD-03E%20Radar.md

2. **MLX90640 Thermal Sensor Setup**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/MLX90640%20IRCamera-OpenCV-PyQt5-ObjectDetection.md

3. **YOLOv8 + ByteTrack Object Detection Setup**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/YOLOv8_ByteTrack_ObjectDetection_Guide.md


## Hailo Version Lock (Critical)

All Hailo libraries and related components must be aligned to the same version.

**Required version: `4.20`**

Make sure the following all match:
- Hailo runtime
- Python packages
- CLI tools
- related system packages

If versions do not match, the vision pipeline may break.


## Hardware

- Raspberry Pi 5
- Hailo AI Kit
- libcamera-compatible camera
  - for example: Arducam AF 16MP or Pi Camera module
- AI-Thinker RD-03E mmWave radar
- MLX90640 thermal infrared sensor

### Interfaces used
- **Radar**: UART / USB-serial
- **Thermal sensor**: I2C
- **Camera**: Raspberry Pi camera stack
- **Hailo**: AI accelerator stack


## Main Pipeline Logic

The pipeline has three stages:

### Step 1 — Radar
The RD-03E radar runs continuously and prints presence lines such as:
- `present=True`
- `present=False`

When the radar reports `present=True`, the orchestrator moves to Step 2.

### Step 2 — Thermal Verification
A short MLX90640 headless verification step runs for a limited timeout window.

If thermal verification confirms likely living heat:
- move to Step 3

If thermal verification fails:
- return to Step 1

### Step 3 — Vision
The Hailo YOLOv8 + ByteTrack vision pipeline starts.

Vision stays on while detections continue.  
If no detections are seen for the configured timeout:
- vision stops
- system returns to Step 1


## Why This Two-Step Design Matters

Compared to one-step radar-gated vision:

### One-step gated
- radar directly activates vision

### Two-step gated
- radar triggers thermal verification first
- thermal must pass before vision starts

This helps reduce unnecessary camera activations and makes the system more selective.


## What the Orchestrator Does

The orchestrator:

- starts the radar process
- reads radar stdout continuously
- watches for `present=True`
- runs MLX90640 thermal verification
- if thermal passes, starts ByteTrack / Hailo vision
- keeps vision on while detections continue
- shuts vision off after a no-detection timeout
- applies a short cooldown
- returns to radar monitoring


## File Layout (Typical)

`/home/<user>/`

- `rd03e_live_fast.py`
- `two_step_gated_orchestrator.py`
- `thermal_verify_mlx90640_headless.py`
- `mlx-env/`
  - `bin/python3`
- `bytetrack_env/`
  - `bytetrack_venv/bin/activate`
  - `ByteTrack/bytetrack_detect.py`


## Important Files

### Radar
- `rd03e_live_fast.py`

This must already work and print `present=True/False` correctly.

### Two-step orchestrator
- `two_step_gated_orchestrator.py`

This is the main controller that links radar, thermal verification, and vision.

### Thermal verification helper
- `thermal_verify_mlx90640_headless.py`

This is created automatically by the orchestrator if it does not already exist.

### Vision
- `bytetrack_detect.py`

This runs the Hailo YOLOv8 + ByteTrack pipeline.


## Main Configuration Parameters

### Radar
- `RADAR_SCRIPT`
- `RADAR_DEVICE`
- `RADAR_ARGS`
- `RADAR_STARTUP_GRACE_SEC`

### Radar cooldown
- `POST_VISION_RADAR_COOLDOWN_SEC`

### Thermal
- `THERMAL_PYTHON`
- `THERMAL_TIMEOUT_SEC`
- `THERMAL_THRESHOLD_C`
- `THERMAL_MIN_HOT_PIXELS`
- `THERMAL_CONSECUTIVE_HITS`
- `THERMAL_MAX_FRAME_RATE_SLEEP`
- `THERMAL_VERIFY_SCRIPT`

### Vision
- `VENV_ACTIVATE`
- `BYTETRACK_SCRIPT`
- `BYTETRACK_CWD`
- `VISION_NO_DETECTION_TIMEOUT_SEC`
- `VISION_STARTUP_GRACE_SEC`
- `VISION_MAX_SESSION_SEC`
- `VISION_STOP_SIGNAL`

### Logging
- `PRINT_RADAR_LINES`
- `PRINT_THERMAL_LINES`
- `PRINT_VISION_LINES`


## Parameter Meaning

### `THERMAL_TIMEOUT_SEC`
How long the thermal stage is allowed to search for valid thermal presence before giving up.

### `THERMAL_THRESHOLD_C`
Temperature threshold above which pixels are considered hot.

### `THERMAL_MIN_HOT_PIXELS`
Minimum number of hot pixels required to count as meaningful thermal presence.

### `THERMAL_CONSECUTIVE_HITS`
How many qualifying frames in a row are needed before thermal presence is confirmed.

### `VISION_NO_DETECTION_TIMEOUT_SEC`
If no detection lines appear for this many seconds, vision shuts off.

### `VISION_STARTUP_GRACE_SEC`
Grace period after vision starts so it is not judged idle too early.

### `POST_VISION_RADAR_COOLDOWN_SEC`
Cooldown after vision stops before radar is allowed to retrigger.


## Expected Runtime Flow

### Successful cycle
1. Radar starts
2. Radar prints `present=False`
3. Radar eventually prints `present=True`
4. Thermal verification starts
5. Thermal confirms living heat
6. Vision starts
7. Detections continue
8. Detections stop
9. Vision times out and shuts off
10. Cooldown happens
11. System returns to radar monitoring

### Failed thermal cycle
1. Radar starts
2. Radar prints `present=True`
3. Thermal verification starts
4. Thermal fails to confirm presence
5. Vision never starts
6. System returns to radar monitoring


## Installation Steps

### Step 0 — Validate all components independently
Before using the full two-step pipeline, confirm:

1. Radar works alone
   - `rd03e_live_fast.py` prints `present=True/False`

2. Thermal works alone
   - MLX90640 can be read in the `mlx-env` environment

3. Vision works alone
   - ByteTrack / Hailo vision runs successfully

### Step 1 — Place the orchestrator in home
Copy the script to:

- `/home/<user>/two_step_gated_orchestrator.py`

### Step 2 — Confirm path values in the script
Check these:
- `RADAR_SCRIPT`
- `THERMAL_PYTHON`
- `THERMAL_VERIFY_SCRIPT`
- `VENV_ACTIVATE`
- `BYTETRACK_SCRIPT`
- `BYTETRACK_CWD`

### Step 3 — Make executable
~~~sh
chmod +x ~/two_step_gated_orchestrator.py
~~~

### Step 4 — Run
~~~sh
python3 ~/two_step_gated_orchestrator.py
~~~

### Step 5 — Stop
~~~sh
Ctrl+C
~~~


## Full Script: `two_step_gated_orchestrator.py`

~~~python
#!/usr/bin/env python3
"""
Two-Step Verification Orchestrator
(RD-03E Radar -> MLX90640 Thermal Verify -> Hailo YOLOv8 + ByteTrack Vision)

Behavior:
  STEP 1 (Radar): continuously reads RD-03E stdout. When present=True => trigger Step 2.
  STEP 2 (Thermal Verify): runs a headless MLX90640 verification for a short timeout window.
      - If thermal "living presence" detected => start Step 3.
      - If not detected within timeout => return to Step 1.
  STEP 3 (Vision): starts ByteTrack/Hailo YOLOv8 pipeline.
      - If detections keep appearing, vision stays ON.
      - If NO detections are seen for VISION_NO_DETECTION_TIMEOUT_SEC, vision stops.
      - Then return to Step 1.

Notes:
  - Thermal verification runs using a dedicated Python interpreter from the mlx-env venv
    to avoid dependency conflicts with the ByteTrack/Hailo environment.
  - Vision runs using your bytetrack_venv activation + script + working directory.
  - Vision child Python is launched with -u so stdout is unbuffered and detection lines
    are visible to the orchestrator in real time.

Run:
  python3 ~/two_step_gated_orchestrator.py

Stop:
  Ctrl+C
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
import select
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List


# =============================================================================
# CONFIG
# =============================================================================

HOME = Path.home()

# --- STEP 1: RD-03E Radar ---
RADAR_SCRIPT = HOME / "rd03e_live_fast.py"
RADAR_DEVICE = "/dev/ttyUSB0"
RADAR_ARGS = ["--low-latency", "--print-hz", "10"]
RADAR_STARTUP_GRACE_SEC = 0.25

# Cooldown after vision shuts off before allowing radar to retrigger
POST_VISION_RADAR_COOLDOWN_SEC = 2.5

# --- STEP 2: Thermal Verify (MLX90640) ---
THERMAL_PYTHON = HOME / "mlx-env" / "bin" / "python3"
THERMAL_TIMEOUT_SEC = 8.0

# Middle ground: slightly stricter than relaxed, much less strict than original
THERMAL_THRESHOLD_C = 31.9
THERMAL_MIN_HOT_PIXELS = 5
THERMAL_CONSECUTIVE_HITS = 2

THERMAL_MAX_FRAME_RATE_SLEEP = 0.02
THERMAL_VERIFY_SCRIPT = HOME / "thermal_verify_mlx90640_headless.py"

# --- STEP 3: Vision (Hailo YOLOv8 + ByteTrack) ---
VENV_ACTIVATE = HOME / "bytetrack_env" / "bytetrack_venv" / "bin" / "activate"
BYTETRACK_SCRIPT = HOME / "bytetrack_env" / "ByteTrack" / "bytetrack_detect.py"
BYTETRACK_CWD = HOME / "bytetrack_env" / "ByteTrack"

# Vision stays ON while detections continue.
# If no detection lines are seen for this many seconds, vision turns OFF.
VISION_NO_DETECTION_TIMEOUT_SEC = 8.0

# Small startup grace so camera/model can initialize before idleness is judged.
VISION_STARTUP_GRACE_SEC = 3.0

# Optional safety cap. Set to 0 to disable.
VISION_MAX_SESSION_SEC = 0.0

VISION_STOP_SIGNAL = signal.SIGINT

# Logging verbosity
PRINT_RADAR_LINES = True
PRINT_THERMAL_LINES = True
PRINT_VISION_LINES = True


# =============================================================================
# INTERNALS / REGEX
# =============================================================================

PRESENT_RE = re.compile(r"\bpresent=(True|False)\b")
DETECTION_RE = re.compile(
    r"Object:\s+([a-zA-Z\s]+)\[\d+\]\s+\((\d+\.\d+)\)\s+@\s+(\d+),(\d+)\s+(\d+)x(\d+)"
)
FRAME_RE = re.compile(r"Viewfinder frame (\d+)")


@dataclass
class ProcHandle:
    name: str
    popen: subprocess.Popen
    pgid: int


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def ensure_file_exists(path: Path, content: str, mode: int = 0o755) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        os.chmod(path, mode)


def start_process_group(name: str, cmd: List[str], cwd: Optional[Path] = None) -> ProcHandle:
    log(f"START {name}: {cmd}")
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    pgid = os.getpgid(p.pid)
    return ProcHandle(name=name, popen=p, pgid=pgid)


def stop_process_group(handle: ProcHandle, sig: int, grace: float = 2.0) -> None:
    p = handle.popen
    if p.poll() is not None:
        return

    log(f"STOP {handle.name}: send {signal.Signals(sig).name} to pgid {handle.pgid}")
    try:
        os.killpg(handle.pgid, sig)
    except ProcessLookupError:
        return

    t0 = time.time()
    while time.time() - t0 < grace:
        if p.poll() is not None:
            log(f"STOP {handle.name}: exited with code {p.returncode}")
            return
        time.sleep(0.05)

    log(f"STOP {handle.name}: still running, force kill pgid {handle.pgid}")
    try:
        os.killpg(handle.pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def read_available_lines(handle: ProcHandle, prefix: str = "", max_lines: int = 100) -> List[str]:
    """
    Nonblocking stdout drain using select.select().
    Reads all currently available complete lines, up to max_lines.
    """
    lines: List[str] = []
    p = handle.popen
    if p.stdout is None:
        return lines

    try:
        for _ in range(max_lines):
            ready, _, _ = select.select([p.stdout], [], [], 0)
            if not ready:
                break

            line = p.stdout.readline()
            if not line:
                break

            line = line.rstrip("\n")
            lines.append(line)
            if prefix:
                log(f"{prefix}{line}")
    except Exception:
        pass

    return lines


def parse_present(line: str) -> Optional[bool]:
    m = PRESENT_RE.search(line)
    if not m:
        return None
    return True if m.group(1) == "True" else False


def line_has_detection(line: str) -> bool:
    return DETECTION_RE.search(line) is not None


THERMAL_HELPER_CODE = r'''#!/usr/bin/env python3
"""
Headless MLX90640 Thermal Verification

Exit codes:
  0 => thermal presence confirmed
  1 => no thermal presence within timeout window
"""

import argparse
import time
import numpy as np
import seeed_mlx90640


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout-sec", type=float, default=8.0)
    ap.add_argument("--threshold-c", type=float, default=31.9)
    ap.add_argument("--min-hot-pixels", type=int, default=5)
    ap.add_argument("--consecutive-hits", type=int, default=2)
    ap.add_argument("--sleep-sec", type=float, default=0.02)
    ap.add_argument("--refresh", type=str, default="REFRESH_16_HZ")
    args = ap.parse_args()

    refresh_map = {
        "REFRESH_0_5_HZ": seeed_mlx90640.RefreshRate.REFRESH_0_5_HZ,
        "REFRESH_1_HZ": seeed_mlx90640.RefreshRate.REFRESH_1_HZ,
        "REFRESH_2_HZ": seeed_mlx90640.RefreshRate.REFRESH_2_HZ,
        "REFRESH_4_HZ": seeed_mlx90640.RefreshRate.REFRESH_4_HZ,
        "REFRESH_8_HZ": seeed_mlx90640.RefreshRate.REFRESH_8_HZ,
        "REFRESH_16_HZ": seeed_mlx90640.RefreshRate.REFRESH_16_HZ,
        "REFRESH_32_HZ": seeed_mlx90640.RefreshRate.REFRESH_32_HZ,
        "REFRESH_64_HZ": seeed_mlx90640.RefreshRate.REFRESH_64_HZ,
    }
    refresh_rate = refresh_map.get(args.refresh, seeed_mlx90640.RefreshRate.REFRESH_16_HZ)

    try:
        sensor = seeed_mlx90640.grove_mxl90640()
        sensor.refresh_rate = refresh_rate
    except Exception as e:
        print(f"[THERMAL] ERROR: failed to init MLX90640: {e}", flush=True)
        return 1

    raw = [0.0] * 768
    hits = 0
    t0 = time.time()

    try:
        sensor.getFrame(raw)
    except Exception:
        pass

    while (time.time() - t0) < args.timeout_sec:
        try:
            sensor.getFrame(raw)
        except Exception as e:
            print(f"[THERMAL] read error: {e}", flush=True)
            time.sleep(args.sleep_sec)
            continue

        arr = np.array(raw, dtype=np.float32)
        if arr.size < 768:
            time.sleep(args.sleep_sec)
            continue

        if float(np.max(arr)) == 0.0 or float(np.min(arr)) == 500.0:
            time.sleep(args.sleep_sec)
            continue

        hot = int(np.sum(arr > args.threshold_c))
        tmax = float(np.max(arr))
        tmean = float(np.mean(arr))

        if tmax > 100.0 or tmean > 80.0:
            print(
                f"[THERMAL] ignoring invalid frame: tmax={tmax:.1f}C mean={tmean:.1f}C",
                flush=True
            )
            time.sleep(args.sleep_sec)
            continue

        print(
            f"[THERMAL] hot_pixels={hot} min_req={args.min_hot_pixels} "
            f"tmax={tmax:.1f}C mean={tmean:.1f}C",
            flush=True
        )

        if hot >= args.min_hot_pixels:
            hits += 1
        else:
            hits = 0

        if hits >= args.consecutive_hits:
            print("[THERMAL] VERIFIED: thermal presence confirmed.", flush=True)
            return 0

        time.sleep(args.sleep_sec)

    print("[THERMAL] NOT VERIFIED: timeout reached with no confirmed presence.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def run_thermal_verification() -> bool:
    ensure_file_exists(THERMAL_VERIFY_SCRIPT, THERMAL_HELPER_CODE)

    if not THERMAL_PYTHON.exists():
        log(f"THERMAL ERROR: expected venv python not found at {THERMAL_PYTHON}")
        log("THERMAL verification cannot run without mlx-env interpreter.")
        return False

    cmd = [
        str(THERMAL_PYTHON),
        str(THERMAL_VERIFY_SCRIPT),
        "--timeout-sec", str(THERMAL_TIMEOUT_SEC),
        "--threshold-c", str(THERMAL_THRESHOLD_C),
        "--min-hot-pixels", str(THERMAL_MIN_HOT_PIXELS),
        "--consecutive-hits", str(THERMAL_CONSECUTIVE_HITS),
        "--sleep-sec", str(THERMAL_MAX_FRAME_RATE_SLEEP),
        "--refresh", "REFRESH_16_HZ",
    ]

    h = start_process_group("THERMAL_VERIFY", cmd, cwd=HOME)

    verified = False
    t0 = time.time()

    while True:
        if h.popen.poll() is not None:
            rc = h.popen.returncode
            verified = (rc == 0)
            log(f"THERMAL_VERIFY exited rc={rc} verified={verified}")
            break

        if PRINT_THERMAL_LINES:
            read_available_lines(h, "THERMAL> ")

        if time.time() - t0 > (THERMAL_TIMEOUT_SEC + 3.0):
            log("THERMAL_VERIFY exceeded orchestrator timeout -> stopping")
            stop_process_group(h, signal.SIGINT)
            verified = False
            break

        time.sleep(0.01)

    if PRINT_THERMAL_LINES:
        for _ in range(100):
            lines = read_available_lines(h, "THERMAL> ")
            if not lines:
                break

    return verified


def start_vision() -> ProcHandle:
    if not VENV_ACTIVATE.exists():
        raise FileNotFoundError(f"VENV_ACTIVATE not found: {VENV_ACTIVATE}")
    if not BYTETRACK_SCRIPT.exists():
        raise FileNotFoundError(f"BYTETRACK_SCRIPT not found: {BYTETRACK_SCRIPT}")
    if not BYTETRACK_CWD.exists():
        raise FileNotFoundError(f"BYTETRACK_CWD not found: {BYTETRACK_CWD}")

    shell_cmd = (
        f"source {shlex.quote(str(VENV_ACTIVATE))} && "
        f"cd {shlex.quote(str(BYTETRACK_CWD))} && "
        f"export PYTHONUNBUFFERED=1 && "
        f"exec python3 -u {shlex.quote(str(BYTETRACK_SCRIPT))}"
    )
    cmd = ["bash", "-lc", shell_cmd]
    return start_process_group("VISION/BYTETRACK", cmd, cwd=BYTETRACK_CWD)


def run_vision_session() -> None:
    """
    Vision stays on while detections continue.
    It stops only when no detections have been seen for VISION_NO_DETECTION_TIMEOUT_SEC.
    """
    h = start_vision()
    log("VISION state: ON")

    session_start_t = time.time()
    last_detection_t: Optional[float] = None
    last_frame_t: Optional[float] = None

    try:
        while True:
            if h.popen.poll() is not None:
                log(f"VISION exited early code={h.popen.returncode}")
                break

            lines = read_available_lines(h, "VISION> " if PRINT_VISION_LINES else "")

            now_t = time.time()

            for line in lines:
                if FRAME_RE.search(line):
                    last_frame_t = now_t

                if line_has_detection(line):
                    last_detection_t = now_t
                    log("VISION activity: detection seen -> keeping camera ON")

            if VISION_MAX_SESSION_SEC > 0 and (now_t - session_start_t) >= VISION_MAX_SESSION_SEC:
                log(f"VISION max session reached ({VISION_MAX_SESSION_SEC}s) -> stopping")
                stop_process_group(h, VISION_STOP_SIGNAL)
                break

            if (now_t - session_start_t) < VISION_STARTUP_GRACE_SEC:
                time.sleep(0.02)
                continue

            if last_detection_t is not None:
                idle_sec = now_t - last_detection_t
                if idle_sec >= VISION_NO_DETECTION_TIMEOUT_SEC:
                    log(
                        f"VISION idle timeout reached: no detections for "
                        f"{idle_sec:.1f}s -> stopping and returning to STEP 1"
                    )
                    stop_process_group(h, VISION_STOP_SIGNAL)
                    break
            else:
                since_start = now_t - session_start_t
                if since_start >= (VISION_STARTUP_GRACE_SEC + VISION_NO_DETECTION_TIMEOUT_SEC):
                    log(
                        f"VISION saw no detections after startup window "
                        f"({since_start:.1f}s total) -> stopping and returning to STEP 1"
                    )
                    stop_process_group(h, VISION_STOP_SIGNAL)
                    break

            time.sleep(0.02)

    finally:
        stop_process_group(h, VISION_STOP_SIGNAL)
        log("VISION state: OFF")


def start_radar() -> ProcHandle:
    if not RADAR_SCRIPT.exists():
        raise FileNotFoundError(f"RADAR_SCRIPT not found: {RADAR_SCRIPT}")

    cmd = ["python3", "-u", str(RADAR_SCRIPT), RADAR_DEVICE] + RADAR_ARGS
    return start_process_group("RADAR", cmd, cwd=HOME)


def main() -> int:
    radar_handle: Optional[ProcHandle] = None

    def cleanup(*_args) -> None:
        nonlocal radar_handle
        log("Received shutdown signal -> cleaning up")
        if radar_handle is not None:
            stop_process_group(radar_handle, signal.SIGTERM)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    log("Two-step orchestrator starting (Radar -> Thermal -> Vision).")

    while True:
        # -------------------------------------------------
        # STEP 1: RADAR LOOP
        # -------------------------------------------------
        radar_handle = start_radar()
        time.sleep(RADAR_STARTUP_GRACE_SEC)

        radar_present = False

        while True:
            if radar_handle.popen.poll() is not None:
                log(f"RADAR exited unexpectedly code={radar_handle.popen.returncode}. Restarting radar...")
                radar_handle = None
                break

            lines = read_available_lines(radar_handle, "RADAR> " if PRINT_RADAR_LINES else "")
            for line in lines:
                p = parse_present(line)
                if p is True:
                    radar_present = True

            if radar_present:
                log("STEP 1 triggered: radar present=True -> moving to STEP 2 (thermal verify)")
                break

            time.sleep(0.01)

        if radar_handle is not None:
            stop_process_group(radar_handle, signal.SIGTERM)
            radar_handle = None

        if not radar_present:
            continue

        # -------------------------------------------------
        # STEP 2: THERMAL VERIFICATION
        # -------------------------------------------------
        verified = run_thermal_verification()
        if not verified:
            log("STEP 2 failed: thermal NOT verified -> returning to STEP 1 (radar)")
            continue

        log("STEP 2 passed: thermal verified -> moving to STEP 3 (vision)")

        # -------------------------------------------------
        # STEP 3: VISION SESSION
        # -------------------------------------------------
        try:
            run_vision_session()
        except Exception as e:
            log(f"VISION error: {e}")

        log(f"Post-vision cooldown: ignoring radar triggers for {POST_VISION_RADAR_COOLDOWN_SEC:.1f}s")
        time.sleep(POST_VISION_RADAR_COOLDOWN_SEC)

        log("Cycle complete -> returning to STEP 1 (radar)")


if __name__ == "__main__":
    raise SystemExit(main())
~~~


## Troubleshooting

### Radar never triggers
Check:
- `/dev/ttyUSB0`
- radar wiring
- `rd03e_live_fast.py` works alone
- radar prints `present=True/False`

### Thermal always fails
Check:
- MLX90640 wiring and I2C access
- `mlx-env` exists and works
- threshold too strict
- hot pixel minimum too strict
- consecutive hits too strict
- invalid frame behavior

### Vision never starts
Check:
- thermal verification is actually passing
- ByteTrack environment path is correct
- Hailo version is still locked to 4.20
- script and working directory are correct

### Vision stops too quickly
Check:
- `VISION_STARTUP_GRACE_SEC`
- `VISION_NO_DETECTION_TIMEOUT_SEC`
- detection lines are visible in stdout

### Vision never stops
Check:
- detection parser is matching correctly
- ByteTrack output is unbuffered
- detections are not still being emitted continuously


## Summary

This pipeline uses staged activation:

- **Radar** stays on continuously
- **Thermal verification** acts as a second filter
- **Vision** only turns on when both earlier stages justify it

That makes it more selective than radar-only gating and more suitable for experiments focused on reducing unnecessary camera use, reducing energy use, and evaluating multimodal edge AI behavior.

