# AI-Thinker RD-03E Radar  
## Low-Latency Presence Detection on Raspberry Pi 

This document describes the configuration used to operate the AI-Thinker RD-03E mmWave radar as an always-on, low-latency presence detector on a Raspberry Pi.

The radar is used as a lightweight **presence trigger** that activates heavier pipelines such as:
- Arducam + YOLO object detection
- MLX90640 thermal detection
- Or both, conditionally, to avoid system overload


## 1. System Goal

- Run RD-03E continuously with minimal CPU usage  
- Achieve near real-time detection (~15–30 ms typical)  
- Eliminate serial buffering, wiring noise, and OS scheduling delays  
- Trigger AI pipelines only when presence is detected  


## 2. Hardware Components

| Component | Purpose |
|----------|---------|
| AI-Thinker RD-03E | mmWave presence & distance sensing |
| SparkFun RedBoard (UNO-compatible) | USB–UART bridge |
| Raspberry Pi 5 | Main processing node |
| Short, high-quality jumper wires | Reduce noise |
| Micro-B USB cable | RedBoard → Raspberry Pi |


## 3. Pin Configuration & Wiring

### RD-03E → RedBoard Wiring

| RD-03E Pin | RedBoard Pin | Notes |
|-----------|--------------|------|
| VCC | 5V | Stable power |
| GND | GND | Common ground |
| OT1 | D1 (TX) | Required for stable output |
| RST | GND | **Critical: prevents freezes & lag** |


## 4. Critical Wiring Requirements

Reset pin **must** be tied to ground:

RD-03E RST → GND

OT1 **must not** be left floating:

RD-03E OT1 → RedBoard D1 (TX)

This configuration resolved:
- Random freezes
- Delayed serial output


## 5. Raspberry Pi Serial Setup

Verify the USB serial device:

```
  ls -l /dev/ttyUSB*
  ls -l /dev/serial/by-id/
```


## 6. Software Environment
```
sudo apt update  
sudo apt install -y python3 python3-pip  
pip3 install pyserial  
```



## 8. rd03e_live_fast.py
```
#!/usr/bin/env python3
import argparse
import time
from collections import deque
import serial

STATUS_MAP = {0x00: "no target", 0x01: "movement", 0x02: "micro-motion"}

def median_int(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return None
    m = n // 2
    return s[m] if n % 2 else (s[m-1] + s[m]) // 2

class Parser:
    HDR = b"\xAA\xAA"
    TAIL = b"\x55\x55"
    L = 7
    def __init__(self):
        self.b = bytearray()

    def feed(self, data):
        self.b.extend(data)

    def pop(self):
        out = []
        while True:
            i = self.b.find(self.HDR)
            if i < 0:
                if len(self.b) > 1:
                    self.b = self.b[-1:]
                break
            if i > 0:
                del self.b[:i]
            if len(self.b) < self.L:
                break
            frame = bytes(self.b[:self.L])
            if not (frame.startswith(self.HDR) and frame.endswith(self.TAIL)):
                del self.b[0:1]
                continue
            status = frame[2]
            dist_cm = frame[3] | (frame[4] << 8)
            out.append((status, dist_cm))
            del self.b[:self.L]
        return out

class Filter:
    def __init__(self, med_w=5, ema_a=0.35, max_cm=650, jump_cm=200):
        self.hist = deque(maxlen=med_w)
        self.ema = None
        self.last_good = None
        self.ema_a = ema_a
        self.max_cm = max_cm
        self.jump_cm = jump_cm

    def update(self, present_raw, dist_cm):
        glitch = False
        if present_raw:
            if dist_cm == 0 or dist_cm > self.max_cm:
                glitch = True
            if self.last_good is not None and abs(dist_cm - self.last_good) > self.jump_cm:
                glitch = True

        if present_raw and not glitch:
            self.last_good = dist_cm
            self.hist.append(dist_cm)
            med = median_int(self.hist)
            if med is None:
                return None, glitch
            if self.ema is None:
                self.ema = float(med)
            else:
                self.ema = self.ema_a * float(med) + (1 - self.ema_a) * self.ema
            return int(round(self.ema)), glitch

        if not present_raw:
            self.hist.clear()
            self.ema = None
            return 0, glitch

        return int(round(self.ema)) if self.ema else None, glitch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=256000)
    ap.add_argument("--print-hz", type=float, default=10.0)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=0)
    ser.reset_input_buffer()

    parser = Parser()
    filt = Filter()

    latest = None
    next_print = time.monotonic()
    period = 1.0 / args.print_hz

    while True:
        data = ser.read(ser.in_waiting or 256)
        if data:
            parser.feed(data)
            for status, dist in parser.pop():
                now = time.time()
                present = status != 0x00
                fcm, glitch = filt.update(present, dist)
                latest = (now, present, status, dist, fcm, glitch)
        else:
            time.sleep(0.001)

        t = time.monotonic()
        if latest and t >= next_print:
            next_print = t + period
            ts, present, status, raw, filt_cm, glitch = latest
            age = (time.time() - ts) * 1000
            print(
                f"\rpresent={present} status={STATUS_MAP.get(status)} "
                f"raw={raw}cm filt={filt_cm}cm age={age:.1f}ms",
                end="", flush=True
            )

if __name__ == "__main__":
    main()
```


## 9. Best Production Command
```
sudo nice -n -10 python3 -u rd03e_live_fast.py /dev/ttyUSB0 --print-hz 10
```


## 10. Final Architecture

RD-03E Radar (always on)  
↓  
Low-latency serial parser  
↓  
Presence detected?  
↓  
YES → Activate YOLO / Thermal object classification pipelines  
NO  → Remain idle
