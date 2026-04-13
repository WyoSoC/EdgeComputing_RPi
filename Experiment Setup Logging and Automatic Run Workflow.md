# Experiment Setup, Logging Workflow, and Full Source Files  
## Energy / Activity Evaluation for Mode 1, Mode 2, and Mode 3

This document explains the full experiment workflow used to generate the final data, summaries, and graphs for the Raspberry Pi multimodal sensing project. The purpose of this workflow was to run controlled timed experiments for the three sensing modes, collect system-level telemetry and process logs, and then summarize the run into structured outputs that could later be aligned with Emporia smart plug power data.

The experiment framework was designed so that each run would:
- execute one selected sensing mode for a fixed duration
- save all outputs into a dedicated run folder
- collect system and process telemetry every second
- capture full stdout from the selected mode
- save metadata about the run
- generate an `events.csv` file and `summary.json` file after completion
- support later manual alignment with Emporia exported CSV data for power analysis


## Experiment Goal

The experiment setup was built to compare three sensing modes:

### Mode 1 — Always-On Vision
- camera + Hailo YOLOv8 + ByteTrack run continuously
- no gating logic
- acts as the baseline

### Mode 2 — Radar-Gated Vision
- RD-03E radar stays on continuously
- radar activates vision only when presence is detected
- vision turns off after absence timeout

### Mode 3 — Two-Step Radar + Thermal + Vision
- RD-03E radar stays on continuously
- radar triggers thermal verification first
- thermal must pass before vision starts
- vision turns off after no-detection timeout

Each mode was tested under fixed scenarios such as:
- empty
- presence

Each run was timed and logged so the final results could be compared in terms of:
- detections
- activations
- vision-on time
- radar trigger behavior
- thermal pass/fail behavior
- CPU and system telemetry
- process-level behavior
- measured power and energy


## Required Supporting Documentation

This experiment workflow depends on the supporting mode/system setup documentation:

1. **RD-03E Radar Setup**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/AI-Thinker%20RD-03E%20Radar.md

2. **MLX90640 Thermal Sensor Setup**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/MLX90640%20IRCamera-OpenCV-PyQt5-ObjectDetection.md

3. **YOLOv8 + ByteTrack Object Detection Setup**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/YOLOv8_ByteTrack_ObjectDetection_Guide.md

4. **Radar-Gated Vision Setup:**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/Radar-Gated%20ByteTrack%20Orchestrator%20(RD-03E%20%2B%20RPi%205%20%2B%20Hailo%20YOLOv8).md

5. **Radar-Thermal-Gated Vision Setup:**
   - https://github.com/WyoSoC/EdgeComputing_RPi/blob/main/Two%20Step%20Gated%20Radar%20-Thermal-Vision%20Pipeline.md


## Required Software / Tools

The experiment wrapper relies on several Linux utilities and Python scripts.

### Linux command-line tools
Make sure these are installed:
- `pidstat`
- `mpstat`
- `iostat`
- `timeout`
- `tee`
- `ps`
- `bash`
- `vcgencmd`
- `glances`

Typical install commands on Raspberry Pi OS / Debian-based systems:

~~~sh
sudo apt update
sudo apt install -y sysstat procps coreutils util-linux python3-pip
python3 -m pip install glances
~~~

### Python / environment assumptions
You should already have:
- a working ByteTrack / Hailo environment
- a working `mlx-env` environment for MLX90640 if Mode 3 is used
- a working radar parser script
- a working RD-03E connection
- working mode scripts
- a working parser script

### Hailo version lock
All Hailo components must remain aligned to:

**`4.20`**

If the Hailo versions drift, the vision pipeline may fail.


## Files Used in the Experiment Workflow

The core experiment automation depended on:
- `run_experiment.sh`
- `parse_experiment_log.py`
- `mode1_always_on.py`
- `mode2_radar_gated.py`
- `mode3_two_step.py`

The wrapper script starts the correct mode, logs everything, and then calls the parser script to produce:
- `events.csv`
- `summary.json`


## High-Level Experiment Flow

For each run:

1. The user starts the wrapper script with:
   - mode
   - scenario
   - run id
   - duration in seconds

2. A new run directory is created

3. Metadata is saved

4. Background logging processes are started:
   - `pidstat`
   - `mpstat`
   - `iostat`
   - Raspberry Pi telemetry loop using `vcgencmd`
   - process snapshots using `ps`
   - `glances` CSV export

5. The selected mode script is launched with a hard timeout

6. Full mode stdout is written to:
   - `mode_stdout.log`

7. When the run ends:
   - logger processes are stopped
   - exit code is saved
   - end time is saved

8. The parser script runs and creates:
   - `events.csv`
   - `summary.json`

9. The user manually exports the matching Emporia CSV and places it into the run folder for later power analysis


## Directory Structure

A typical run folder looks like this:

~~~text
/home/<user>/energy_tests/
  mode1_empty_run2_20260405_235801/
    metadata.env
    notes.txt
    run_command.sh
    mode_stdout.log
    run_exit_code.txt
    end_time.txt
    pidstat.log
    mpstat.log
    iostat.log
    pi_telemetry.csv
    process_snapshots.log
    glances.csv
    events.csv
    summary.json
    detections/
    emporia_export.csv
~~~

Each run is self-contained so it can be analyzed independently.


## Metadata Saved Per Run

The wrapper saves:
- mode
- scenario
- run id
- start timestamp
- planned duration
- run directory
- hostname
- kernel info

This information is stored in:
- `metadata.env`
- `notes.txt`

That makes each run easier to track and reproduce.


## Logged Outputs Collected Per Run

### `pidstat.log`
Per-process CPU, memory, and I/O activity over time.

### `mpstat.log`
CPU activity by core.

### `iostat.log`
Disk and device activity.

### `pi_telemetry.csv`
Per-second Raspberry Pi telemetry from `vcgencmd`, including:
- temperature
- throttling state
- arm clock speed

### `process_snapshots.log`
Top CPU processes over time using `ps`.

### `glances.csv`
Additional system-level monitoring data.

### `mode_stdout.log`
The full stdout of the selected mode script. This is the main input later parsed into events and summary metrics.

### `events.csv`
Structured event timeline extracted from mode stdout.

### `summary.json`
Compact run summary such as:
- detections seen
- vision activations
- total vision-on time
- radar triggers
- thermal passes / fails
- run duration


## Experiment Scenarios

The same wrapper was used across scenarios such as:
- `empty`
- `presence`

The scenario name is passed into the run and stored in the metadata so each run folder remains clearly labeled.


## Typical Run Commands

### Mode 1, empty, run 2, 300 seconds
~~~sh
./run_experiment.sh mode1 empty 2 300
~~~

### Mode 2, presence, run 2, 300 seconds
~~~sh
./run_experiment.sh mode2 presence 2 300
~~~

### Mode 3, empty, run 2, 300 seconds
~~~sh
./run_experiment.sh mode3 empty 2 300
~~~

### Mode 3, presence, run 2, 300 seconds
~~~sh
./run_experiment.sh mode3 presence 2 300
~~~


## Full Script: `mode1_always_on.py`

~~~python
import re
import subprocess
import sys
import signal
import torch
import numpy as np
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import random

OUTPUT_DIR = Path(os.environ.get("DETECTIONS_DIR", str(Path.home() / "detections"))).expanduser().resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_saved = False


###############################################################################
# REGEX
###############################################################################
DETECTION_REGEX = r"Object: ([a-zA-Z\s]+)\[\d+\] \((\d+\.\d+)\) @ (\d+),(\d+) (\d+)x(\d+)"
FRAME_REGEX     = r"Viewfinder frame (\d+)"


###############################################################################
# UTILITY: IoU, REID, COLOR
###############################################################################
def box_iou_xyxy(boxA, boxB):
    """
    boxA, boxB => [x1,y1,x2,y2]
    Returns IoU for these two boxes.
    """
    xA1,yA1,xA2,yA2 = boxA
    xB1,yB1,xB2,yB2 = boxB
    inter_x1= max(xA1, xB1)
    inter_y1= max(yA1, yB1)
    inter_x2= min(xA2, xB2)
    inter_y2= min(yA2, yB2)
    iw= max(0, inter_x2- inter_x1)
    ih= max(0, inter_y2- inter_y1)
    inter_area= iw*ih

    areaA= (xA2- xA1)*(yA2- yA1)
    areaB= (xB2- xB1)*(yB2- yB1)
    denom= areaA+ areaB- inter_area + 1e-6
    return inter_area/ denom

def generate_random_reid(box_xy, frame_id):
    """
    Stub: produce a stable random 128D vector based on (x+y+frame_id).
    Real system => CNN embedding.
    """
    seed_val= int(box_xy[0] + box_xy[1] + frame_id)
    rng= np.random.RandomState(seed_val)
    return rng.rand(128).astype(np.float32)

def extract_color_hist(box_xy, frame_id):
    """
    Another stub: produce stable random 64D color hist based on (x+y+some factor).
    Real system => actual color histogram from image ROI.
    """
    seed_val= int(box_xy[0]*3 + box_xy[1]*7 + frame_id*11)
    rng= np.random.RandomState(seed_val)
    return rng.rand(64).astype(np.float32)

def color_hist_dist(a, b):
    """
    Simple L2 distance for color hist.
    """
    return np.linalg.norm(a- b)

###############################################################################
# TRACK STATES
###############################################################################
class TrackState:
    Tracked = 1
    Lost    = 2
    Removed = 3

###############################################################################
# DETECTION STUB
###############################################################################
class DetectionStub:
    def __init__(self, tlwh, score, reid_vec, color_hist):
        self.tlwh       = np.array(tlwh, dtype=np.float32)
        self.score      = float(score)
        self.reid_vec   = reid_vec
        self.color_hist = color_hist

###############################################################################
# STRACK
###############################################################################
class STrack:
    """
    Track object with multiple reliability metrics:
    - tlwh
    - reid_vec
    - color_hist
    - matched_frames => how many consecutive frames we matched
    - stable => once matched_frames≥ stable_frames_needed => we consider it stable
    - counted => whether we incremented object_counts
    - velocity check => prev_center
    """
    _count=0
    def __init__(self, tlwh, score, reid_vec, color_hist):
        self.tlwh        = np.array(tlwh, dtype=np.float32)
        self.score       = float(score)
        self.reid_vec    = reid_vec
        self.color_hist  = color_hist

        self.state       = TrackState.Tracked
        STrack._count   +=1
        self.track_id    = STrack._count

        self.end_frame      = 0
        self.matched_frames = 0
        self.counted        = False

        # velocity
        x,y,w,h= self.tlwh
        self.prev_center= ((x + (x+w))/2., (y + (y+h))/2.)

    def predict(self):
        pass

    def update(self, new_det, frame_id, alpha=0.9):
        """
        Overwrite tlwh, blend reid+color, increment matched_frames.
        """
        # compute velocity
        old_x, old_y, old_w, old_h= self.tlwh
        old_cx= (old_x + (old_x+ old_w))/2.
        old_cy= (old_y + (old_y+ old_h))/2.

        self.tlwh= new_det.tlwh
        self.score= new_det.score

        # blend reid
        self.reid_vec= alpha*self.reid_vec + (1-alpha)* new_det.reid_vec
        # blend color
        self.color_hist= alpha*self.color_hist + (1-alpha)* new_det.color_hist

        self.state= TrackState.Tracked
        self.end_frame= frame_id
        self.matched_frames+=1

        # new center
        x2,y2,w2,h2= self.tlwh
        new_cx= (x2+ (x2+ w2))/2.
        new_cy= (y2+ (y2+ h2))/2.
        self.prev_center= (new_cx, new_cy)

    def mark_lost(self):
        self.state= TrackState.Lost

    def mark_removed(self):
        self.state= TrackState.Removed

    @property
    def is_tracked(self):
        return (self.state== TrackState.Tracked)

    @property
    def center(self):
        x,y,w,h= self.tlwh
        return ( x+ w/2., y+ h/2.)

###############################################################################
# ADVANCED 2-PHASE SINGLE PASS TRACKER
###############################################################################
class AdvancedTracker:
    """
    A multi-phase approach:
     1) Partition detections => high_conf >= high_conf_thresh => can spawn new tracks
                                low_conf >= min_conf but < high_conf => match existing only
     2) Hungarian approach for cost => iou + reid + color
     3) stable_frames_needed => how many consecutive matches to confirm counting
     4) velocity check => skip huge jumps
     5) remove lost if older than max_time_lost
    """
    def __init__(
        self,
        min_conf=0.5,
        high_conf_thresh=0.7,
        match_thresh=0.6,
        stable_frames_needed=3,
        max_time_lost=30,
        iou_weight=0.5,
        reid_weight=0.3,
        color_weight=0.2,
        max_velocity=150.0
    ):
        self.min_conf           = min_conf
        self.high_conf_thresh   = high_conf_thresh
        self.match_thresh       = match_thresh
        self.stable_frames_needed= stable_frames_needed
        self.max_time_lost      = max_time_lost

        self.iou_weight   = iou_weight
        self.reid_weight  = reid_weight
        self.color_weight = color_weight
        self.max_velocity = max_velocity

        self.frame_id=0
        self.tracked= []
        self.lost= []
        self.removed= []

    def update(self, detections, frame_id):
        self.frame_id= frame_id

        # Partition detections => high_conf + low_conf
        high_conf_det= []
        low_conf_det= []
        for (x1,y1,x2,y2,sc,_lbl) in detections:
            if sc< self.min_conf:
                continue
            w= x2- x1
            h= y2- y1
            reidv= generate_random_reid((x1,y1), frame_id)
            colorv= extract_color_hist((x1,y1), frame_id)
            ds= DetectionStub([x1,y1,w,h], sc, reidv, colorv)
            if sc>= self.high_conf_thresh:
                high_conf_det.append(ds)
            else:
                low_conf_det.append(ds)

        for t in self.tracked:
            t.predict()
        for t in self.lost:
            t.predict()

        pool= self.tracked+ self.lost

        # Phase1 => match high_conf to existing
        cost_mat= self._build_cost(pool, high_conf_det)
        matchA, u_trkA, u_detA= self._hungarian_match(cost_mat)

        # update matched
        used_detA= set()
        for (i_t, i_d) in matchA:
            if not self._velocity_check(pool[i_t], high_conf_det[i_d]):
                continue
            pool[i_t].update(high_conf_det[i_d], frame_id)
            if pool[i_t] in self.lost:
                self.lost.remove(pool[i_t])
            if pool[i_t] not in self.tracked:
                self.tracked.append(pool[i_t])
            used_detA.add(i_d)

        # unmatched high => new track
        for i in u_detA:
            if i not in used_detA:
                # spawn new track
                dd= high_conf_det[i]
                trk= STrack(dd.tlwh, dd.score, dd.reid_vec, dd.color_hist)
                trk.end_frame= frame_id
                self.tracked.append(trk)

        unmatched_trkA= u_trkA
        # mark them => lost
        for it in unmatched_trkA:
            track= pool[it]
            if track.is_tracked:
                track.mark_lost()
                if track in self.tracked:
                    self.tracked.remove(track)
                if track not in self.lost:
                    self.lost.append(track)
            else:
                track.mark_lost()

        # Phase2 => match leftover tracks with low_conf (cannot spawn new tracks)
        # only existing tracks can be matched if iou≥ match_thresh
        # gather leftover tracks => the newly minted "lost" + the "tracked" that didn't match
        # effectively 'pool' again
        pool2= self.tracked+ self.lost

        cost_mat2= self._build_cost(pool2, low_conf_det)
        matchB, u_trkB, u_detB= self._hungarian_match(cost_mat2)
        used_detB= set()

        for (i_t, i_d) in matchB:
            # we do not spawn new track from low conf => so only update existing
            if not self._velocity_check(pool2[i_t], low_conf_det[i_d]):
                continue
            iou_val= self._compute_iou_val(pool2[i_t], low_conf_det[i_d])
            if iou_val>= self.match_thresh:
                pool2[i_t].update(low_conf_det[i_d], frame_id)
                if pool2[i_t] in self.lost:
                    self.lost.remove(pool2[i_t])
                if pool2[i_t] not in self.tracked:
                    self.tracked.append(pool2[i_t])
                used_detB.add(i_d)

        unmatched_trkB= u_trkB
        # those remain lost
        for idx in unmatched_trkB:
            track= pool2[idx]
            if track.is_tracked:
                track.mark_lost()
                if track in self.tracked:
                    self.tracked.remove(track)
                if track not in self.lost:
                    self.lost.append(track)
            else:
                track.mark_lost()

        # unmatched low_conf => do nothing => cannot spawn new track
        # remove lost older than max_time_lost
        keep_lost=[]
        for trk in self.lost:
            if (self.frame_id- trk.end_frame)> self.max_time_lost:
                trk.mark_removed()
                self.removed.append(trk)
            else:
                keep_lost.append(trk)
        self.lost= keep_lost

        # Return current tracked
        return [t for t in self.tracked if t.is_tracked]


    def _build_cost(self, track_pool, dets):
        """
        cost= iou_weight*(1- iou) + reid_weight*( l2dist( reid )/20 ) + color_weight*( l2dist( color )/30)
        """
        if not track_pool or not dets:
            return np.zeros((len(track_pool), len(dets)), dtype=np.float32)

        M,N= len(track_pool), len(dets)
        cost= np.zeros((M,N), dtype=np.float32)
        for i, trk in enumerate(track_pool):
            x_t,y_t,w_t,h_t= trk.tlwh
            boxA= [x_t,y_t, x_t+w_t, y_t+h_t]
            for j, dd in enumerate(dets):
                x_d,y_d,w_d,h_d= dd.tlwh
                boxB= [x_d, y_d, x_d+w_d, y_d+h_d]
                iou_val= box_iou_xyxy(boxA, boxB)
                rdist= np.linalg.norm(trk.reid_vec- dd.reid_vec)
                cdist= color_hist_dist(trk.color_hist, dd.color_hist)

                cost_val= self.iou_weight*(1- iou_val)\
                    + self.reid_weight*(rdist/ 20.)\
                    + self.color_weight*(cdist/30.)
                cost[i,j]= cost_val
        return cost

    def _hungarian_match(self, cost_mat):
        """
        We do a simple greedy approach based on ascending cost.
        For real Hungarian, you'd do from scipy: linear_sum_assignment(cost_mat).
        But to keep consistent with your prior approach, we do the "greedy ascending cost."
        """
        matched=[]
        used_rows= set()
        used_cols= set()

        M,N= cost_mat.shape
        pairs= [(r,c) for r in range(M) for c in range(N)]
        pairs.sort(key= lambda x: cost_mat[x[0], x[1]])

        for (r,c) in pairs:
            if r in used_rows or c in used_cols:
                continue
            iou_val= 1.- cost_mat[r,c]*(1./ self.iou_weight) if self.iou_weight>0 else 0.
            # check iou≥ match_thresh
            matched.append((r,c))
            used_rows.add(r)
            used_cols.add(c)

        unmatched_tracks= [r for r in range(M) if r not in used_rows]
        unmatched_dets  = [c for c in range(N) if c not in used_cols]
        return matched, unmatched_tracks, unmatched_dets

    def _velocity_check(self, track, new_det):
        """
        If the center jumps > self.max_velocity => discard match
        """
        old_x, old_y, old_w, old_h= track.tlwh
        old_cx= (old_x+ (old_x+ old_w))/2.
        old_cy= (old_y+ (old_y+ old_h))/2.

        x_d,y_d,w_d,h_d= new_det.tlwh
        new_cx= (x_d + (x_d+ w_d))/2.
        new_cy= (y_d + (y_d+ h_d))/2.

        dist= np.sqrt( (new_cx- old_cx)**2 + (new_cy- old_cy)**2 )
        if dist> self.max_velocity:
            return False
        return True

    def _compute_iou_val(self, track, det):
        x_t,y_t,w_t,h_t= track.tlwh
        boxA= [x_t,y_t,x_t+w_t,y_t+h_t]
        x_d,y_d,w_d,h_d= det.tlwh
        boxB= [x_d,y_d,x_d+w_d,y_d+h_d]
        return box_iou_xyxy(boxA, boxB)



###############################################################################
# POSITION MEMORY
###############################################################################
POSITION_IOU_THRESH= 0.8
REID_DIST_THRESH   = 0.2
COLOR_DIST_THRESH  = 0.2

position_memory= defaultdict(list)

def is_same_stationary_object(label, box_xyxy, reid_vec, color_hist):
    for (old_box, old_reid, old_color) in position_memory[label]:
        iou_val= box_iou_xyxy(box_xyxy, old_box)
        rd= np.linalg.norm(reid_vec- old_reid)
        cd= color_hist_dist(color_hist, old_color)
        if iou_val>= POSITION_IOU_THRESH and rd<=REID_DIST_THRESH and cd<=COLOR_DIST_THRESH:
            return True
    return False


###############################################################################
# LOGGING
###############################################################################
CSV_FILE= "object_tracking_log.xlsx"
object_counts= defaultdict(int)

tracker= AdvancedTracker(
    min_conf=0.5,
    high_conf_thresh=0.7,
    match_thresh=0.6,
    stable_frames_needed=3,
    max_time_lost=30,
    iou_weight=0.5,
    reid_weight=0.3,
    color_weight=0.2,
    max_velocity=150.0
)

frame_id= 0
detections_buffer=[]

def signal_handler(sig, frame):
    print("\n🛑 Interrupted by user. Saving results...")
    save_to_excel()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler) 

def run_rpicam_and_parse():
    global frame_id, detections_buffer
    print("🚀 Starting advanced multi-phase tracker with IoU+ReID+Color+Velocity+PositionMemory...\n")

    try:
        proc= subprocess.Popen(
            [
                "rpicam-hello",
                "-t","0",
                "--post-process-file","/usr/share/rpi-camera-assets/hailo_yolov8_inference.json",
                "--lores-width","640",
                "--lores-height","640",
                "--verbose"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in iter(proc.stdout.readline, ''):
            line_s= line.strip()
            print(line_s)

            match_frame= re.search(FRAME_REGEX, line_s)
            match_det  = re.search(DETECTION_REGEX, line_s)

            if match_frame:
                frame_id+=1
                if detections_buffer:
                    track_and_count(detections_buffer)
                    detections_buffer.clear()

            elif match_det:
                lbl= match_det.group(1).strip()
                sc= float(match_det.group(2))
                x,y,w,h= map(int, match_det.groups()[2:])
                if sc>=0.01:
                    x2= x+w
                    y2= y+h
                    detections_buffer.append([x,y,x2,y2, sc, lbl])

        if detections_buffer:
            frame_id+=1
            track_and_count(detections_buffer)
            detections_buffer.clear()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        proc.terminate()
        proc.wait()
        save_to_excel()

def track_and_count(dets):
    global frame_id, tracker, object_counts

    if not dets:
        return
    arr=[]
    lbl_map=[]
    for(x1,y1,x2,y2, sc, lb) in dets:
        lbl_map.append(lb)
        arr.append([x1,y1,x2,y2, sc, 0.0])

    torch_dets= torch.tensor(arr, dtype=torch.float32)
    print(f"\n[DEBUG] Frame {frame_id}, Detections => {torch_dets.tolist()}")

    active_tracks= tracker.update(torch_dets, frame_id)
    print(f"[DEBUG] {len(active_tracks)} track(s) after advanced multi-phase matching.\n")

    # stable + not counted => pick best label => see if position memory => increment if new
    for trk in active_tracks:
        if trk.matched_frames< tracker.stable_frames_needed:
            continue
        if trk.counted:
            continue

        # find best label by overlap
        x_t,y_t,w_t,h_t= trk.tlwh
        x2t= x_t+ w_t
        y2t= y_t+ h_t

        best_iou=0.
        best_lbl="unknown"
        for i, row in enumerate(arr):
            dx1,dy1,dx2,dy2, sc,d0= row
            inter_x1= max(x_t, dx1)
            inter_y1= max(y_t, dy1)
            inter_x2= min(x2t, dx2)
            inter_y2= min(y2t, dy2)
            iw= max(0, inter_x2- inter_x1)
            ih= max(0, inter_y2- inter_y1)
            inter_area= iw*ih
            track_area= w_t*h_t
            det_area= (dx2- dx1)*(dy2- dy1)
            iou_val= inter_area/(track_area+ det_area- inter_area+1e-6)
            if iou_val> best_iou:
                best_iou= iou_val
                best_lbl= lbl_map[i]

        # check position memory
        xyxy_box= [x_t,y_t,x2t,y2t]
        if is_same_stationary_object(best_lbl, xyxy_box, trk.reid_vec, trk.color_hist):
            continue

        # else increment
        trk.counted= True
        object_counts[best_lbl]+=1
        position_memory[best_lbl].append( (xyxy_box, trk.reid_vec, trk.color_hist) )



def save_to_excel():
    global _saved
    if _saved:
        return
    _saved = True

    if object_counts:
        df = pd.DataFrame(list(object_counts.items()), columns=["Object", "Count"])
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = (OUTPUT_DIR / f"object_tracking_log_{ts}.xlsx").resolve()
        df.to_excel(out_path, index=False)
        print(f"✅ Results saved to {out_path}", flush=True)
    else:
        print("⚠️ No detections to save.", flush=True)

if __name__=="__main__":
    run_rpicam_and_parse()
~~~


## Full Script: `mode2_radar_gated.py`

~~~python
#!/usr/bin/env python3
"""
mode2_radar_gated.py

Radar-gated vision pipeline:
- Always run RD-03E radar script and parse its stdout for presence.
- When presence detected -> start bytetrack_detect.py inside the bytetrack venv.
- When presence absent for OFF_DELAY_S -> stop bytetrack (and its rpicam child processes).
- IMPORTANT: vision stdout is continuously drained and re-logged so detection lines
  are visible to the experiment wrapper and parser.
"""

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

# =========================
# CONFIG
# =========================

RADAR_SCRIPT = str(Path.home() / "rd03e_live_fast.py")
RADAR_DEVICE = "/dev/ttyUSB0"
RADAR_ARGS = ["--low-latency", "--print-hz", "10"]

VENV_ACTIVATE = str(Path.home() / "bytetrack_env" / "bytetrack_venv" / "bin" / "activate")
BYTETRACK_SCRIPT = str(Path.home() / "bytetrack_env" / "ByteTrack" / "bytetrack_detect.py")
BYTETRACK_CWD = str(Path.home() / "bytetrack_env" / "ByteTrack")

PRESENCE_HOLD_S = 1.5
OFF_DELAY_S = 8.0
COOLDOWN_S = 2.0
RESTART_ON_CRASH = True

DEBUG_PRESENCE_PARSE = False
PRINT_VISION_LINES = True

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


PRESENT_FIELD_RX = re.compile(
    r"\bpresent\b\s*[:=]\s*(true|false|1|0|yes|no|on|off)\b",
    re.IGNORECASE
)

PRESENCE_REGEXES = [
    re.compile(r"\bpresence\b\s*[:=]\s*(1|true|on|yes)\b", re.IGNORECASE),
    re.compile(r"\boccupied\b\s*[:=]\s*(1|true|on|yes)\b", re.IGNORECASE),
    re.compile(r"\bmotion\b\s*[:=]\s*(1|true|on|yes)\b", re.IGNORECASE),
    re.compile(r"\btarget(s)?\b.*\b([1-9]\d*)\b", re.IGNORECASE),
    re.compile(r"\bdetect(ed|ion)?\b.*\b(1|true|on|yes)\b", re.IGNORECASE),
]

TRUE_WORDS = {"1", "true", "yes", "on"}
FALSE_WORDS = {"0", "false", "no", "off"}


def parse_presence(line: str) -> Optional[bool]:
    s = line.strip()
    if not s:
        return None

    m = PRESENT_FIELD_RX.search(s)
    if m:
        v = m.group(1).lower()
        if v in TRUE_WORDS:
            return True
        if v in FALSE_WORDS:
            return False

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
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid
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


def read_available_lines(ph: ProcHandle, prefix: str = "", max_lines: int = 100) -> List[str]:
    lines = []
    p = ph.popen
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


def build_bytetrack_command() -> list:
    bash_cmd = (
        f"source {shlex.quote(VENV_ACTIVATE)} && "
        f"cd {shlex.quote(BYTETRACK_CWD)} && "
        f"export PYTHONUNBUFFERED=1 && "
        f"exec python3 -u {shlex.quote(BYTETRACK_SCRIPT)}"
    )
    return ["bash", "-lc", bash_cmd]


def build_radar_command() -> list:
    return ["python3", "-u", RADAR_SCRIPT, RADAR_DEVICE, *RADAR_ARGS]


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

        if (now() - self.last_action_t) < COOLDOWN_S:
            return

        if self.vision and self.vision.popen.poll() is None:
            return

        self.last_action_t = now()
        self.vision = spawn_process(build_bytetrack_command(), "VISION/BYTETRACK", cwd=BYTETRACK_CWD)
        log("VISION state: ON")

    def stop_vision(self):
        if (now() - self.last_action_t) < COOLDOWN_S:
            return

        if not self.vision:
            return
        if self.vision.popen.poll() is not None:
            self.vision = None
            return

        self.last_action_t = now()

        stop_process_tree(self.vision, sig=signal.SIGINT, timeout_s=2.0)
        if self.vision.popen.poll() is None:
            stop_process_tree(self.vision, sig=signal.SIGTERM, timeout_s=2.0)

        self.vision = None
        log("VISION state: OFF")

    def _drain_vision_output(self):
        if self.vision and self.vision.popen.poll() is None and PRINT_VISION_LINES:
            read_available_lines(self.vision, "VISION> ")

    def radar_loop(self):
        while self.running:
            radar_lines = read_available_lines(self.radar, "RADAR> ")
            for line in radar_lines:
                p = parse_presence(line)
                if p is True:
                    self.last_presence_t = now()
                    if DEBUG_PRESENCE_PARSE:
                        log("PARSE presence=True -> updating last_presence_t")

            self._drain_vision_output()

            if self.radar.popen.poll() is not None:
                rc = self.radar.popen.returncode
                log(f"RADAR exited (code={rc}).")
                self.running = False
                break

            self._gate_step()
            time.sleep(0.01)

    def _presence_is_active(self) -> bool:
        return self.last_presence_t > 0 and (now() - self.last_presence_t) <= PRESENCE_HOLD_S

    def _presence_is_stale(self) -> bool:
        return self.last_presence_t <= 0 or (now() - self.last_presence_t) >= OFF_DELAY_S

    def _gate_step(self):
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
~~~


## Full Script: `mode3_two_step.py`

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


## Full Script: `parse_experiment_log.py`

~~~python
#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime
from email.utils import parsedate_to_datetime

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(.*)$")

DETECTION_RE = re.compile(
    r"Object:\s+([a-zA-Z\s]+)\[\d+\]\s+\((\d+\.\d+)\)\s+@\s+(\d+),(\d+)\s+(\d+)x(\d+)"
)
DETECTION_ACTIVITY_RE = re.compile(r"VISION activity:\s*detection seen", re.IGNORECASE)

VISION_ON_RE = re.compile(r"VISION state:\s*ON")
VISION_OFF_RE = re.compile(r"VISION state:\s*OFF")

THERMAL_START_RE = re.compile(r"START THERMAL_VERIFY")
THERMAL_PASS_RE = re.compile(r"STEP 2 passed: thermal verified")
THERMAL_FAIL_RE = re.compile(r"STEP 2 failed: thermal NOT verified")
THERMAL_VERIFIED_RE = re.compile(r"\[THERMAL\]\s+VERIFIED", re.IGNORECASE)
THERMAL_INVALID_RE = re.compile(r"\[THERMAL\]\s+ignoring invalid frame", re.IGNORECASE)

COOLDOWN_RE = re.compile(r"Post-vision cooldown")
IDLE_TIMEOUT_RE = re.compile(r"VISION idle timeout reached")

PRESENT_TRUE_RE = re.compile(r"\bpresent=True\b")
PRESENT_FALSE_RE = re.compile(r"\bpresent=False\b")


def parse_ts(line: str):
    m = TS_RE.match(line)
    if not m:
        return None, line.rstrip("\n")
    ts_str, msg = m.group(1), m.group(2)
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S"), msg


def load_metadata_env(path: Path):
    data = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def parse_end_time(path: Path):
    if not path.exists():
        return None
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()

    # Most robust for standard `date` output with timezone
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt.replace(tzinfo=None)
    except Exception:
        pass

    # Fallbacks
    for fmt in [
        "%a %b %d %H:%M:%S %Z %Y",
        "%a %b %d %H:%M:%S %Y",
    ]:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--metadata", required=False)
    ap.add_argument("--end-time", required=False)
    args = ap.parse_args()

    log_path = Path(args.input)
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    metadata = load_metadata_env(Path(args.metadata)) if args.metadata else {}
    meta_start_ts = None
    meta_duration_sec = None

    if "START_TS" in metadata:
        try:
            meta_start_ts = datetime.strptime(metadata["START_TS"], "%Y%m%d_%H%M%S")
        except ValueError:
            meta_start_ts = None

    if "DURATION_SEC" in metadata:
        try:
            meta_duration_sec = float(metadata["DURATION_SEC"])
        except ValueError:
            meta_duration_sec = None

    end_ts = parse_end_time(Path(args.end_time)) if args.end_time else None

    events = []
    detections = 0
    invalid_thermal_frames = 0
    vision_activations = 0
    thermal_passes = 0
    thermal_fails = 0
    radar_triggers = 0

    vision_on_ts = None
    total_vision_on_sec = 0.0
    first_ts = None
    last_ts = None

    # Radar edge detection
    radar_present_state = None  # None / False / True

    # Thermal cycle de-duplication
    thermal_cycle_open = False
    thermal_cycle_counted = False

    # If raw object lines are absent, use fallback detection activity count
    raw_detection_lines_seen = 0
    fallback_detection_activity_seen = 0

    def add_event(ts, event, details=""):
        events.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            "event": event,
            "details": details
        })

    for raw in lines:
        ts, msg = parse_ts(raw)
        if ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

        # -------------------------
        # Vision state
        # -------------------------
        if VISION_ON_RE.search(msg):
            vision_activations += 1
            if vision_on_ts is None:
                vision_on_ts = ts if ts else meta_start_ts
            add_event(ts, "VISION_ON")

        if VISION_OFF_RE.search(msg):
            add_event(ts, "VISION_OFF")
            if vision_on_ts and ts:
                total_vision_on_sec += max(0.0, (ts - vision_on_ts).total_seconds())
            vision_on_ts = None

        # -------------------------
        # Radar trigger counting via edge detection
        # -------------------------
        if msg.startswith("RADAR>"):
            if PRESENT_TRUE_RE.search(msg):
                add_event(ts, "RADAR_PRESENT_TRUE")
                if radar_present_state is not True:
                    radar_triggers += 1
                    add_event(ts, "RADAR_TRIGGER")
                radar_present_state = True

            elif PRESENT_FALSE_RE.search(msg):
                add_event(ts, "RADAR_PRESENT_FALSE")
                radar_present_state = False

        # -------------------------
        # Thermal verification
        # -------------------------
        if THERMAL_START_RE.search(msg):
            thermal_cycle_open = True
            thermal_cycle_counted = False
            add_event(ts, "THERMAL_START")

        if THERMAL_INVALID_RE.search(msg):
            invalid_thermal_frames += 1
            add_event(ts, "THERMAL_INVALID_FRAME")

        if thermal_cycle_open and not thermal_cycle_counted:
            if THERMAL_PASS_RE.search(msg) or THERMAL_VERIFIED_RE.search(msg):
                thermal_passes += 1
                thermal_cycle_counted = True
                thermal_cycle_open = False
                add_event(ts, "THERMAL_PASS")

            elif THERMAL_FAIL_RE.search(msg):
                thermal_fails += 1
                thermal_cycle_counted = True
                thermal_cycle_open = False
                add_event(ts, "THERMAL_FAIL")

        # -------------------------
        # Other state lines
        # -------------------------
        if COOLDOWN_RE.search(msg):
            add_event(ts, "POST_VISION_COOLDOWN")

        if IDLE_TIMEOUT_RE.search(msg):
            add_event(ts, "VISION_IDLE_TIMEOUT")

        # -------------------------
        # Detections
        # -------------------------
        dm = DETECTION_RE.search(msg)
        if dm:
            raw_detection_lines_seen += 1
            label = dm.group(1).strip()
            score = dm.group(2)
            add_event(ts, "DETECTION_SEEN", f"{label},{score}")

        elif DETECTION_ACTIVITY_RE.search(msg):
            fallback_detection_activity_seen += 1

    # Prefer raw object line count; otherwise use activity fallback
    if raw_detection_lines_seen > 0:
        detections = raw_detection_lines_seen
    else:
        detections = fallback_detection_activity_seen

    # Fallback timestamps
    if first_ts is None:
        first_ts = meta_start_ts
    if last_ts is None:
        last_ts = end_ts

    total_run_sec = 0.0
    if first_ts and last_ts:
        total_run_sec = max(0.0, (last_ts - first_ts).total_seconds())

    # Final fallback: use planned duration if end_time parsing failed
    if total_run_sec <= 0 and meta_duration_sec is not None:
        total_run_sec = meta_duration_sec

    # Mode 1 fallback: always-on vision
    if args.mode == "mode1":
        if total_run_sec > 0:
            vision_activations = 1
            total_vision_on_sec = total_run_sec
            vision_on_ts = None  # prevent any later double-count
            if not any(e["event"] == "VISION_ON" for e in events):
                add_event(first_ts, "VISION_ON")
            if not any(e["event"] == "VISION_OFF" for e in events):
                add_event(last_ts if last_ts else first_ts, "VISION_OFF")

    # If vision was still open in timestamped modes, close at run end
    if args.mode != "mode1" and vision_on_ts and total_run_sec > 0 and last_ts:
        total_vision_on_sec += max(0.0, (last_ts - vision_on_ts).total_seconds())

    vision_duty_cycle = None
    if total_run_sec > 0:
        vision_duty_cycle = total_vision_on_sec / total_run_sec

    summary = {
        "mode": args.mode,
        "log_file": str(log_path),
        "total_run_sec": total_run_sec,
        "vision_activations": vision_activations,
        "total_vision_on_sec": total_vision_on_sec,
        "vision_duty_cycle": vision_duty_cycle,
        "detections_seen": detections,
        "radar_triggers": radar_triggers,
        "thermal_passes": thermal_passes,
        "thermal_fails": thermal_fails,
        "thermal_invalid_frames": invalid_thermal_frames,
    }

    with open(args.events, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "event", "details"])
        w.writeheader()
        w.writerows(events)

    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
~~~


## Full Script: `run_experiment.sh`

~~~bash
#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_experiment.sh <mode> <scenario> <run_id> <duration_sec>
#
# Example:
#   ./run_experiment.sh mode3 person 1 300

MODE="${1:?mode required: mode1|mode2|mode3}"
SCENARIO="${2:?scenario required}"
RUN_ID="${3:?run id required}"
DURATION_SEC="${4:?duration required}"

TS="$(date +%Y%m%d_%H%M%S)"
BASE_DIR="$HOME/energy_tests"
RUN_DIR="$BASE_DIR/${MODE}_${SCENARIO}_run${RUN_ID}_${TS}"
mkdir -p "$RUN_DIR"

# -----------------------------
# Edit these paths if needed
# -----------------------------
MODE1_SCRIPT="$HOME/experiments/mode1_always_on.py"
MODE2_SCRIPT="$HOME/experiments/mode2_radar_gated.py"
MODE3_SCRIPT="$HOME/experiments/mode3_two_step.py"

# Optional: if mode1 needs the ByteTrack venv activated
BYTETRACK_VENV_ACTIVATE="$HOME/bytetrack_env/bytetrack_venv/bin/activate"
BYTETRACK_CWD="$HOME/bytetrack_env/ByteTrack"

# Detection output folder per run
DETECTIONS_DIR="$RUN_DIR/detections"
mkdir -p "$DETECTIONS_DIR"

# Optional Glances logging
ENABLE_GLANCES="1"

# -----------------------------
# Choose command by mode
# -----------------------------
case "$MODE" in
  mode1)
    RUN_CMD="source \"$BYTETRACK_VENV_ACTIVATE\" && cd \"$BYTETRACK_CWD\" && export DETECTIONS_DIR=\"$DETECTIONS_DIR\" && export PYTHONUNBUFFERED=1 && exec python3 -u \"$MODE1_SCRIPT\""
    ;;
  mode2)
    RUN_CMD="export PYTHONUNBUFFERED=1 && exec python3 -u \"$MODE2_SCRIPT\""
    ;;
  mode3)
    RUN_CMD="export PYTHONUNBUFFERED=1 && exec python3 -u \"$MODE3_SCRIPT\""
    ;;
  *)
    echo "Unknown mode: $MODE"
    exit 1
    ;;
esac

# -----------------------------
# Save metadata
# -----------------------------
cat > "$RUN_DIR/metadata.env" <<EOF
MODE=$MODE
SCENARIO=$SCENARIO
RUN_ID=$RUN_ID
START_TS=$TS
DURATION_SEC=$DURATION_SEC
RUN_DIR=$RUN_DIR
HOSTNAME=$(hostname)
KERNEL=$(uname -a)
EOF

cat > "$RUN_DIR/notes.txt" <<EOF
Mode: $MODE
Scenario: $SCENARIO
Run: $RUN_ID
Planned duration: $DURATION_SEC seconds
Start wall-clock: $(date)
Emporia export needed after run: YES
EOF

echo "Run directory: $RUN_DIR"

# -----------------------------
# Start loggers
# -----------------------------
pidstat -rudhl 1 > "$RUN_DIR/pidstat.log" &
echo $! > "$RUN_DIR/pidstat.pid"

mpstat -P ALL 1 > "$RUN_DIR/mpstat.log" &
echo $! > "$RUN_DIR/mpstat.pid"

iostat -xz 1 > "$RUN_DIR/iostat.log" &
echo $! > "$RUN_DIR/iostat.pid"

bash -c 'while true; do
  echo "$(date +%F_%T),$(vcgencmd measure_temp),$(vcgencmd get_throttled),$(vcgencmd measure_clock arm)"
  sleep 1
done' > "$RUN_DIR/pi_telemetry.csv" &
echo $! > "$RUN_DIR/pi_telemetry.pid"

bash -c 'while true; do
  echo "===== $(date +%F_%T) ====="
  ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -25
  sleep 1
done' > "$RUN_DIR/process_snapshots.log" &
echo $! > "$RUN_DIR/process_snapshots.pid"

# Glances automatic CSV logging
glances --export csv --export-csv-file "$RUN_DIR/glances.csv" --quiet > /dev/null 2>&1 &
echo $! > "$RUN_DIR/glances.pid"


cleanup() {
  for f in pidstat.pid mpstat.pid iostat.pid pi_telemetry.pid glances.pid process_snapshots.pid; do
    if [[ -f "$RUN_DIR/$f" ]]; then
      kill "$(cat "$RUN_DIR/$f")" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

# -----------------------------
# Run the mode with exact timeout
# -----------------------------
echo "Starting mode command..."
echo "$RUN_CMD" > "$RUN_DIR/run_command.sh"

set +e
timeout "${DURATION_SEC}s" bash -lc "$RUN_CMD" | tee "$RUN_DIR/mode_stdout.log"
RUN_EXIT=$?
set -e

echo "$RUN_EXIT" > "$RUN_DIR/run_exit_code.txt"
date > "$RUN_DIR/end_time.txt"

# -----------------------------
# Stop loggers
# -----------------------------
for f in pidstat.pid mpstat.pid iostat.pid pi_telemetry.pid glances.pid process_snapshots.pid; do
  if [[ -f "$RUN_DIR/$f" ]]; then
    kill "$(cat "$RUN_DIR/$f")" 2>/dev/null || true
  fi
done

sleep 1
rm -f "$RUN_DIR"/*.pid

# -----------------------------
# Parse stdout into events + summary
# -----------------------------
python3 "$HOME/experiments/parse_experiment_log.py" \
  --mode "$MODE" \
  --input "$RUN_DIR/mode_stdout.log" \
  --events "$RUN_DIR/events.csv" \
  --summary "$RUN_DIR/summary.json" \
  --metadata "$RUN_DIR/metadata.env" \
  --end-time "$RUN_DIR/end_time.txt"

echo "Run complete."
echo "Folder: $RUN_DIR"
echo "Next manual step: export Emporia CSV for the same time window into:"
echo "  $RUN_DIR/emporia_export.csv"
~~~


## How the Wrapper Selects Modes

The wrapper picks one mode script based on the first input argument:

- `mode1` -> `mode1_always_on.py`
- `mode2` -> `mode2_radar_gated.py`
- `mode3` -> `mode3_two_step.py`

### Important behavior
- Mode 1 activates the ByteTrack virtual environment before starting
- Mode 2 and Mode 3 run directly as Python scripts
- all mode stdout is captured into `mode_stdout.log`


## Parser Stage

After each run completes, the wrapper calls:

~~~sh
python3 "$HOME/experiments/parse_experiment_log.py" \
  --mode "$MODE" \
  --input "$RUN_DIR/mode_stdout.log" \
  --events "$RUN_DIR/events.csv" \
  --summary "$RUN_DIR/summary.json" \
  --metadata "$RUN_DIR/metadata.env" \
  --end-time "$RUN_DIR/end_time.txt"
~~~

The parser converts the raw mode stdout into:
- a structured event log
- a summary JSON

It looks for patterns such as:
- vision on/off
- radar trigger edges
- thermal start/pass/fail
- thermal invalid frames
- object detection lines
- detection activity lines
- idle timeouts
- cooldowns


## What the Parser Summary Captures

Typical summary information includes:
- mode
- total run time
- vision activations
- total vision-on seconds
- vision duty cycle
- detections seen
- radar triggers
- thermal passes
- thermal fails
- invalid thermal frames

This summary is what was later used to build comparison graphs.


## Emporia Power Data Workflow

The wrapper does not collect Emporia data directly.  
Instead, after the run finishes, the user manually exports the matching Emporia data and saves it into the run folder.

The wrapper prints this reminder at the end:

~~~text
Next manual step: export Emporia CSV for the same time window into:
  $RUN_DIR/emporia_export.csv
~~~

Find the Emporia Energy Smart Plug at:
https://shop.emporiaenergy.com/products/emporia-smart-plug-home-energy-monitoring-outlets?srsltid=AfmBOopGYXSByMI5kYGxdJe7jaOh5oaB6xMo52QP1Fxp0J-0Ix1KjSxi

### Manual Emporia step
After each run:
1. open the Emporia app or export interface
2. export the relevant CSV for the same time window
3. save it into the run folder as:
   - `emporia_export.csv`

This later allows alignment between:
- mode start/end timestamps
- run metadata
- measured plug-level power data


## Experiment Folder Naming Convention

Run folders follow this pattern:

~~~text
$HOME/energy_tests/<mode>_<scenario>_run<id>_<timestamp>
~~~

Example:

~~~text
/home/abalhas1/energy_tests/mode3_presence_run2_20260405_234741
~~~

This makes runs easy to sort and compare.


## Example Run Sequence Used in Final Evaluation

The final structured experiments were run as 300-second runs across:
- Mode 1 empty
- Mode 2 empty
- Mode 3 empty
- Mode 1 presence
- Mode 2 presence
- Mode 3 presence

This allowed comparison of:
- always-on behavior
- radar-gated behavior
- radar + thermal gated behavior

under both low-activity and real-presence conditions.


## What Was Measured

The experiment framework was designed to support analysis of:

### Activity behavior
- detections logged
- number of activations
- radar trigger counts
- thermal passes/fails

### Vision duty behavior
- total vision-on time
- vision duty cycle
- no-detection shutdown behavior

### System telemetry
- CPU usage
- process activity
- clocks and temperature
- process-level snapshots

### Power / energy
- Emporia smart plug average power
- total run energy
- baseline-adjusted energy
- mode-to-mode comparison


## Recommended Environment Setup Before Running Experiments

Before starting experiments, verify:

### Radar
- `rd03e_live_fast.py` works alone
- radar prints `present=True/False`
- `/dev/ttyUSB0` is correct

### Vision
- ByteTrack/Hailo pipeline works alone
- Hailo stack is stable and version-matched
- `bytetrack_detect.py` runs correctly

### Thermal (for mode3)
- MLX90640 works over I2C
- `mlx-env` is working
- thermal helper runs correctly

### Logging tools
- `pidstat`, `mpstat`, `iostat`, and `glances` are installed
- `vcgencmd` works on the Pi

### Disk space
- make sure enough space is available for logs, CSVs, and mode stdout


## Troubleshooting

### Wrapper exits immediately
Check:
- arguments are correct
- mode is one of `mode1`, `mode2`, `mode3`
- paths in the wrapper are correct

### Run folder created but mode fails
Check:
- mode script path is correct
- Python environment is correct
- Hailo version is correct
- mode runs independently outside wrapper

### No detections folder created for mode1
Check:
- `DETECTIONS_DIR` is being passed correctly
- mode1 script respects the environment variable

### No parser output appears
Check:
- `parse_experiment_log.py` exists
- input file path is correct
- parser arguments are correct

### Glances CSV missing
Check:
- `glances` is installed
- command works from terminal
- permissions allow writing to run directory

### Emporia data missing later
Check:
- manual export was done after the run
- exported file matches the run time window
- file was copied into the correct folder


## Suggested Minimal Workflow

### 1. Verify each mode independently
~~~sh
python3 ~/experiments/mode1_always_on.py
python3 ~/experiments/mode2_radar_gated.py
python3 ~/experiments/mode3_two_step.py
~~~

### 2. Run controlled timed experiments
~~~sh
./run_experiment.sh mode1 empty 2 300
./run_experiment.sh mode2 empty 2 300
./run_experiment.sh mode3 empty 2 300
./run_experiment.sh mode1 presence 2 300
./run_experiment.sh mode2 presence 2 300
./run_experiment.sh mode3 presence 2 300
~~~

### 3. Export matching Emporia CSVs
- save them into each run folder

### 4. Use summaries + Emporia data to build comparison graphs


## Summary

The experiment framework automated the final evaluation of the multimodal sensing system. It allowed timed, repeatable runs across all three modes and scenarios, saved each run in a dedicated folder, captured telemetry and process logs, and prepared structured outputs for later graph creation and power analysis. This workflow was essential for turning the sensing system into something that could be measured and compared.

