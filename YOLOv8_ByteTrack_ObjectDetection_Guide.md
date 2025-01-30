# Enhanced General Object Detection & Count Logging with YOLOv8 & ByteTrack

This guide provides step-by-step instructions to set up and run an enhanced object detection and tracking system using YOLOv8 and ByteTrack on a Raspberry Pi.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation of ByteTrack Dependencies](#installation-of-bytetrack-dependencies)
   - [Create and Activate a Virtual Environment](#1-create-and-activate-a-virtual-environment)
   - [Upgrade Essential Tools](#2-upgrade-essential-tools)
   - [Install PyTorch and TorchVision](#3-install-pytorch-and-torchvision)
   - [Install Requirements from ByteTrack](#4-install-requirements-from-bytetrack)
   - [Install Additional Dependencies Individually](#5-install-additional-dependencies-individually)
   - [Install YOLOX in Development Mode](#6-install-yolox-in-development-mode)
   - [Verify Installations](#7-verify-installations)
3. [Fix the `matching.py` File Code](#-fix-the-matchingpy-file-code)
4. [Run the Code](#-run-the-following-code)

---
## Prerequisites

- **General Object Detection with YOLOv8**: Ensure you have installed all required dependencies for YOLOv8.
- **LibreOffice**: Required for handling Excel files.

## Installation of ByteTrack Dependencies

Follow these steps to install PyTorch, YOLOX, ONNX, and pycocotools on your Raspberry Pi.

### 1️⃣ Create and Activate a Virtual Environment

```bash
python3 -m venv ~/bytetrack_env/bytetrack_venv
source ~/bytetrack_env/bytetrack_venv/bin/activate
```
This ensures a clean, isolated Python environment.
###2️⃣ Upgrade Essential Tools

```bash
pip install --upgrade pip setuptools wheel
```
###3️⃣ Install PyTorch and TorchVision
```bash
pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cpu
```
Explicit versions are critical for Raspberry Pi compatibility.

### 4️⃣ Install Requirements from ByteTrack
Navigate to your ByteTrack directory:

```bash
cd ~/bytetrack_env/ByteTrack
pip install -r requirements.txt
```
This installs core dependencies listed in the project's requirements.txt.
If there are version issues (like numpy or incompatible dependencies), install specific versions manually.
```bash
pip install numpy==1.25.2 cython==3.0.11 setuptools==75.6.0
```
### 5️⃣ Install Additional Dependencies Individually
a. ONNX
```bash
pip install onnx==1.15.0
```

Why? ONNX compatibility with Raspberry Pi requires an older version.

b. ONNX Runtime
```bash
pip install onnxruntime==1.15.1
```
Why? Runtime required for ONNX operations.

pycocotools
```bash
CFLAGS="-Wno-cpp -Wno-unused-function -std=c99" pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
Why? Custom flags prevent compilation errors on ARM architecture.

If you face any errors with this part, proceed with the following steps:

Reinstall packaging:

```bash
pip uninstall -y packaging
pip install packaging==23.1
```
Clear pip Cache:

```bash
pip cache purge
rm -rf ~/.cache/pip
```
Reinstall wheel:

```bash
pip uninstall -y wheel
pip install wheel==0.41.2
```
Verify the Installation:

```bash
pip show wheel
```
Reinstall pycocotools:

```bash
CFLAGS="-Wno-cpp -Wno-unused-function -std=c99" pip install --no-cache-dir 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```
d. thop
```bash
pip install thop
```
Why? Required by YOLOX for model profiling.

e. OpenCV (cv2)
```bash
pip install opencv-python
```
Why? Required for image and video processing.

f. Loguru
```bash
pip install loguru
```
g. Tabulate
```bash
pip install tabulate
```
h. Scipy
```bash
pip install scipy
```
i. Lap
```bash
pip install lap
```
j. Pandas
```bash
pip install pandas
```
k. Openpyxl
```bash
pip install openpyxl
```
l. Torchvision
```bash
pip install torchvision
```


###6️⃣ Install YOLOX in Development Mode
Navigate to the ByteTrack directory:

```bash
cd ~/bytetrack_env/ByteTrack
python3 setup.py develop
```
Why? Installs YOLOX in a way that allows code modifications without reinstallation.

If an error occurs, proceed with the following steps:

Reinstall sympy:

```bash
pip uninstall -y sympy
pip install sympy==1.12
```
Verify Installation:

```bash
python -c "import sympy; print(sympy.__version__)"
```
###7️⃣ Verify Installations
Run the following commands to ensure everything is correctly installed:

```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import yolox; print('YOLOX imported successfully')"
python3 -c "import onnx; print('ONNX version:', onnx.__version__)"
python3 -c "import pycocotools; print('Pycocotools installed successfully')"
```
Expected output:

```bash
PyTorch version: 2.0.1
YOLOX imported successfully
ONNX version: 1.15.0
Pycocotools installed successfully
```

###✅ Fix the matching.py File Code
Open the file using a text editor:

```bash
nano /home/………/bytetrack_env/ByteTrack/yolox/tracker/matching.py
```
Find the problematic line (line 61) and replace:

```python
ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
```
With this:

```python
ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
```
Alternatively, you can also use float (both will work):

```python
ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
```
Save the changes:

Press Ctrl+O to write the file.

Press Enter to confirm.

Press Ctrl+X to exit nano.



```python
import re
import subprocess
import sys
import signal
import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
import pandas as pd
import random

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
    if object_counts:
        df= pd.DataFrame(list(object_counts.items()), columns=["Object","Count"])
        ts= datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        f= f"object_tracking_log_{ts}.xlsx"
        df.to_excel(f, index=False)
        print(f"✅ Results saved to {f}")
    else:
        print("⚠️ No detections to save.")


if __name__=="__main__":
    run_rpicam_and_parse()
```
