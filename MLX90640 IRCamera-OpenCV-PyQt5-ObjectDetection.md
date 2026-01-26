# MLX90640 IR Thermal Sensor (Seeed-Python-IRCamera) — Raspberry Pi Setup Guide

This document includes:
- Baseline Seeed thermal viewer setup (`seeed_python_ircamera.py`)
- Performance optimizations (I2C speed + refresh rate)
- Optional enhanced OpenCV + PyQt5 + object-detection viewer (`seeed_python_ircamera2.py`) with full code


## 1) Hardware Requirements

### 1.1 Components
- Raspberry Pi (Raspberry Pi OS)
- MLX90640 Thermal IR Sensor (Seeed / Grove MLX90640)
- Jumper wires / Grove connector (depending on module)

### 1.2 Wiring (I2C)
Typical I2C wiring:
- VCC  -> 3.3V
- GND  -> GND
- SDA  -> SDA
- SCL  -> SCL


## 2) Step 1 — System Preparations

### 2.1 Update & Upgrade
Run:
```
    sudo apt update && sudo apt upgrade -y
```

### 2.2 Enable I2C
Enable I2C:
1) Run:
```
    sudo raspi-config
```

3) Navigate:
    Interfacing Options -> I2C -> Enable
4) Reboot:
```
    sudo reboot
```

### 2.3 Verify I2C Detection (Address 0x33)
After reboot:
```
    i2cdetect -y 1
```

Expected output includes 0x33 (example):
```
    0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- 33 -- -- -- -- -- -- -- -- -- -- -- -- 
```

If 0x33 does NOT appear:
- Re-check wiring (SDA/SCL)
- Confirm I2C is enabled
- Reboot and retry


## 3) Step 2 — Set Up a Virtual Environment

### 3.1 Install venv
    sudo apt install python3-venv -y

### 3.2 Create + Activate the venv
From your home directory:
```
    python3 -m venv mlx-env
    source mlx-env/bin/activate
```

## 4) Step 3 — Install Required Libraries

All `pip install` commands below are run INSIDE the venv (after `source mlx-env/bin/activate`).

### 4.1 Install Seeed grove.py (Pinned)
Install version 0.7:
```
    pip install Seeed-grove.py==0.7
```

Note:
- Version 0.7 fixes an SMBus error by correctly exposing the `msg` attribute.

### 4.2 Install MLX90640 Driver
    pip install seeed-python-mlx90640

### 4.3 Install Additional Dependencies
    pip install pyserial

### 4.4 Install PyQt5 (System-Wide) and Link Into venv
PyQt5 often fails via pip on Raspberry Pi. Install system package:
```
    sudo apt install python3-pyqt5 -y
```

Then link it into the venv.

IMPORTANT: The example path below assumes your venv is using Python 3.11. If your venv uses a different version, adjust the folder name accordingly.

Link command (example for Python 3.11):
```
    ln -s /usr/lib/python3/dist-packages/PyQt5 ~/mlx-env/lib/python3.11/site-packages/
```

To confirm the correct venv site-packages path, run:
```
    python3 -c "import site; print(site.getsitepackages())"
```

Test PyQt5:
```
    python3 -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 is working')"
```

## 5) Step 4 — Download the IR Camera Code

### 5.1 Clone Repository
    git clone https://github.com/gobuyun/seeed_ircamera.git
    cd seeed_ircamera


## 6) Step 5 — Run the Thermal Camera (Baseline Viewer)

Activate venv:
```
    source ~/mlx-env/bin/activate
```

Run the baseline script:
```
    cd ~/seeed_ircamera
    python3 seeed_python_ircamera.py
```

Expected:
- A thermal image window appears on the Raspberry Pi display.

Primary reference:
- https://www.instructables.com/MLX90640-IR-Thermal-Camera-Working-With-Raspberry-/



## 7) Step 6 — Optimize the Implementation (Recommended)

### 7.1 Increase I2C Baudrate (400 kHz)
Edit boot configuration:
```
    sudo nano /boot/firmware/config.txt
```

Add to the bottom:
```
    dtparam=i2c_arm_baudrate=400000
```

Save and reboot:
```
    sudo reboot
```

### 7.2 Fix Script Ownership (So You Can Edit)
If `seeed_python_ircamera.py` is owned by root:
```
    cd ~/seeed_ircamera
    sudo chown $USER:$USER seeed_python_ircamera.py
```

Now you can edit it:
```
    nano seeed_python_ircamera.py
```

### 7.3 Increase Refresh Rate (Reduce Lag)
Edit:
```
    nano seeed_python_ircamera.py
```

Find (example):
```
    self.dataHandle.refresh_rate = seeed_mlx90640.RefreshRate.REFRESH_0_5_HZ
```

Replace with:
```
    self.dataHandle.refresh_rate = seeed_mlx90640.RefreshRate.REFRESH_16_HZ
```

Save and exit.

### 7.4 Run the Optimized Script
    source ~/mlx-env/bin/activate
    cd ~/seeed_ircamera
    python3 seeed_python_ircamera.py

Result:
- Faster data transfer
- Improved frame rate
- Smoother thermal video


## 8) Optional — Enhanced OpenCV + PyQt5 Viewer with Object Detection

This enhanced script:
- Displays thermal feed with colormap and smoothing
- Draws center crosshair and shows center temperature
- Detects “hot regions” above a threshold
- Auto-saves cropped screenshots on detections (5-second cooldown)
- Logs detection events to `detections.csv`
- Supports manual screenshot and video recording

### 8.1 Install Extra Python Packages (Inside venv)
Activate venv:
```
    source ~/mlx-env/bin/activate
```

Install:
```
    pip install numpy opencv-python pyserial
```

PyQt5 should already be installed system-wide and linked into the venv (Step 4.4).

### 8.2 Save the Script
Create a new file:
- `seeed_python_ircamera2.py`

Example:
```
    cd ~
    nano seeed_python_ircamera2.py
```

Paste the full code below, save, exit.

### 8.3 Run the Enhanced Script
    source ~/mlx-env/bin/activate
    python3 ~/seeed_python_ircamera2.py

Output files created in the run directory:
- `detections.csv`
- `ultimate_thermal_viewer_final.log`
- `object_YYYYMMDD_HHMMSS_#.png` (auto-detection crops)
- `screenshot_YYYYMMDD_HHMMSS.png` (manual capture)
- `recording_YYYYMMDD_HHMMSS.avi` (if recording is used)

---

## 9) Enhanced Viewer Code (Copy/Paste Entire File)

Save this entire block as: `seeed_python_ircamera2.py`
```
#!/usr/bin/env python3
"""
Ultimate Enhanced Thermal Viewer for MLX90640 on Raspberry Pi (Final Optimized Build)

Features:
  • Real-time thermal imaging using MLX90640 via I2C (or Serial).
  • Frame processing: normalization, colormap application, and smoothing.
  • Object detection: automatically captures a screenshot of detected objects (5-sec cooldown).
  • Crosshair: A fixed crosshair is overlaid at the center of the thermal image, with the
    exact temperature reading at that point displayed.
  • Diagnostics: Displays min, max, mean, and center temperatures, FPS, and frame count.
  • Settings: Allows adjustment of the temperature detection threshold with a live numeric display,
      and selection of temperature unit (°C or °F). Settings persist across sessions.
  • Sensor Calibration: Simulated calibration to compute an offset applied to sensor data.
  • Screenshot capture and video recording.
  • Minimal menu (File→Exit) and status bar.
  • CSV Logging: Logs timestamp, average temperature, object area, and screenshot filename
    for each auto-screenshot event.
  • Optimized for performance and clarity.

Requirements:
    pip install numpy opencv-python PyQt5 seeed_mlx90640 pyserial
"""

import sys, time, queue, numpy as np, cv2, datetime, logging, csv, os
from serial import Serial
from typing import List, Dict, Any, Optional, Tuple
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout,
                             QHBoxLayout, QComboBox, QSlider, QPushButton, QTabWidget,
                             QFormLayout, QTextEdit, QAction, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QSettings
from PyQt5.QtGui import QImage, QPixmap, QIcon
import seeed_mlx90640

FRAME_QUEUE_MAXSIZE = 10
frame_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)

BILATERAL_FILTER_D = 9
SIGMA_COLOR = 75
SIGMA_SPACE = 75

OBJECT_SCREENSHOT_COOLDOWN = 5

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("ultimate_thermal_viewer_final.log", mode='w')
    ]
)

CSV_FILENAME = "detections.csv"

def init_csv_logging(filename: str = CSV_FILENAME) -> None:
    """
    Creates the CSV file with header if it doesn't exist.
    The columns are:
      - timestamp: ISO-formatted date/time
      - temperature: average temperature in the detected region
      - object_area: bounding box area (in sensor coordinates)
      - screenshot_filename: name of the saved screenshot
    """
    if not os.path.exists(filename):
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "temperature", "object_area", "screenshot_filename"])
        logging.info("CSV log initialized: %s", filename)

def log_detection_event_csv(temperature: float, object_area: float,
                            screenshot_filename: str, filename: str = CSV_FILENAME) -> None:
    """
    Appends a row to the CSV file for each detection event.
    """
    timestamp = datetime.datetime.now().isoformat()
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([timestamp, temperature, object_area, screenshot_filename])
    logging.info("CSV logged event: Temp=%.1f, Area=%.1f, File=%s", temperature, object_area, screenshot_filename)

class DataReader(QThread):
    """
    Acquires thermal frames continuously from the MLX90640 sensor.
    Supports I2C (default) or Serial mode.
    Emits newFrame signal each time a valid frame is queued.
    """
    newFrame = pyqtSignal()

    MODE_I2C = 0
    MODE_SERIAL = 1

    def __init__(self, port: Optional[str] = None,
                 refresh_rate: int = seeed_mlx90640.RefreshRate.REFRESH_16_HZ) -> None:
        super().__init__()
        self.frameCount = 0
        self.mode = self.MODE_I2C if port is None else self.MODE_SERIAL
        self.port = port
        try:
            if self.mode == self.MODE_I2C:
                self.dataHandle = seeed_mlx90640.grove_mxl90640()
                self.dataHandle.refresh_rate = refresh_rate
                self.readData = self.i2cRead
                logging.info("Sensor initialized in I2C mode.")
            else:
                self.dataHandle = Serial(self.port, 2000000, timeout=5)
                self.readData = self.serialRead
                logging.info("Sensor initialized in Serial mode on port %s.", self.port)
        except Exception as e:
            logging.error("Sensor initialization error: %s", e)

    def i2cRead(self) -> List[float]:
        raw_data = [0.0] * 768
        self.dataHandle.getFrame(raw_data)
        return raw_data

    def serialRead(self) -> List[float]:
        raw_data = self.dataHandle.read_until(terminator=b'\r\n')
        try:
            return [float(val) for val in str(raw_data, encoding="utf8").split(",") if val]
        except Exception as e:
            logging.error("Serial conversion error: %s", e)
            return []

    def run(self) -> None:
        try:
            self.readData()
        except Exception as e:
            logging.error("Pre-read error: %s", e)
        while True:
            try:
                data = self.readData()
                if len(data) < 768:
                    continue
                data_array = np.array(data, dtype=np.float32)
                if np.max(data_array) == 0 or np.min(data_array) == 500:
                    continue
                frame_info: Dict[str, Any] = {
                    "frame": data_array,
                    "max": float(np.max(data_array)),
                    "min": float(np.min(data_array)),
                    "timestamp": time.time()
                }
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame_info)
                self.newFrame.emit()
                self.frameCount += 1
                if self.frameCount % 10 == 0:
                    logging.info("Frames acquired: %d", self.frameCount)
            except Exception as e:
                logging.error("Acquisition error: %s", e)

class DiagnosticsPanel(QWidget):
    """
    Displays sensor diagnostics: min, max, mean, and center temperatures,
    FPS, frame count, and timestamp. Temperatures are displayed in the selected unit.
    """
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.start_time = time.time()
        self.initUI()

    def initUI(self) -> None:
        layout = QFormLayout()
        self.minLabel = QLabel("N/A")
        self.maxLabel = QLabel("N/A")
        self.meanLabel = QLabel("N/A")
        self.centerTempLabel = QLabel("N/A")
        self.timestampLabel = QLabel("N/A")
        self.fpsLabel = QLabel("N/A")
        self.frameCountLabel = QLabel("0")
        self.resetButton = QPushButton("Reset Stats", self)
        self.resetButton.clicked.connect(self.resetDiagnostics)
        layout.addRow("Min Temp:", self.minLabel)
        layout.addRow("Max Temp:", self.maxLabel)
        layout.addRow("Mean Temp:", self.meanLabel)
        layout.addRow("Center Temp:", self.centerTempLabel)
        layout.addRow("Timestamp:", self.timestampLabel)
        layout.addRow("FPS:", self.fpsLabel)
        layout.addRow("Frame Count:", self.frameCountLabel)
        layout.addRow(self.resetButton)
        self.setLayout(layout)

    def updateDiagnostics(self, thermal_frame: np.ndarray, timestamp: float,
                          current_fps: float, total_frames: int, unit: str = "°C",
                          center_temp: Optional[float] = None) -> None:
        if unit == "°F":
            min_val = np.min(thermal_frame) * 9/5 + 32
            max_val = np.max(thermal_frame) * 9/5 + 32
            mean_val = np.mean(thermal_frame) * 9/5 + 32
        else:
            min_val = np.min(thermal_frame)
            max_val = np.max(thermal_frame)
            mean_val = np.mean(thermal_frame)
        self.minLabel.setText(f"{min_val:.1f} {unit}")
        self.maxLabel.setText(f"{max_val:.1f} {unit}")
        self.meanLabel.setText(f"{mean_val:.1f} {unit}")
        if center_temp is not None:
            self.centerTempLabel.setText(f"{center_temp:.1f} {unit}")
        self.timestampLabel.setText(datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        self.fpsLabel.setText(f"{current_fps:.1f}")
        self.frameCountLabel.setText(str(total_frames))

    def resetDiagnostics(self) -> None:
        self.start_time = time.time()
        for widget in (self.minLabel, self.maxLabel, self.meanLabel, self.centerTempLabel,
                       self.timestampLabel, self.fpsLabel):
            widget.setText("N/A")
        self.frameCountLabel.setText("0")
        logging.info("Diagnostics reset.")

class SettingsPanel(QWidget):
    """
    Provides controls to adjust viewer settings:
      - Colormap selection.
      - Temperature threshold with live numeric display.
      - Temperature unit selection (°C or °F).
      - Dummy calibration routine.
      - Persistence via QSettings.
    Emits settingsChanged with keys: 'colormap', 'threshold', 'unit'.
    """
    settingsChanged = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.settings = QSettings("UltimateThermalViewer", "Settings")
        self.initUI()
        self.loadSettings()

    def initUI(self) -> None:
        layout = QFormLayout()
        self.colormapCombo = QComboBox(self)
        self.colormapCombo.setToolTip("Select colormap")
        self.colormapOptions = {
            "Inferno": cv2.COLORMAP_INFERNO,
            "Jet": cv2.COLORMAP_JET,
            "Hot": cv2.COLORMAP_HOT,
            "Ocean": cv2.COLORMAP_OCEAN,
            "Rainbow": cv2.COLORMAP_RAINBOW
        }
        for name in self.colormapOptions:
            self.colormapCombo.addItem(name)
        self.colormapCombo.currentIndexChanged.connect(self.onSettingsChanged)
        layout.addRow("Colormap:", self.colormapCombo)

        self.thresholdSlider = QSlider(Qt.Horizontal, self)
        self.thresholdSlider.setToolTip("Adjust detection threshold (°C)")
        self.thresholdSlider.setMinimum(20)
        self.thresholdSlider.setMaximum(50)
        self.thresholdSlider.setTickInterval(1)
        self.thresholdSlider.setValue(32)
        self.thresholdSlider.valueChanged.connect(self.updateThresholdDisplay)
        self.thresholdSlider.valueChanged.connect(self.onSettingsChanged)
        self.thresholdDisplay = QLabel("32 °C", self)
        threshLayout = QHBoxLayout()
        threshLayout.addWidget(self.thresholdSlider)
        threshLayout.addWidget(self.thresholdDisplay)
        layout.addRow("Threshold:", threshLayout)

        self.unitCombo = QComboBox(self)
        self.unitCombo.setToolTip("Select temperature unit")
        self.unitCombo.addItems(["°C", "°F"])
        self.unitCombo.currentIndexChanged.connect(self.updateThresholdDisplay)
        self.unitCombo.currentIndexChanged.connect(self.onSettingsChanged)
        layout.addRow("Unit:", self.unitCombo)

        self.calibrateButton = QPushButton("Run Calibration", self)
        self.calibrateButton.setToolTip("Simulate calibration")
        self.calibrateButton.clicked.connect(self.runCalibration)
        layout.addRow(self.calibrateButton)

        self.saveButton = QPushButton("Save Settings", self)
        self.saveButton.setToolTip("Save current settings")
        self.saveButton.clicked.connect(self.saveSettings)
        layout.addRow(self.saveButton)
        self.setLayout(layout)

    def updateThresholdDisplay(self) -> None:
        threshold = self.thresholdSlider.value()
        unit = self.unitCombo.currentText()
        if unit == "°F":
            threshold_display = threshold * 9/5 + 32
        else:
            threshold_display = threshold
        self.thresholdDisplay.setText(f"{threshold_display:.1f} {unit}")

    def onSettingsChanged(self) -> None:
        new_settings = {
            "colormap": self.colormapCombo.currentText(),
            "threshold": self.thresholdSlider.value(),
            "unit": self.unitCombo.currentText()
        }
        logging.info("Settings changed: %s", new_settings)
        self.settingsChanged.emit(new_settings)

    def runCalibration(self) -> None:
        logging.info("Dummy calibration initiated...")
        time.sleep(1.5)
        logging.info("Dummy calibration completed.")

    def saveSettings(self) -> None:
        self.settings.setValue("colormap", self.colormapCombo.currentText())
        self.settings.setValue("threshold", self.thresholdSlider.value())
        self.settings.setValue("unit", self.unitCombo.currentText())
        logging.info("Settings saved.")

    def loadSettings(self) -> None:
        colormap = self.settings.value("colormap", "Inferno")
        threshold = int(self.settings.value("threshold", 32))
        unit = self.settings.value("unit", "°C")
        idx = self.colormapCombo.findText(colormap)
        if idx >= 0:
            self.colormapCombo.setCurrentIndex(idx)
        self.thresholdSlider.setValue(threshold)
        idx_unit = self.unitCombo.findText(unit)
        if idx_unit >= 0:
            self.unitCombo.setCurrentIndex(idx_unit)
        self.updateThresholdDisplay()
        logging.info("Settings loaded: colormap=%s, threshold=%d, unit=%s", colormap, threshold, unit)

class SensorCalibrationPanel(QWidget):
    """
    Simulates advanced sensor calibration.
    Computes a random offset (°C) to correct sensor drift.
    Emits calibrationPerformed with the computed offset.
    """
    calibrationPerformed = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.calibration_log: List[str] = []
        self.initUI()

    def initUI(self) -> None:
        layout = QVBoxLayout()
        self.calibButton = QPushButton("Perform Sensor Calibration", self)
        self.calibButton.setToolTip("Simulate advanced sensor calibration")
        self.calibButton.clicked.connect(self.performCalibration)
        layout.addWidget(self.calibButton)
        self.logTextEdit = QTextEdit(self)
        self.logTextEdit.setToolTip("Calibration log")
        self.logTextEdit.setReadOnly(True)
        self.logTextEdit.setFixedHeight(150)
        layout.addWidget(self.logTextEdit)
        self.resetLogButton = QPushButton("Reset Calibration Log", self)
        self.resetLogButton.setToolTip("Clear calibration log")
        self.resetLogButton.clicked.connect(self.resetCalibrationLog)
        layout.addWidget(self.resetLogButton)
        self.setLayout(layout)

    def performCalibration(self) -> None:
        start_time = time.time()
        time.sleep(2)
        calibration_offset = float(np.random.uniform(-0.3, 0.3))
        duration = round(time.time() - start_time, 2)
        log_entry = (f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                     f"Offset: {calibration_offset:+.2f} °C, Duration: {duration} s")
        self.calibration_log.append(log_entry)
        self.logTextEdit.append(log_entry)
        logging.info("Calibration performed: %s", log_entry)
        self.calibrationPerformed.emit(calibration_offset)

    def resetCalibrationLog(self) -> None:
        self.calibration_log.clear()
        self.logTextEdit.clear()
        logging.info("Calibration log reset.")

class ThermalViewer(QWidget):
    """
    Main thermal imaging interface with crosshair, diagnostics, settings, calibration,
    object detection, auto screenshots (cooldown), recording, and CSV logging.
    """
    def __init__(self, display_width: int = 640, display_height: int = 480) -> None:
        super().__init__()
        self.display_width = display_width
        self.display_height = display_height
        self.native_cols, self.native_rows = 32, 24
        self.frameCounter = 0
        self.fps = 0.0
        self.lastTime = time.time()
        self.totalFrames = 0
        self.presence_threshold = 32.0
        self.selectedColormap = cv2.COLORMAP_INFERNO
        self.temp_unit = "°C"
        self.calibration_offset = 0.0
        self.recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.current_frame: Optional[np.ndarray] = None
        self.last_object_screenshot_time = 0.0

        self.diagnosticsPanel = DiagnosticsPanel(self)
        self.settingsPanel = SettingsPanel(self)
        self.settingsPanel.settingsChanged.connect(self.applySettings)
        self.calibrationPanel = SensorCalibrationPanel(self)
        self.calibrationPanel.calibrationPerformed.connect(self.updateCalibrationOffset)

        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)

    def initUI(self) -> None:
        self.setWindowTitle("Ultimate Enhanced Thermal Viewer")
        self.resize(self.display_width + 40, self.display_height + 240)

        self.imageLabel = QLabel(self)
        self.imageLabel.setFixedSize(self.display_width, self.display_height)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("background-color: black;")

        self.captureButton = QPushButton("Capture Screenshot", self)
        self.captureButton.clicked.connect(self.captureScreenshot)

        self.recordButton = QPushButton("Start Recording", self)
        self.recordButton.setCheckable(True)
        self.recordButton.clicked.connect(self.toggleRecording)

        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.captureButton)
        controlLayout.addWidget(self.recordButton)

        self.tabs = QTabWidget(self)
        viewerTab = QWidget()
        viewerLayout = QVBoxLayout()
        viewerLayout.addWidget(self.imageLabel)
        viewerLayout.addLayout(controlLayout)
        viewerTab.setLayout(viewerLayout)
        self.tabs.addTab(viewerTab, "Live Viewer")
        self.tabs.addTab(self.diagnosticsPanel, "Diagnostics")
        self.tabs.addTab(self.settingsPanel, "Settings")
        self.tabs.addTab(self.calibrationPanel, "Sensor Calibration")

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.tabs)
        self.setLayout(mainLayout)
        self.show()

    def applySettings(self, settings_dict: Dict[str, Any]) -> None:
        colormap_name = settings_dict.get("colormap", "Inferno")
        self.presence_threshold = float(settings_dict.get("threshold", 32))
        self.temp_unit = settings_dict.get("unit", "°C")
        self.selectedColormap = self.settingsPanel.colormapOptions.get(colormap_name, cv2.COLORMAP_INFERNO)

    def updateCalibrationOffset(self, offset: float) -> None:
        self.calibration_offset = offset

    def detectObjects(self, thermal_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        mask = np.uint8((thermal_frame > self.presence_threshold) * 255)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 2:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            scale_x = self.display_width / self.native_cols
            scale_y = self.display_height / self.native_rows
            bboxes.append((int(x * scale_x), int(y * scale_y),
                           int(w * scale_x), int(h * scale_y)))
        return bboxes

    def updateFrame(self) -> None:
        if frame_queue.empty():
            return
        frame_info = frame_queue.get()
        raw_frame = frame_info["frame"]
        min_val, max_val = frame_info["min"], frame_info["max"]
        timestamp = frame_info["timestamp"]

        processed_img = self.processFrame(raw_frame, min_val, max_val)
        self.current_frame = processed_img.copy()

        reshaped = raw_frame.reshape((self.native_rows, self.native_cols))
        adjusted_frame = reshaped - self.calibration_offset

        detected_objects = self.detectObjects(adjusted_frame)

        center_x = self.display_width // 2
        center_y = self.display_height // 2
        cv2.line(processed_img, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(processed_img, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)

        sensor_center_x = self.native_cols // 2
        sensor_center_y = self.native_rows // 2
        center_temp = adjusted_frame[sensor_center_y, sensor_center_x]
        if self.temp_unit == "°F":
            center_temp = center_temp * 9/5 + 32

        temp_text = f"{center_temp:.1f} {self.temp_unit.replace('°', '')}"
        cv2.putText(processed_img, temp_text,
                    (center_x + 15, center_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for bbox in detected_objects:
            x, y, w, h = bbox
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(processed_img, "Object", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_time = time.time()
        if detected_objects and (current_time - self.last_object_screenshot_time > OBJECT_SCREENSHOT_COOLDOWN):
            scale_x = self.display_width / self.native_cols
            scale_y = self.display_height / self.native_rows
            for i, bbox_disp in enumerate(detected_objects):
                x_disp, y_disp, w_disp, h_disp = bbox_disp
                cropped = processed_img[y_disp:y_disp+h_disp, x_disp:x_disp+w_disp]
                filename = f"object_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.png"
                cv2.imwrite(filename, cropped)

                sx, sy = int(x_disp / scale_x), int(y_disp / scale_y)
                sw, sh = int(w_disp / scale_x), int(h_disp / scale_y)
                region = adjusted_frame[sy:sy+sh, sx:sx+sw]

                avg_temp = float(np.mean(region)) if region.size > 0 else float(center_temp)
                object_area = float(sw * sh)
                log_detection_event_csv(avg_temp, object_area, filename)

            self.last_object_screenshot_time = current_time

        self.frameCounter += 1
        self.totalFrames += 1
        now_t = time.time()
        if now_t - self.lastTime >= 1.0:
            self.fps = self.frameCounter / (now_t - self.lastTime)
            self.frameCounter = 0
            self.lastTime = now_t

        cv2.putText(processed_img, f"FPS: {self.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        self.diagnosticsPanel.updateDiagnostics(adjusted_frame, timestamp, self.fps,
                                                self.totalFrames, self.temp_unit, float(center_temp))
        self.imageLabel.setPixmap(QPixmap.fromImage(self.convertCvToQImage(processed_img)))

        if self.recording and self.video_writer is not None:
            self.video_writer.write(processed_img)

    def processFrame(self, raw_frame: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
        reshaped = raw_frame.reshape((self.native_rows, self.native_cols))
        adjusted = reshaped - self.calibration_offset
        norm_frame = np.interp(adjusted,
                               (min_val - self.calibration_offset, max_val - self.calibration_offset),
                               (0, 255)).astype(np.uint8)
        upscaled = cv2.resize(norm_frame, (self.display_width, self.display_height), interpolation=cv2.INTER_LANCZOS4)
        colored = cv2.applyColorMap(upscaled, self.selectedColormap)
        return cv2.bilateralFilter(colored, d=BILATERAL_FILTER_D, sigmaColor=SIGMA_COLOR, sigmaSpace=SIGMA_SPACE)

    def convertCvToQImage(self, cv_img: np.ndarray) -> QImage:
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        return QImage(rgb_img.data, w, h, ch*w, QImage.Format_RGB888)

    def captureScreenshot(self) -> None:
        if self.current_frame is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{ts}.png"
        cv2.imwrite(filename, self.current_frame)
        QMessageBox.information(self, "Screenshot", f"Screenshot saved as:\n{filename}")

    def toggleRecording(self) -> None:
        if not self.recording:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{ts}.avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10, (self.display_width, self.display_height))
            if not self.video_writer.isOpened():
                self.recordButton.setChecked(False)
                return
            self.recording = True
            self.recordButton.setText("Stop Recording")
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recordButton.setText("Start Recording")

    def closeEvent(self, event) -> None:
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ultimate Enhanced Thermal Viewer")
        self.setWindowIcon(QIcon("icon.png"))
        self.viewer = ThermalViewer(display_width=640, display_height=480)
        self.setCentralWidget(self.viewer)
        self.createMenus()
        self.statusBar().showMessage("Ready")

    def createMenus(self) -> None:
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        exitAction = QAction("E&xit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

def main() -> None:
    init_csv_logging()
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sensorThread = DataReader(port=None)
    sensorThread.newFrame.connect(lambda: None)
    sensorThread.start()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
