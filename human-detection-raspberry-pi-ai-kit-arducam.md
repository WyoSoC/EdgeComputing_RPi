# Raspberry Pi M.2 HAT+ AI Kit with Arducam IMX519 16 MP AF

## Step 1: Install Hailo Dependencies
Ensure you have the required dependencies installed for the NPU. If you haven’t already, run:

```bash
sudo apt install hailo-all
sudo reboot
```
After rebooting, verify that the Hailo NPU is set up correctly by running:
```bash
hailortcli fw-control identify
```
This command should output details about the Hailo NPU, confirming it’s recognized by the system.

##Step 2: Set Up the rpicam-apps Environment

Clone the rpicam-apps repository to access the JSON configuration files required for the demos:
```bash
git clone --depth 1 https://github.com/raspberrypi/rpicam-apps.git ~/rpicam-apps
```
These JSON files contain the configuration settings for various AI models that run on the Hailo NPU.

##Step 3: Run Object Detection Demo Using rpicam-hello

To utilize the Hailo NPU, you can start by running the provided demos. For example, to perform object detection using the YOLOv5 model:

YOLOv5 Person and Face Detection
Run the following command:
```bash
rpicam-hello -t 0 --post-process-file ~/rpicam-apps/assets/hailo_yolov5_personface.json --lores-width 640 --lores-height 640
```

This command:
Runs rpicam-hello with the Hailo NPU performing object detection.
Uses the YOLOv5 model specified in the hailo_yolov5_personface.json file.
Sets the resolution for low-resolution input processing to 640x640.
