# Getting the Arducam 16MP IMX519 Camera to Work with a Raspberry Pi 5

This guide will walk you through the steps to set up the Arducam 16MP IMX519 camera on a Raspberry Pi 5.

### Step 1: Connect the Camera

1. Connect the camera to the ribbon slot on the Raspberry Pi (for this example, we’re using slot index 1).
2. Ensure the ribbon's connection is metal-on-metal for each side.

### Step 2: Download and Install the Required Packages

1. Download the installation script:
    ```bash
    wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
    ```

2. Make the script executable:
    ```bash
    chmod +x install_pivariety_pkgs.sh
    ```

3. Install the necessary packages:
    ```bash
    ./install_pivariety_pkgs.sh -p libcamera_dev
    ./install_pivariety_pkgs.sh -p libcamera_apps
    ```

    - If you encounter an error, try running:
        ```bash
        sudo dpkg -i rpicam-apps_1.5.2-3_arm64.deb
        ```

### Step 3: Update Configuration

1. Open the config file:
    ```bash
    sudo nano /boot/firmware/config.txt
    ```

2. Find the line `[all]` and add the following line beneath it:
    ```plaintext
    dtoverlay=imx519
    ```
   - This is for slot index 1.

3. Save and exit:
    - Press `^O` (Control + O) to Write Out, then `Enter` to confirm.
    - Press `^X` (Control + X) to Exit.

4. Reboot the system to apply the changes:
    ```bash
    sudo reboot
    ```

### Step 4: Test the Camera

After rebooting, test the camera by running:
```bash
rpicam-hello
```

[https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/]
