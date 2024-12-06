# MLX90640 Qwiic IR Array Guide

## 1. Ensure Proper Hookup of the Qwiic IR Array (MLX90640)

Before starting, ensure that your MLX90640 sensor is correctly connected to your Raspberry Pi via I2C. Use the Qwiic connectors or jumper wires for the connection.

- **VCC**: 3.3V (or 5V depending on your specific setup)
- **GND**: Ground
- **SCL**: Clock (connect to Raspberry Pi's SCL pin)
- **SDA**: Data (connect to Raspberry Pi's SDA pin)

## 2. Software Setup

### Set Up a Virtual Environment (Recommended)

It’s a good practice to use a virtual environment for Python projects to avoid dependency conflicts.

#### Install `virtualenv`:
If you don’t have `virtualenv` installed, open a terminal and run:
```bash
sudo apt install python3-venv
```

### Create a virtual environment for your project:

```bash
python3 -m venv mlx90640-env
```

### Activate the Virtual Environment:
```bash
source mlx90640-env/bin/activate
```
Now you're working within the virtual environment.

### Install Required Libraries
Next, install the necessary libraries for I2C communication and working with the MLX90640 sensor.

```bash
pip install adafruit-blinka        # Required for I2C communication on Raspberry Pi
pip install adafruit-circuitpython-mlx90640 # Library for the MLX90640 sensor
pip install numpy                  # For numerical operations
pip install matplotlib             # For visualizing the sensor output
```

### Enable I2C on Your Raspberry Pi
Make sure that the I2C interface is enabled on your Raspberry Pi. This is typically enabled by default on recent Raspberry Pi OS versions.
To enable I2C on your Raspberry Pi:

Run the following command:
```bash
sudo raspi-config
```
Navigate to Interfacing Options → I2C and enable it.
Reboot the Raspberry Pi:
```bash
sudo reboot
```

## 3. Check if the Sensor is Detected
After ensuring that the sensor is correctly connected and I2C is enabled, check if the Raspberry Pi can detect the MLX90640 sensor.

Run the following command to check the I2C devices:

```bash
i2cdetect -y 1
```

You should see an output like this with a grid, and a number (usually 0x33) where your MLX90640 is located:

```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:                         -- -- -- -- -- -- -- -- 
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
20: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
30: -- -- -- 33 -- -- -- -- -- -- -- -- -- -- -- -- 
40: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
50: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
60: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
70: -- -- -- -- -- -- -- --
```

This confirms that the Raspberry Pi recognizes the sensor.

## 4. Reactivate the Virtual Environment
If the virtual environment isn't active, you need to reactivate it first:

```bash
source mlx90640-env/bin/activate
```

## 5. Create the Python Script
To run the MLX90640 sensor, create a Python script using nano or your preferred text editor. For example, create the file test2_mlx90640.py:

```bash
nano test2_mlx90640.py
```

Paste the following code into the file:

```python
import time
import board
import busio
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from adafruit_mlx90640 import MLX90640

# Create I2C bus and MLX90640 object
i2c = busio.I2C(board.SCL, board.SDA, frequency=1000000)  # 1 MHz I2C speed
mlx90640 = MLX90640(i2c)

# Start reading thermal data
frame = [0] * 768  # 24 * 32 = 768
plt.ion()  # Interactive mode for live plotting
fig, ax = plt.subplots()
cbar = None  # Placeholder for the color bar

# Remove axis ticks for cleaner display
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

while True:
    mlx90640.getFrame(frame)
    reshaped_frame = np.reshape(frame, (24, 32))

    # Upscale the frame for higher resolution
    upscaled_frame = zoom(reshaped_frame, (10, 10), order=3)  # 10x scaling

    # Clear the axis for updating
    ax.clear()
    img = ax.imshow(upscaled_frame, cmap='plasma', interpolation='bilinear')

    # Add the color bar only once
    if cbar is None:
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)

    # Update the plot
    plt.draw()
    plt.pause(0.1)  # Pause for a short time to allow the plot to update
    time.sleep(0.2)
```

### Save and Exit:
- Press Ctrl + O to write out the file.
- Press Enter to confirm the filename.
- Press Ctrl + X to exit the editor.

## 6. Run the Script

Once the script is saved, you can run it to view the sensor output.

Execute the following command in the terminal:

```bash
python test2_mlx90640.py
```

This script will read the temperature data from the MLX90640 sensor and visualize it in real-time using a heatmap.
