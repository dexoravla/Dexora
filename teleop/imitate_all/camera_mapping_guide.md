# Persistent Camera Device Mapping Guide

## Overview

This project sets up persistent camera device symlinks based on USB port location, solving the problem of camera device numbers changing after they are unplugged and reconnected.

## Mapping Table

| Camera type | USB port | Symlink | Current device |
|-------------|----------|---------|----------------|
| Left camera | 1-13.1 | `/dev/camera_left` | `/dev/video12` |
| Right camera | 1-13.3 | `/dev/camera_right` | `/dev/video2` |
| High-precision camera | 1-13.4 | `/dev/camera_high` | `/dev/video0` |

## Rules File

**File location**: `/etc/udev/rules.d/99-camera-symlinks.rules`

```bash
# Left camera - USB port 1-13.1 -> /dev/camera_left
SUBSYSTEM=="video4linux", ATTRS{devpath}=="13.1", SYMLINK+="camera_left"

# Right camera - USB port 1-13.3 -> /dev/camera_right
SUBSYSTEM=="video4linux", ATTRS{devpath}=="13.3", SYMLINK+="camera_right"

# High-precision camera - USB port 1-13.4 -> /dev/camera_high
SUBSYSTEM=="video4linux", ATTRS{devpath}=="13.4", SYMLINK+="camera_high"
```

## Advantages

1. **Stability**: Based on USB port location, independent of the device number.
2. **Persistence**: The mapping remains valid after a system reboot.
3. **Consistency**: The mapping is preserved across camera unplug/replug events.

## Usage

### In code

```python
# Use the symlinks instead of device numbers
import cv2

# Previous approach (fragile)
# cap_left = cv2.VideoCapture(12)  # video12
# cap_right = cv2.VideoCapture(2)  # video2
# cap_high = cv2.VideoCapture(0)  # video0

# New approach (stable and reliable)
cap_left = cv2.VideoCapture('/dev/camera_left')
cap_right = cv2.VideoCapture('/dev/camera_right')
cap_high = cv2.VideoCapture('/dev/camera_high')
```

### On the command line

```bash
# Using ffmpeg
ffmpeg -f v4l2 -i /dev/camera_left output_left.mp4
ffmpeg -f v4l2 -i /dev/camera_right output_right.mp4
ffmpeg -f v4l2 -i /dev/camera_high output_high.mp4

# Use v4l2-ctl to inspect device info
v4l2-ctl -d /dev/camera_left --list-formats-ext
v4l2-ctl -d /dev/camera_right --list-formats-ext
v4l2-ctl -d /dev/camera_high --list-formats-ext
```

## Management Commands

### Reload rules
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### View current mappings
```bash
ls -la /dev/camera*
```

### Test the mapping status
```bash
./test_camera_mapping.sh
```

## Troubleshooting

### If the mapping stops working

1. Check that the rules file exists:
   ```bash
   ls -la /etc/udev/rules.d/99-camera-symlinks.rules
   ```

2. Reload the rules:
   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

3. Inspect device information:
   ```bash
   udevadm info -a -n /dev/video0
   udevadm info -a -n /dev/video2
   udevadm info -a -n /dev/video12
   ```

### If the USB port changes

If a camera's USB port location changes, you need to update the `devpath` value in the rules file:

1. Look up the new port assignment:
   ```bash
   udevadm info -a -n /dev/video0 | grep "ATTRS{devpath}"
   ```

2. Update the `devpath` value in the rules file.

3. Reload the rules.

## Notes

1. **Port location**: Make sure cameras are always connected to the same USB ports.
2. **Permissions**: Ensure the user has access to `/dev/video*` devices.
3. **Device type**: The current rules are tailored for the LRCP F1080P camera; other models may require adjustments.

## Extensions

To add more cameras or support other models, you can:

1. Match by USB device ID:
   ```bash
   SUBSYSTEM=="video4linux", ATTRS{idVendor}=="1bcf", ATTRS{idProduct}=="2cc8", SYMLINK+="camera_name"
   ```

2. Match by device serial number:
   ```bash
   SUBSYSTEM=="video4linux", ATTRS{serial}=="unique_serial", SYMLINK+="camera_name"
   ```

3. Match by device name:
   ```bash
   SUBSYSTEM=="video4linux", ATTR{name}=="Camera Name", SYMLINK+="camera_name"
   ```
