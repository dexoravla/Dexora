# Data Consistency Validation Script Guide

## What the script does

`validate_data_consistency.py` verifies that, under every action folder, the image counts of
the four camera folders and the frame counts of the BSON files are all consistent with each other.

## What gets validated

1. **Consistency of camera-folder image counts**:
   - Checks the four camera folders `camera0_head`, `camera1_left_wrist`,
     `camera2_right_wrist`, and `camera3_third_view`.
   - Verifies that every camera folder contains the same number of images.
   - Supports jpg, jpeg, png, bmp, and tiff image formats.

2. **Consistency of BSON frame counts**:
   - Checks the `episode_0.bson` and `xhand_control_data.bson` files.
   - Verifies that the frame counts inside the BSON files match the image counts.
   - `episode_0.bson`: checks the length of each array under the `data` field.
   - `xhand_control_data.bson`: checks the length of the `frames` field.

## Usage

### 1. Validate every action folder
```bash
python3 validate_data_consistency.py
```

### 2. Validate a specific action folder
```bash
python3 validate_data_consistency.py --action action8
```

### 3. Show verbose output
```bash
python3 validate_data_consistency.py --action action8 --verbose
```

### 4. Specify a base path
```bash
python3 validate_data_consistency.py --path /path/to/your/data
```

## Command-line arguments

- `--path, -p`: Base path to validate (default: current directory).
- `--action, -a`: Validate only the specified action folder.
- `--verbose, -v`: Show verbose output.
- `--help, -h`: Show help.

## Output

### Result for a single action
```
action8 validation result:
  Total episodes: 50
  Consistent episodes: 50
  Inconsistent episodes: 0

Validation log saved to: ./logs/action_action8_validation_20250908_150820.log
```

### Aggregate report across all actions
```
Overall stats:
  Total action folders: 100
  Fully consistent actions: 95
  Actions with inconsistencies: 5

Validation log saved to: ./logs/data_validation_20250908_150820.log

Details of inconsistent action folders:
action15:
  Total episodes: 50
  Consistent episodes: 48
  Inconsistent episodes: 2
  episode_5:
    Camera image counts: {'camera0_head': 435, 'camera1_left_wrist': 434, 'camera2_right_wrist': 435, 'camera3_third_view': 435}
    Error: Camera image counts are inconsistent
```

### Example log file contents
```
Action: action8
  Total episodes: 50
  Consistent episodes: 50
  Inconsistent episodes: 0
    Episode: episode_0
      Camera image counts:
        camera0_head: 435 images
        camera1_left_wrist: 435 images
        camera2_right_wrist: 435 images
        camera3_third_view: 435 images
      BSON frame counts:
        episode_0.bson:
          /observation/left_arm/pose: 435 entries
          /observation/left_arm/joint_state: 435 entries
          /action/left_arm/joint_state: 435 entries
          ...
        xhand_control_data.bson:
          frames: 435 entries
```

## Log files

The script automatically writes the validation results to log files under the `logs` directory:

- Validate all actions: `logs/data_validation_YYYYMMDD_HHMMSS.log`
- Validate a single action: `logs/action_ACTIONNAME_validation_YYYYMMDD_HHMMSS.log`

Each log file contains detailed per-action and per-episode statistics.

## Dependencies

The script requires the following Python packages:
- `bson`: for reading BSON files
- `pathlib`: for path manipulation
- `glob`: for file matching

Install:
```bash
pip install pymongo  # provides the bson module
```

## Notes

1. The script automatically skips missing camera folders or BSON files.
2. For BSON files that cannot be read, a warning is logged but validation continues.
3. Validation may take a while, especially when there are many action folders.
4. We recommend first testing on a single action folder with `--action` to confirm the script
   behaves correctly, and then running it across all actions.
