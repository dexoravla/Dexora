# LeRobot API version of the Airbot data converter

This is the Airbot data converter rewritten with the official LeRobot API. Compared to the original, the key improvements are:

## Key advantages

1. **Uses the official LeRobot API end-to-end**: no more manual parquet writing, video encoding, and so on.
2. **Standard LeRobot v2.1 format**: the produced dataset fully conforms to the LeRobot v2.1 standard.
3. **Simplified code**: removes a large amount of duplicated custom implementations.
4. **Better error handling**: takes advantage of LeRobot's built-in validation.
5. **Automatic video encoding**: uses LeRobot's efficient video encoding pipeline.

## File layout

```
dataprocess/
├── airbot_lerobot.py     # LeRobot API-based processor (current)
├── airbot.py             # Legacy manual processor (kept for reference)
├── airbot_config.py      # Shared configuration
├── code/                 # Cross-embodiment configs (aloha, realman)
├── lerobot_split_merge_prcessor-main/   # Dataset split/merge toolkit
└── README.md             # This document
```

## Core changes

### Functions removed (replaced by the LeRobot API)

1. `create_video_from_images()` -> `LeRobotDataset.encode_episode_videos()`
2. `update_episode_metadata()` -> `LeRobotDatasetMetadata.save_episode()`
3. `create_meta_info()` -> `LeRobotDatasetMetadata.create()`
4. `save_device_info()`, `save_label_info()` -> handled automatically by LeRobot
5. Manual parquet writing -> `LeRobotDataset.save_episode()`
6. Manual chunk-directory management -> handled automatically by LeRobot

### New functions

1. `create_lerobot_features()` - defines the dataset feature schema
2. `convert_frame_to_lerobot_format()` - converts per-frame data into LeRobot format
3. `setup_lerobot_dataset()` - initializes the dataset via the LeRobot API
4. `process_episode_with_lerobot()` - processes an episode through the LeRobot API

### Modified core flow

Original flow:
```python
# Manually create the directory structure
# Manually write parquet files
# Manually create video files
# Manually update metadata
```

New flow:
```python
# 1. Create the LeRobot dataset
dataset = LeRobotDataset.create(repo_id, fps, features, root, robot_type)

# 2. For each episode:
for episode in episodes:
    # 3. For each frame:
    for frame_data in episode_frames:
        frame = convert_frame_to_lerobot_format(frame_data)
        dataset.add_frame(frame, task, timestamp)
    
    # 4. Save the episode (parquet + video handled automatically)
    dataset.save_episode()
```

## How to use

### 1. Basic usage

```bash
cd dataprocess

# Use the LeRobot API version
python airbot_lerobot.py
```

### 2. Configuration changes

The data converter reuses the same `airbot_config.py` file. Key configuration entries:

```python
# Data paths
source_data_root = "/path/to/source/data"     # BSON + image folders
output_data_root = "/path/to/output/data"     # LeRobot-format output

# Dataset parameters
fps = 20.0                                    # Sampling frequency
robot = "airbot_dexterous"                    # Robot type
overwrite = True                              # Whether to overwrite an existing dataset

# BSON filenames
robot_bson_name = "episode_0.bson"            # Arm data
hand_bson_name = "xhand_control_data.bson"    # Dexterous-hand data
```

## Data format

### Input format (unchanged)

```
source_data_root/
├── action8/
│   ├── episode_001/
│   │   ├── episode_0.bson          # Arm data
│   │   ├── xhand_control_data.bson # Dexterous-hand data
│   │   ├── camera_4/               # High-mounted camera
│   │   ├── camera_2/               # Left-side camera
│   │   └── camera_6/               # Right-side camera
│   └── episode_002/
└── action27/
```

### Output format (LeRobot v2.1 standard)

```
output_data_root/airbot_dexterous_bimanual_dexterous_manipulation/
├── data/
│   ├── chunk-000/
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   └── chunk-001/
├── meta/
│   ├── info.json                   # Dataset info
│   ├── episodes.jsonl              # Episode metadata
│   ├── stats.json                  # Data statistics
│   └── tasks.jsonl                 # Task definitions
└── videos/
    ├── chunk-000/
    │   ├── observation.images.camera_high/
    │   │   ├── episode_000000.mp4
    │   │   └── ...
    │   ├── observation.images.camera_left/
    │   ├── observation.images.camera_right/
    │   └── observation.images.camera_front/
    └── chunk-001/
```

## Data feature definitions

- **states**: 36-dim (left arm 6 + right arm 6 + left hand 12 + right hand 12)
- **actions**: 36-dim (same dimensionality as states)
- **Observation images**: 4 cameras (camera_high, camera_left, camera_right, camera_front)
- **Task mapping**: supports automatic action_id -> task_index mapping

## Compatibility

- **LeRobot version**: v0.3.4 (v2.1 format)
- **Python version**: 3.8+
- **Keeps original configuration**: fully compatible with the existing `airbot_config.py`
- **Source data format**: keeps the BSON + image folder layout unchanged

## Performance optimizations

1. **Batched video encoding**: supports batched encoding to improve throughput
2. **Parallel image writing**: supports multi-process image writes
3. **Memory optimizations**: uses LeRobot's memory management
4. **Automatic validation**: built-in data validation and error checking

## Troubleshooting

### Common issues

1. **Import errors**: make sure the correct lerobot version is installed
   ```bash
   pip install lerobot==0.3.4
   ```

2. **Feature-dimension errors**: verify that the joint data matches the expected dimensionality
   - Arm: 6 DOF per side
   - Dexterous hand: 12 DOF per side

3. **Camera mapping issues**: review the camera mapping in `airbot_config.py`

4. **Missing BSON files**: ensure every episode directory contains the required BSON files

### Debug mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Comparison with the original version

| Capability | Original version | LeRobot API version |
|------|--------|------------------|
| parquet writing | Manual | LeRobot API |
| Video encoding | OpenCV + ffmpeg | LeRobot pipeline |
| Metadata management | Manual JSONL | LeRobot automatic |
| Directory structure | Manually built | LeRobot standard |
| Data validation | Basic checks | Full validation |
| Error handling | Limited | Comprehensive |
| Code complexity | High | Significantly reduced |

## Migrating from the legacy processor

If you previously used `airbot.py`, switch to `airbot_lerobot.py` —
the two processors share `airbot_config.py`, so the only step is:

```bash
python airbot_lerobot.py
```

Output is the standard LeRobot v2.1 layout described above and is
directly consumed by `data/lerobot_vla_dataset.py` and `s1_pretrain.sh`.
