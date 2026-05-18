# Data Collection and Replay

## Environment Setup

!!! tip "Supported Operation Systems"

    * Ubuntu AMD64(x86_64)

It is recomended to use **anaconda** to manage python environments. You can download and install it by running the following commands(if download very slowly, you can [click here](https://mirrors.bfsu.edu.cn/anaconda/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh) to download manually):

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && bash miniconda.sh
```

Restart your terminal and you can now use conda:
```bash
conda config --set auto_activate_base false && conda deactivate
```

Clone the reposity and enter it:
```bash
git clone https://github.com/OpenGHz/Imitate-All.git
cd Imitate-All
```

Install the required python packages:
```bash
conda create -n imitall python=3.12.8 && conda activate imitall
pip install -r requirements/data_collection.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt install ffmpeg
```

Some computers need to execute the following command to support simultaneous reading of multiple external cameras:
```bash
sudo rmmod uvcvideo && sudo modprobe uvcvideo nodrop=1 timeout=5000 quirks=0x80
```

For AIRBOTPlay3.0 usage, make sure you have installed the `airbot_python_sdk` package (version >= 3.0.4, refer to `Environment-setup-for-AIRBOT-Play.md`), you can install the grpc packages to your conda environment first:
```bash
python3 -m pip install --upgrade pip setuptools -i https://pypi.mirrors.ustc.edu.cn/simple/
python3 -m pip install grpcio grpcio-tools -i https://pypi.mirrors.ustc.edu.cn/simple/
cd /path/to/your/airbot_sdk/tools  # Don't put this package in the Imitate-All folder
chmod +x ./grpc_generate.sh
bash ./grpc_generate.sh
cd ../python3
pip install -e . -i https://pypi.mirrors.ustc.edu.cn/simple/
```

## Data Collection

> The task name should be reasonable, and it is recommended to include time in the name to distinguish the same task data collected at different times.

### Starting Robotic Arms
1. Prepare all teacher-follower robotic arms.
2. Connect the power sources of all robotic arms (order doesn't matter).
3. **First**, connect the teacher arm via Type-C data cable (CAN0), **then** connect the follower arm (CAN1). For dual-arm operations, follow the above sequence for the left-side robotic arm first, then the right-side arm.
4. Long-press the power button on each robotic arm to turn them on.
5. Ensure that the robotic arms are at the zero pose; otherwise, perform a zero calibration.
6. Start the control service of all the robots.

> Note: Other devices connected to your computer may occupy the CAN interfaces, you may need to change the default can interfaces manually. Please refer to [Explanation of Parameters](#explanation-of-parameters).

### Connecting Cameras
Data collection typically requires multiple cameras, and the connection order can be arbitrary, for example:

- Single-arm task: Left arm-mounted camera -> Right arm-mounted camera -> Environment camera
- Dual-arm task: Left arm camera -> Right arm camera -> Environment top camera -> Environment bottom camera

> Note: Your computer may not be able to activate too many cameras at the same time.

### Modify Default Configurations

For AIRBOT Play 2.0, the defualt configurations are in `configurations/basic_configs/example/robot/airbots/play/airbot_play_2_demonstration.yaml`, which is used for one airbot_play with a E2B encoder to teleoprate another one airbot_play with a G2 gripper and one USB camera.

For AIRBOT Play 3.0, the defualt configurations are in `configurations/basic_configs/example/robot/airbots/play/airbot_play_demonstration.yaml`. You can modify these according to your usage, for example:

- One short Play teacher with one short follower:
    ```
    leader_arm_type: ["play_short"]
    follower_arm_type: ["play_short"]
    ```
- Two short Play teachers with two short followers:
    ```
    leader_arm_type: ["play_short"， "play_short"]
    follower_arm_type: ["play_short"， "play_short"]
    leader_end_effector: ["E2B"， "E2B"]
    follower_end_effector: ["G2"， "G2"]
    leader_can_interface: ["can0", "can2"]
    follower_can_interface: ["can1", "can3"]
    leader_domain_id: [33, 34]
    follower_domain_id: [44, 45]
    start_arm_joint_position: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    start_eef_joint_position: [0.0, 0.0]
    ```

For recording with TOK2/4, please refer to `airbot_tok2_2_demonstration.yaml`/`airbot_tok4_2_demonstration.yaml` in `configurations/basic_configs/example/robot/airbots/tok` folder.

For recording with RealSense cameras, please refer to `configurations/basic_configs/example/robot/airbots/tok/airbot_tok2_2_no_base_realsense_demonstration.yaml` and run `pip install -r requirements/realsense.txt -i https://pypi.mirrors.ustc.edu.cn/simple/` to install pyrealsense2.

### Starting Data Collection / Demonstration

The common usage of parameters for data collection command is as follows:

```bash
python3 control_robot.py record \
  --robot-path configurations/basic_configs/example/robot/airbots/play/airbot_play_demonstration.yaml \
  --root data \
  --repo-id raw/example \
  --fps 20 \
  --num-frames-per-episode 500 \
  --num-episodes 10000 \
  --start-episode 0 \
```

#### Explanation of Parameters

  - `--robot-path`: path to the robot yaml file used to instantiate the robot
  - `--root`: root directory where the dataset will be stored locally at `{root}/{repo_id}`, defaulting to `data`
  - `--repo-id`: dataset identifier, defaulting to `raw/example`
  - `--fps`: frames per second
  - `--num-episodes`: number of episodes to record
  - `--num-frames-per-episode`: number of frames for data recording for each episode
  - `--start-episode`: index of the first episode to record

#### Key Descriptions

> Note: Avoid continuous key presses to prevent repeated behavior.

After excuting the command above, you can use the keyboard of your computer to control. The key descriptions are as follows:

- `Space Bar`: Start recording one episode data. The data will be automatically saved upon recording all frames.
- `s`: Stop current recording and save the data before recording all frames. This will result in different recorded data lengths
- `q`: Discard current recording or rerecording the last episode
- `g`: Toggle demonstration mode on/off
- `0`: Set the robotic arms to the initial states
- `p`: Print current robot states information in terminal
- `i`: Show these instructions again
- `z`: Exit this program after converting all saved images to mp4 videos
- `ESC`: Exit this program without converting data

#### Operational Steps

1. Start the program, and the real-time windows of each camera will appear (if not all cameras start, try adjusting device numbers or checking camera connections; try connecting only one camera per docking station; some computers may support only 1-2 external cameras when USB ports share the same bus, consider changing the computer).
2. Press `Space Bar` to start recording data and teleoperating the robotic arms to complete the target task.
3. If the demonstration opration is not acceptable, press `q` to discard the current recording process.
4. After task completion:
    - Wait to collect the specified number of frames (the number of frames actually spent to complete the task should be as close as possible to the maximum collection frames) or press `s` to save the data immediately.
    - Data will be automatically saved upon collecting all frames. After saving, the robotic arms will automatically return to the initial position.
5. Data will be saved to the `data/raw/<task_name>` folder in the current directory by default. Each collected episode data includes:
    - `mp4`: videos of the cameras
    - `low_dim.json`: the robots' states
    - `meta.json`: number of frames and collecting fps
    - `timestamps.json`: timestamp of each frame (low_dim data and images)

**Additional Suggestions:**

1. Try to ensure the task are completed **just before reaching the maximum frame count**, i.e., do not end the action too early.

2. The robotic arm movement speed should not be too fast; otherwise, the collected information will be sparse, and the image quality will not be high.

3. It is recommended to store the collected task data folder <task_name> in the same directory structure on a **portable hard drive** as a backup.

## Data Replay

### Modify Default Configurations

The defualt configurations are `play/airbot_play_with_usbcams.yaml` and `tok/airbot_tok.yaml` in `configurations/basic_configs/example/robot/airbots/` folder. The former is used for one long airbot_plays with a G2 gripper while the latter is used for two long airbot_plays with G2 grippers. You can modify them according to your usage.

### Starting Data Replay

> Data replay can be used to verify if there are issues with collected data, init states of the environment, etc. (requires prior [Environment Setup](#environment-setup)).

The data replay command and its parameters are as follows:

```bash
python3 control_robot.py replay \
  --robot-path configurations/basic_configs/example/robot/airbots/play/airbot_play.yaml \
  --root data \
  --repo-id raw/example \
  --fps 25 \
  --num-episodes 1 \
  --start-episode 0 \
  --num-rollouts 50
```

**Parameter explanation:**

  - `--robot-path`: path to the robot yaml file used to instantiate the robot
  - `--fps`: frames per second
  - `--root`: root directory where the datasets are stored
  - `--repo-id`: dataset identifier
  - `--num-episodes`: number of episodes to record, defaulting to 1
  - `--start-episode`: index of the first episode to replay
  - `--num-rollouts`: number of times to replay the episode, defaulting to 50

After executing the command above, you can see the instruction ``Press Enter to replay. ...`` in the terminal. Then press `Enter` to replay. And you will see the same prompt after the current replay ends.
