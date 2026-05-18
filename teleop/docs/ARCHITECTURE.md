# 项目架构 / Architecture

本项目是把分散的三个上游仓库合并为一个可迁移工作集。
本文档解释**模块边界 / 进程边界 / 数据流向 / 同步机制**。

## 1. 进程拓扑

```
                   ┌────────────────────────────────┐
                   │   scripts/record_delete.py     │  ← 主进程 (imitall env)
                   │   或 record.py / record_intrpt │
                   └──────┬──────────────────┬──────┘
              subprocess.Popen          subprocess.Popen
                 (imitall env)           (xhand_tele_env)
                          │                       │
                          ▼                       ▼
         imitate_all/record_4_rgb_cam.py    teleop_pkg/receive_from_vision_pro.py
         ─────────────────────────────       ────────────────────────────────────
         · 4× cv2.VideoCapture                · XHandTeleOps("config.yaml")
         · airbot_py.airbot_mmk2.AirbotMMK2    · Vision Pro → retarget → XHand
         · 异步 ImageSaver 线程池              · 帧循环：读手位 + 推目标 + 录制
         · 录制 episode_N 目录                 · 帧循环输出 frames 列表
         · 调 airbot_data.io.save_bson         · 自己写 xhand_control_data.bson
         · 落到 ./data/raw/example/            · 落到 ./xhand_control_data.bson
                          │                       │
                          └──── wait() ────┬──────┘
                                           │
                                  record_delete.copy()
                                           │
                                           ▼
                          /media/<drive>/data/action<K>/
                              └── episode_<N>/
                                  ├── camera0_head/  (frame_000000.jpg ...)
                                  ├── camera1_left_wrist/
                                  ├── camera2_right_wrist/
                                  ├── camera3_third_view/
                                  ├── episode_0.bson            ← 本体动作 + 观测
                                  └── xhand_control_data.bson   ← 灵巧手动作 + 观测
```

## 2. 数据结构

### `episode_0.bson`（机器人本体）

由 `airbot_data.io.save_bson` 写出，顶层是单个 BSON document：

```text
{
  "id":        <int>,
  "timestamp": <ISO 时间>,
  "metadata": {
      "topics": { "/observation/left_arm/joint_state": {...}, ... }
  },
  "data": {
      "/observation/left_arm/joint_state":  [frame_0, frame_1, ...],
      "/observation/right_arm/joint_state": [...],
      "/observation/head/joint_state":      [...],
      "/observation/spine/joint_state":     [...],
      "/observation/left_arm/pose":         [...],
      "/observation/right_arm/pose":        [...],
      "/action/left_arm/joint_state":       [...],
      "/action/right_arm/joint_state":      [...],
      "/action/head/joint_state":           [...],
      "/action/spine/joint_state":          [...]
  }
}
```

### `xhand_control_data.bson`（灵巧手）

由 `receive_from_vision_pro.py` 自己拼装：

```text
{
  "frames": [
    { "t": <float, 秒，相对帧 0>,
      "action":      { "left_hand": [12 个关节], "right_hand": [12 个关节] },
      "observation": { "left_hand": [12 个关节], "right_hand": [12 个关节] }
    },
    ... 共 N 帧
  ]
}
```

> ⚠️ 默认 `receive_from_vision_pro.py` 写的是相对时间 `t`（从录制起点开始算）。
> 如果你需要 epoch 时间戳，改用 `receive_from_vision_pro_timestemp.py`，
> 它会把同一帧的 perf_counter 也存进 BSON。

### `frame_*.jpg`

`record_4_rgb_cam.py` 用 `ImageSaver` 线程池异步写，命名格式
`frame_{frame_index:06d}.jpg`，质量 95，编码 JPEG。

## 3. 同步机制

| 阶段 | 当前实现 | 备注 |
| ---- | -------- | ---- |
| 数采启动 | 两个子进程独立启动，**没有同步握手** | 改进：互写 `start_recording` 文件，sync_helper.py 已经写好原语只是没接进来 |
| 数采运行 | 两侧各自 20 fps 自循环 | 由 `time.perf_counter` 校时，没有跨进程对齐 |
| 数采结束 | 都靠用户在两个窗口分别按 `s` | 改进：一处按键 broadcast |
| **回放启动** | **`replay.py` 已有 READY/START 握手** | 子进程往 stdout 打 `READY`，主进程等两路都 READY 后等用户回车，再 `START` 一起触发 |
| 回放运行 | 各自按 BSON 里的 `t` 字段重放 | 没有运行时对齐，依赖录制时的时间一致性 |

## 4. 路径策略

所有「外层启动器」都用：

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

来锚定项目根，子进程的脚本路径、源数据路径都从 `PROJECT_ROOT` 拼出来。
**`PYTHON_PATH` 仍然是 conda env 绝对路径**，因为这跟具体机器的安装位置强相关，
不适合自动推断。

`record_4_rgb_cam.py` 和 `receive_from_vision_pro.py` 在文件顶部都做了
`os.chdir(os.path.dirname(os.path.abspath(__file__)))`，所以它们的输出目录是
**相对于自己所在文件的位置**，搬到本仓库后产物分别落在：

| 源数据 | 落到 |
| ------ | ---- |
| 本体 4 路相机 + episode_0.bson | `imitate_all/data/raw/example/episode_0/` |
| xhand_control_data.bson       | `teleop_pkg/xhand_control_data.bson`      |

然后 `record_delete.py:copy()` 再把它们搬到 `ARCHIVE_ROOT/episode_{order}/`。

## 5. 模块图（导入关系）

```
scripts/record_delete.py     ──spawn──▶ imitate_all/record_4_rgb_cam.py
                                          ├── habitats.common.robot_devices.utils
                                          ├── habitats.common.utils.utils
                                          ├── data_process.dataset.raw_dataset
                                          ├── robots.common (make_robot_from_yaml)
                                          ├── airbot_data.io (pip 包)
                                          └── airbot_py.airbot_mmk2 (pip 包)

scripts/record_delete.py     ──spawn──▶ teleop_pkg/receive_from_vision_pro.py
                                          ├── xhand_tele_ops (wheel 私有包)
                                          ├── bson (pymongo)
                                          └── pynput (键盘监听)

scripts/replay.py            ──spawn──▶ imitate_all/mmk_replay.py
                                          └── airbot_py.airbot_mmk2

scripts/replay.py            ──spawn──▶ teleop_pkg/control_from_bson.py
                                          └── xhand_tele_ops

data_tools/validate_data_consistency.py     ← stand-alone
data_tools/swap_action_observation_bson.py  ← stand-alone
data_tools/bson_to_json_converter.py        ← stand-alone

video_tools/*                ← stand-alone (依赖 ffmpeg + opencv + imageio)
camera_tools/*               ← stand-alone (依赖 opencv + pyapriltags)
```
