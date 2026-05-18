"""
机器人控制工具。

用于录制数据集、回放录制的episode、在机器人上运行策略
并录制评估数据集，以及根据需要重新校准机器人。
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# from habitats.common.robot_devices.cameras.utils import prepare_cv2_imshow
# prepare_cv2_imshow()  这个方法是预热一下opencv的imshow，避免第一次显示时卡顿 既然不显示了就注释了

import argparse
import concurrent.futures
import json
import logging
import shutil
import time
import traceback
from pathlib import Path
from threading import Event
from functools import partial
import cv2
import tqdm
from omegaconf import DictConfig
from PIL import Image
from termcolor import colored

from habitats.common.robot_devices.utils import busy_wait
from habitats.common.utils.utils import init_logging

from typing import Optional, Callable, Dict
from data_process.dataset.raw_dataset import RawDataset
from robots.common import Robot, make_robot_from_yaml
import numpy as np
from airbot_data.io import save_bson

# USB摄像头相关导入
import threading
from queue import Queue
import time


class ImageSaver(threading.Thread):
    """图像保存器线程类，用于异步保存图像文件"""
    def __init__(self, name="cam_saver", quality=95):
        super().__init__()
        self.queue = Queue()
        self.quality = quality
        self.running = True
        self.name = name
        self.start()

    def save(self, path, image):
        """将图像添加到保存队列"""
        self.queue.put((path, image))

    def run(self):
        """线程运行函数，持续从队列中取出图像并保存"""
        while self.running or not self.queue.empty():
            try:
                path, img = self.queue.get(timeout=0.1)
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            except:
                continue

    def stop(self):
        """停止线程"""
        self.running = False
        self.join()

def save_image(img: np.ndarray, frame_index, images_dir: Path):
    """保存图像到指定目录"""
    img = Image.fromarray(img)
    path = images_dir / f"frame_{frame_index:06d}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def none_or_int(value):
    """将字符串转换为None或整数"""
    if value == "None":
        return None
    return int(value)


def log_control_info(robot, dt_s, episode_index=None, frame_index=None, fps=None):
    """记录控制信息，包括时间间隔和频率"""
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1/ dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # 总步骤时间（毫秒）及其频率
    log_dt("dt", dt_s)

    for name in robot.cameras:
        key = f"read_camera_{name}_dt_s"
        if key in robot.logs:
            log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


def is_headless():
    """检测Python是否在没有显示器的情况下运行"""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "尝试导入pynput时出错。切换到无头模式。"
            "因此，摄像头的视频流将不会显示，"
            "您将无法通过键盘更改控制流程。"
            "更多信息请参见下面的跟踪信息。\n"
        )
        traceback.print_exc()
        print()
        return True


def show_info_on_image(episode, fps, steps):
    """在图像上显示演示信息"""
    # 创建一个白色背景的图像 (height, width, channels)
    height, width = 400, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景

    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_up = 1
    font_scale_down = 6
    thickness_up = 2
    thickness_down = 5

    # 文字内容
    text_top = f"Episode:{episode}  FPS:{fps}"
    text_bottom = f"{steps}"

    # 计算文本大小，以便居中
    (text_width_top, text_height_top), _ = cv2.getTextSize(
        text_top, font, font_scale_up, thickness_up
    )
    (text_width_bottom, text_height_bottom), _ = cv2.getTextSize(
        text_bottom, font, font_scale_down, thickness_down
    )

    # 设置文本位置（使文字居中）
    x_top = (width - text_width_top) // 2
    y_top = int(height * 0.25)  # 上栏位置

    x_bottom = (width - text_width_bottom) // 2
    y_bottom = int(height * 0.75)  # 下栏位置

    # 在图像上添加文字
    cv2.putText(
        image, text_top, (x_top, y_top), font, font_scale_up, (0, 0, 255), thickness_up
    )
    cv2.putText(
        image,
        text_bottom,
        (x_bottom, y_bottom),
        font,
        font_scale_down,
        (0, 255, 0),
        thickness_down,
    )

    # 显示图像
    # cv2.imshow("演示信息", image)
    # RealTimeDisplay.imshow("演示信息", image)


########################################################################################
# 控制模式
########################################################################################
# === 初始化摄像头 ===
# 使用系统级符号链接，支持插拔相机不修改代码
camera_devices = {
    "camera_left_wrist": "/dev/camera_left",    # 左相机
    "camera_right_wrist": "/dev/camera_right",  # 右相机  
    "camera_third_view": "/dev/camera_high",     # 第三视角相机
    "camera_head": "/dev/camera_head"          # 头相机
}

caps = {}
savers = {}

import subprocess

# 初始化 USB 相机
for camera_name, device_path in camera_devices.items():
    # 检查设备文件是否存在
    if not os.path.exists(device_path):
        print(f"[{camera_name}] ❌ 设备不存在: {device_path}")
        # 如果找不到设备，尝试刷新udev规则再试一次
        try:
            print(f"尝试刷新udev规则以重新识别设备: {device_path}")
            subprocess.run(['udevadm', 'control', '--reload-rules'], check=True)
            subprocess.run(['udevadm', 'trigger'], check=True)
            print("✅ 已刷新udev规则，重新检测设备...")
            time.sleep(1)  # 等待udev生效
        except Exception as e:
            print(f"❌ 刷新udev规则失败: {e}")
        # 再次检查设备
        if not os.path.exists(device_path):
            print(f"[{camera_name}] ❌ 刷新udev后设备仍然不存在: {device_path}")
            continue

    cap = cv2.VideoCapture(device_path)  # 直接使用设备路径
    if not cap.isOpened():
        print(f"[{camera_name}] ❌ 无法打开设备: {device_path}")
        continue
        
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    caps[camera_name] = cap
    savers[camera_name] = ImageSaver(name=f"saver_{camera_name}")


# 已删除Realsense初始化和相关测试代码

for cam_id, cap in caps.items():
    ret, frame = cap.read()
    if not ret:
        print(f"[video{cam_id}] ⚠️ 采图失败")
        continue

for cam_id, cap in caps.items():
    ret, frame = cap.read()
    if not ret:
        print(f"[video{cam_id}] ⚠️ 采图失败")
        continue

# 已删除 realsense_pipeline 检查和等待帧部分

def teleoperate(
    robot: Robot, fps: Optional[int] = None, teleop_time_s: Optional[float] = None
):
    """远程操作机器人"""
    # TODO(rcadene): 添加记录日志的选项
    if not robot.is_connected:
        robot.connect()

    start_teleop_t = time.perf_counter()
    while True:
        start_loop_t = time.perf_counter()
        robot.teleop_step()

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        if (
            teleop_time_s is not None
            and time.perf_counter() - start_teleop_t > teleop_time_s
        ):
            break


def record(
    robot: Robot,
    root: str,
    repo_id: str,
    fps: Optional[int] = None,
    episode_time_s=None,
    num_frames_per_episode=None,
    warmup_time_s=2,
    reset_time_s=5,
    num_episodes=50,
    video=True,
    num_image_writers_per_camera=4,
    force_override=False,
    start_episode=-1,
    policy: Optional[Callable] = None,
    hydra_cfg: Optional[DictConfig] = None,
    run_compute_stats=True,
    push_to_hub=True,
    tags=None,
    *args,
    **kwargs,
):
    """录制机器人数据"""
    # 允许在特定时间或帧数内录制数据
    assert (episode_time_s, num_frames_per_episode).count(None) == 1
    if episode_time_s is None:
        episode_time_s = np.inf
    elif num_frames_per_episode is None:
        num_frames_per_episode = np.inf

    if not video:
        raise NotImplementedError()

    local_dir = Path(root) / repo_id  # 数据/原始数据
    if local_dir.exists() and force_override:
        shutil.rmtree(local_dir)

    # episodes_dir = local_dir / "episodes"
    episodes_dir = local_dir
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # 恢复数据录制的逻辑
    raw_start_episode = start_episode
    rec_info_path = episodes_dir / "data_recording_info.json"
    if start_episode < 0:
        start_episode += 1
        if rec_info_path.exists():
            with open(rec_info_path) as f:
                rec_info = json.load(f)
            episode_index = rec_info["last_episode_index"] + 1 + start_episode
        else:
            if start_episode < 0:
                logging.warning(
                    "未找到数据录制信息。从episode 0开始。"
                )
            episode_index = 0
        start_episode = episode_index
    else:
        episode_index = start_episode

    if is_headless():
        logging.info(
            "检测到无头环境。屏幕摄像头显示和键盘输入将不可用。"
        )

    def show_cameras(robot: Robot):
        """显示摄像头画面"""
        observation = robot.capture_observation()
        start = time.time()
        image_keys = [key for key in observation if "image" in key]
        for key in image_keys:
            image = observation[key]["data"]
            # print(key, image.shape)
            # cv2.imshow(key.split("/")[-1], image)
            #     displayer.show_frame(image[:, :, ::-1])
            #     displayer.clock.tick(30)
            # print("show_cameras time:", time.time() - start)
        cv2.waitKey(1)
        #     RealTimeDisplay.imshow(key, image[:, :, ::-1])
        # RealTimeDisplay.waitKey(1)

    # 允许在录制episode或重置环境时提前退出，
    # 通过点击右箭头键'->'。这可能需要sudo权限
    # 来允许您的终端监控键盘事件。

    class KeyboardHandler(object):
        """键盘处理器类，处理用户按键输入"""
        def __init__(self) -> None:
            self.exit_early: bool = False
            self._stop_recording: bool = False
            self.record_event: Event = Event()
            self._is_waiting_start_recording: bool = False

        def show_instruction(self):
            """显示操作指令"""
            print(
                """(按键说明:
                '空格键' 开始录制数据,
                's键' 立即保存当前episode,
                'ESC键' 退出程序.
            )"""
            )

        def wait_start_recording(self):
            """等待开始录制"""
            self._is_waiting_start_recording = True
            self.record_event.wait()
            self.record_event.clear()
            self._is_waiting_start_recording = False
            return False, self._stop_recording

        def wait_and_show_camera(self, robot: Robot):
            """等待开始录制并显示摄像头画面"""
            self._is_waiting_start_recording = True
            while not self.record_event.is_set():
                show_cameras(robot)
            self.record_event.clear()
            self._is_waiting_start_recording = False
            return False, self._stop_recording

        def is_recording(self):
            """检查是否正在录制"""
            return not self._is_waiting_start_recording

        def set_record_event(self):
            """设置录制事件"""
            if not self.record_event.is_set():
                self.record_event.set()
                return True
            else:
                print("\n 出现错误，录制数据已经开始")
                return False

        def on_press(self, key, robot: Robot = None):
            """按键处理函数"""
            try:
                print()
                if key == keyboard.Key.space:
                    if (not self.is_recording()) and self.set_record_event():
                        robot.enter_passive_mode()
                        print("开始录制数据")
                    else:
                        print(
                            "仍在录制数据，请等待或按's'立即保存..."
                        )
                elif key == keyboard.Key.esc:
                    print("停止数据录制...")
                    self.exit_early = True
                    self._stop_recording = True
                    if not self.is_recording():
                        self.set_record_event()
                elif key.char == "s":
                    if self.is_recording():
                        print("立即保存当前episode")
                        self.exit_early = True
                    else:
                        print("未在录制数据，无需保存")
                else:
                    print(
                        "未知按键:",
                        key,
                        f"类型:{type(key)}, 字符串值 {str(key)}",
                    )
            except Exception as e:
                print(
                    "未知按键:",
                    key,
                    f"类型:{type(key)}, 字符串值 {str(key)}",
                )

        @property
        def stop_recording(self):
            """获取停止录制状态"""
            return self._stop_recording

    keyer = KeyboardHandler()
    # 仅在非无头环境中导入pynput
    if not is_headless():
        from pynput import keyboard

        listener = keyboard.Listener(on_press=partial(keyer.on_press, robot=robot))
        listener.start()

    # 使用线程保存图像以达到高帧率（30及以上）
    # 使用`with`语句在发生异常时平滑退出
    futures = []
    camera_num = len(robot.cameras)
    num_image_writers = max(1, num_image_writers_per_camera) * max(1, len(robot.cameras))
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_image_writers
    ) as executor:

        # 向用户显示指令
        keyer.show_instruction()
        # 开始录制所有episode

        while episode_index < num_episodes:

            # 创建episode目录
            episode_dir = episodes_dir / f"episode_{episode_index}"
            episode_dir.mkdir(exist_ok=True)


            # cam_ids = [0, 2, 6]  # 你可以改为你实际的摄像头编号
            # caps = {}
            # savers = {}
            # savers = {}
            # for cam_id in cam_ids:
            #     cap = cv2.VideoCapture(cam_id)
            #     if not cap.isOpened():
            #         print(f"[video{cam_id}] ❌ 无法打开")
            #         break
            #     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            #     caps[cam_id] = cap
            #     savers[cam_id] = ImageSaver(name=f"saver_cam{cam_id}")
            
            # === 创建保存目录 ===
            save_root = episode_dir
            for camera_name in camera_devices.keys():
                (save_root / camera_name).mkdir(parents=True, exist_ok=True)
                
            # 按键开启
            logging.info(
                f"Press 'Space Bar' to start recording episode {episode_index}."
            )
            if is_headless():
                _, stop_record = keyer.wait_start_recording()
            else:
                _, stop_record = keyer.wait_and_show_camera(robot)
            if stop_record:
                break
            logging.info(f"Start recording episode {episode_index}")



            frame_index = 0
            timestamp = 0
            start_episode_t = time.perf_counter()
            # 录制一个episode
            bson_dict: Dict[str, Dict[str, list]] = {
                "id": "734ad1c8-66ee-4479-b3cb-41d16c9b2e22",
                "timestamp": 1734076528859,
                # "timestamp": time.perf_counter(),
                "metadata": {
                    "driver_version": "1.0.0",
                    "operator": "manual",
                    "station_id": "3784D4BA-87AF-47E7-B86D-42CA1904AA77",
                    "task": "example",
                    "topics": {
                        "/action/head/joint_state": {
                            "description": "",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/action/spine/joint_state": {
                            "description": "",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/action/left_arm/joint_state": {
                            "description": "replay",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        # "/action/left_arm_eef/joint_state": {
                        #     "description": "replay",
                        #     "type": "jointstate",
                        #     "sn": "",
                        #     "firmware_version": "0.0.0",
                        # },
                        "/action/right_arm/joint_state": {
                            "description": "replay",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        # "/action/right_arm_eef/joint_state": {
                        #     "description": "replay",
                        #     "type": "jointstate",
                        #     "sn": "",
                        #     "firmware_version": "0.0.0",
                        # },
                        # "/action/eef/pose": {
                        #     "description": "",
                        #     "type": "jointstate",
                        #     "sn": "",
                        #     "firmware_version": "0.0.0",
                        # },
                        # "/action/base/joint_state": {
                        #     "description": "slamtec-athena",
                        #     "type": "jointstate",
                        #     "sn": "",
                        #     "firmware_version": "0.0.0",
                        # },
                        "/observation/head/joint_state": {
                            "description": "",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/observation/spine/joint_state": {
                            "description": "",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/observation/left_arm/joint_state": {
                            "description": "airbot-play-short",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/observation/right_arm/joint_state": {
                            "description": "airbot-play-short",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/observation/left_arm_eef/joint_state": {
                            "description": "airbot-play-short",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/observation/right_arm_eef/joint_state": {
                            "description": "airbot-play-short",
                            "type": "jointstate",
                            "sn": "",
                            "firmware_version": "0.0.0",
                        },
                        "/observation/left_arm/pose": {
                            "description": "",
                            "type": "pose",
                        },
                        "/observation/right_arm/pose": {
                            "description": "",
                            "type": "pose",
                        },
                        # "/observation/base/joint_state": {
                        #     "description": "slamtec-athena",
                        #     "type": "jointstate",
                        #     "sn": "",
                        #     "firmware_version": "0.0.0",
                        # },

                        ####################
                        #只注释这个
                        ####################

                        # "/images/head_camera": {
                        #     "description": "DSJ-2062-309",
                        #     "type": "image",
                        #     "width": 640,
                        #     "height": 480,
                        #     "encoding": "H264",
                        #     "distortion_model": None,
                        #     "distortion_params": None,
                        #     "intrinsics": None,
                        #     "fov": 120.0,
                        #     "start_time": 1733377253041,
                        # },
                        # "/images/left_camera": {
                        #     "description": "DSJ-2062-309",
                        #     "type": "image",
                        #     "width": 640,
                        #     "height": 480,
                        #     "encoding": "H264",
                        #     "distortion_model": None,
                        #     "distortion_params": None,
                        #     "intrinsics": None,
                        #     "fov": 120.0,
                        #     "start_time": 1733377253041,
                        # },
                        # "/images/right_camera": {
                        #     "description": "DSJ-2062-309",
                        #     "type": "image",
                        #     "width": 640,
                        #     "height": 480,
                        #     "encoding": "H264",
                        #     "distortion_model": None,
                        #     "distortion_params": None,
                        #     "intrinsics": None,
                        #     "fov": 120.0,
                        #     "start_time": 1733377253041,
                        # },
                    },
                    # "version": "1.2.1",
                    "version": "1.2.2",
                },
                "data": {},
            }
                # 修改后的记录循环部分
            while timestamp < episode_time_s:
                start_loop_t = time.perf_counter()

                # 保存USB相机数据
                for camera_name, cap in caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"[{camera_name}] ⚠️ 采图失败")
                        continue
                    save_path = save_root / camera_name / f"frame_{frame_index:06d}.jpg"
                    savers[camera_name].save(save_path, frame)


                # 捕获机器人状态
                observation: dict = robot.capture_observation()
                
                # # 仅当配置了头部相机时才处理头部相机数据
                # if hasattr(robot.config, 'cameras') and 'head_camera' in robot.config.cameras:
                #     head_cam_keys = [k for k in observation if "head_camera" in k]
                #     if head_cam_keys:
                #         head_cam_key = head_cam_keys[0]
                #         head_cam_img = observation[head_cam_key]["data"]
                        
                #         # 创建头部相机目录并保存图像
                #         head_cam_dir = save_root / "camera_0"
                #         head_cam_dir.mkdir(parents=True, exist_ok=True)
                #         head_cam_path = head_cam_dir / f"frame_{frame_index:06d}.jpg"
                #         cv2.imwrite(str(head_cam_path), head_cam_img)
                        
                #         # 从observation中移除头部相机数据，避免被保存到bson中
                #         observation.pop(head_cam_key, None)
                #     else:
                #         logging.warning("Head camera configured but no data found in observation")

                # # 显示当前图像 注释掉可以加快帧率，如果不注释掉的话会影响帧率
                # if not is_headless():
                #     # 优先显示USB相机图像
                #     if caps:
                #         # 显示第一个USB相机的图像
                #         cam_id = list(caps.keys())[0]
                #         cap = caps[cam_id]
                #         ret, frame = cap.read()
                #         if ret:
                #             cv2.imshow(f"USB Camera {cam_id}", frame)
                #             show_info_on_image(episode_index, fps, frame_index + 1)
                #             cv2.waitKey(1)
                #     else:
                #         # 如果没有USB相机，尝试显示机器人状态中的图像
                #         image_keys = [key for key in observation if "image" in key]
                #         if image_keys:
                #             for key in image_keys:
                #                 image = observation[key]["data"]
                #                 cv2.imshow(key.split("/")[-1], image)
                #             show_info_on_image(episode_index, fps, frame_index + 1)
                #             cv2.waitKey(1)
                
                # 构造episode dict
                for key, value in observation.items():
                    if key not in bson_dict["data"]:
                        bson_dict["data"][key] = []
                    bson_dict["data"][key].append(value)

                frame_index += 1
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / fps - dt_s)
                dt_s = time.perf_counter() - start_loop_t
                log_control_info(robot, dt_s, fps=fps)
                timestamp = time.perf_counter() - start_episode_t
                
                if keyer.exit_early:
                    keyer.exit_early = False
                    break
                elif frame_index >= num_frames_per_episode:
                    break
            timestamp = 0
            start_vencod_t = time.perf_counter()
            save_bson(episodes_dir / f"episode_{episode_index}.bson", bson_dict)
            # 保存录制信息
            rec_info = {
                "last_episode_index": episode_index,
            }
            with open(rec_info_path, "w") as f:
                json.dump(rec_info, f)

            if not keyer.stop_recording:
                # 在执行器完成时开始重置环境
                logging.info("重置环境")
                # say("重置环境")
                robot.reset()

            # 检查当前episode是否是最后一个
            is_last_episode = keyer.stop_recording or (
                episode_index == (num_episodes - 1)
            )
            # 如有必要则等待
            with tqdm.tqdm(total=reset_time_s, desc="Waiting") as pbar:
                while timestamp < reset_time_s and not is_last_episode:
                    time.sleep(1)
                    timestamp = time.perf_counter() - start_vencod_t
                    pbar.update(1)
                    if keyer.exit_early:
                        keyer.exit_early = False
                        break

            # 更新episode索引
            episode_index += 1
            if is_last_episode:
                break

    # if not is_headless():
    #     cv2.destroyAllWindows()

    logging.info("退出程序")


def replay(
    robot: Robot,
    root: str,
    repo_id: str,
    start_episode: int,
    num_episodes: int,
    num_rollouts: int,
    fps: int,
):
    """回放录制的数据"""
    # TODO(rcadene): 添加记录日志的选项
    local_dir = Path(root) / repo_id
    assert local_dir.exists(), f"本地目录未找到: {local_dir}"
    logging.info(f"从 {local_dir} 加载数据集")
    dataset = RawDataset(repo_id, root=root)

    for episode_index in range(start_episode, start_episode + num_episodes):
        logging.info(f"回放episode {episode_index}")

        dataset.warm_up_episodes([start_episode], low_dim_only=True)

        meta = dataset.raw_data[start_episode]["meta"]
        low_dim = dataset.raw_data[start_episode]["low_dim"]

        # 连接不同的机械臂

        for roll in range(num_rollouts):
            # 使用轨迹模式移动到第一帧
            action = robot.low_dim_to_action(low_dim, 0)
            logging.info("移动到episode的第一帧")
            robot.enter_traj_mode()
            robot.send_action(action)
            # time.sleep(1)
            key = input(
                f"按Enter键回放episode {episode_index}，编号 {roll}，或按'x和Enter'退出当前episode，或按'z和Enter'退出所有episode"
            )
            if key in ["z", "Z"]:
                return
            elif key in ["x", "X"]:
                break
            logging.info("回放episode")
            robot.enter_servo_mode()
            for i in tqdm.tqdm(range(meta["length"])):
                start_episode_t = time.perf_counter()
                action = robot.low_dim_to_action(low_dim, i)
                # print("当前关节:", robot.get_low_dim_data()["observation/arm/joint_state"])
                # print("目标动作:", action)
                robot.send_action(action)
                dt_s = time.perf_counter() - start_episode_t
                busy_wait(1.0 / fps - dt_s)
                dt_s = time.perf_counter() - start_episode_t
                # log_control_info(robot, dt_s, fps=fps)


def cleanup():
    """清理资源"""
    # 关闭 USB 相机
    for cam_id, cap in caps.items():
        cap.release()

    # 停止保存器
    for saver in savers.values():
        saver.stop()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="mode", required=True)

        # 为所有子解析器设置通用选项
        base_parser = argparse.ArgumentParser(add_help=False)
        base_parser.add_argument(
            "--robot-path",
            type=str,
            help="用于实例化机器人的yaml文件路径，使用`make_robot`工厂函数。",
        )
        base_parser.add_argument(
            "--robot-overrides",
            type=str,
            nargs="*",
            help="覆盖配置值的任何键值参数（使用点号进行嵌套覆盖）",
        )
        base_parser.add_argument(
            "--fps",
            type=none_or_int,
            help="每秒帧数（设置为None以禁用）",
        )

        dataused_parser = argparse.ArgumentParser(add_help=False)
        dataused_parser.add_argument(
            "--root",
            type=Path,
            default="data",
            help="数据集本地存储的根目录，位于'{root}/{repo_id}'（例如'data/hf_username/dataset_name'）。",
        )
        dataused_parser.add_argument(
            "--repo-id",
            type=str,
            default="raw/example",
            help="数据集标识符。按照惯例，它应该匹配'{hf_username}/{dataset_name}'（例如`lerobot/test`）。",
        )
        dataused_parser.add_argument(
            "--num-episodes", type=int, default=1, help="要录制的episode数量。"
        )
        dataused_parser.add_argument(
            "--start-episode",
            type=int,
            help="要录制的第一个episode的索引；值<0表示从'data_recording_info.json'获取最后一个episode索引并添加(value + 1)。",
        )

        parser_teleop = subparsers.add_parser("teleoperate", parents=[base_parser])

        parser_record = subparsers.add_parser(
            "record", parents=[base_parser, dataused_parser]
        )
        parser_record.add_argument(
            "--warmup-time-s",
            type=int,
            default=10,
            help="在开始数据收集之前等待的秒数。它允许机器人设备预热和同步。",
        )
        parser_record_length = parser_record.add_mutually_exclusive_group(required=True)
        parser_record_length.add_argument(
            "--episode-time-s",
            type=int,
            help="每个episode的数据记录秒数。",
        )
        parser_record_length.add_argument(
            "--num-frames-per-episode",
            type=int,
            help="每个episode的数据记录帧数。",
        )
        parser_record.add_argument(
            "--reset-time-s",
            type=int,
            default=0,
            help="每个episode后重置环境的秒数。",
        )
        parser_record.add_argument(
            "--run-compute-stats",
            type=int,
            default=1,
            help="默认情况下，在数据收集结束时运行数据统计计算。计算密集型，不要求只回放一个episode。",
        )
        parser_record.add_argument(
            "--push-to-hub",
            type=int,
            default=1,
            help="将数据集上传到Hugging Face Hub。",
        )
        parser_record.add_argument(
            "--tags",
            type=str,
            nargs="*",
            help="在Hub上为您的数据集添加标签。",
        )
        parser_record.add_argument(
            "--num-image-writers-per-camera",
            type=int,
            default=4,
            help=(
                "每个摄像头在磁盘上写入帧作为jpg图像的线程数。"
                "太多线程可能会导致主进程阻塞，从而导致不稳定的远程操作fps。"
                "线程太少可能会导致相机fps低。"
            ),
        )
        parser_record.add_argument(
            "--force-override",
            type=int,
            default=0,
            help="默认情况下，数据记录是继续的。当设置为1时，删除本地目录并从头开始数据记录。",
        )
        parser_record.add_argument(
            "-p",
            "--pretrained-policy-name-or-path",
            type=str,
            help=(
                "Hub上托管的模型repo ID或包含使用`Policy.save_pretrained`保存的权重的目录。"
            ),
        )
        parser_record.add_argument(
            "--policy-overrides",
            type=str,
            nargs="*",
            help="覆盖配置值的任何键值参数（使用点号进行嵌套覆盖）",
        )

        parser_replay = subparsers.add_parser(
            "replay", parents=[base_parser, dataused_parser]
        )
        parser_replay.add_argument(
            "--num-rollouts",
            type=int,
            default=50,
            help="每个episode回放的次数。",
        )

        args = parser.parse_args()

        init_logging()

        control_mode = args.mode
        robot_path = args.robot_path
        robot_overrides = args.robot_overrides
        kwargs = vars(args)
        del kwargs["mode"]
        del kwargs["robot_path"] # airbot_com_mmk_demonstration_bson.yaml 
        del kwargs["robot_overrides"]

        robot = make_robot_from_yaml(robot_path, robot_overrides)

        if control_mode == "teleoperate":
            teleoperate(robot, **kwargs)
        elif control_mode == "record":
            record(robot, **kwargs)
        elif control_mode == "replay":
            replay(robot, **kwargs)
        robot.exit()

    except KeyboardInterrupt:
        print("\n程序被用户中断")

    except Exception as e:
        print(f"程序异常: {str(e)}")
        traceback.print_exc()
    finally:
        # 确保所有资源正确释放
        cleanup()
        cv2.destroyAllWindows()
        if 'robot' in locals():
            robot.exit()
        # # 确保所有摄像头线程停止
        # if 'usb_cameras' in locals():
        #     for cam in usb_cameras:
        #         cam.stop()