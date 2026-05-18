# coding: utf-8
"""
早期版本的数采主程序：流程同 record_delete.py，但归档时是「复制不删源」。
保留作为参考，正式数采推荐使用 record_delete.py。
"""
import subprocess
import os
import shutil
import time
import argparse
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROBOT_PYTHON_PATH = "/home/slam/miniconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH  = "/home/slam/miniconda3/envs/xhand_tele_env/bin/python"

# 当前指向 4-cam 录制，如果想用 2-cam 版本，把下面这行改成
#   PROJECT_ROOT / "imitate_all" / "control_robot_bson_2cam.py"
ROBOT_SCRIPT = str(PROJECT_ROOT / "imitate_all" / "record_4_rgb_cam.py")
HAND_SCRIPT  = str(PROJECT_ROOT / "teleop_pkg"  / "receive_from_vision_pro.py")

ARCHIVE_ROOT = "/home/slam/Desktop/wzr/data_collection/action4"
SOURCE_EPISODE_DIR = PROJECT_ROOT / "imitate_all" / "data" / "raw" / "example" / "episode_0"
SOURCE_BSON        = PROJECT_ROOT / "imitate_all" / "data" / "raw" / "example" / "episode_0.bson"
SOURCE_HAND_BSON   = PROJECT_ROOT / "teleop_pkg"  / "xhand_control_data.bson"

ROBOT_ARGS = [
    "record",
    "--robot-path", "configurations/basic_configs/example/robot/airbots/mmk/airbot_com_mmk_demonstration_no_eef_bson.yaml",
    "--root", "data",
    "--repo-id", "raw/example",
    "--fps", "20",
    "--warmup-time-s", "1",
    "--num-frames-per-episode", "1000",
    "--reset-time-s", "1",
    "--num-episodes", "10000",
    "--start-episode", "0",
    "--num-image-writers-per-camera", "1"
]


def copy_dir_contents(src, dst):
    """递归复制目录中的所有内容"""
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)

        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
            print(f"已复制目录: {item} 从 {src}")
        else:
            shutil.copy2(src_path, dst_path)
            print(f"已复制文件: {item} 从 {src}")


def copy():
    parser = argparse.ArgumentParser(description='文件复制脚本')
    parser.add_argument('--order', type=int, required=True, help='episode序号')
    args = parser.parse_args()

    time.sleep(0.5)

    path_A = ARCHIVE_ROOT
    source_episode_dir = str(SOURCE_EPISODE_DIR)
    source_bson        = str(SOURCE_BSON)
    source_bsond       = str(SOURCE_HAND_BSON)

    episode_dir = os.path.join(path_A, f"episode_{args.order}")
    os.makedirs(episode_dir, exist_ok=True)
    print(f"已创建文件夹: {episode_dir}")

    if os.path.exists(source_episode_dir):
        copy_dir_contents(source_episode_dir, episode_dir)
    else:
        print(f"错误: 源目录 {source_episode_dir} 不存在")
        return

    if os.path.exists(source_bson):
        dest_bson = os.path.join(episode_dir, os.path.basename(source_bson))
        shutil.copy2(source_bson, dest_bson)
        print(f"已复制: {os.path.basename(source_bson)} 到 {episode_dir}")
    else:
        print(f"警告: 文件 {source_bson} 不存在")

    if os.path.exists(source_bsond):
        dest_bsond = os.path.join(episode_dir, os.path.basename(source_bsond))
        shutil.copy2(source_bsond, dest_bsond)
        print(f"已复制: {os.path.basename(source_bsond)} 到 {episode_dir}")
    else:
        print(f"警告: 文件 {source_bsond} 不存在")

    print("所有文件复制完成！")


def main():
    print("🔧 启动机器人控制程序...")
    robot_proc = subprocess.Popen([ROBOT_PYTHON_PATH, ROBOT_SCRIPT] + ROBOT_ARGS)

    print("🔧 启动灵巧手控制程序...")
    hand_proc = subprocess.Popen([HAND_PYTHON_PATH, HAND_SCRIPT])

    try:
        robot_proc.wait()
        hand_proc.wait()
    except KeyboardInterrupt:
        print("🔴 检测到中断，终止两个进程...")
        robot_proc.terminate()
        hand_proc.terminate()

    copy()


if __name__ == "__main__":
    main()
