# coding: utf-8
"""
实际数采主程序 —— 同时拉起：
  - 机器人本体 + 4 相机的数据录制   (imitate_all/record_4_rgb_cam.py)
  - 灵巧手 Vision Pro 遥操与记录    (teleop_pkg/receive_from_vision_pro.py)
两个子进程都结束后，调用 copy() 把本次 episode 的产物搬运到归档目录并清空源。

⚠️ 仅做了「把绝对路径替换为项目内相对路径」的迁移改造，其它逻辑保持与原版一致。
    如需修复参数解析顺序、进程联动退出、--order 校验等问题，请参见项目根目录
    README.md 的「建议改进」一节。
"""
import subprocess
import os
import shutil
import time
import argparse
import sys
from pathlib import Path

# ----------------------------------------------------------------------------- 
# 路径配置（迁移时只需要改这一块） 
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 两个 conda 环境的 python 解释器（按部署机器实际情况修改）
ROBOT_PYTHON_PATH = "/home/slam/miniconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH  = "/home/slam/miniconda3/envs/xhand_tele_env/bin/python"

# 真正的录制 / 遥操脚本（位于本项目中）
ROBOT_SCRIPT = str(PROJECT_ROOT / "imitate_all" / "record_4_rgb_cam.py")
HAND_SCRIPT  = str(PROJECT_ROOT / "teleop_pkg"  / "receive_from_vision_pro.py")
# HAND_SCRIPT_TIME = str(PROJECT_ROOT / "teleop_pkg" / "receive_from_vision_pro_timestemp.py")
# ↑ 切换为带时间戳的版本时，把上一行注释取消，把 HAND_SCRIPT 注释掉

# 录制完成后归档的目标根目录（外部存储，按需修改）
ARCHIVE_ROOT = "/media/slam/data/action6"

# 录制脚本会把数据保存在哪里 —— 来自 ROBOT_ARGS 的 --root + --repo-id：
#   ROBOT_SCRIPT 内部会 chdir 到自身所在目录，所以 data 目录是
#   imitate_all/data/raw/example
SOURCE_EPISODE_DIR = PROJECT_ROOT / "imitate_all" / "data" / "raw" / "example" / "episode_0"
SOURCE_BSON        = PROJECT_ROOT / "imitate_all" / "data" / "raw" / "example" / "episode_0.bson"
SOURCE_HAND_BSON   = PROJECT_ROOT / "teleop_pkg"  / "xhand_control_data.bson"

# -----------------------------------------------------------------------------
# 录制脚本的参数
# -----------------------------------------------------------------------------
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


def safe_remove(path):
    """安全删除文件或目录"""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"已删除目录: {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"已删除文件: {path}")
    except Exception as e:
        print(f"删除失败 {path}: {str(e)}")


def copy_and_remove(src, dst):
    """复制文件/文件夹后删除源"""
    try:
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"已复制目录: {src} 到 {dst}")
        else:
            shutil.copy2(src, dst)
            print(f"已复制文件: {src} 到 {dst}")
        return True
    except Exception as e:
        print(f"复制失败 {src}: {str(e)}")
        return False


def copy():
    parser = argparse.ArgumentParser(description='文件复制并删除源脚本')
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

    sources_to_remove = []

    # 1. 递归复制 episode_0 目录下的所有文件夹和文件
    if os.path.exists(source_episode_dir):
        for item in os.listdir(source_episode_dir):
            src_path = os.path.join(source_episode_dir, item)
            dst_path = os.path.join(episode_dir, item)
            if copy_and_remove(src_path, dst_path):
                sources_to_remove.append(src_path)
        sources_to_remove.append(source_episode_dir)
    else:
        print(f"错误: 源目录 {source_episode_dir} 不存在")
        return

    # 2. 复制 episode_0.bson 文件
    if os.path.exists(source_bson):
        dest_bson = os.path.join(episode_dir, os.path.basename(source_bson))
        if copy_and_remove(source_bson, dest_bson):
            sources_to_remove.append(source_bson)
    else:
        print(f"警告: 文件 {source_bson} 不存在")

    # 3. 复制 xhand_control_data.bson 文件
    if os.path.exists(source_bsond):
        dest_bsond = os.path.join(episode_dir, os.path.basename(source_bsond))
        if copy_and_remove(source_bsond, dest_bsond):
            sources_to_remove.append(source_bsond)
    else:
        print(f"警告: 文件 {source_bsond} 不存在")

    for path in sources_to_remove:
        safe_remove(path)

    print("所有文件复制并删除源操作完成！")


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
