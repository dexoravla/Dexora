# coding: utf-8
"""
带 ProcessManager 进程管理的数采主程序。
相比 record_delete.py，使用 psutil 在 Ctrl-C / SIGTERM 时递归终止所有子进程。
"""
import subprocess
import os
import shutil
import time
import argparse
import signal
import sys
from pathlib import Path
import psutil

# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROBOT_PYTHON_PATH = "/home/slam/miniconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH  = "/home/slam/miniconda3/envs/xhand_tele_env/bin/python"

ROBOT_SCRIPT = str(PROJECT_ROOT / "imitate_all" / "record_4_rgb_cam.py")
HAND_SCRIPT  = str(PROJECT_ROOT / "teleop_pkg"  / "receive_from_vision_pro.py")

ARCHIVE_ROOT = "/home/slam/Desktop/wzr/data_collection/action2"
SOURCE_EPISODE_DIR = PROJECT_ROOT / "imitate_all" / "data" / "raw" / "example" / "episode_0"
SOURCE_BSON        = PROJECT_ROOT / "imitate_all" / "data" / "raw" / "example" / "episode_0.bson"
SOURCE_HAND_BSON   = PROJECT_ROOT / "teleop_pkg"  / "xhand_control_data.bson"

ROBOT_ARGS = [
    "record",
    "--robot-path", "configurations/basic_configs/example/robot/airbots/mmk/airbot_com_mmk_demonstration_bson.yaml",
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


class ProcessManager:
    """进程管理类，确保所有子进程都能被正确终止"""
    def __init__(self):
        self.processes = []
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\n🔴 收到中断信号({signum})，正在终止所有子进程...")
        self.terminate_all()
        sys.exit(0)

    def add_process(self, proc):
        self.processes.append(proc)

    def terminate_all(self):
        """终止所有子进程及其子进程"""
        for proc in self.processes:
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                proc.terminate()
            except psutil.NoSuchProcess:
                pass
        time.sleep(1)


def safe_remove(path):
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

    try:
        if os.path.exists(source_episode_dir):
            for item in os.listdir(source_episode_dir):
                src_path = os.path.join(source_episode_dir, item)
                dst_path = os.path.join(episode_dir, item)
                if copy_and_remove(src_path, dst_path):
                    sources_to_remove.append(src_path)
            sources_to_remove.append(source_episode_dir)

        for src_file in [source_bson, source_bsond]:
            if os.path.exists(src_file):
                dst_file = os.path.join(episode_dir, os.path.basename(src_file))
                if copy_and_remove(src_file, dst_file):
                    sources_to_remove.append(src_file)

        for path in sources_to_remove:
            safe_remove(path)

        print("✅ 所有文件复制并删除源操作完成！")
    except Exception as e:
        print(f"❌ 复制过程中发生错误: {str(e)}")
        raise


def main():
    proc_manager = ProcessManager()

    try:
        print("🔧 启动机器人控制程序...")
        robot_proc = subprocess.Popen(
            [ROBOT_PYTHON_PATH, ROBOT_SCRIPT] + ROBOT_ARGS,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        proc_manager.add_process(robot_proc)

        print("🔧 启动灵巧手控制程序...")
        hand_proc = subprocess.Popen(
            [HAND_PYTHON_PATH, HAND_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        proc_manager.add_process(hand_proc)

        while True:
            if robot_proc.poll() is not None and hand_proc.poll() is not None:
                break
            time.sleep(0.1)

        print("🔄 所有控制程序已完成，开始复制数据...")
        copy()

    except KeyboardInterrupt:
        print("\n🛑 用户中断操作")
    except Exception as e:
        print(f"❌ 主程序发生错误: {str(e)}")
    finally:
        proc_manager.terminate_all()
        print("🛑 程序已完全退出")


if __name__ == "__main__":
    main()
