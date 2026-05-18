# coding: utf-8
"""
只回放机器人本体轨迹（不动灵巧手）的简化版。
适合在没有灵巧手 / 不想驱动灵巧手的场景下检查机器人侧的轨迹。
"""
import subprocess
import os
import threading
from pathlib import Path

# -----------------------------------------------------------------------------
# 路径配置
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ROBOT_PYTHON_PATH = "/home/slam/miniconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH  = "/home/slam/miniconda3/envs/xhand_tele_env/bin/python"

ROBOT_SCRIPT = str(PROJECT_ROOT / "imitate_all" / "mmk_replay.py")
HAND_SCRIPT  = str(PROJECT_ROOT / "teleop_pkg"  / "control_from_bson.py")

DEFAULT_EPISODE_BSON = str(PROJECT_ROOT / "samples" / "episode_0.bson")

ROBOT_ARGS = [
    DEFAULT_EPISODE_BSON,
    "--ip", "192.168.11.200",
    "--freq", "20",
]


def wait_for_ready(process, name, ready_event):
    while True:
        line = process.stdout.readline()
        if not line:
            break
        decoded = line.decode("utf-8").strip()
        print(f"[{name}] {decoded}")
        if decoded == "READY":
            ready_event.set()
            break


def main():
    from threading import Event

    print("🔧 启动机器人控制程序...")
    robot_proc = subprocess.Popen(
        [ROBOT_PYTHON_PATH, ROBOT_SCRIPT] + ROBOT_ARGS,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    # 不启动灵巧手回放
    # hand_proc = subprocess.Popen(
    #     [HAND_PYTHON_PATH, HAND_SCRIPT],
    #     stdout=subprocess.PIPE,
    #     stdin=subprocess.PIPE,
    #     stderr=subprocess.STDOUT
    # )

    robot_ready = Event()
    threading.Thread(target=wait_for_ready, args=(robot_proc, "Robot", robot_ready)).start()

    robot_ready.wait()

    print("✅ 机器人侧初始化完成。请输入启动命令。")
    print("🔔 请输入空行（直接按回车）以启动回放。")
    while True:
        user_input = input("▶️ 等待回车启动 >> ").strip()
        if user_input == "":
            break
        else:
            print("❌ 非法输入。请输入空行（只按回车）以启动。")

    print("🚀 正在向机器人发送 START 指令...")
    robot_proc.stdin.write(b"START\n")
    robot_proc.stdin.flush()

    try:
        robot_proc.wait()
    except KeyboardInterrupt:
        print("🔴 检测到中断，终止进程...")
        robot_proc.terminate()


if __name__ == "__main__":
    main()
