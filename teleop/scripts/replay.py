# coding: utf-8
"""
回放主程序 —— 同时拉起机器人本体回放 + 灵巧手回放，两路握手到 READY 后
等待用户按回车，同步发送 START 指令，按 BSON 中记录的时间戳重放轨迹。

默认回放 samples/episode_0.bson + teleop_pkg 自带的 xhand_control_data.bson。
如果想换数据，把 ROBOT_ARGS 里的 BSON 路径改成你的 episode 路径。
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

# 要回放的 episode BSON（机器人本体轨迹）
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

    print("🔧 启动灵巧手控制程序...")
    hand_proc = subprocess.Popen(
        [HAND_PYTHON_PATH, HAND_SCRIPT],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    robot_ready = Event()
    hand_ready = Event()
    threading.Thread(target=wait_for_ready, args=(robot_proc, "Robot", robot_ready)).start()
    threading.Thread(target=wait_for_ready, args=(hand_proc, "Hand", hand_ready)).start()

    robot_ready.wait()
    hand_ready.wait()

    print("✅ 两个子系统初始化完成。请输入启动命令。")
    print("🔔 请输入空行（直接按回车）以同步启动两个子系统。")
    while True:
        user_input = input("▶️ 等待回车启动 >> ").strip()
        if user_input == "":
            break
        else:
            print("❌ 非法输入。请输入空行（只按回车）以启动。")

    print("🚀 正在向两个子系统发送 START 指令...")
    robot_proc.stdin.write(b"START\n")
    robot_proc.stdin.flush()
    hand_proc.stdin.write(b"START\n")
    hand_proc.stdin.flush()

    try:
        robot_proc.wait()
        hand_proc.wait()
    except KeyboardInterrupt:
        print("🔴 检测到中断，终止两个进程...")
        robot_proc.terminate()
        hand_proc.terminate()


if __name__ == "__main__":
    main()
