# coding: utf-8
"""
回放（支持 lerobot parquet）—— 如果设置了 PARQUET_PATH 且文件存在，
会把 lerobot 风格的 parquet（含 actions/timestamp）传给机器人侧的回放脚本；
否则回退到 BSON 回放逻辑。

⚠️ 把 PARQUET_PATH 改成你本地的绝对路径再使用，或者留空走 BSON 流程。
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

# 统一的 Parquet 绝对路径（lerobot 格式，包含 actions+timestamp）。
# 留空或不存在时，自动回退到 BSON 路径。
PARQUET_PATH = ""    # 例： "/home/slam/data/episode_000005.parquet"


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

    parquet_path = PARQUET_PATH
    use_parquet = bool(parquet_path) and os.path.isfile(parquet_path)
    if parquet_path and not use_parquet:
        print(f"⚠️ 未找到文件: {parquet_path}，将继续使用 BSON 逻辑。")

    print("🔧 启动机器人控制程序...")
    robot_cmd = [ROBOT_PYTHON_PATH, ROBOT_SCRIPT] + ROBOT_ARGS
    if use_parquet:
        robot_cmd += ["--parquet", parquet_path, "--only-arms"]
    robot_proc = subprocess.Popen(
        robot_cmd,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )

    print("🔧 启动灵巧手控制程序...")
    hand_cmd = [HAND_PYTHON_PATH, HAND_SCRIPT]
    if use_parquet:
        hand_cmd += ["--parquet", parquet_path]
    hand_proc = subprocess.Popen(
        hand_cmd,
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
