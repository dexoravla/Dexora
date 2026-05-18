#!/usr/bin/env python3
import argparse
import datetime
import shlex
import sys
from typing import List, Tuple

import paramiko


def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class RemoteDocker:
    def __init__(self, host: str, username: str, password: str, container: str = "airbot_mmk2"):
        self.host = host
        self.username = username
        self.password = password
        self.container = container
        self.client = None  # type: ignore

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(self.host, username=self.username, password=self.password, look_for_keys=False)

    def close(self):
        if self.client:
            self.client.close()

    def run_host(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        if not self.client:
            raise RuntimeError("SSH client not connected")
        stdin, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="ignore")
        err = stderr.read().decode("utf-8", errors="ignore")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    def run_in_container(self, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        quoted = shlex.quote(command)
        docker_cmd = f"docker exec -i {shlex.quote(self.container)} bash -lc {quoted}"
        return self.run_host(docker_cmd, timeout=timeout)


def list_can_interfaces(docker: RemoteDocker) -> List[str]:
    # Get interface names that include 'can' but exclude 'vcan'
    cmd = (
        "ip -br link show | awk '{print $1}' | grep -i can || true"
    )
    rc, out, err = docker.run_in_container(cmd)
    if rc != 0:
        print(f"[WARN] list can ifaces rc={rc} err={err.strip()}")
    names = [line.strip() for line in out.splitlines() if line.strip()]
    # Exclude vcan
    names = [n for n in names if not n.lower().startswith("vcan")]
    # Ensure typical aliases are included if present in system
    for alias in ("can_right", "can_left"):
        if alias not in names:
            # check existence individually to avoid noise
            rc, _, _ = docker.run_in_container(f"ip link show {alias} >/dev/null 2>&1 || true")
            if rc == 0:
                names.append(alias)
    # De-dup while preserving order
    seen = set()
    ordered = []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def kill_can_users(docker: RemoteDocker):
    # Best-effort kill typical processes that may hold the CAN sockets
    patterns = [
        "ros2",
        "candump",
        "cansend",
        "airbot",
        "params_check",
    ]
    grep = "|".join(patterns)
    docker.run_in_container(f"ps -eo pid,cmd | grep -E '{grep}' | grep -v grep || true")
    docker.run_in_container(f"pkill -f '{grep}' || true")
    docker.run_in_container("sleep 1")


def set_can_bitrate(docker: RemoteDocker, iface: str, bitrate: int) -> Tuple[bool, str]:
    cmds = [
        f"ip link set {iface} down || true",
        # Try classical CAN first
        f"ip link set {iface} type can bitrate {bitrate} || true",
        f"ip link set {iface} up || true",
        f"ip -details -br link show {iface} || true",
    ]
    output_logs = []
    success = True
    for c in cmds:
        rc, out, err = docker.run_in_container(c)
        output_logs.append(f"$ {c}\n{out}{err}")
        # Non-strict; link set may return non-zero for 'already up', etc. We'll validate after.
    # Validate bitrate by inspecting output
    rc, out, err = docker.run_in_container(f"ip -details link show {iface} | sed -n '1,10p' || true")
    output_logs.append(f"$ ip -details link show {iface}\n{out}{err}")
    if str(bitrate) not in out:
        success = False
    return success, "\n".join(output_logs)


def run_params_check(docker: RemoteDocker, mode: str) -> Tuple[int, str, str]:
    # Use timeout to avoid hanging
    cmd = f"timeout 8s ros2 run airbot_tools params_check -m {mode} || true"
    return docker.run_in_container(cmd, timeout=20)


def main():
    parser = argparse.ArgumentParser(description="Reset CAN interfaces inside Docker and test with ros2 params_check")
    parser.add_argument("--host", default="192.168.11.200")
    parser.add_argument("--user", default="orangepi")
    parser.add_argument("--password", default="airbot")
    parser.add_argument("--container", default="airbot_mmk2")
    parser.add_argument("--log", default="log.md", help="Path to append local log")
    args = parser.parse_args()

    header = f"\n\n### {now_str()} 远程CAN修复执行\n"
    print(header)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(header)

    docker = RemoteDocker(args.host, args.user, args.password, args.container)
    try:
        docker.connect()
        print(f"[{now_str()}] 已连接 {args.user}@{args.host}")

        # Show running container info
        rc, out, err = docker.run_host("docker ps --format '{{.Names}}\t{{.Status}}\t{{.Image}}' | grep -F '" + args.container + "' || true")
        print(out)

        # Ensure no processes hold CAN
        print(f"[{now_str()}] 结束可能占用CAN的进程 ...")
        kill_can_users(docker)

        # Enumerate interfaces
        ifaces = list_can_interfaces(docker)
        print(f"[{now_str()}] 检测到CAN接口: {ifaces if ifaces else '无'}")

        # Candidate bitrates to try
        candidate_bitrates = [1000000, 500000, 250000, 125000]

        overall_success = False
        all_logs: List[str] = []
        last_check_output = ""

        for bitrate in candidate_bitrates:
            print(f"[{now_str()}] 尝试配置波特率: {bitrate}")
            step_logs = [f"配置 {bitrate} 日志:"]
            # Down all ifaces first
            for iface in ifaces:
                docker.run_in_container(f"ip link set {iface} down || true")
            # Apply bitrate for each
            success_all = True
            for iface in ifaces:
                ok, logs = set_can_bitrate(docker, iface, bitrate)
                step_logs.append(logs)
                success_all = success_all and ok

            # Test with params_check for both sides if present
            modes_to_test = []
            # If alias present then test; otherwise still try because tool may map internally
            if "can_right" in ifaces or True:
                modes_to_test.append("can_right")
            if "can_left" in ifaces or True:
                modes_to_test.append("can_left")

            combined = []
            for mode in modes_to_test:
                rc, out, err = run_params_check(docker, mode)
                combined.append(f"$ params_check -m {mode}\n{out}{err}")
                last_check_output = out + err

            step_logs.append("\n".join(combined))
            all_logs.append("\n\n".join(step_logs))

            # Determine success: no 'used by other process' and has Arm type or no errors
            if ("used by other process" not in last_check_output) and ("Error List:" not in last_check_output or "[]" in last_check_output):
                overall_success = True
                print(f"[{now_str()}] 成功：波特率 {bitrate}")
                break
            else:
                print(f"[{now_str()}] 仍有错误，继续尝试其他波特率 ...")

        # Append logs to local file
        with open(args.log, "a", encoding="utf-8") as f:
            for block in all_logs:
                f.write("\n\n")
                f.write(block)

        if overall_success:
            print(f"[{now_str()}] CAN接口恢复完成。")
        else:
            print(f"[{now_str()}] 未能完全恢复，请查看log.md并手动检查占用进程。")

    finally:
        docker.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)



