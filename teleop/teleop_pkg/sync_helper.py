import os
import json
import time
import argparse

def wait_for_start_signal(sync_dir, system_name, timeout=120):
    """等待开始信号"""
    status_file = os.path.join(sync_dir, "status.json")
    start_signal_file = os.path.join(sync_dir, "start_recording")
    
    # 标记系统已准备
    try:
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status = json.load(f)
        else:
            status = {}
        
        if system_name == "robot":
            status["robot_ready"] = True
        elif system_name == "hand":
            status["hand_ready"] = True
        
        with open(status_file, 'w') as f:
            json.dump(status, f)
        
        print(f"🔄 {system_name}系统已准备，等待开始信号...")
    except Exception as e:
        print(f"❌ 标记系统准备状态失败: {e}")
        return None
    
    # 等待开始信号
    start_wait = time.time()
    while time.time() - start_wait < timeout:
        if os.path.exists(start_signal_file):
            try:
                with open(start_signal_file, 'r') as f:
                    start_timestamp = float(f.read().strip())
                print(f"✅ {system_name}系统收到开始信号: {start_timestamp}")
                return start_timestamp
            except Exception as e:
                print(f"❌ 读取开始信号失败: {e}")
        time.sleep(0.01)
    
    print(f"⏰ {system_name}系统等待开始信号超时")
    return None

def check_stop_signal(sync_dir):
    """检查停止信号"""
    stop_signal_file = os.path.join(sync_dir, "stop_recording")
    if os.path.exists(stop_signal_file):
        try:
            with open(stop_signal_file, 'r') as f:
                stop_timestamp = float(f.read().strip())
            return stop_timestamp
        except:
            return time.time()
    return None

def wait_for_stop_signal(sync_dir, check_interval=0.01):
    """等待停止信号"""
    while True:
        stop_time = check_stop_signal(sync_dir)
        if stop_time is not None:
            print(f"🛑 收到停止信号: {stop_time}")
            return stop_time
        time.sleep(check_interval)

def get_sync_status(sync_dir):
    """获取同步状态"""
    status_file = os.path.join(sync_dir, "status.json")
    try:
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='同步辅助工具')
    parser.add_argument('--sync-dir', required=True, help='同步目录')
    parser.add_argument('--system', required=True, choices=['robot', 'hand'], help='系统类型')
    parser.add_argument('--action', required=True, choices=['wait-start', 'wait-stop', 'status'], help='操作类型')
    
    args = parser.parse_args()
    
    if args.action == 'wait-start':
        start_time = wait_for_start_signal(args.sync_dir, args.system)
        if start_time:
            print(f"开始时间: {start_time}")
        else:
            exit(1)
    elif args.action == 'wait-stop':
        stop_time = wait_for_stop_signal(args.sync_dir)
        print(f"停止时间: {stop_time}")
    elif args.action == 'status':
        status = get_sync_status(args.sync_dir)
        print(json.dumps(status, indent=2))