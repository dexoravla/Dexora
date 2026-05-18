#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XHand ZMQ Forwarder
Runs in the XHand environment and forwards hand control/observation data via ZMQ
"""
import serial
import threading
import argparse
import sys, os
import time
import yaml
import logging
import json

import numpy as np
import zmq
ser_ttl = None
# 初始化串口 (根据你的 Linux 实际挂载点修改，如 /dev/ttyUSB0)
try:
    # 建议波特率保持一致，115200 是常用标准
    ser_ttl = serial.Serial(
        port='/dev/ttyACM0',  # 请根据实际情况修改
        baudrate=115200, 
        timeout=0.1, 
        write_timeout=0.05  # 关键：防止串口阻塞主循环
    )
except Exception as e:
    logging.error(f"无法初始化 TTL 串口: {e}")
    ser_ttl = None

def sendTTL(action_list):
    """
    将右手的 12 个弧度值发送到串口硬件
    """
    global ser_ttl
    if ser_ttl is None or not ser_ttl.is_open:
        # 可选：尝试在此处重连或直接返回
        return

    try:
        # 1. 验证数据长度
        if len(action_list) != 12:
            logging.error(f"TTL数据长度错误: 期望12, 实际{len(action_list)}")
            return

        # 2. 格式化数据
        # 使用 :.4f 限制精度，既保证了舵机精度（约0.08度），又缩短了数据包长度
        data_str = "[" + ",".join([f"{val:.4f}" for val in action_list]) + "]\n"
        
        # 3. 编码并写入
        # encode('utf-8') 将字符串转为字节流
        ser_ttl.write(data_str.encode('utf-8'))
        
        # 4. 强制刷新发送缓冲区，确保指令立即发出
        ser_ttl.flush()

    except serial.SerialTimeoutException:
        logging.warning("TTL 串口发送超时，缓冲区可能已满")
    except Exception as e:
        logging.error(f"TTL 发送异常: {e}")

# XHand-specific imports will be added at runtime
# from xhand_tele_ops import XHandTeleOps


# Joint limits (radians) per joint index 0..11 based on the provided spec image
# Mapping assumption:
# 0: thumb_bend_joint           [0, 105°]
# 1: thumb_rota_joint1          [-40°, 90°]
# 2: thumb_rota_joint2          [0, 90°]
# 3: index_bend_joint           [-10°, 10°]
# 4: index_joint1               [0, 110°]
# 5: index_joint2               [0, 110°]
# 6: mid_joint1                 [0, 110°]
# 7: mid_joint2                 [0, 110°]
# 8: ring_joint1                [0, 110°]
# 9: ring_joint2                [0, 110°]
# 10: pinky_joint1              [0, 110°]
# 11: pinky_joint2              [0, 110°]
JOINT_LIMITS_RAD = [
    (0.0, 1.832595715),     # 105°
    (-0.698131701, 1.570796327),  # -40° ~ 90°
    (0.0, 1.570796327),     # 90°
    (-0.174532925, 0.174532925),  # -10° ~ 10°
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
    (0.0, 1.919862177),
        (0.0, 1.919862177),
        (0.0, 1.919862177),
    (0.0, 1.919862177),
]

# 初始关节位置（单位：度），每次启动后归位到此姿态
INIT_JOINTS_DEG = {
    "left_hand": [0.75, 47.75, 0.58, 0.05, 0.75, -0.5, 0.92, -0.75, 0.75, -0.58, 1.0, -0.58], 
    "right_hand": [30, 55.33, 3.0, 0.17, 1.08, 0.92, 1.25, 1.25, 1.33, 0.33, 1.33, -0.08]
}


def _clamp_hand_action(hand_action):
    """Clamp a 12-DoF hand action (radians) to JOINT_LIMITS_RAD.
    Pads/truncates to 12 dims if needed and logs a warning.
    Returns a list of 12 floats.
    """
    arr = np.asarray(hand_action, dtype=np.float32).reshape(-1)
    # Sanitize NaN/Inf to avoid sending invalid packets (common source of CRC/communication faults)
    if not np.all(np.isfinite(arr)):
        logging.warning("[XHAND] action contains NaN/Inf, sanitizing to finite values")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure exactly 12 dims
    if arr.size < 12:
        logging.warning(f"[XHAND] action dim {arr.size} < 12, padding with zeros")
        arr = np.pad(arr, (0, 12 - arr.size), mode="constant", constant_values=0.0)
    elif arr.size > 12:
        logging.warning(f"[XHAND] action dim {arr.size} > 12, truncating to 12")
        arr = arr[:12]
    mins = np.array([mn for mn, _ in JOINT_LIMITS_RAD], dtype=np.float32)
    maxs = np.array([mx for _, mx in JOINT_LIMITS_RAD], dtype=np.float32)
    clipped = np.clip(arr, mins, maxs)
    return clipped.tolist()


class XHandForwarder:
    """ZMQ forwarder for XHand communication"""
    
    def __init__(self, config_path, zmq_port=5557):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        logging.info(f"XHand forwarder listening on port {zmq_port}")
        
        # XHand configuration
        self.xhand_code_path = self.config['xhand_code_path']
        self.xhand_config = self.config['xhand_config']
        
        # Initialize XHand controller
        self._initialize_xhand()

        # Local throttling / retry (defensive against CRC / bus pressure)
        self._last_send_t = 0.0
        self._last_left = np.zeros(12, dtype=np.float32)
        self._last_right = np.zeros(12, dtype=np.float32)
        self._crc_count = 0
        self._send_count = 0
        self._skip_count = 0
        self._last_err = None

        # 初始化完成，将双手归位到初始姿态
        self._reset_to_init_pose()
    
    def _initialize_xhand(self):
        """Initialize the XHand controller"""
        os.chdir(self.xhand_code_path)
        from xhand_tele_ops import XHandTeleOps
        
        self.xhand_controller = XHandTeleOps(self.xhand_config)
        logging.info("XHand controller initialized successfully")

    def _reset_to_init_pose(self):
        """启动时将双手归位到 INIT_JOINTS_DEG 定义的初始姿态。
        INIT_JOINTS_DEG 单位为度，execute_action 期望弧度，此处做转换。
        """
        left_rad = np.deg2rad(INIT_JOINTS_DEG["left_hand"]).tolist()
        right_rad = np.deg2rad(INIT_JOINTS_DEG["right_hand"]).tolist()
        logging.info(
            f"[XHAND] Resetting to init pose | "
            f"left_deg={INIT_JOINTS_DEG['left_hand']} | "
            f"right_deg={INIT_JOINTS_DEG['right_hand']}"
        )
        result = self.execute_action({"left_hand": left_rad, "right_hand": right_rad})
        logging.info(f"[XHAND] Init pose result: {result}")

    def get_observations(self):
        """Get current hand state observations"""
        observations = {}
        
        # Get left hand data
        resp = self.xhand_controller.get_hand_full_info("hand_a", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            left_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
            observations['left_hand'] = left_hand_pos
        else:
            logging.error("Failed to get left hand data")
            observations['left_hand'] = [0.0] * 12
        
        # Get right hand data
        resp = self.xhand_controller.get_hand_full_info("hand_b", force_update=False, is_print=False)
        if resp and resp["code"] == 200:
            result = resp['data']
            right_hand_pos = [result['joint_position_dic'][f'joint{i}'] for i in range(12)]
            observations['right_hand'] = right_hand_pos
        else:
            logging.error("Failed to get right hand data")
            observations['right_hand'] = [0.0] * 12
        
        return observations
    
    def execute_action(self, action_data):
        """Execute hand actions"""
        try:
            # Extract hand actions
            left_hand_action = action_data.get('left_hand', [0.0] * 12)
            right_hand_action = action_data.get('right_hand', [0.0] * 12)
            
            # Clamp actions to safe joint limits (radians)
            left_hand_action = _clamp_hand_action(left_hand_action)
            right_hand_action = _clamp_hand_action(right_hand_action)

            #sendTTL(right_hand_action)

            # Forwarder-side throttling: avoid flooding RS485 even if upstream is noisy.
            min_interval = float(self.config.get("xhand_min_send_interval_s", 0.12))  # ~8.3Hz
            eps = float(self.config.get("xhand_forwarder_eps", 0.0))  # mean abs delta (rad)
            now = time.time()
            left_arr = np.asarray(left_hand_action, dtype=np.float32)
            right_arr = np.asarray(right_hand_action, dtype=np.float32)
            mean_delta = float(np.mean(np.abs(left_arr - self._last_left)) + np.mean(np.abs(right_arr - self._last_right))) / 2.0

            if (now - self._last_send_t) < min_interval or (eps > 0.0 and mean_delta < eps):
                self._skip_count += 1
                return {"status": "skipped", "mean_delta": mean_delta, "skip_count": self._skip_count}

            # Optional unit conversion for SDK expectations
            action_unit = str(self.config.get("xhand_action_unit", "rad")).lower()
            if action_unit in ("deg", "degree", "degrees"):
                left_send = np.rad2deg(left_arr).tolist()
                right_send = np.rad2deg(right_arr).tolist()
            else:
                left_send = left_arr.tolist()
                right_send = right_arr.tolist()

            transform_data = {"left_hand": left_send, "right_hand": right_send}
            # Retry on CRC-like failures (best effort)
            retries = int(self.config.get("xhand_send_retries", 1))
            backoff = float(self.config.get("xhand_crc_backoff_s", 0.08))
            last_exc = None
            resp = None
            for attempt in range(retries + 1):
                try:
                    resp = self.xhand_controller.send_data_xhand(transform_data)
                    self._last_send_t = now
                    self._last_left = left_arr
                    self._last_right = right_arr
                    self._send_count += 1
                    self._last_err = None
                    # Some SDKs return dict with 'code'/'msg'
                    if isinstance(resp, dict):
                        msg = str(resp.get("msg") or resp.get("message") or "")
                        if "crc" in msg.lower() or "CRC" in msg:
                            self._crc_count += 1
                            if attempt < retries:
                                time.sleep(backoff)
                                continue
                            return {"status": "error", "error": msg, "crc_count": self._crc_count}
                    return {"status": "success", "resp": resp, "crc_count": self._crc_count}
                except Exception as e:
                    last_exc = e
                    if "crc" in str(e).lower():
                        self._crc_count += 1
                    if attempt < retries:
                        time.sleep(backoff)
                        continue
                    break

            self._last_err = str(last_exc) if last_exc is not None else "unknown"
            return {"status": "error", "error": self._last_err, "crc_count": self._crc_count}
            
        except Exception as e:
            logging.error(f"Error executing action: {e}")
            return {'error': str(e)}
    
    def handle_request(self, request):
        """Handle incoming ZMQ request"""
        command = request.get('command')
        
        if command == 'get_observations':
            return self.get_observations()
        
        elif command == 'execute_action':
            action_data = request.get('action_data')
            return self.execute_action(action_data)
        
        elif command == 'ping':
            return {'status': 'pong'}
        
        else:
            return {'error': f'Unknown command: {command}'}
    
    def run(self):
        """Main loop for handling requests"""
        logging.info("XHand forwarder started")
        
        try:
            while True:
                # Wait for request
                message = self.socket.recv_json()
                logging.debug(f"Received request: {message.get('command')}")
                
                # Process request
                try:
                    response = self.handle_request(message)
                except Exception as e:
                    logging.error(f"Error handling request: {e}")
                    response = {'error': str(e)}
                
                # Send response
                self.socket.send_json(response)
                
        except KeyboardInterrupt:
            logging.info("Shutting down XHand forwarder...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        # Close ZMQ socket
        self.socket.close()
        self.context.term()

def serial_listener(forwarder, default_left_rad, default_right_rad):
    """
    串口监听线程函数
    """
    try:
        global ser_ttl
        # 配置串口（请根据实际情况修改端口号和波特率）
        ser = ser_ttl #serial.Serial(port_name, 115200, timeout=1)
        #logging.info(f"Started listening on {port_name}")
        
        # 初始的右手数据
        current_right_rad = default_right_rad 

        while True:
            if ser.in_waiting > 0:
                # 读取一行并解码
                line = ser.readline().decode('utf-8').strip()
                try:
                    # 使用 ast.literal_eval 安全地将字符串 "[...]" 转为 list
                    data = eval(line)
                    
                    if isinstance(data, list) and len(data) == 12:
                        #current_right_rad[0] = data[0]
                        #current_right_rad[1] = data[1]
                        #current_right_rad[2] = data[2]
                        num=11
                        
                        current_right_rad = data
                        logging.info(f"Received new joints: {current_right_rad}")
                        
                        # 触发执行动作
                        # 注意：这里假设 execute_action 是线程安全的
                        forwarder.execute_action({
                            "left_hand": default_left_rad, 
                            "right_hand": current_right_rad
                        })
                    else:
                        logging.warning(f"Invalid data length or type: {line}")
                        
                except Exception as e:
                    logging.error(f"Failed to parse serial data '{line}': {e}")
                    
    except Exception as e:
        logging.error(f"Serial Error: {e}")
def main():
    parser = argparse.ArgumentParser(description="XHand ZMQ Forwarder")
    parser.add_argument("--config", type=str, default="deploy/gr00t_mmk_xhand_config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--port", type=int, default=5557,
                        help="ZMQ port to listen on")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run forwarder
    forwarder = XHandForwarder(args.config, args.port)
    


    # 2. 准备初始数据
    left_rad = np.deg2rad(INIT_JOINTS_DEG["left_hand"]).tolist()
    right_rad = np.deg2rad(INIT_JOINTS_DEG["right_hand"]).tolist()
    
    # 3. 启动串口监听线程 (假设串口设备为 /dev/ttyUSB0)
    # 你可以把串口名也加进 argparse 参数里
    serial_thread = threading.Thread(
        target=serial_listener, 
        args=(forwarder, left_rad,right_rad),
        daemon=True # 随主程序退出
    )
    serial_thread.start()
    forwarder.run()



if __name__ == "__main__":
    main()
    
