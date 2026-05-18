#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MMK 机器人 ZMQ 转发器
在 MMK 环境下运行，通过 ZMQ 转发机器人控制和观测数据
"""

import argparse
import sys
import time
import yaml
import logging
import json

sys.path.append("/home/ubuntu/mmk_dev/Imitate-All/")

import base64

import numpy as np
import cv2
import zmq

# MMK 相关模块在运行时动态导入
# from robots.airbots.airbot_mmk.airbot_com_mmk2_bson import AIRBOTMMK2
# from robots.airbots.airbot_mmk.airbot_com_mmk2 import AIRBOTMMK2Config


class MMKForwarder:
    """MMK 机器人 ZMQ 通信转发器"""
    
    def __init__(self, config_path, zmq_port=5556):
        # 读取配置文件
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化 ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        logging.info(f"MMK 转发器监听端口 {zmq_port}")
        
        # 机器人配置参数
        self.ip = self.config["mmk_ip"]
        self.port = self.config["mmk_port"]
        self.mmk_code_path = self.config["mmk_code_path"]
        
        # 相机配置，仅处理内部摄像头
        self.internal_camera_name = "head_camera"
        
        # 组件列表
        self.components = ["left_arm", "right_arm", "head", "spine"]
        self.robot_cameras = {"head_camera": ["color"]}
        
        # 默认位置
        self.default_action = [0.3236820101737976, -1.0759518146514893, 1.6138323545455933, 2.174982786178589, -1.2834744453430176, -0.9786755442619324,-0.5533303022384644, -0.6879911422729492, 1.0378042459487915, -0.9946975111961365, 0.8146410584449768, 0.06351567804813385,
            0.0, -1.0,
            0.15
        ]
        
        # 只初始化机器人
        self._initialize_robot()
    
    def _initialize_robot(self):
        """初始化 MMK 机器人"""
        sys.path.append(self.mmk_code_path)
        from robots.airbots.airbot_mmk.airbot_com_mmk2_bson import AIRBOTMMK2
        from robots.airbots.airbot_mmk.airbot_com_mmk2 import AIRBOTMMK2Config
        
        config = AIRBOTMMK2Config(
            ip=self.ip,
            port=self.port,
            components=self.components,
            cameras=self.robot_cameras,
            default_action=self.default_action,
        )
        
        self.robot = AIRBOTMMK2(config=config)
        self.robot.reset(sleep_time=2)
        logging.info(f"成功连接 MMK 机器人 {self.ip}")
    
    def get_observations(self):
        """获取当前机器人状态"""
        robot_state_data = self.robot.get_low_dim_data()
        
        # 获取关节位置
        left_joint_data = robot_state_data["/observation/left_arm/joint_state"]["data"]["pos"]
        right_joint_data = robot_state_data["/observation/right_arm/joint_state"]["data"]["pos"]
        
        qpos = np.array(left_joint_data + right_joint_data)
        
        return {
            'qpos': qpos.tolist(),
        }
    
    def execute_action(self, action):
        """让 MMK 机器人执行动作"""
        # 关节限位
        # J1: [-π, 2π/3]
        # J2: [-17π/18, π/18]
        # J3: [-π/36, π]
        # J4: [-17π/18, 17π/18]
        # J5: [-5π/9, 5π/9]
        # J6: [-17π/18, 17π/18]
        
        # 每个关节的范围（左右相同）
        joint_limits = np.array([
            [-np.pi, 2 * np.pi / 3],              # J1
            [-17 * np.pi / 18, np.pi / 18],       # J2
            [-np.pi / 36, np.pi],                 # J3
            [-17 * np.pi / 18, 17 * np.pi / 18],  # J4
            [-5 * np.pi / 9, 5 * np.pi / 9],      # J5
            [-17 * np.pi / 18, 17 * np.pi / 18]   # J6
        ])
        
        action_array = np.array(action)
        
        # 动作维度检测
        if len(action_array) != 12:
            logging.warning(f"动作维度不符: 应为12，实际为 {len(action_array)}")
            return {'status': 'error', 'message': f'无效的动作维度: {len(action_array)}'}
        
        # 左臂限位
        left_arm = action_array[:6]
        left_arm_clamped = np.clip(left_arm, joint_limits[:, 0], joint_limits[:, 1])
        
        # 右臂限位
        right_arm = action_array[6:12]
        right_arm_clamped = np.clip(right_arm, joint_limits[:, 0], joint_limits[:, 1])
        
        # 是否有超出关节范围
        if not np.allclose(left_arm, left_arm_clamped) or not np.allclose(right_arm, right_arm_clamped):
            logging.warning(f"动作超限，已裁剪: {action_array.tolist()}")
        
        clamped_action = np.concatenate([left_arm_clamped, right_arm_clamped]).tolist()
        
        # 拼接三维夹爪数据
        full_action = clamped_action + self.default_action[-3:]
        # DEBUG 日志降低 IO，减少控制延迟
        logging.debug(f"[MMK EXECUTE] 发送动作 (15维): {full_action}")
        try:
            result = self.robot.send_action(full_action)
            logging.debug(f"[MMK RESULT] send_action 返回: {result}")
            return {'status': 'success', 'send_result': str(result)}
        except Exception as e:
            logging.error(f"[MMK ERROR] 发送动作失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def reset(self):
        """复位机器人"""
        self.robot.reset(sleep_time=2)
        return {'status': 'success'}
    
    def handle_request(self, request):
        """处理 ZMQ 请求"""
        command = request.get('command')
        
        if command == 'get_observations':
            return self.get_observations()
        
        elif command == 'execute_action':
            action = request.get('action')
            return self.execute_action(action)
        
        elif command == 'reset':
            return self.reset()
        
        else:
            return {'error': f'未知命令: {command}'}
    
    def run(self):
        """循环处理请求"""
        logging.info("MMK 转发器已启动")
        
        try:
            while True:
                # 等待请求
                message = self.socket.recv_json()
                logging.debug(f"收到请求: {message.get('command')}")
                
                # 处理请求
                try:
                    response = self.handle_request(message)
                except Exception as e:
                    logging.error(f"处理请求出错: {e}")
                    import traceback
                    traceback.print_exc()
                    response = {'error': str(e)}
                
                # 响应发送回客户端
                self.socket.send_json(response)
                
        except KeyboardInterrupt:
            logging.info("MMK 转发器关闭中...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """释放资源"""
        # 关闭 ZMQ
        self.socket.close()
        self.context.term()


def main():
    parser = argparse.ArgumentParser(description="MMK 机器人 ZMQ 转发器")
    parser.add_argument("--config", type=str, default="deploy/gr00t_mmk_xhand_config.yaml",
                        help="配置文件路径")
    parser.add_argument("--port", type=int, default=5556,
                        help="监听 ZMQ 端口")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="日志等级")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并运行转发器
    forwarder = MMKForwarder(args.config, args.port)
    forwarder.run()


if __name__ == "__main__":
    main()
