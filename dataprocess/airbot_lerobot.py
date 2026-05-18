#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 LeRobot 官方 API 的 Airbot 数据转换器
基于 lerobot v0.3.4 (v2.1 format)
"""
import json
import os
import numpy as np
import struct
import bson
import cv2
import tempfile
import shutil
from collections import defaultdict
from tqdm import tqdm
import datetime
import logging
from typing import Dict, List, Any, Union, Optional
from pathlib import Path

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import DEFAULT_FEATURES

from airbot_config import AirbotConfig


class AirbotLeRobotProcessor:
    """使用 LeRobot 官方 API 的数据处理器"""
    
    def __init__(self, config: AirbotConfig):
        self.config = config
        self.setup_logging()
        
        # 任务映射相关
        self.action_to_task_index = {}
        self.current_task_index = 0
        self.task_index_to_info = {}
        
        # 初始化 LeRobot 数据集
        self.dataset = None
        self.processed_episodes = set()
        
        # 初始化任务映射
        self._initialize_action_task_mapping()
        
    def setup_logging(self):
        """配置日志"""
        log_dir = os.path.join(self.config.log_root, self.config.task_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.get_today_time()}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("AirbotLeRobotProcessor")

    def get_today_time(self):
        """获取当前时间字符串"""
        today = datetime.datetime.now()
        return today.strftime("%Y%m%d%H%M%S")

    def _initialize_action_task_mapping(self):
        """初始化action到task_index的映射"""
        self.logger.info("=== 初始化Action到Task_Index映射 ===")
        
        # 收集所有独特的task_name
        task_names = set()
        action_to_task_name = {}
        
        # 从task_categories获取action和对应的task_name映射
        if hasattr(self.config, 'task_categories'):
            for category, info in self.config.task_categories.items():
                actions = info.get('actions', [])
                task_name = info.get('task_name', category)
                for action_id in actions:
                    action_to_task_name[action_id] = task_name
                    task_names.add(task_name)
        
        # 从action_instruction_mapping获取剩余的actions
        if hasattr(self.config, 'action_instruction_mapping'):
            for action_id in self.config.action_instruction_mapping.keys():
                if action_id not in action_to_task_name:
                    # 使用默认task_name
                    default_task_name = getattr(self.config, 'task_name', 'default_task')
                    action_to_task_name[action_id] = default_task_name
                    task_names.add(default_task_name)
        
        # 为每个独特的task_name分配task_index
        sorted_task_names = sorted(list(task_names))
        task_name_to_index = {}
        
        for idx, task_name in enumerate(sorted_task_names):
            task_name_to_index[task_name] = idx
            task_info = self._find_task_info_by_task_name(task_name)
            self.task_index_to_info[idx] = {
                'task_name': task_name,
                'task_index': idx,
                'description': task_info.get('description', f'Task {task_name}'),
                'category': task_info.get('category', 'uncategorized'),
                'actions': []
            }
        
        # 为每个action分配对应的task_index
        for action_id, task_name in action_to_task_name.items():
            task_index = task_name_to_index[task_name]
            self.action_to_task_index[action_id] = task_index
            self.task_index_to_info[task_index]['actions'].append(action_id)
        
        self.logger.info(f"总共创建了 {len(task_name_to_index)} 个独特的task_index")
        for task_index, info in self.task_index_to_info.items():
            self.logger.info(f"Task {task_index}: {info['task_name']} -> {len(info['actions'])} actions")

    def _find_task_info_by_task_name(self, task_name: str) -> dict:
        """根据task_name查找任务详细信息"""
        if hasattr(self.config, 'task_categories'):
            for category, info in self.config.task_categories.items():
                if info.get('task_name', category) == task_name:
                    return info
        
        # 默认返回
        return {
            'description': f'Task {task_name}',
            'category': 'uncategorized'
        }

    def create_lerobot_features(self) -> Dict[str, Any]:
        """创建 LeRobot 数据集的 features 定义"""
        features = {}
        
        # 状态特征 (机械臂 + 灵巧手)
        # 假设: 左臂6DOF + 右臂6DOF + 左手12DOF + 右手12DOF = 36维
        state_dim = 36
        features["states"] = {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [
                # 左臂关节
                "left_arm_joint_0", "left_arm_joint_1", "left_arm_joint_2", 
                "left_arm_joint_3", "left_arm_joint_4", "left_arm_joint_5",
                # 右臂关节
                "right_arm_joint_0", "right_arm_joint_1", "right_arm_joint_2",
                "right_arm_joint_3", "right_arm_joint_4", "right_arm_joint_5",
                # 左手关节
                "left_hand_joint_0", "left_hand_joint_1", "left_hand_joint_2", "left_hand_joint_3",
                "left_hand_joint_4", "left_hand_joint_5", "left_hand_joint_6", "left_hand_joint_7",
                "left_hand_joint_8", "left_hand_joint_9", "left_hand_joint_10", "left_hand_joint_11",
                # 右手关节
                "right_hand_joint_0", "right_hand_joint_1", "right_hand_joint_2", "right_hand_joint_3",
                "right_hand_joint_4", "right_hand_joint_5", "right_hand_joint_6", "right_hand_joint_7",
                "right_hand_joint_8", "right_hand_joint_9", "right_hand_joint_10", "right_hand_joint_11",
            ]
        }
        
        # 动作特征 (与状态维度相同)
        action_dim = 36
        features["actions"] = {
            "dtype": "float32", 
            "shape": (action_dim,),
            "names": [name.replace("joint", "action") for name in features["states"]["names"]]
        }
        
        # 图像特征 (4个相机)
        for camera_name in ['camera_high', 'camera_left', 'camera_right', 'camera_front']:
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (480, 640, 3),  # 假设的图像尺寸
                "names": ["height", "width", "channels"]
            }
        
        # 添加默认特征
        features.update(DEFAULT_FEATURES)
        
        return features

    def setup_lerobot_dataset(self) -> LeRobotDataset:
        """使用 LeRobot API 初始化数据集"""
        self.logger.info("=== 使用 LeRobot API 初始化数据集 ===")
        
        # 创建输出目录
        dataset_name = f"{self.config.robot}_{self.config.task_name}"
        dataset_root = Path(self.config.output_data_root) / dataset_name
        
        # 如果存在且允许覆写，则删除
        if dataset_root.exists() and self.config.overwrite:
            import shutil
            shutil.rmtree(dataset_root)
            self.logger.info(f"已删除现有数据集: {dataset_root}")
        
        # 创建 features 定义
        features = self.create_lerobot_features()
        
        # 使用 LeRobot API 创建数据集
        try:
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=int(self.config.fps),
                features=features,
                root=str(dataset_root),
                robot_type=self.config.robot,
                use_videos=True,
                video_backend=getattr(self.config, 'video_backend', 'pyav')
            )
            
            self.logger.info(f"LeRobot 数据集创建成功: {dataset_root}")
            self.logger.info(f"Features: {list(features.keys())}")
            
            return dataset
            
        except Exception as e:
            self.logger.error(f"创建 LeRobot 数据集失败: {e}")
            raise

    def read_bson_file(self, bson_path: str) -> Dict[str, Any]:
        """读取BSON文件"""
        try:
            with open(bson_path, 'rb') as f:
                data = bson.decode_all(f.read())
                return data[0] if data else {}
        except Exception as e:
            self.logger.error(f"读取BSON文件失败 {bson_path}: {e}")
            return {}

    def load_images_from_folders(self, episode_path: str, action_name: str) -> Dict[str, List[np.ndarray]]:
        """从文件夹加载图像序列 - 包含所有4个相机"""
        images = {
            'camera_high': [],
            'camera_left': [],
            'camera_right': [],
            'camera_front': []
        }
        
        # 获取当前action的相机映射
        camera_mapping = self.config.get_camera_mapping(action_name)
        self.logger.info(f"使用 {action_name} 的相机映射: {camera_mapping}")
        
        # 统一处理所有4个相机，包括front相机
        for camera_name, folder_name in camera_mapping.items():
            camera_dir = os.path.join(episode_path, folder_name)
            
            if os.path.exists(camera_dir):
                image_files = [f for f in os.listdir(camera_dir) if f.endswith(('.jpg', '.png'))]
                if not image_files:
                    self.logger.warning(f"相机文件夹中没有图像文件: {camera_dir}")
                    continue
                    
                # 按文件名中的数字排序
                try:
                    image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                except (IndexError, ValueError):
                    # 如果文件名格式不符合预期，使用默认排序
                    image_files.sort()
                
                camera_images = []
                for img_file in image_files:
                    img_path = os.path.join(camera_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        camera_images.append(img_rgb)
                
                images[camera_name] = camera_images
                self.logger.info(f"{camera_name}: 从文件夹加载了 {len(images[camera_name])} 张图像")
            else:
                self.logger.warning(f"{camera_name} 文件夹不存在: {camera_dir}")
        
        return images

    def extract_images_from_bson(self, robot_data: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        """从BSON文件中提取front相机图像数据"""
        images = {'camera_front': []}
        
        try:
            if 'frames' in robot_data:
                for frame_data in robot_data['frames']:
                    if 'camera' in frame_data and 'video_data' in frame_data['camera']:
                        video_bytes = frame_data['camera']['video_data']
                        frames = self.extract_frames_from_mp4_bytes(video_bytes)
                        images['camera_front'].extend(frames)
            
            self.logger.info(f"camera_front: 从BSON提取了 {len(images['camera_front'])} 张图像")
        except Exception as e:
            self.logger.error(f"从BSON提取前置相机图像失败: {e}")
        
        return images

    def extract_frames_from_mp4_bytes(self, video_bytes: bytes) -> List[np.ndarray]:
        """从MP4视频字节数据中提取帧"""
        frames = []
        temp_video_path = None
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_video_path = temp_file.name
                temp_file.write(video_bytes)
            
            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(temp_video_path)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            cap.release()
            
        except Exception as e:
            self.logger.error(f"从MP4字节提取帧失败: {e}")
        
        finally:
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        return frames

    def extract_robot_joint_data(self, robot_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """从机器人BSON数据中提取关节数据 - 使用与airbot.py相同的逻辑"""
        joint_data = {}
        
        try:
            data_section = robot_data.get('data', {})
            
            # 提取左臂关节数据
            left_arm_joint_obs = data_section.get('/observation/left_arm/joint_state', [])
            left_arm_joint_action = data_section.get('/action/left_arm/joint_state', [])
            
            # 提取右臂关节数据
            right_arm_joint_obs = data_section.get('/observation/right_arm/joint_state', [])
            right_arm_joint_action = data_section.get('/action/right_arm/joint_state', [])
            
            # 处理各类数据
            for key, data_list in [
                ('left_arm_obs', left_arm_joint_obs),
                ('left_arm_action', left_arm_joint_action),
                ('right_arm_obs', right_arm_joint_obs),
                ('right_arm_action', right_arm_joint_action)
            ]:
                if data_list:
                    positions = []
                    for frame_data in data_list:
                        if 'data' in frame_data and 'pos' in frame_data['data']:
                            pos_data = frame_data['data']['pos']
                            if len(pos_data) >= 6:
                                positions.append(pos_data[:6])
                            else:
                                padded_pos = pos_data + [0.0] * (6 - len(pos_data))
                                positions.append(padded_pos)
                    joint_data[key] = np.array(positions)
                    self.logger.info(f"机械臂数据 {key}: {joint_data[key].shape}")
            
        except Exception as e:
            self.logger.error(f"提取机械臂关节数据失败: {e}")
        
        return joint_data

    def extract_hand_data(self, hand_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """从灵巧手BSON数据中提取手部数据 - 使用与airbot.py相同的逻辑"""
        hand_joint_data = {}
        
        try:
            frames = hand_data.get('frames', [])
            
            left_hand_actions = []
            right_hand_actions = []
            left_hand_obs = []
            right_hand_obs = []
            
            for frame in frames:
                action_data = frame.get('action', {})
                left_action = action_data.get('left_hand', [])
                right_action = action_data.get('right_hand', [])
                
                left_action = self.pad_or_truncate(left_action, 12)
                right_action = self.pad_or_truncate(right_action, 12)
                
                left_hand_actions.append(left_action)
                right_hand_actions.append(right_action)
                
                obs_data = frame.get('observation', {})
                left_obs = obs_data.get('left_hand', [])
                right_obs = obs_data.get('right_hand', [])
                
                left_obs = self.pad_or_truncate(left_obs, 12)
                right_obs = self.pad_or_truncate(right_obs, 12)
                
                # 将手部observation从角度转换为弧度，与action保持一致
                left_obs = [np.deg2rad(angle) for angle in left_obs]
                right_obs = [np.deg2rad(angle) for angle in right_obs]
                
                left_hand_obs.append(left_obs)
                right_hand_obs.append(right_obs)
            
            hand_joint_data['left_hand_action'] = np.array(left_hand_actions)
            hand_joint_data['right_hand_action'] = np.array(right_hand_actions)
            hand_joint_data['left_hand_obs'] = np.array(left_hand_obs)
            hand_joint_data['right_hand_obs'] = np.array(right_hand_obs)
            
            # 记录数据信息
            for key, data in hand_joint_data.items():
                self.logger.info(f"灵巧手数据 {key}: {data.shape}")
            
            self.logger.info("手部数据单位转换: observation 角度→弧度")
            
        except Exception as e:
            self.logger.error(f"提取灵巧手数据失败: {e}")
        
        return hand_joint_data

    def pad_or_truncate(self, data: List[float], target_length: int) -> List[float]:
        """填充或截断数据到目标长度"""
        if len(data) == 0:
            return [0.0] * target_length
        elif len(data) < target_length:
            return data + [0.0] * (target_length - len(data))
        else:
            return data[:target_length]

    def convert_frame_to_lerobot_format(self, 
                                       frame_idx: int,
                                       robot_joint_data: Dict[str, np.ndarray], 
                                       hand_joint_data: Dict[str, np.ndarray],
                                       images: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
        """将单帧数据转换为 LeRobot 格式 - 使用正确的数据键名"""
        frame = {}
        
        # 构建状态向量 (36维: 6+6+12+12) - 使用observation数据作为状态
        left_arm = robot_joint_data.get('left_arm_obs', np.zeros((1, 6)))[frame_idx] if frame_idx < len(robot_joint_data.get('left_arm_obs', [])) else np.zeros(6)
        right_arm = robot_joint_data.get('right_arm_obs', np.zeros((1, 6)))[frame_idx] if frame_idx < len(robot_joint_data.get('right_arm_obs', [])) else np.zeros(6)
        left_hand = hand_joint_data.get('left_hand_obs', np.zeros((1, 12)))[frame_idx] if frame_idx < len(hand_joint_data.get('left_hand_obs', [])) else np.zeros(12)
        right_hand = hand_joint_data.get('right_hand_obs', np.zeros((1, 12)))[frame_idx] if frame_idx < len(hand_joint_data.get('right_hand_obs', [])) else np.zeros(12)
        
        # 组合状态
        state = np.concatenate([left_arm, right_arm, left_hand, right_hand])
        frame["states"] = state.astype(np.float32)
        
        # 构建动作向量 (36维: 6+6+12+12) - 使用action数据
        left_arm_action = robot_joint_data.get('left_arm_action', np.zeros((1, 6)))[frame_idx] if frame_idx < len(robot_joint_data.get('left_arm_action', [])) else np.zeros(6)
        right_arm_action = robot_joint_data.get('right_arm_action', np.zeros((1, 6)))[frame_idx] if frame_idx < len(robot_joint_data.get('right_arm_action', [])) else np.zeros(6)
        left_hand_action = hand_joint_data.get('left_hand_action', np.zeros((1, 12)))[frame_idx] if frame_idx < len(hand_joint_data.get('left_hand_action', [])) else np.zeros(12)
        right_hand_action = hand_joint_data.get('right_hand_action', np.zeros((1, 12)))[frame_idx] if frame_idx < len(hand_joint_data.get('right_hand_action', [])) else np.zeros(12)
        
        # 组合动作
        action = np.concatenate([left_arm_action, right_arm_action, left_hand_action, right_hand_action])
        frame["actions"] = action.astype(np.float32)
        
        # 添加图像
        for camera_name in ['camera_high', 'camera_left', 'camera_right', 'camera_front']:
            if camera_name in images and frame_idx < len(images[camera_name]):
                frame[f"observation.images.{camera_name}"] = images[camera_name][frame_idx]
            else:
                # 如果没有图像，创建一个黑色图像占位符
                frame[f"observation.images.{camera_name}"] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return frame

    def get_instruction_for_action(self, action_id: str) -> str:
        """根据action ID获取对应的指令"""
        if hasattr(self.config, 'action_instruction_mapping'):
            instruction = self.config.action_instruction_mapping.get(action_id)
            if instruction:
                return instruction
        
        # 如果没有找到映射，返回默认指令
        return f"{self.config.default_instruction}_{action_id}"

    def get_task_info_for_action(self, action_id: str) -> tuple:
        """获取action对应的任务信息"""
        task_index = self.action_to_task_index.get(action_id, 0)
        
        if task_index in self.task_index_to_info:
            task_info = self.task_index_to_info[task_index]
            return task_index, task_info['task_name'], task_info['description']
        else:
            return 0, self.config.task_type, f"Default task for {action_id}"

    def get_action_id_from_path(self, action_folder_path: str) -> str:
        """从路径中提取action ID"""
        return os.path.basename(action_folder_path)

    def process_episode_with_lerobot(self, episode_path: str, action_folder_path: str):
        """使用 LeRobot API 处理单个episode"""
        action_id = self.get_action_id_from_path(action_folder_path)
        task_index, task_name, task_description = self.get_task_info_for_action(action_id)
        instruction = self.get_instruction_for_action(action_id)
        
        self.logger.info(f"开始处理episode: {episode_path}")
        self.logger.info(f"Action: {action_id} -> Task: {task_name} (index: {task_index})")
        
        # 读取BSON文件 - 使用固定的文件名
        robot_bson_path = os.path.join(episode_path, self.config.robot_bson_name)
        hand_bson_path = os.path.join(episode_path, self.config.hand_bson_name)
    
        if not os.path.exists(robot_bson_path) or not os.path.exists(hand_bson_path):
            self.logger.error(f"BSON文件不存在: {robot_bson_path} 或 {hand_bson_path}")
            return
        
        # 读取数据
        robot_data = self.read_bson_file(robot_bson_path)
        hand_data = self.read_bson_file(hand_bson_path)
        
        # 加载图像
        images = self.load_images_from_folders(episode_path, action_id)
        
        # 如果camera_front没有从文件夹加载到数据，尝试从BSON获取
        # 判断逻辑：如果配置中没有映射camera_front，或者从文件夹没有加载到数据，则从BSON获取
        camera_mapping = self.config.get_camera_mapping(action_id)
        
        if not images.get('camera_front') or 'camera_front' not in camera_mapping:
            self.logger.info("尝试从BSON提取camera_front数据...")
            front_images = self.extract_images_from_bson(robot_data)
            if front_images.get('camera_front'):
                images['camera_front'] = front_images['camera_front']
                self.logger.info(f"从BSON成功提取camera_front: {len(images['camera_front'])} 帧")
            else:
                self.logger.warning("从BSON也无法提取到camera_front数据")
        
        # 提取关节数据
        robot_joint_data = self.extract_robot_joint_data(robot_data)
        hand_joint_data = self.extract_hand_data(hand_data)
        
        # 确定帧数
        frame_counts = []
        for key, data in robot_joint_data.items():
            if len(data) > 0:
                frame_counts.append(len(data))
        for key, data in hand_joint_data.items():
            if len(data) > 0:
                frame_counts.append(len(data))
        for camera_name, img_list in images.items():
            if len(img_list) > 0:
                frame_counts.append(len(img_list))
        
        if not frame_counts:
            self.logger.error(f"Episode {episode_path} 没有有效数据")
            return
        
        num_frames = min(frame_counts)
        self.logger.info(f"Episode 帧数: {num_frames}")
        
        # 使用 LeRobot API 添加数据
        try:
            for frame_idx in range(num_frames):
                # 转换为 LeRobot 格式
                frame = self.convert_frame_to_lerobot_format(
                    frame_idx, robot_joint_data, hand_joint_data, images
                )
                
                # 计算时间戳
                timestamp = frame_idx / self.config.fps
                
                # 添加帧到数据集
                self.dataset.add_frame(frame, task=instruction, timestamp=timestamp)
            
            # 保存episode
            self.dataset.save_episode()
            
            self.logger.info(f"Episode 处理完成: {num_frames} 帧")
            
        except Exception as e:
            self.logger.error(f"处理episode失败: {e}")
            raise

    def get_processed_episodes(self) -> set:
        """获取已处理的episode集合 (如果数据集已存在)"""
        processed = set()
        
        if self.dataset and hasattr(self.dataset, 'meta') and hasattr(self.dataset.meta, 'episodes'):
            episodes_stats = self.dataset.meta.episodes_stats
            for ep_idx, ep_info in episodes_stats.items():
                if 'original_path' in ep_info:
                    processed.add(ep_info['original_path'])
        
        return processed

    def process_all_episodes(self):
        """处理所有episodes"""
        self.logger.info("=== 开始处理所有episodes ===")
        
        # 初始化 LeRobot 数据集
        self.dataset = self.setup_lerobot_dataset()
        
        # 获取已处理的episodes (如果继续处理)
        if not self.config.overwrite:
            self.processed_episodes = self.get_processed_episodes()
            self.logger.info(f"检测到已处理的episodes: {len(self.processed_episodes)}")
        
        # 扫描源数据目录
        source_root = self.config.source_data_root
        processed_count = 0
        
        for action_folder in os.listdir(source_root):
            action_folder_path = os.path.join(source_root, action_folder)
            if not os.path.isdir(action_folder_path):
                continue
            
            # 只处理action开头的文件夹
            if not action_folder.startswith('action'):
                self.logger.debug(f"跳过非action文件夹: {action_folder}")
                continue
            
            action_id = self.get_action_id_from_path(action_folder_path)
            self.logger.info(f"\n处理Action: {action_id}")
            
            # 遍历该action下的所有episodes
            for episode_folder in os.listdir(action_folder_path):
                episode_path = os.path.join(action_folder_path, episode_folder)
                if not os.path.isdir(episode_path):
                    continue
                
                # 检查是否已处理
                if episode_path in self.processed_episodes:
                    self.logger.info(f"跳过已处理的episode: {episode_path}")
                    continue
                
                try:
                    self.process_episode_with_lerobot(episode_path, action_folder_path)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        self.logger.info(f"已处理 {processed_count} 个episodes")
                    
                except Exception as e:
                    self.logger.error(f"处理episode失败 {episode_path}: {e}")
                    continue
        
        self.logger.info(f"=== 处理完成，共处理 {processed_count} 个episodes ===")
        
        # 保存最终统计信息
        if self.dataset:
            self.logger.info(f"数据集总episodes: {self.dataset.num_episodes}")
            self.logger.info(f"数据集总帧数: {len(self.dataset)}")


def main():
    """主函数"""
    # 配置参数
    config = AirbotConfig()
    
    # 创建处理器并开始处理
    processor = AirbotLeRobotProcessor(config)
    processor.process_all_episodes()


if __name__ == "__main__":
    main()
