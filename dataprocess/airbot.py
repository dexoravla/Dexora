import json
import os
import numpy as np
import shutil
import struct
import bson
import cv2
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from collections import defaultdict
from tqdm import tqdm
import datetime
import time
import logging
import tempfile
from typing import Dict, List, Any, Union, Optional
from airbot_config import AirbotConfig
import subprocess


class AirbotDataProcessor:
    def __init__(self, config: AirbotConfig):
        self.config = config
        self.setup_logging()
        self.current_episode_index = 0
        self.total_frames = 0
        self.episode_data_buffer = []
        self.action_to_task_index = {}
        self.current_task_index = 0
        self.task_index_to_info = {}
        self.init_dataset_structure()
        # 检测已处理的episode数量并更新
        self.current_episode_index = self.detect_existing_episodes()
        self.total_frames = self.calculate_existing_frames()
        
        # 如果检测到已有数据，需要重新计算chunk
        if self.current_episode_index > 0:
            new_chunk = self.calculate_current_chunk()
            if new_chunk != self.current_chunk:
                self.current_chunk = new_chunk
                # 更新目录路径
                self.data_dir = os.path.join(self.dataset_root, "data", f"chunk-{self.current_chunk:03d}")
                self.video_dir = os.path.join(self.dataset_root, "videos", f"chunk-{self.current_chunk:03d}")
                self.logger.info(f"更新到chunk: {self.current_chunk}")
        
        # 🔧 加载已有的task映射
        self.load_existing_task_mapping()
        self._initialize_action_task_mapping()
        
    def detect_existing_episodes(self) -> int:
        """检测已存在的episode数量"""
        try:
            episodes_file = os.path.join(self.meta_dir, "episodes.jsonl")
            if not os.path.exists(episodes_file):
                self.logger.info("未找到已有episodes，从0开始")
                return 0
            
            max_episode_index = -1
            with open(episodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        episode_data = json.loads(line.strip())
                        max_episode_index = max(max_episode_index, episode_data['episode_index'])
            
            next_index = max_episode_index + 1
            self.logger.info(f"检测到已处理 {next_index} 个episodes，从index {next_index} 继续")
            return next_index
            
        except Exception as e:
            self.logger.warning(f"检测已有episodes失败: {e}，从0开始")
            return 0
        
    def calculate_existing_frames(self) -> int:
        """计算已有episodes的总帧数"""
        try:
            episodes_stats_file = os.path.join(self.meta_dir, "episodes_stats.jsonl")
            if not os.path.exists(episodes_stats_file):
                return 0
            
            total_frames = 0
            with open(episodes_stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        stats = json.loads(line.strip())
                        total_frames += stats.get('length', 0)
            
            self.logger.info(f"已有数据总帧数: {total_frames}")
            return total_frames
            
        except Exception as e:
            self.logger.warning(f"计算已有帧数失败: {e}")
            return 0
        
    def load_existing_task_mapping(self):
        """加载已有的task映射"""
        try:
            tasks_file = os.path.join(self.meta_dir, "tasks.jsonl")
            if not os.path.exists(tasks_file):
                return
            
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task_data = json.loads(line.strip())
                        task_index = task_data['task_index']
                        
                        self.task_index_to_info[task_index] = {
                            'task_name': task_data['task'],
                            'description': task_data['description'],
                            'category': task_data.get('category', 'uncategorized'),
                            'actions': task_data.get('actions', [])
                        }
                        
                        # 重建action到task_index的映射
                        for action_id in task_data.get('actions', []):
                            self.action_to_task_index[action_id] = task_index
            
            if self.task_index_to_info:
                self.current_task_index = max(self.task_index_to_info.keys()) + 1
                self.logger.info(f"加载已有task映射: {len(self.task_index_to_info)} 个tasks")
            
        except Exception as e:
            self.logger.warning(f"加载已有task映射失败: {e}")
    def _initialize_action_task_mapping(self):
        """初始化action到task_index的映射 - 按task_name分配task_index"""
        self.logger.info("=== 初始化Action到Task_Index映射(按Task_Name) ===")
        
        # 🔧 第一步：收集所有独特的task_name
        task_names = set()
        action_to_task_name = {}
        
        # 从task_categories获取action和对应的task_name映射
        if hasattr(self.config, 'task_categories'):
            for category_name, category_info in self.config.task_categories.items():
                task_name = category_info.get('task_name', category_name)
                actions = category_info.get('actions', [])
                
                task_names.add(task_name)
                for action_id in actions:
                    action_to_task_name[action_id] = task_name
        
        # 从action_instruction_mapping获取剩余的actions
        if hasattr(self.config, 'action_instruction_mapping'):
            for action_id in self.config.action_instruction_mapping.keys():
                if action_id not in action_to_task_name:
                    # 如果没有在task_categories中找到，使用默认task_name
                    default_task_name = f'task_{action_id}'
                    action_to_task_name[action_id] = default_task_name
                    task_names.add(default_task_name)
        
        # 🔧 第二步：为每个独特的task_name分配task_index
        sorted_task_names = sorted(list(task_names))
        task_name_to_index = {}
        
        for idx, task_name in enumerate(sorted_task_names):
            task_name_to_index[task_name] = idx
            self.logger.info(f"Task_Name '{task_name}' -> Task_Index {idx}")
        
        # 🔧 第三步：为每个action分配对应的task_index
        for action_id, task_name in action_to_task_name.items():
            task_index = task_name_to_index[task_name]
            self.action_to_task_index[action_id] = task_index
            
            # 创建任务信息映射
            if task_index not in self.task_index_to_info:
                # 查找该task_name的详细信息
                task_info = self._find_task_info_by_task_name(task_name)
                
                self.task_index_to_info[task_index] = {
                    'task_name': task_name,
                    'description': task_info.get('description', f'Task {task_name}'),
                    'category': task_info.get('category', 'uncategorized'),
                    'actions': []  # 存储属于这个task的所有actions
                }
            
            # 将action添加到对应task的actions列表中
            self.task_index_to_info[task_index]['actions'].append(action_id)
            
            self.logger.info(f"Action {action_id} -> Task_Name '{task_name}' -> Task_Index {task_index}")
        
        # 🔧 第四步：更新task信息，显示每个task包含的actions
        for task_index, task_info in self.task_index_to_info.items():
            actions_list = task_info['actions']
            self.logger.info(f"Task_Index {task_index} ('{task_info['task_name']}') 包含 {len(actions_list)} 个actions: {actions_list}")
        
        self.logger.info(f"总共创建了 {len(task_name_to_index)} 个独特的task_index")

    def _find_task_info_by_task_name(self, task_name: str) -> dict:
        """根据task_name查找任务详细信息"""
        if hasattr(self.config, 'task_categories'):
            for category_name, category_info in self.config.task_categories.items():
                if category_info.get('task_name', category_name) == task_name:
                    return {
                        'description': category_info.get('description', ''),
                        'category': category_name
                    }
        
        # 默认返回
        return {
            'description': f'Task {task_name}',
            'category': 'uncategorized'
        }
        

    def get_today_time(self):
        """获取当前时间字符串"""
        today = datetime.datetime.now()
        return today.strftime("%Y%m%d%H%M%S")
    
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
        self.logger = logging.getLogger("AirbotDataProcessor")

    def init_dataset_structure(self):
        """初始化数据集目录结构"""
        self.dataset_name = f"{self.config.robot}_{self.config.task_name}"
        self.dataset_root = os.path.join(self.config.output_data_root, self.dataset_name)
        
        # 动态计算当前应该使用的chunk
        self.current_chunk = self.calculate_current_chunk()
        
        # 创建必要的目录 - 使用动态chunk
        self.data_dir = os.path.join(self.dataset_root, "data", f"chunk-{self.current_chunk:03d}")
        self.video_dir = os.path.join(self.dataset_root, "videos", f"chunk-{self.current_chunk:03d}")
        self.meta_dir = os.path.join(self.dataset_root, "meta")
        self.device_dir = os.path.join(self.dataset_root, "device")
        self.label_dir = os.path.join(self.dataset_root, "label")
        
        for dir_path in [self.data_dir, self.video_dir, self.meta_dir, 
                        self.device_dir, self.label_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 为每个相机创建视频目录
        for camera_name in ['camera_high', 'camera_left', 'camera_right', 'camera_front']:
            camera_video_dir = os.path.join(self.video_dir, f"observation.images.{camera_name}")
            os.makedirs(camera_video_dir, exist_ok=True)
        
        self.logger.info(f"初始化数据集结构: {self.dataset_root}")
        self.logger.info(f"当前使用chunk: {self.current_chunk}")

    def calculate_current_chunk(self) -> int:
        """计算当前应该使用的chunk编号"""
        chunk_size = 1000  # 每个chunk最多1000个episodes
        return self.current_episode_index // chunk_size
    
    def get_chunk_for_episode(self, episode_index: int) -> int:
        """获取指定episode应该所在的chunk"""
        chunk_size = 1000
        return episode_index // chunk_size

    def ensure_chunk_directories(self, chunk_index: int):
        """确保指定chunk的目录存在"""
        chunk_data_dir = os.path.join(self.dataset_root, "data", f"chunk-{chunk_index:03d}")
        chunk_video_dir = os.path.join(self.dataset_root, "videos", f"chunk-{chunk_index:03d}")
        
        os.makedirs(chunk_data_dir, exist_ok=True)
        os.makedirs(chunk_video_dir, exist_ok=True)
        
        # 为每个相机创建视频目录
        for camera_name in ['camera_high', 'camera_left', 'camera_right', 'camera_front']:
            camera_video_dir = os.path.join(chunk_video_dir, f"observation.images.{camera_name}")
            os.makedirs(camera_video_dir, exist_ok=True)
        
        return chunk_data_dir, chunk_video_dir
    def read_bson_file(self, bson_path: str) -> Dict[str, Any]:
        """读取BSON文件"""
        try:
            with open(bson_path, 'rb') as f:
                size_bytes = f.read(4)
                if len(size_bytes) < 4:
                    raise ValueError("文件太小，无法包含有效的BSON文档")
                
                f.seek(0)
                document_size = struct.unpack('<i', size_bytes)[0]
                f.seek(0)
                bson_data = f.read(document_size)
                document = bson.decode(bson_data)
                return document
                
        except Exception as e:
            self.logger.error(f"读取BSON文件失败: {bson_path}, 错误: {e}")
            return {}

    def load_images_from_folders(self, episode_path: str, action_name: str) -> Dict[str, List[np.ndarray]]:
        """从文件夹加载图像序列 - 包含所有4个相机"""
        images = {
            'camera_high': [],
            'camera_left': [],
            'camera_right': [],
            'camera_front': []  # 🔧 front相机也从文件夹获取
        }
        
        # 获取当前action的相机映射
        camera_mapping = self.config.get_camera_mapping(action_name)
        self.logger.info(f"使用 {action_name} 的相机映射: {camera_mapping}")
        
        # 🔧 统一处理所有4个相机，包括front相机
        for camera_name, folder_name in camera_mapping.items():
            camera_folder = os.path.join(episode_path, folder_name)
            if not os.path.exists(camera_folder):
                self.logger.warning(f"相机文件夹不存在: {camera_folder}")
                continue
                
            image_files = [f for f in os.listdir(camera_folder) if f.endswith('.jpg')]
            if not image_files:
                self.logger.warning(f"相机文件夹中没有图像文件: {camera_folder}")
                continue
                
            # 按文件名中的数字排序
            try:
                image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            except (IndexError, ValueError):
                # 如果文件名格式不符合预期，使用默认排序
                image_files.sort()
            
            camera_images = []
            for img_file in image_files:
                img_path = os.path.join(camera_folder, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    camera_images.append(img_rgb)
                else:
                    self.logger.warning(f"无法加载图像: {img_path}")
            
            images[camera_name] = camera_images
            self.logger.info(f"从 {folder_name} 加载 {camera_name}: {len(camera_images)} 张图像")
        
        return images

    def extract_images_from_bson(self, robot_data: Dict[str, Any]) -> Dict[str, List[np.ndarray]]:
        """从BSON文件中提取相机图像数据 - 主要用于提取front相机数据"""
        images = {
            'camera_front': []  # 只处理front相机
        }
        
        try:
            data_section = robot_data.get('data', {})
            
            # 检查 /images/head_camera 路径（前置相机数据）
            head_camera_data = data_section.get('/images/head_camera', None)
            
            if head_camera_data:
                self.logger.info("找到 /images/head_camera 数据")
                self.logger.info(f"数据类型: {type(head_camera_data)}")
                
                # 处理字节数据（MP4视频流）
                if isinstance(head_camera_data, bytes):
                    self.logger.info(f"发现压缩的视频数据流，大小: {len(head_camera_data) / 1024 / 1024:.2f} MB")
                    
                    # 检查数据前几个字节以确定格式
                    header = head_camera_data[:20]
                    self.logger.info(f"数据头(hex): {header.hex()}")
                    
                    # 检查是否为MP4格式
                    is_mp4 = False
                    if b'ftyp' in head_camera_data[:20]:  # MP4文件标识
                        is_mp4 = True
                        self.logger.info("检测到MP4格式视频")
                    elif head_camera_data.startswith(b'\x00\x00\x00'):
                        # 可能是MP4但文件头稍有不同
                        is_mp4 = True
                        self.logger.info("检测到可能的MP4格式，尝试解析")
                    
                    if is_mp4:
                        # 从MP4视频流中提取帧
                        frames = self.extract_frames_from_mp4_bytes(head_camera_data)
                        if frames:
                            images['camera_front'] = frames
                            self.logger.info(f"✅ 成功从MP4视频提取 {len(frames)} 帧")
                        else:
                            self.logger.warning("❌ 无法从MP4视频提取帧")
                    else:
                        self.logger.warning("数据不是MP4格式，尝试其他解析方法")
                        # 可以添加其他格式的解析逻辑
                
                elif isinstance(head_camera_data, list):
                    # 处理图像列表
                    self.logger.info(f"发现图像列表，共 {len(head_camera_data)} 项")
                    frames = []bsonvladataset
                    for item in head_camera_data:
                        if isinstance(item, bytes):
                            # 尝试解码字节数据为图像
                            try:
                                img_array = np.frombuffer(item, dtype=np.uint8)
                                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                                if img is not None:
                                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    frames.append(img_rgb)
                            except Exception as e:
                                self.logger.warning(f"无法解码图像数据: {e}")
                    if frames:
                        images['camera_front'] = frames
                        self.logger.info(f"✅ 成功从列表提取 {len(frames)} 帧")
                
                else:
                    self.logger.warning(f"未知的数据类型: {type(head_camera_data)}")
            
            else:
                self.logger.warning("未找到 /images/head_camera 数据")
                
                # 打印所有可用的键以帮助调试
                all_keys = list(data_section.keys())
                self.logger.info(f"BSON data中所有的键 (前20个): {all_keys[:20]}")
                
                # 尝试查找任何包含图像数据的键
                image_keys = [key for key in all_keys if 'image' in key.lower() or 'camera' in key.lower()]
                if image_keys:
                    self.logger.info(f"发现可能的图像键: {image_keys}")
                        
        except Exception as e:
            self.logger.error(f"从BSON提取图像失败: {e}")
            import traceback
            self.logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return images

    def extract_frames_from_mp4_bytes(self, video_bytes: bytes) -> List[np.ndarray]:
        """从MP4视频字节数据中提取帧"""
        frames = []
        temp_video_path = None
        
        try:
            # 创建临时目录（如果不存在）
            temp_dir = tempfile.gettempdir()
            
            # 创建唯一的临时文件名
            import uuid
            temp_filename = f"temp_video_{self.current_episode_index}_{uuid.uuid4().hex}.mp4"
            temp_video_path = os.path.join(temp_dir, temp_filename)
            
            # 保存视频数据到临时文件
            self.logger.info(f"保存视频到临时文件: {temp_video_path}")
            with open(temp_video_path, 'wb') as f:
                f.write(video_bytes)
            
            # 验证文件是否成功创建
            if not os.path.exists(temp_video_path):
                self.logger.error("临时视频文件创建失败")
                return frames
            
            file_size = os.path.getsize(temp_video_path)
            self.logger.info(f"临时视频文件大小: {file_size} 字节")
            
            # 使用OpenCV读取视频
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                self.logger.error("OpenCV无法打开视频文件")
                # 尝试使用ffmpeg验证文件
                try:
                    import subprocess
                    result = subprocess.run(['ffmpeg', '-i', temp_video_path], 
                                        capture_output=True, text=True)
                    self.logger.info(f"FFmpeg输出: {result.stderr[:500]}")
                except:
                    pass
                return frames
            
            # 获取视频属性
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"视频属性 - 分辨率: {width}x{height}, FPS: {fps:.2f}, 总帧数: {frame_count}")
            
            # 读取所有帧
            successfully_read = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                successfully_read += 1
                
                # 进度报告
                if successfully_read % 50 == 0:
                    self.logger.info(f"  已读取 {successfully_read}/{frame_count} 帧...")
            
            cap.release()
            self.logger.info(f"✅ 成功提取 {len(frames)} 帧")
            
        except Exception as e:
            self.logger.error(f"提取MP4帧时出错: {e}")
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
        
        finally:
            # 清理临时文件
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                    self.logger.info("已删除临时视频文件")
                except Exception as e:
                    self.logger.warning(f"删除临时文件失败: {e}")
        
        return frames

    def create_video_from_images(self, images: List[np.ndarray], output_path: str):
        """从图像序列创建视频文件 - 符合LeRobot v2.1标准"""
        if not images:
            self.logger.warning(f"没有图像可用于创建视频: {output_path}")
            return
        
        height, width = images[0].shape[:2]
        self.logger.info(f"🎬 创建视频: {output_path}")
        self.logger.info(f"   - 帧数: {len(images)}")
        self.logger.info(f"   - 尺寸: {width}x{height}")
        self.logger.info(f"   - FPS: {self.config.fps}")
        
        # 🔧 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 🔧 直接使用ffmpeg创建H.264视频
            cmd = [
                'ffmpeg',
                '-y',  # 覆盖输出文件
                '-f', 'rawvideo',  # 输入格式：原始视频
                '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}',  # 尺寸
                '-pix_fmt', 'rgb24',  # 输入像素格式
                '-r', str(self.config.fps),  # 帧率
                '-i', '-',  # 从stdin读取
                '-c:v', 'libx264',  # 🔧 H.264编码器
                '-pix_fmt', 'yuv420p',  # 🔧 LeRobot标准像素格式
                '-crf', '18',  # 🔧 高质量设置
                '-preset', 'medium',  # 编码速度vs质量平衡
                '-movflags', '+faststart',  # 优化流媒体播放
                output_path
            ]
            
            # 启动ffmpeg进程
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 将图像数据写入ffmpeg
            for i, img in enumerate(images):
                # 确保图像是RGB格式且为uint8
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                if img.shape[2] == 3:  # RGB
                    img_bytes = img.tobytes()
                    process.stdin.write(img_bytes)
                else:
                    self.logger.error(f"图像格式错误: {img.shape}")
                    process.terminate()
                    return
                
                # 进度报告
                if (i + 1) % 50 == 0:
                    self.logger.info(f"   - 已处理 {i + 1}/{len(images)} 帧")
            
            # 关闭stdin并等待完成
            process.stdin.close()
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                # 验证输出文件
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path)
                    self.logger.info(f"✅ 成功创建H.264视频: {file_size} bytes")
                    
                    # 🔧 验证视频编码格式
                    self.verify_video_codec(output_path)
                else:
                    self.logger.error("ffmpeg命令执行成功但输出文件不存在")
            else:
                self.logger.error(f"ffmpeg执行失败 (返回码: {process.returncode})")
                self.logger.error(f"ffmpeg stderr: {stderr.decode()[:500]}")
                
        except FileNotFoundError:
            self.logger.error("未找到ffmpeg，请安装ffmpeg: sudo apt-get install ffmpeg")
        except Exception as e:
            self.logger.error(f"创建视频时出错: {e}")
            
    def verify_video_codec(self, video_path: str):
        """验证视频编码格式是否符合LeRobot标准"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', 
                '-show_streams', '-select_streams', 'v:0', 
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                
                if 'streams' in probe_data and len(probe_data['streams']) > 0:
                    video_stream = probe_data['streams'][0]
                    codec_name = video_stream.get('codec_name', 'unknown')
                    pix_fmt = video_stream.get('pix_fmt', 'unknown')
                    
                    # 验证编码格式
                    if codec_name == 'h264':
                        self.logger.info("✅ 视频编码验证: H.264 ✓")
                    else:
                        self.logger.warning(f"⚠️  视频编码不是H.264: {codec_name}")
                    
                    # 验证像素格式
                    if pix_fmt == 'yuv420p':
                        self.logger.info("✅ 像素格式验证: yuv420p ✓")
                    else:
                        self.logger.warning(f"⚠️  像素格式不是yuv420p: {pix_fmt}")
                    
                    # 输出完整信息
                    fps = video_stream.get('r_frame_rate', 'unknown')
                    width = video_stream.get('width', 'unknown')
                    height = video_stream.get('height', 'unknown')
                    
                    self.logger.info(f"📹 视频信息: {width}x{height}, {fps} fps, {codec_name}/{pix_fmt}")
                    
                else:
                    self.logger.warning("无法获取视频流信息")
            else:
                self.logger.warning("ffprobe执行失败，跳过编码验证")
                
        except FileNotFoundError:
            self.logger.warning("未找到ffprobe，跳过编码验证")
        except Exception as e:
            self.logger.warning(f"验证视频编码时出错: {e}")

    def extract_robot_joint_data(self, robot_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """从机器人BSON数据中提取关节数据"""
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
            
        except Exception as e:
            self.logger.error(f"提取机器人关节数据失败: {e}")
        
        return joint_data

    def extract_hand_data(self, hand_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """从灵巧手BSON数据中提取手部数据"""
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
                
                # ✅ 将手部observation从角度转换为弧度，与action保持一致
                left_obs = [np.deg2rad(angle) for angle in left_obs]
                right_obs = [np.deg2rad(angle) for angle in right_obs]
                
                left_hand_obs.append(left_obs)
                right_hand_obs.append(right_obs)
            
            hand_joint_data['left_hand_action'] = np.array(left_hand_actions)
            hand_joint_data['right_hand_action'] = np.array(right_hand_actions)
            hand_joint_data['left_hand_obs'] = np.array(left_hand_obs)
            hand_joint_data['right_hand_obs'] = np.array(right_hand_obs)
            
            # 记录转换信息
            self.logger.info(f"手部数据单位转换: observation 角度→弧度")
            
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

    def read_instruction_from_action_txt(self, action_folder_path: str) -> str:
        """优化后的指令读取方法"""
        action_id = self.get_action_id_from_path(action_folder_path)
        
        # 首先尝试从配置映射获取
        instruction = self.get_instruction_for_action(action_id)
        if instruction and instruction != self.config.default_instruction:
            self.logger.info(f"从配置映射获取 {action_id} 指令: {instruction}")
            return instruction
        
        # 其次尝试从action.txt文件读取
        if self.config.use_action_txt:
            action_txt_path = os.path.join(action_folder_path, "action.txt")
            try:
                if os.path.exists(action_txt_path):
                    with open(action_txt_path, 'r', encoding='utf-8') as f:
                        txt_instruction = f.read().strip()
                        if txt_instruction:
                            self.logger.info(f"从action.txt读取到 {action_id} 指令: {txt_instruction}")
                            return txt_instruction
            except Exception as e:
                self.logger.error(f"读取action.txt失败: {e}")
        
        # 最后使用默认指令
        self.logger.warning(f"使用默认指令给 {action_id}")
        return f"{self.config.default_instruction}_{action_id}"

    def detect_abnormal_episodes(self, robot_joint_data: Dict, hand_joint_data: Dict, num_frames: int) -> str:
        """检测异常episode，返回状态：'normal', 'abnormal' 或具体异常类型"""
        try:
            # 检查数据完整性
            min_required_frames = int(num_frames * 0.8)  # 至少80%的数据
            
            for key, data in {**robot_joint_data, **hand_joint_data}.items():
                if len(data) < min_required_frames:
                    self.logger.warning(f"{key} 数据不完整: {len(data)}/{num_frames} 帧")
                    return "incomplete_data"
            
            # 检查动作幅度是否异常
            for key, data in robot_joint_data.items():
                if len(data) > 1:
                    # 计算相邻帧之间的最大变化
                    max_change = np.max(np.abs(np.diff(data, axis=0)))
                    if max_change > 1.0:  # 单帧变化超过1弧度认为异常
                        self.logger.warning(f"{key} 动作幅度异常: 最大变化 {max_change}")
                        return "large_motion_jump"
                    
                    # 检查是否有NaN或Inf
                    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
                        self.logger.warning(f"{key} 包含无效数值")
                        return "invalid_values"
            
            # 检查手部数据异常 - 更新为弧度范围
            for key, data in hand_joint_data.items():
                if len(data) > 0:
                    # 检查数值范围 - 更新为弧度范围：-π到π
                    if np.any(data < -np.pi) or np.any(data > np.pi):
                        self.logger.warning(f"{key} 数值超出正常范围(弧度): [{np.min(data):.3f}, {np.max(data):.3f}]")
                        return "out_of_range"
            
            return "normal"
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return "detection_error"
    def validate_data_units(self, robot_joint_data: Dict, hand_joint_data: Dict):
        """验证数据单位是否正确"""
        self.logger.info("=== 数据单位验证 ===")
        
        # 验证机械臂数据（应该在合理的弧度范围内）
        for key, data in robot_joint_data.items():
            if len(data) > 0:
                min_val, max_val = np.min(data), np.max(data)
                self.logger.info(f"{key}: 范围 [{min_val:.3f}, {max_val:.3f}] 弧度")
                if max_val - min_val > 2 * np.pi:
                    self.logger.warning(f"{key}: 数据范围可能过大")
        
        # 验证手部数据（应该在弧度范围内）
        for key, data in hand_joint_data.items():
            if len(data) > 0:
                min_val, max_val = np.min(data), np.max(data)
                self.logger.info(f"{key}: 范围 [{min_val:.3f}, {max_val:.3f}] 弧度")
                
                # 检查是否像角度数据（通常会很大）
                if np.any(np.abs(data) > 10):  # 如果有值大于10弧度，可能是角度
                    self.logger.warning(f"{key}: 数值可能仍为角度单位，请检查转换")
                
                # 检查转换是否成功
                if 'obs' in key and np.any(np.abs(data) > np.pi):
                    self.logger.warning(f"{key}: observation数据超出[-π, π]范围")
    def validate_episode_data(self, episode_index: int, num_frames: int, instruction: str, images: Dict) -> bool:
        """验证episode数据质量"""
        is_valid = True
        
        # 检查帧数是否合理
        if num_frames < 10:
            self.logger.warning(f"Episode {episode_index} 帧数过少: {num_frames}")
            is_valid = False
        elif num_frames > 10000:
            self.logger.warning(f"Episode {episode_index} 帧数过多: {num_frames}")
            is_valid = False
        
        # 检查instruction是否有效
        if not instruction or instruction == "" or len(instruction) < 3:
            self.logger.warning(f"Episode {episode_index} instruction无效: '{instruction}'")
            is_valid = False
        
        # 检查图像数据 - 只检查front相机有数据
        if 'camera_front' not in images or len(images['camera_front']) == 0:
            self.logger.warning(f"Episode {episode_index} 缺少前置相机数据")
            is_valid = False
        
        # 检查front相机的视频文件是否成功生成
        video_path = os.path.join(self.video_dir, f"observation.images.camera_front", 
                                 f"episode_{episode_index:06d}.mp4")
        if os.path.exists(video_path):
            # 检查文件大小
            file_size = os.path.getsize(video_path)
            if file_size < 1000:  # 小于1KB认为是无效视频
                self.logger.warning(f"Episode {episode_index} 前置相机视频文件过小")
                is_valid = False
        else:
            self.logger.warning(f"Episode {episode_index} 前置相机视频文件缺失")
            is_valid = False
        
        return is_valid
    def update_episode_metadata(self, episode_index: int, num_frames: int, 
                            instruction: str, task_index: int = 0, 
                            task_name: Optional[str] = None, action_id: Optional[str] = None,
                            original_episode_path: Optional[str] = None):  # 🔧 添加原始路径参数
        """更新episode元数据"""
        # 更新episodes.jsonl
        episodes_file = os.path.join(self.meta_dir, "episodes.jsonl")
        episode_entry = {
            "episode_index": episode_index,
            "length": num_frames,
            "timestamp": datetime.datetime.now().isoformat(),
            "split": "train",
            "action_id": action_id,
            "task_name": task_name,
            "original_path": original_episode_path  # 🔧 存储原始episode路径
        }
        
        with open(episodes_file, 'a', encoding='utf-8') as f:
            json.dump(episode_entry, f, ensure_ascii=False)
            f.write('\n')
        
        # 更新episodes_stats.jsonl
        episodes_stats_file = os.path.join(self.meta_dir, "episodes_stats.jsonl")
        stats_entry = {
            "episode_index": episode_index,
            "instruction": instruction,
            "length": num_frames,
            "fps": self.config.fps,
            "duration": num_frames / self.config.fps,
            "task_index": task_index,
            "task": task_name or self.config.task_type,
            "action_id": action_id,
            "original_path": original_episode_path  # 🔧 存储原始episode路径
        }
        
        with open(episodes_stats_file, 'a', encoding='utf-8') as f:
            json.dump(stats_entry, f, ensure_ascii=False)
            f.write('\n')

    def get_action_id_from_path(self, action_folder_path: str) -> str:
        """从路径中提取action ID"""
        return os.path.basename(action_folder_path)
    
    def get_instruction_for_action(self, action_id: str) -> str:
        """根据action ID获取对应的指令"""
        if hasattr(self.config, 'action_instruction_mapping'):
            instruction = self.config.action_instruction_mapping.get(action_id, '')
            if instruction:
                return instruction
        
        # 如果没有找到映射，尝试从action.txt读取
        return self.config.default_instruction
    
    def get_task_info_for_action(self, action_id: str) -> tuple:
        """获取action对应的任务信息 - 使用自动生成的task_index"""
        
        # 从映射中获取task_index
        task_index = self.action_to_task_index.get(action_id, 0)
        
        # 获取任务信息
        if task_index in self.task_index_to_info:
            task_info = self.task_index_to_info[task_index]
            return (
                task_index,
                task_info['task_name'],
                task_info['description']
            )
        else:
            # 备用逻辑：如果映射中没有找到，创建新的
            self.logger.warning(f"Action {action_id} 未在映射中找到，创建新的task_index")
            new_task_index = len(self.action_to_task_index)
            self.action_to_task_index[action_id] = new_task_index
            
            task_info = self._find_task_info_by_task_name(f'task_{action_id}')
            
            # 🔧 修复：使用正确的数据结构
            self.task_index_to_info[new_task_index] = {
                'task_name': f'task_{action_id}',
                'description': task_info.get('description', f'Task for {action_id}'),
                'category': task_info.get('category', 'uncategorized'),
                'actions': [action_id]  # 🔧 修复：使用actions列表而不是action_id
            }
            
            return (
                new_task_index,
                f'task_{action_id}',
                task_info.get('description', f'Task for {action_id}')
            )
    

    def process_episode(self, episode_path: str, episode_index: int, action_folder_path: str):
        """处理单个episode并保存为parquet格式"""
        self.logger.info(f"开始处理episode {episode_index}: {episode_path}")
        
        # 确定episode所属的chunk
        episode_chunk = self.get_chunk_for_episode(episode_index)
        chunk_data_dir, chunk_video_dir = self.ensure_chunk_directories(episode_chunk)
        
        self.logger.info(f"Episode {episode_index} 将保存到 chunk-{episode_chunk:03d}")
        
        # 提取action信息
        action_id = self.get_action_id_from_path(action_folder_path)
        task_index, task_name, task_description = self.get_task_info_for_action(action_id)
        
        self.logger.info(f"Action: {action_id} -> Task_Index: {task_index}, Task: {task_name}")
        
        # 读取instruction
        instruction = self.read_instruction_from_action_txt(action_folder_path)
        
        # 读取BSON文件
        robot_bson_path = os.path.join(episode_path, self.config.robot_bson_name)
        hand_bson_path = os.path.join(episode_path, self.config.hand_bson_name)
        
        if not os.path.exists(robot_bson_path) or not os.path.exists(hand_bson_path):
            self.logger.error(f"Episode {episode_index} BSON文件缺失")
            return
        
        robot_data = self.read_bson_file(robot_bson_path)
        hand_data = self.read_bson_file(hand_bson_path)
        
        # 从文件夹加载图像（high, left, right相机）
        images = self.load_images_from_folders(episode_path, action_id)
        
        #  如果front相机文件夹没有数据，尝试从BSON获取作为备用
        if not images.get('camera_front'):
            self.logger.info("从文件夹未找到front相机数据，尝试从BSON提取...")
            bson_images = self.extract_images_from_bson(robot_data)
            if bson_images.get('camera_front'):
                images['camera_front'] = bson_images['camera_front']
                self.logger.info(f"从BSON成功提取front相机: {len(images['camera_front'])} 帧")
        
        #  打印各相机的帧数信息
        for camera_name, img_list in images.items():
            self.logger.info(f"{camera_name}: {len(img_list)} 帧")
        
        #  提取关节数据
        robot_joint_data = self.extract_robot_joint_data(robot_data)
        hand_joint_data = self.extract_hand_data(hand_data)
        
        #  验证数据单位
        self.validate_data_units(robot_joint_data, hand_joint_data)
        
        #  确定帧数 - 使用所有可用数据的最小帧数
        frame_counts = []
        for key, data in robot_joint_data.items():
            if len(data) > 0:
                frame_counts.append(len(data))
        for key, data in hand_joint_data.items():
            if len(data) > 0:
                frame_counts.append(len(data))
        
        # 添加有数据的相机帧数
        for camera_name, img_list in images.items():
            if len(img_list) > 0:
                frame_counts.append(len(img_list))
        
        if not frame_counts:
            self.logger.warning(f"Episode {episode_index} 没有有效数据")
            return
        
        num_frames = min(frame_counts)
        self.logger.info(f"确定帧数: {num_frames} (来自: {frame_counts})")
        
        #  检测异常
        abnormal_status = self.detect_abnormal_episodes(robot_joint_data, hand_joint_data, num_frames)
        if abnormal_status != "normal":
            self.logger.warning(f"Episode {episode_index} 检测到异常: {abnormal_status}")
            # 如果是异常episode，更新instruction
            instruction = f"{instruction} [ABNORMAL: {abnormal_status}]"
        
        #  构建episode数据 - 参考要求的列顺序和数据类型
        episode_data = {
            'states': [],
            'actions': [],
            'next.done': [],
            'timestamp': [],
            'frame_index': [],
            'episode_index': [],
            'index': [],
            'task_index': []
        }
        
        for frame_idx in range(num_frames):
            # 添加状态数据（36维） - 存储为object类型
            states = np.zeros(36, dtype=np.float32)
            # 左臂
            if 'left_arm_obs' in robot_joint_data and frame_idx < len(robot_joint_data['left_arm_obs']):
                states[0:6] = robot_joint_data['left_arm_obs'][frame_idx]
            # 左手
            if 'left_hand_obs' in hand_joint_data and frame_idx < len(hand_joint_data['left_hand_obs']):
                states[6:18] = hand_joint_data['left_hand_obs'][frame_idx]
            # 右臂
            if 'right_arm_obs' in robot_joint_data and frame_idx < len(robot_joint_data['right_arm_obs']):
                states[18:24] = robot_joint_data['right_arm_obs'][frame_idx]
            # 右手
            if 'right_hand_obs' in hand_joint_data and frame_idx < len(hand_joint_data['right_hand_obs']):
                states[24:36] = hand_joint_data['right_hand_obs'][frame_idx]
            
            episode_data['states'].append(states)
            
            # 添加动作数据（36维） - 存储为object类型
            actions = np.zeros(36, dtype=np.float32)
            # 左臂
            if 'left_arm_action' in robot_joint_data and frame_idx < len(robot_joint_data['left_arm_action']):
                actions[0:6] = robot_joint_data['left_arm_action'][frame_idx]
            # 左手
            if 'left_hand_action' in hand_joint_data and frame_idx < len(hand_joint_data['left_hand_action']):
                actions[6:18] = hand_joint_data['left_hand_action'][frame_idx]
            # 右臂
            if 'right_arm_action' in robot_joint_data and frame_idx < len(robot_joint_data['right_arm_action']):
                actions[18:24] = robot_joint_data['right_arm_action'][frame_idx]
            # 右手
            if 'right_hand_action' in hand_joint_data and frame_idx < len(hand_joint_data['right_hand_action']):
                actions[24:36] = hand_joint_data['right_hand_action'][frame_idx]
            
            episode_data['actions'].append(actions)
            
            # 其他字段按照要求的数据类型
            episode_data['next.done'].append(frame_idx == num_frames - 1)  # bool类型
            episode_data['timestamp'].append(np.float32(frame_idx / self.config.fps))  # float32类型
            episode_data['frame_index'].append(np.int64(frame_idx))  # int64类型
            episode_data['episode_index'].append(np.int64(episode_index))  # int64类型
            episode_data['index'].append(np.int64(self.total_frames + frame_idx))  # int64类型
            episode_data['task_index'].append(np.int64(task_index))  # 使用动态的task_index
        
        #  创建DataFrame并确保数据类型正确
        df = pd.DataFrame(episode_data)
        
        # 显式设置数据类型
        df['states'] = df['states'].astype('object')
        df['actions'] = df['actions'].astype('object')
        df['next.done'] = df['next.done'].astype('bool')
        df['timestamp'] = df['timestamp'].astype('float32')
        df['frame_index'] = df['frame_index'].astype('int64')
        df['episode_index'] = df['episode_index'].astype('int64')
        df['index'] = df['index'].astype('int64')
        df['task_index'] = df['task_index'].astype('int64')
        
        # 确保列顺序正确
        df = df[['states', 'actions', 'next.done', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']]
        
        # 保存parquet文件到正确的chunk目录
        parquet_path = os.path.join(chunk_data_dir, f"episode_{episode_index:06d}.parquet")
        df.to_parquet(parquet_path, index=False)
        
        #  创建视频文件到正确的chunk目录
        self.logger.info("=== 开始创建视频文件 ===")
        video_created_count = 0

        for camera_name, img_list in images.items():
            if img_list:
                img_list_trimmed = img_list[:num_frames]
                #  使用正确的chunk目录
                camera_video_dir_full = os.path.join(chunk_video_dir, f"observation.images.{camera_name}")
                video_path = os.path.join(camera_video_dir_full, f"episode_{episode_index:06d}.mp4")
                
                self.logger.info(f"🎬 准备创建 {camera_name} 视频:")
                self.logger.info(f"   - 输入帧数: {len(img_list_trimmed)}")
                self.logger.info(f"   - 输出路径: {video_path}")
                self.logger.info(f"   - 图像尺寸: {img_list_trimmed[0].shape if img_list_trimmed else 'N/A'}")
                
                self.create_video_from_images(img_list_trimmed, video_path)
                
                # 验证视频文件
                if os.path.exists(video_path):
                    file_size = os.path.getsize(video_path)
                    self.logger.info(f"✅ 创建 {camera_name} 视频成功: {video_path} ({file_size} bytes)")
                    video_created_count += 1
                else:
                    self.logger.error(f"❌ 创建 {camera_name} 视频失败: {video_path}")
            else:
                self.logger.warning(f"⚠️  {camera_name} 没有图像数据，跳过视频创建")

        self.logger.info(f"=== 视频创建完成: {video_created_count}/4 个相机 ===")

        # 数据验证
        if not self.validate_episode_data(episode_index, num_frames, instruction, images):
            self.logger.error(f"Episode {episode_index} 数据验证失败")
        
        #  更新元数据
        self.update_episode_metadata(episode_index, num_frames, instruction, 
                                task_index, task_name, action_id, episode_path)
        
        self.total_frames += num_frames
        self.logger.info(f"完成处理episode {episode_index}, 共 {num_frames} 帧")
        
        # 打印数据类型信息用于验证
        self.logger.info(f"Parquet文件数据类型: {df.dtypes.to_dict()}")


    def get_processed_episodes(self) -> set:
        """获取已处理的episode路径集合"""
        processed_episodes = set()
        
        episodes_file = os.path.join(self.meta_dir, "episodes.jsonl")
        if os.path.exists(episodes_file):
            with open(episodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        episode_data = json.loads(line.strip())
                        original_path = episode_data.get('original_path')
                        if original_path and os.path.exists(original_path):
                            processed_episodes.add(original_path)
                        else:
                            # 兼容旧格式：根据action_id重建路径
                            action_id = episode_data.get('action_id', '')
                            if action_id:
                                action_path = os.path.join(self.config.source_data_root, action_id)
                                if os.path.exists(action_path):
                                    episode_folders = [d for d in os.listdir(action_path) 
                                                    if d.startswith('episode_')]
                                    episode_folders.sort()
                                    # 使用时间戳或其他方式匹配
                                    # 这里简化处理，可能需要更精确的匹配逻辑
                                    for episode_folder in episode_folders:
                                        episode_full_path = os.path.join(action_path, episode_folder)
                                        processed_episodes.add(episode_full_path)
        
        return processed_episodes
    def create_meta_info(self):
        """创建meta/info.json文件"""
        # 计算实际的chunk数量
        max_episode_index = self.current_episode_index - 1 if self.current_episode_index > 0 else 0
        total_chunks = self.get_chunk_for_episode(max_episode_index) + 1 if self.current_episode_index > 0 else 1
        # 统计每个chunk的episode数量
        chunk_stats = {}
        for chunk_idx in range(total_chunks):
            chunk_data_dir = os.path.join(self.dataset_root, "data", f"chunk-{chunk_idx:03d}")
            if os.path.exists(chunk_data_dir):
                parquet_files = [f for f in os.listdir(chunk_data_dir) if f.endswith('.parquet')]
                chunk_stats[chunk_idx] = len(parquet_files)
            else:
                chunk_stats[chunk_idx] = 0
        # 计算任务统计
        task_counts = {}
        task_names = {}
        
        # 统计每个task_index对应的episode数量
        for task_index, task_info in self.task_index_to_info.items():
            actions_list = task_info.get('actions', [])
            task_counts[task_index] = len(actions_list)
            task_names[task_index] = task_info['task_name']
        
        total_tasks = len(self.task_index_to_info)
        total_videos = self.current_episode_index * 4
        info = {
            "codebase_version": "v2.1",
            "robot_type": self.config.robot,
            "total_episodes": self.current_episode_index,
            "total_frames": self.total_frames,
            "total_tasks": total_tasks,
            #"task_distribution": task_counts,
            #"task_names": task_names,
            "total_videos": total_videos,  
            "total_chunks": 1,  
            "chunks_size": 1000,  
            "fps": self.config.fps,
            "splits": {
                "train": f"0:{self.current_episode_index}"
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.camera_high": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.config.fps,
                        "video.height": 480,
                        "video.width": 640,
                        "video.channels": 3,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.camera_left": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.config.fps,
                        "video.height": 480,
                        "video.width": 640,
                        "video.channels": 3,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.camera_right": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.config.fps,
                        "video.height": 480,
                        "video.width": 640,
                        "video.channels": 3,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "observation.images.camera_front": {
                    "dtype": "video",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.fps": self.config.fps,
                        "video.height": 480,
                        "video.width": 640,
                        "video.channels": 3,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "has_audio": False
                    }
                },
                "states": {
                    "dtype": "float32",
                    "shape": [36],
                    "name": self.config.state_names
                },
                "actions": {
                    "dtype": "float32",
                    "shape": [36],
                    "name": self.config.action_names
                },
                "next.done": {
                    "dtype": "bool",
                    "shape": [1],
                    "names": None
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": None
                }
            }
        }
        
        info_path = os.path.join(self.meta_dir, "info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    def create_tasks_jsonl(self):
        """创建包含多个任务的tasks.jsonl文件 - 按task_name分组"""
        tasks_file = os.path.join(self.meta_dir, "tasks.jsonl")
        
        # 🔧 使用task_name分组的task_index映射
        for task_index in sorted(self.task_index_to_info.keys()):
            task_info = self.task_index_to_info[task_index]
            
            task_entry = {
                "task_index": task_index,
                "task": task_info['task_name'],
                "description": task_info['description'],
                "actions": task_info['actions'],  # 🔧 显示属于这个task的所有actions
                "category": task_info['category']
            }
            
            with open(tasks_file, 'a', encoding='utf-8') as f:
                json.dump(task_entry, f, ensure_ascii=False)
                f.write('\n')
        
        self.logger.info(f"生成了 {len(self.task_index_to_info)} 个任务的tasks.jsonl文件(按task_name分组)")
    def save_device_info(self):
        """保存设备信息 - 使用配置文件中的设备信息"""
        
        # 标准格式的设备信息 - 使用配置文件中的信息
        device_info = {
            "device_list": [
                {
                    "device_id": self.config.device_id,
                    "device_type": self.config.device_type,
                    "device_type_info": self.config.device_type_info,
                    "device_info": {
                        "camera_high": {
                            "type": self.config.camera_models["camera_high"],
                            "resolution": "640x480",
                            "description": "顶部俯视相机",
                            "dimension": {
                                "width": 640,
                                "height": 480,
                                "channels": 3
                            }
                        },
                        "camera_left": {
                            "type": self.config.camera_models["camera_left"],
                            "resolution": "640x480", 
                            "description": "左侧相机",
                            "dimension": {
                                "width": 640,
                                "height": 480,
                                "channels": 3
                            }
                        },
                        "camera_right": {
                            "type": self.config.camera_models["camera_right"],
                            "resolution": "640x480",
                            "description": "右侧相机", 
                            "dimension": {
                                "width": 640,
                                "height": 480,
                                "channels": 3
                            }
                        },
                        "camera_front": {
                            "type": self.config.camera_models["camera_front"],
                            "resolution": "640x480",
                            "description": "前置头部相机",
                            "dimension": {
                                "width": 640,
                                "height": 480,
                                "channels": 3
                            }
                        },
                        "left_arm": {
                            "type": self.config.arm_models["left_arm"],
                            "joints": 6,
                            "dimension": {
                                "left_arm_joint_1": "rad",
                                "left_arm_joint_2": "rad", 
                                "left_arm_joint_3": "rad",
                                "left_arm_joint_4": "rad",
                                "left_arm_joint_5": "rad",
                                "left_arm_joint_6": "rad",
                                "left_end_effector_positions_x": "m",
                                "left_end_effector_positions_y": "m", 
                                "left_end_effector_positions_z": "m",
                                "left_end_effector_quat_x": None,
                                "left_end_effector_quat_y": None,
                                "left_end_effector_quat_z": None,
                                "left_end_effector_quat_w": None
                            }
                        },
                        "right_arm": {
                            "type": self.config.arm_models["right_arm"], 
                            "joints": 6,
                            "dimension": {
                                "right_arm_joint_1": "rad",
                                "right_arm_joint_2": "rad",
                                "right_arm_joint_3": "rad", 
                                "right_arm_joint_4": "rad",
                                "right_arm_joint_5": "rad",
                                "right_arm_joint_6": "rad",
                                "right_end_effector_positions_x": "m",
                                "right_end_effector_positions_y": "m",
                                "right_end_effector_positions_z": "m", 
                                "right_end_effector_quat_x": None,
                                "right_end_effector_quat_y": None,
                                "right_end_effector_quat_z": None,
                                "right_end_effector_quat_w": None
                            }
                        },
                        "left_hand": {
                            "type": self.config.hand_models["left_hand"],
                            "dof": 12,
                            "has_force_feedback": True,
                            "dimension": {
                                "left_hand_joint_1": "rad",
                                "left_hand_joint_2": "rad",
                                "left_hand_joint_3": "rad",
                                "left_hand_joint_4": "rad", 
                                "left_hand_joint_5": "rad",
                                "left_hand_joint_6": "rad",
                                "left_hand_joint_7": "rad",
                                "left_hand_joint_8": "rad",
                                "left_hand_joint_9": "rad",
                                "left_hand_joint_10": "rad",
                                "left_hand_joint_11": "rad",
                                "left_hand_joint_12": "rad"
                            }
                        },
                        "right_hand": {
                            "type": self.config.hand_models["right_hand"],
                            "dof": 12, 
                            "has_force_feedback": True,
                            "dimension": {
                                "right_hand_joint_1": "rad",
                                "right_hand_joint_2": "rad",
                                "right_hand_joint_3": "rad",
                                "right_hand_joint_4": "rad",
                                "right_hand_joint_5": "rad",
                                "right_hand_joint_6": "rad",
                                "right_hand_joint_7": "rad",
                                "right_hand_joint_8": "rad",
                                "right_hand_joint_9": "rad",
                                "right_hand_joint_10": "rad",
                                "right_hand_joint_11": "rad", 
                                "right_hand_joint_12": "rad"
                            }
                        }
                    }
                }
            ]
        }
        
        device_info_path = os.path.join(self.device_dir, "device_info.json")
        with open(device_info_path, "w", encoding="utf-8") as f:
            json.dump(device_info, f, indent=4, ensure_ascii=False)
        
        # 相机内参和标定信息 - 使用配置文件
        calibration_info = self.create_calibration_info_from_config()
        camera_intrinsic_path = os.path.join(self.device_dir, "camera_intrinsic.json")
        with open(camera_intrinsic_path, "w", encoding="utf-8") as f:
            json.dump(calibration_info, f, indent=4, ensure_ascii=False)
        
        # 设备episode映射
        device_episode_path = os.path.join(self.device_dir, "device_episode.json")
        device_episodes = []
        for i in range(self.current_episode_index):
            device_episodes.append({
                "episode_index": i,
                "device_id": self.config.device_id
            })
        with open(device_episode_path, "w", encoding="utf-8") as f:
            json.dump(device_episodes, f, indent=4, ensure_ascii=False)

    def create_calibration_info_from_config(self) -> Dict[str, Any]:
        """从配置文件创建标定信息 """
        import datetime
        
        # 参考格式：使用device_list结构
        calibration_info = {
            "device_list": [
                {
                    "device_id": self.config.device_id,
                    "calibration_info": {
                        "version": self.config.device_version,
                        "date": datetime.datetime.now().strftime("%Y-%m-%d"),  # 简化日期格式
                        "source": "calibration_tool",  # 标定工具来源
                        "reprojection_error": getattr(self.config, 'reprojection_error', 0.8)
                    },
                    "camera_intrinsics": {
                        "top": {  # camera_high -> top
                            "model": "PINHOLE",
                            "parameters": {
                                "width": 640,
                                "height": 480,
                                "fx": self.config.camera_calibration["camera_high"].get("fx", 605.849),
                                "fy": self.config.camera_calibration["camera_high"].get("fy", 605.742),
                                "cx": self.config.camera_calibration["camera_high"].get("cx", 309.673),
                                "cy": self.config.camera_calibration["camera_high"].get("cy", 245.429)
                            },
                            "distortion": {
                                "model": "Inverse Brown Conrady",
                                "k1": self.config.camera_calibration["camera_high"].get("k1", 0),
                                "k2": self.config.camera_calibration["camera_high"].get("k2", 0),
                                "k3": self.config.camera_calibration["camera_high"].get("k3", 0),
                                "p1": self.config.camera_calibration["camera_high"].get("p1", 0),
                                "p2": self.config.camera_calibration["camera_high"].get("p2", 0)
                            }
                        },
                        "left": {  # camera_left -> left
                            "model": "PINHOLE",
                            "parameters": {
                                "width": 640,
                                "height": 480,
                                "fx": self.config.camera_calibration["camera_left"].get("fx", 607.072),
                                "fy": self.config.camera_calibration["camera_left"].get("fy", 607.943),
                                "cx": self.config.camera_calibration["camera_left"].get("cx", 315.199),
                                "cy": self.config.camera_calibration["camera_left"].get("cy", 245.262)
                            },
                            "distortion": {
                                "model": "Inverse Brown Conrady",
                                "k1": self.config.camera_calibration["camera_left"].get("k1", 0),
                                "k2": self.config.camera_calibration["camera_left"].get("k2", 0),
                                "k3": self.config.camera_calibration["camera_left"].get("k3", 0),
                                "p1": self.config.camera_calibration["camera_left"].get("p1", 0),
                                "p2": self.config.camera_calibration["camera_left"].get("p2", 0)
                            }
                        },
                        "right": {  # camera_right -> right
                            "model": "PINHOLE",
                            "parameters": {
                                "width": 640,
                                "height": 480,
                                "fx": self.config.camera_calibration["camera_right"].get("fx", 606.736),
                                "fy": self.config.camera_calibration["camera_right"].get("fy", 605.451),
                                "cx": self.config.camera_calibration["camera_right"].get("cx", 322.341),
                                "cy": self.config.camera_calibration["camera_right"].get("cy", 255.021)
                            },
                            "distortion": {
                                "model": "Inverse Brown Conrady",
                                "k1": self.config.camera_calibration["camera_right"].get("k1", 0),
                                "k2": self.config.camera_calibration["camera_right"].get("k2", 0),
                                "k3": self.config.camera_calibration["camera_right"].get("k3", 0),
                                "p1": self.config.camera_calibration["camera_right"].get("p1", 0),
                                "p2": self.config.camera_calibration["camera_right"].get("p2", 0)
                            }
                        },
                        "front": {  # camera_front -> front
                            "model": "PINHOLE",
                            "parameters": {
                                "width": 640,
                                "height": 480,
                                "fx": self.config.camera_calibration["camera_front"].get("fx", 600.0),
                                "fy": self.config.camera_calibration["camera_front"].get("fy", 600.0),
                                "cx": self.config.camera_calibration["camera_front"].get("cx", 320.0),
                                "cy": self.config.camera_calibration["camera_front"].get("cy", 240.0)
                            },
                            "distortion": {
                                "model": "Inverse Brown Conrady",
                                "k1": self.config.camera_calibration["camera_front"].get("k1", 0),
                                "k2": self.config.camera_calibration["camera_front"].get("k2", 0),
                                "k3": self.config.camera_calibration["camera_front"].get("k3", 0),
                                "p1": self.config.camera_calibration["camera_front"].get("p1", 0),
                                "p2": self.config.camera_calibration["camera_front"].get("p2", 0)
                            }
                        }
                    }
                }
            ]
        }
        
        return calibration_info
    def save_label_info(self):
        """保存标签信息 - 调整为参考格式的嵌套结构"""
        annotation_path = os.path.join(self.label_dir, "data_annotation.json")
        
        # 读取episodes_stats.jsonl获取instruction信息
        episodes_stats_path = os.path.join(self.meta_dir, "episodes_stats.jsonl")
        episode_instructions = {}
        episode_action_ids = {}
        
        if os.path.exists(episodes_stats_path):
            with open(episodes_stats_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        stats = json.loads(line.strip())
                        episode_instructions[stats['episode_index']] = stats.get('instruction', self.config.default_instruction)
                        episode_action_ids[stats['episode_index']] = stats.get('action_id', 'unknown')
        
        annotations = []
        for i in range(self.current_episode_index):
            instruction = episode_instructions.get(i, self.config.default_instruction)
            action_id = episode_action_ids.get(i, 'unknown')
            
            # 获取episode的帧数
            episodes_file = os.path.join(self.meta_dir, "episodes.jsonl")
            episode_length = 100  # 默认值
            if os.path.exists(episodes_file):
                with open(episodes_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            episode_data = json.loads(line.strip())
                            if episode_data['episode_index'] == i:
                                episode_length = episode_data['length']
                                break
            
            # 构建路径信息
            episode_path = f"{self.config.source_data_root}/{action_id}/episode_{i % 10}"
            video_path = f"{self.dataset_root}/videos/chunk-000/observation.images.camera_high/episode_{i:06d}.mp4"
            
            # 根据instruction判断是否为异常episode
            timelinelabels = [instruction]
            if "[ABNORMAL:" in instruction:
                timelinelabels.append("abnormal")
            else:
                timelinelabels.append("end")
            
            # 🔧 修复：完整构建videoLabels，移除ellipsis
            annotations.append({
                "episode_index": i,
                "annotation": {
                    # 添加路径信息
                    "path": episode_path,
                    "video": video_path,
                    "id": 96397 + i,
                    # 🔧 完整的videoLabels结构
                    "videoLabels": [
                        {
                            "ranges": [
                                {
                                    "start": 0,
                                    "end": episode_length
                                }
                            ],
                            "timelinelabels": timelinelabels
                        }
                    ],
                    # 将annotator等字段移到annotation内部
                    "annotator": 1,
                    "annotation_id": 80000 + i,
                    "created_at": datetime.datetime.now().isoformat() + "Z",
                    "updated_at": datetime.datetime.now().isoformat() + "Z", 
                    "lead_time": 100.0
                    # 移除result字段 - 不再包含冗余数据
                }
            })
        
        with open(annotation_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=4, ensure_ascii=False)

    def create_readme(self):
        """创建README.md文件"""
        readme_content = f"""# {self.dataset_name}

## Dataset Information
- **Robot Type**: {self.config.robot}
- **Task**: {self.config.task_type}
- **Total Episodes**: {self.current_episode_index}
- **Total Frames**: {self.total_frames}
- **FPS**: {self.config.fps}

## Structure
- `data/`: Episode data in parquet format
- `videos/`: Video recordings for each camera
- `meta/`: Metadata files
- `device/`: Device configuration and calibration
- `label/`: Data annotations

## Usage
This dataset is formatted for LeRobot v2.1.
"""
        
        readme_path = os.path.join(self.dataset_root, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def process_all_episodes(self):
        """处理所有episode - 支持增量处理"""
        processed_episodes = set()
        # 从episodes.jsonl读取已处理的episodes
        episodes_file = os.path.join(self.meta_dir, "episodes.jsonl")
        if os.path.exists(episodes_file):
            with open(episodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        episode_data = json.loads(line.strip())
                        original_path = episode_data.get('original_path')
                        if original_path:
                            if os.path.exists(original_path):
                                processed_episodes.add(original_path)
                                self.logger.debug(f"已处理: {original_path}")
                            else:
                                self.logger.warning(f"已处理的episode路径不存在: {original_path}")
        
        self.logger.info(f"从episodes.jsonl检测到已处理的episodes: {len(processed_episodes)} 个")
        
        # 🔧 如果没有original_path信息，使用episodes_stats.jsonl作为备用检测方法
        if len(processed_episodes) == 0:
            self.logger.info("episodes.jsonl中没有original_path信息，尝试从episodes_stats.jsonl检测")
            processed_episodes = self.detect_processed_episodes_from_stats()
        
        # 🔧 调试：打印前5个已处理的episode路径
        if processed_episodes:
            sample_paths = list(processed_episodes)[:5]
            self.logger.info(f"已处理episode示例: {sample_paths}")
        
        # 遍历所有存在的episodes，找出未处理的
        action_folders = [d for d in os.listdir(self.config.source_data_root) 
                        if d.startswith('action')]
        action_folders.sort()
        
        new_episodes_count = 0
        total_found_episodes = 0
        skipped_episodes = 0
        
        for action_folder in action_folders:
            action_path = os.path.join(self.config.source_data_root, action_folder)
            
            episode_folders = [d for d in os.listdir(action_path) 
                            if d.startswith('episode_')]
            episode_folders.sort()
            
            total_found_episodes += len(episode_folders)
            self.logger.info(f"Action {action_folder}: 发现 {len(episode_folders)} 个episodes")
            
            for episode_folder in episode_folders:
                episode_path = os.path.join(action_path, episode_folder)
                
                # 🔧 严谨：精确匹配episode路径
                if episode_path in processed_episodes:
                    self.logger.debug(f"跳过已处理的episode: {episode_path}")
                    skipped_episodes += 1
                    continue
                
                try:
                    # 检查是否需要切换到新的chunk
                    current_episode_chunk = self.get_chunk_for_episode(self.current_episode_index)
                    if current_episode_chunk != self.current_chunk:
                        self.logger.info(f"切换到新的chunk: {current_episode_chunk}")
                        self.current_chunk = current_episode_chunk
                        # 更新目录路径
                        self.data_dir = os.path.join(self.dataset_root, "data", f"chunk-{self.current_chunk:03d}")
                        self.video_dir = os.path.join(self.dataset_root, "videos", f"chunk-{self.current_chunk:03d}")
                    
                    self.logger.info(f"🆕 处理新episode: {episode_path} (episode_index: {self.current_episode_index})")
                    self.process_episode(episode_path, self.current_episode_index, action_path)
                    self.current_episode_index += 1
                    new_episodes_count += 1
                except Exception as e:
                    self.logger.error(f"处理episode失败: {episode_path}, 错误: {e}")
                    import traceback
                    self.logger.error(f"详细错误信息: {traceback.format_exc()}")
                    continue
        
        # 🔧 详细统计信息
        self.logger.info(f"=== 处理统计 ===")
        self.logger.info(f"发现的总episodes: {total_found_episodes}")
        self.logger.info(f"跳过的已处理episodes: {skipped_episodes}")
        self.logger.info(f"新处理的episodes: {new_episodes_count}")
        self.logger.info(f"当前总episodes: {self.current_episode_index}")
        
        # 🔧 验证一致性
        expected_processed = self.current_episode_index - new_episodes_count
        if skipped_episodes != expected_processed:
            self.logger.warning(f"检测不一致: 跳过{skipped_episodes}个，期望{expected_processed}个")
        
        if new_episodes_count == 0:
            self.logger.info("✅ 没有新的episodes需要处理，所有数据都是最新的")
        else:
            self.logger.info(f"✅ 新处理了 {new_episodes_count} 个episodes")
            
            # 只有在有新数据时才更新元数据
            self.update_meta_info()
            self.update_tasks_jsonl()
            self.save_device_info()
            self.save_label_info()
            self.create_readme()
        
        self.logger.info(f"增量处理完成！")
        max_chunk = self.get_chunk_for_episode(self.current_episode_index - 1) if self.current_episode_index > 0 else 0
        self.logger.info(f"数据分布在 {max_chunk + 1} 个chunks中 (chunk-000 到 chunk-{max_chunk:03d})")



    def detect_processed_episodes_from_stats(self) -> set:
        """从episodes_stats.jsonl检测已处理的episodes（备用方法）"""
        processed_episodes = set()
        
        episodes_stats_file = os.path.join(self.meta_dir, "episodes_stats.jsonl")
        if os.path.exists(episodes_stats_file):
            # 🔧 收集所有已处理的action_id和episode信息
            processed_actions = {}  # action_id -> [episode_indices]
            
            with open(episodes_stats_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        stats = json.loads(line.strip())
                        action_id = stats.get('action_id', '')
                        episode_index = stats.get('episode_index', 0)
                        
                        if action_id:
                            if action_id not in processed_actions:
                                processed_actions[action_id] = []
                            processed_actions[action_id].append(episode_index)
            
            # 🔧 为每个action，按episode_index顺序匹配到具体的episode文件夹
            for action_id, episode_indices in processed_actions.items():
                action_path = os.path.join(self.config.source_data_root, action_id)
                if os.path.exists(action_path):
                    episode_folders = [d for d in os.listdir(action_path) 
                                    if d.startswith('episode_')]
                    episode_folders.sort()
                    
                    # 🔧 按处理顺序匹配：第一个处理的对应第一个文件夹
                    episode_indices.sort()
                    for i, processed_idx in enumerate(episode_indices):
                        if i < len(episode_folders):
                            episode_full_path = os.path.join(action_path, episode_folders[i])
                            processed_episodes.add(episode_full_path)
                            self.logger.debug(f"从stats匹配: episode_index {processed_idx} -> {episode_full_path}")
            
            self.logger.info(f"从episodes_stats.jsonl检测到已处理的episodes: {len(processed_episodes)} 个")
        
        return processed_episodes
    def update_tasks_jsonl(self):
        """增量更新tasks.jsonl - 不覆盖已有数据"""
        tasks_file = os.path.join(self.meta_dir, "tasks.jsonl")
        
        # 🔧 读取已有的task_index
        existing_tasks = set()
        if os.path.exists(tasks_file):
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        task_data = json.loads(line.strip())
                        existing_tasks.add(task_data['task_index'])
        
        # 🔧 只添加新的tasks
        new_tasks = []
        for task_index in sorted(self.task_index_to_info.keys()):
            if task_index not in existing_tasks:
                task_info = self.task_index_to_info[task_index]
                task_entry = {
                    "task_index": task_index,
                    "task": task_info['task_name'],
                    "description": task_info['description'],
                    "actions": task_info['actions'],
                    "category": task_info['category']
                }
                new_tasks.append(task_entry)
        
        # 🔧 追加新的tasks
        if new_tasks:
            with open(tasks_file, 'a', encoding='utf-8') as f:
                for task_entry in new_tasks:
                    json.dump(task_entry, f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"新增 {len(new_tasks)} 个tasks到tasks.jsonl")
        else:
            self.logger.info("没有新的tasks需要添加")

    def update_meta_info(self):
        """更新meta/info.json - 使用最新的统计信息"""
        # 重新计算所有统计信息
        total_tasks = len(self.task_index_to_info)
        total_videos = self.current_episode_index * 4
        
        # 使用最新的数据重新生成info.json
        self.create_meta_info()  # 这个方法会重新生成完整的info.json

def main():
    # 配置参数
    config = AirbotConfig()
    
    # 创建处理器并开始处理
    processor = AirbotDataProcessor(config)
    processor.process_all_episodes()


if __name__ == "__main__":
    main()