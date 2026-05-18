import json
import os
import numpy as np
import shutil
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import cv2

from realman_config import DataConfig
import h5py
import datetime
import time
import pandas as pd
from nas_sdk import NASAuthenticator
import logging


class DataProcessor(object):
    def __init__(self, config,json_path,nas_path,task_name):
        self.config = config

        # if self.config.overwrite:
        #     data_root = self.config.data_root
        #     shutil.rmtree(os.path.join(data_root, self.config.repo_id), ignore_errors=True)
        self.data_root = None
        self.json_path = json_path
        self.nas_path = nas_path
        self.task_name = task_name
        self.create_dataset()
        self.nas_auth = NASAuthenticator() # 初始化nas
        self.setup_logging()  # 初始化日志

    def get_today_time(self):
        # 获取当前日期和时间
        today = datetime.datetime.now()
        
        # 格式化日期为字符串，格式为 "YYYY-MM-DD"
        date_string = today.strftime("%Y%m%d%H%M%S")
        return date_string
    
    def setup_logging(self):
        """配置日志输出到文件和控制台"""
        log_dir = os.path.join(self.config.log_root, self.task_name)
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
        self.logger = logging.getLogger("DataProcessor")

    def create_dataset(self):
        # 构建 features 字典
        features = {}
        _repo_id = self.config.robot + '_' + self.task_name
        self.data_root = os.path.join(self.config.data_root, self.get_today_time(),_repo_id)
       
        for camera_dict in self.config.rgb_names:
            rgb_name = next(iter(camera_dict))  # 获取字典的键（如 'cam_high'）
            features[rgb_name] = camera_dict[rgb_name]  # 赋值配置
        
        # rgb_config = {
        #     'dtype': 'video',
        #     'shape': (self.config.image_height, self.config.image_width, 3),
        #     'name': ['height', 'width', 'channels'],
        # }
        # features = {rgb_name: self.config.rgb_names[rgb_name] for rgb_name in self.config.rgb_names}
        features['states'] = {
            'dtype': 'float32',
            'shape': (self.config.action_len,),
            'name': self.config.action_name,
        }
        features['actions'] = {
            'dtype': 'float32',
            'shape': (self.config.action_len,),
            'name': self.config.action_name,
        }
        features['next.done'] = {
            "dtype": "bool",
            "shape": (1,),
            "names": None
        }
        self.dataset = LeRobotDataset.create(
            repo_id=_repo_id,
            fps=self.config.fps,
            root=self.data_root,
            robot_type = self.config.robot,
            video_backend = self.config.video_backend,
            image_writer_processes = self.config.num_image_writer_processes,
            image_writer_threads = self.config.num_image_writer_threads_per_camera, # add_frame 线程
            features=features,
        )
    

    def recover_rotation_matrix(self, col1, col2):
        """
        从旋转矩阵的前两列恢复完整的3x3旋转矩阵
        
        参数:
        col1, col2 (array-like): 旋转矩阵的前两列，形状为(3,)
        
        返回:
        numpy.ndarray: 3x3的旋转矩阵
        """
        col1 = np.array(col1, dtype=np.float32)
        col2 = np.array(col2, dtype=np.float32)
        
        # 检查向量是否接近零
        eps = 1e-10  # 设定一个小的阈值
        
        # 确保col1是单位向量
        norm_col1 = np.linalg.norm(col1)
        if norm_col1 < eps:
            raise ValueError("col1 几乎为零向量，无法归一化")
        col1 = col1 / norm_col1
        
        # 从col2中移除col1的投影，使其正交
        col2 = col2 - np.dot(col2, col1) * col1
        
        # 确保col2也是单位向量
        norm_col2 = np.linalg.norm(col2)
        if norm_col2 < eps:
            # 如果正交后的col2几乎为零，尝试寻找与col1正交的向量
            # 这里选择一个与col1正交的任意向量
            if abs(col1[0]) < 0.5:
                temp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                temp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            col2 = temp - np.dot(temp, col1) * col1
            col2 = col2 / np.linalg.norm(col2)
        else:
            col2 = col2 / norm_col2
        
        # 计算第三列，已经是单位向量
        col3 = np.cross(col1, col2)
        
        # 组合成旋转矩阵
        R = np.column_stack((col1, col2, col3))
        
        # 检查行列式是否为1（确保是旋转而非反射）
        if np.linalg.det(R) < 0:
            # 如果行列式为负，翻转第三列
            R[:, 2] = -R[:, 2]
        
        return R

    def rotation_matrix_to_quaternion(self,R):
        """
        将3x3旋转矩阵转换为四元数表示
        
        参数:
        R (numpy.ndarray): 3x3的旋转矩阵
        
        返回:
        numpy.ndarray: 四元数，格式为[w, x, y, z]
        """
        R = np.array(R, dtype=np.float32)
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        # 归一化四元数，提高数值稳定性
        quaternion = np.array([w, x, y, z], dtype=np.float32)
        return quaternion / np.linalg.norm(quaternion)
    
    def rotation_recovery(self, col1, col2): # 将6D pose转成四元数
        R = self.recover_rotation_matrix(col1, col2)
        return self.rotation_matrix_to_quaternion(R)
    
    def read_csv(self, path):# 读取任务nas列表
        df = pd.read_csv(path)   
        # 提取 'json_path' 和 'nas_path' 两列
        selected_columns = df[['json_path', 'nas_path']]        
        # 转换成嵌套列表（每行是一个子列表）
        nested_list = selected_columns.values.tolist()       
        # 打印前3行示例
        #self.logger.info("嵌套列表（前3行）:", nested_list[:3])        
        return nested_list
    
    def download_from_nas(self,nas_path,an_path): # 下载hdf5数据到本地，并返回nas路径、本地数据路径
        # 根据规则拼接nas路径
        parts = []
        last_name = ""
        for i in range(10):
            path,tail = os.path.split(an_path)
            if tail:
                if 'realman' in tail:
                    break
                parts.append(tail)
            an_path = path
        for i in reversed(parts):
            last_name += i + '/'
        last_name = last_name[:-1]
        local_path_dir = os.path.basename(nas_path)
        local_path = os.path.join(self.config.hdf5_root,local_path_dir,last_name)
        nas_path = os.path.join(nas_path,last_name)
        # 下载文件
        if self.nas_auth.download_file(nas_path,local_path):
            return nas_path,local_path
        else:
            return None,None

        


    def parse_annotation(self):
        if not self.nas_auth.get_auth_sid():
            self.logger.info("nas not connect")
            return
        new_annotation = []
        new_device = []
        if os.path.exists(self.json_path):
            with open(self.json_path,"r",encoding='utf-8') as file:
                data = json.load(file)
            for source_idx,hdf5_file in enumerate(data):
                episode_path = hdf5_file["path"]
                nas_path,local_path = self.download_from_nas(self.nas_path,episode_path)
                # 开始数据转换
                self.logger.info(f'Processing source {source_idx + 1}, {local_path}')
                self._add_episode2(local_path)
                hdf5_file["nas_path"] = nas_path # 将nas路径放到标注文件中
                episode_annotation = {
                    "episode_index": source_idx,
                    "annotation":hdf5_file
                }
                episode_device = {
                    "episode_index": source_idx,
                    "device_id":"7xK992pQ"
                }
                new_annotation.append(episode_annotation)
                new_device.append(episode_device)
            
            # 标注文件
            new_annotation_path = os.path.join(self.data_root,"label","data_annotation.json")
            os.makedirs(os.path.dirname(new_annotation_path) or ".", exist_ok=True)  # 创建目录（如果不存在）
            with open(new_annotation_path, "w", encoding="utf-8") as file:
                json.dump(new_annotation, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved new annotation to: {new_annotation_path}")
            # 设备信息
            new_machine_path = os.path.join(self.data_root,"device","device_info.json")
            os.makedirs(os.path.dirname(new_machine_path) or ".", exist_ok=True)  # 创建目录（如果不存在）
            with open(new_machine_path, "w", encoding="utf-8") as file:
                new_machine = self.config.device_info
                json.dump(new_machine, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved new device to: {new_machine_path}")
            #相机参数
            new_camera_path = os.path.join(self.data_root,"device","camera_intrinsic.json")
            os.makedirs(os.path.dirname(new_camera_path) or ".", exist_ok=True)  # 创建目录（如果不存在）
            with open(new_camera_path, "w", encoding="utf-8") as file:
                new_camera = self.config.camera_info
                json.dump(new_camera, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved new device to: {new_camera_path}")
            #任务的设备编号（同个任务数据可由不同的设备采集）
            new_device_path = os.path.join(self.data_root,"device","device_episode.json")
            os.makedirs(os.path.dirname(new_device_path) or ".", exist_ok=True)  # 创建目录（如果不存在）
            with open(new_device_path, "w", encoding="utf-8") as file:
                json.dump(new_device, file, indent=4, ensure_ascii=False)
            self.logger.info(f"Saved new device to: {new_device_path}")


    def process_data(self):
        for source_idx, source_data_root in enumerate(self.config.source_data_roots):
            episode_path = source_data_root
            self.logger.info(f'Processing source {source_idx + 1}, {episode_path}')
            #self._load_episode2(episode_path)
            self._add_episode2(episode_path)
    
    # def _add_episode(self, episode_path):
    #     raw_images, raw_actions, instruction = self._load_episode(episode_path)
    #     indexs = list(range(len(raw_images[self.config.rgb_names[0]])))
        
    #     for i in tqdm(indexs[:-1], desc=f'Adding episode {episode_path}'):
    #         states = np.concatenate([raw_actions[action_dir][i] for action_dir in self.config.action_dirs])
    #         actions = np.concatenate([raw_actions[action_dir][i + 1] for action_dir in self.config.action_dirs])
    #         if not self._check_nonoop_actions(states, actions):
    #             continue

    #         frame = {rgb_name: load_image(raw_images[rgb_name][i]) for rgb_name in self.config.rgb_names}
    #         frame['states'] = states
    #         frame['actions'] = actions
    #         self.dataset.add_frame(frame, task=instruction)
            
    #     self.dataset.save_episode()
        
    # def _load_episode(self, episode_path):
    #     raw_images = defaultdict(list)
    #     for rgb_dir, rgb_name in zip(self.config.rgb_dirs, self.config.rgb_names):
    #         rgb_dir = os.path.join(episode_path, rgb_dir)
    #         for file_name in sorted(os.listdir(rgb_dir), key=lambda x: float(x[:-5])):
    #             image_path = os.path.join(rgb_dir, file_name)
    #             raw_images[rgb_name].append(image_path)
            
    #     raw_actions = defaultdict(list)
    #     for action_dir, action_keys in zip(self.config.action_dirs, self.config.action_keys_list):
    #         action_dir_ = os.path.join(episode_path, action_dir)
    #         for file_name in sorted(os.listdir(action_dir_), key=lambda x: float(x[:-5])):
    #             action_path = os.path.join(action_dir_, file_name)
    #             with open(action_path, 'r') as f:
    #                 action_data = json.load(f)
    #             action_data = np.array([action_data[key] for key in action_keys])
    #             raw_actions[action_dir].append(action_data)
        
    #     instruction_path = os.path.join(episode_path, self.config.instruction_path)
    #     with open(instruction_path, 'r') as f:
    #         instruction_data = json.load(f)
    #     instruction = instruction_data['instructions'][0]
    #     if instruction == 'null':
    #         instruction = self.config.default_instruction
        
    #     lens = []
    #     for rgb_name, images_list in raw_images.items():
    #         lens.append(len(images_list))
    #     for action_dir, actions_list in raw_actions.items():
    #         lens.append(len(actions_list))
        
    #     assert all(lens[0] == l for l in lens), "All lists must have the same length"
        
    #     return raw_images, raw_actions, instruction
    
    def _add_episode2(self, episode_path):
        index, raw_images, raw_actions, instruction = self._load_episode2(episode_path)
        self.logger.info(index)
        
        
        for i in tqdm(range(index), desc=f'Adding episode {episode_path}'):
            # 转换四元数
            r1 = raw_actions[i,0:11]
            r2 = self.rotation_recovery(raw_actions[i,11:14],raw_actions[i,14:17])
            r3 = raw_actions[i,17:28]
            r4 = self.rotation_recovery(raw_actions[i,28:31],raw_actions[i,31:34])
            actions = np.concatenate([r1,r2,r3,r4])
            states = np.concatenate([r1,r2,r3,r4])

           # rgb 转换
            def trans_img(images):
                img = cv2.imdecode(np.frombuffer(images, dtype=np.uint8),cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            frame = {next(iter(rgb_name)): trans_img(raw_images[next(iter(rgb_name))][i]) for rgb_name in self.config.rgb_names}
            frame['states'] = states
            frame['actions'] = actions
            if i == index - 1:
                frame['next.done'] = np.array([1], dtype=np.bool_)  # 明确使用 NumPy 数组
            else:
                frame['next.done'] = np.array([0], dtype=np.bool_)  # 明确使用 NumPy 数组
            #self.logger.info(f'1:{time.time()}')
            self.dataset.add_frame(frame,task=instruction)
            #self.logger.info(f'2:{time.time()}')
            
        self.dataset.save_episode()

    

    def _load_episode2(self, episode_path):
        try:
            with h5py.File(episode_path, "r") as file:
                self.logger.info("Datasets in HDF5 file:")
                
                def info_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        self.logger.info(f"Dataset: {name}")
                        self.logger.info(f"  Shape: {obj.shape}, DType: {obj.dtype}")
                        if obj.attrs:
                            self.logger.info("  Attributes:")
                            for key, val in obj.attrs.items():
                                self.logger.info(f"    {key}: {val}")
                
                #file.visititems(info_datasets)  # 打印 dataset 信息
                
                # 存储所有 dataset 的字典（使用 name 作为键）
                datasets_dict = {}
                frame_count = {}
                def load_datasets(group, path=''):
                    for key in group.keys():
                        current_path = f"{path}/{key}" if path else key
                        item = group[key]
                        
                        if isinstance(item, h5py.Dataset):
                            # 直接使用 current_path（即 name）作为字典键
                            datasets_dict[current_path] = item[()]  # 读取数据
                            self.logger.info(f"Loaded dataset: {current_path}")
                            
                        elif isinstance(item, h5py.Group):
                            load_datasets(item, current_path)  # 递归处理子组

                raw_images = defaultdict(list)
                load_datasets(file)  # 加载所有 dataset
                for dataset_path in datasets_dict:
                    if "depth" in dataset_path: # 跳过深度，因为现在的数据没有深度信息，后面会补全。
                        self.logger.info(f"depth:{dataset_path} exist in task:{self.task_name}")
                        return 
                    if "action" in dataset_path:
                        frame = datasets_dict[dataset_path].shape
                        raw_actions = datasets_dict[dataset_path][:,self.config.action_index] 
                        frame_count["action"] = frame[0]
                    if "cam_high" in dataset_path:
                        frame = datasets_dict[dataset_path].shape
                        raw_images["observation.images.cam_high"] = datasets_dict[dataset_path]
                        frame_count["cam_high"] = frame[0]
                    if "cam_left_wrist" in dataset_path:
                        frame = datasets_dict[dataset_path].shape
                        raw_images["observation.images.cam_left_wrist"] = datasets_dict[dataset_path]
                        frame_count["cam_left_wrist"] = frame[0]
                    if "cam_right_wrist" in dataset_path:
                        frame = datasets_dict[dataset_path].shape
                        raw_images["observation.images.cam_right_wrist"] = datasets_dict[dataset_path]
                        frame_count["cam_right_wrist"] = frame[0]
                if len(set(frame_count.values())) > 1:
                    self.logger.info(f"frame error")
                    self.logger.info(self.task_name)
                    return

                instruction = self.task_name
                return frame[0],raw_images,raw_actions,instruction
                
        except Exception as e:
            self.logger.info(f"Error loading episode: {str(e)}")
            return None
        
    def _check_nonoop_actions(self, states, actions):
        return np.abs(states - actions).max() > self.config.nonoop_threshold
    
def read_csv(path):
    df = pd.read_csv(path)   
    # 提取 'json_path' 和 'nas_path' 两列
    selected_columns = df[['json_path', 'nas_path']]        
    # 转换成嵌套列表（每行是一个子列表）
    nested_list = selected_columns.values.tolist()       
    # 打印前3行示例
    #self.logger.info("嵌套列表（前3行）:", nested_list[:3])        
    return nested_list


def main():
    config = DataConfig()
    nested_list = read_csv(config.source_data_csv)
    raw_data_dir = os.path.dirname(config.source_data_csv)
    for both_data in nested_list:
        json_path = os.path.join(raw_data_dir,both_data[0])
        nas_path = both_data[1]
        task_name = os.path.basename(nas_path)
        processor = DataProcessor(config,json_path,nas_path,task_name)
        processor.parse_annotation()



if __name__ == "__main__":
    main()
