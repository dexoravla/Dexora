import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import fnmatch
import subprocess
import pickle
import re
from datetime import datetime
import time
from threading import Thread, Event
import logging
from visualize_episodes import save_videos
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional
from pprint import pprint


logger = logging.getLogger(__name__)
np.random.seed(0)


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        norm_stats: dict,
        augmentors: dict,
        data_indexes: dict,
        chunk_sizes: dict = None,
        action_bias: int = 1,
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = data_indexes["camera_names"]
        self.observation_indexes = data_indexes.get("observation", None)
        self.action_indexes = data_indexes.get("action", None)
        self.observation_chunk_size = chunk_sizes.get("observation", None)
        self.action_chunk_size = chunk_sizes.get("action", None)
        assert self.observation_chunk_size is None, "Not implemented"
        self.norm_stats = norm_stats.copy()
        self.augmentors = augmentors
        self.augment_images = augmentors.get("image", None)
        self.action_bias = action_bias
        if self.observation_indexes is not None:
            self.norm_stats["qpos_mean"] = self.norm_stats["qpos_mean"][
                self.observation_indexes
            ]
            self.norm_stats["qpos_std"] = self.norm_stats["qpos_std"][
                self.observation_indexes
            ]
        if self.action_indexes is not None:
            self.norm_stats["action_mean"] = self.norm_stats["action_mean"][
                self.action_indexes
            ]
            self.norm_stats["action_std"] = self.norm_stats["action_std"][
                self.action_indexes
            ]
        with h5py.File(self._get_dataset_path(0), "r") as root:
            # logger.info(f"Keys in the dataset: {list(root.keys())}")
            cam_names = set(root["/observations/images"].keys())
            wrong_cam_names = set(self.camera_names) - cam_names
            if len(wrong_cam_names) > 0:
                raise ValueError(
                    f"Wrong camera names: {wrong_cam_names}, "
                    f"available names: {cam_names}. "
                    "Please modify the camera names in the task configuration file."
                )
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def _get_dataset_path(self, index) -> str:
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        return dataset_path

    def __getitem__(self, index):
        sample_full_episode = False  # TODO:hardcode
        dataset_path = self._get_dataset_path(index)
        action_chunk = self.action_chunk_size
        with h5py.File(dataset_path, "r") as root:
            compressed = root.attrs.get("compress", False)
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root["/observations/qpos"][start_ts]
            if self.observation_indexes is not None:
                qpos = qpos[self.observation_indexes]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f"/observations/images/{cam_name}"][
                    start_ts
                ]
            if compressed:
                for cam_name in image_dict.keys():
                    import cv2

                    decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                    image_dict[cam_name] = np.array(decompressed_image)
            # TODO: pass the chunk_size as a arg to load only the needed actions
            # get all actions after and including start_ts
            # hack, to make timesteps more aligned
            bias = self.action_bias
            action_start = max(0, start_ts - bias)
            action = root["/action"][action_start : action_start + action_chunk]
            if self.action_indexes is not None:
                action = action[:, self.action_indexes]
                # chunked_action_shape = (action_chunk, action.shape[1])
            # else:
            # chunked_action_shape = (action_chunk, original_action_shape[1])
            action_len = len(action)
            # print(f"action_shape: {action.shape}")
        # padded_action = np.zeros(chunked_action_shape, dtype=np.float32)
        padded_action = np.tile(action[-1], (action_chunk, 1))
        padded_action[:action_len] = action
        # print(f"padded_action shape: {padded_action.shape}")
        is_pad = np.zeros(action_chunk)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        # construct image metrix (the same as np.array)
        # each row is all images from one camera
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last LABEL
        image_data = torch.einsum("k h w c -> k c h w", image_data)

        # augment images
        if self.augment_images is not None and self.augment_images.activated:
            image_data = self.augment_images(image_data)

        # TODO: configure the data process in the config file
        # normalize image to [0, 1]
        # TODO: will standardize image in the loss function, not good
        image_data = image_data / 255.0
        # standardize lowdim data
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats[
            "action_std"
        ]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]
        # print(f"qpos_data_shape: {qpos_data.shape}")
        # print(f"action_data_shape: {action_data.shape}")
        return image_data, qpos_data, action_data, is_pad


def find_all_hdf5(dataset_dir, skip_mirrored_data=True):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            if "features" in filename:
                continue
            if skip_mirrored_data and "mirror" in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f"Found {len(hdf5_files)} hdf5 files")
    return hdf5_files


def get_pkl_info(path):
    with open(path, "rb") as f:
        key_info = pickle.load(f)
    return key_info


def get_init_states(dir, episode_idx=None):
    if isinstance(episode_idx, int):
        dataset_path = os.path.join(dir, f"episode_{episode_idx}.hdf5")
        dataset_path = os.path.abspath(dataset_path)
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][0]
            action = root["/action"][0]
    else:
        # dir is info dir
        key_info_path = os.path.join(dir, f"key_info.pkl")
        with open(key_info_path, "rb") as f:
            key_info = pickle.load(f)
            qpos = key_info["init_info"]["init_joint"]
            action = key_info["init_info"]["init_action"]
    return qpos, action


def multi_slices_to_indexes(slices: Union[List[tuple], tuple]):
    def process_tuple(num_episodes: tuple) -> list:
        if len(num_episodes) == 2:
            start, end = num_episodes
            postfix = None
        elif len(num_episodes) == 3:
            start, end, postfix = num_episodes
        num_episodes = list(range(start, end))
        if postfix is not None:
            for index, ep in enumerate(num_episodes):
                num_episodes[index] = f"{ep}_{postfix}"
        return num_episodes

    if isinstance(slices, tuple):
        slices = process_tuple(slices)
    elif isinstance(slices, list):
        for index, element in enumerate(slices):
            if isinstance(element, int):
                element = (element, element + 1)
            slices[index] = process_tuple(element)
        # flatten the list
        flattened = []
        for sublist in slices:
            flattened.extend(sublist)
        slices = flattened
    else:
        raise ValueError("slices should be tuple or list of tuples")
    return slices


def process_num_episodes(
    num_episodes: Union[list, tuple, int], dataset_dir: str
) -> list:
    """Change num_episodes to list of ids"""

    if num_episodes in ["ALL", "all", "All", 0]:
        num_episodes = len(find_all_hdf5(dataset_dir))
        num_episodes = list(range(num_episodes))
    else:
        num_episodes = multi_slices_to_indexes(num_episodes)
    return num_episodes


@dataclass
class LoadDataConfig(object):
    dataset_dir: str
    num_episodes: Union[list, tuple, int]
    batch_size_train: int
    batch_size_validate: int
    train_ratio: float
    num_workers_train: int
    num_workers_validate: int
    observation_slice: Optional[Union[List[tuple], tuple]]
    action_slice: Optional[Union[List[tuple], tuple]]
    augmentors: dict
    check_episodes: bool
    camera_names: list
    chunk_sizes: dict

    def __post_init__(self):
        # change dataset_dir to absolute path
        self.dataset_dir = os.path.abspath(self.dataset_dir)
        # change num_episodes to list of ids
        self.num_episodes = process_num_episodes(self.num_episodes, self.dataset_dir)
        # check if all episodes exist
        assert os.path.isdir(
            self.dataset_dir
        ), f"dataset_dir {self.dataset_dir} not found"
        if self.check_episodes:
            for ep in self.num_episodes:
                assert os.path.exists(
                    os.path.join(self.dataset_dir, f"episode_{ep}.hdf5")
                ), f"episode {ep} not found"
        if self.observation_slice is not None:
            self.observation_slice = multi_slices_to_indexes(self.observation_slice)
            print(f"Observation slice: {self.observation_slice}")
        if self.action_slice is not None:
            self.action_slice = multi_slices_to_indexes(self.action_slice)
            print(f"Action slice: {self.action_slice}")


def get_norm_stats(dataset_dir, num_episodes, config: LoadDataConfig):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in num_episodes:
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        dataset_path = os.path.abspath(dataset_path)
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
        all_qpos_data.append(qpos)
        all_action_data.append(action)

    all_qpos_data = np.concatenate(all_qpos_data)
    all_action_data = np.concatenate(all_action_data)
    if config.observation_slice is not None:
        all_qpos_data = all_qpos_data[:, config.observation_slice]
    if config.action_slice is not None:
        all_action_data = all_action_data[:, config.action_slice]

    # normalize action data
    action_mean = np.mean(all_action_data, 0)
    action_std = np.std(all_action_data, 0)
    action_std = np.clip(action_std, 1e-2, np.inf)  # clipping

    # normalize qpos data
    qpos_mean = np.mean(all_qpos_data, 0)
    qpos_std = np.std(all_qpos_data, 0)
    qpos_std = np.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean,
        "action_std": action_std,
        "qpos_mean": qpos_mean,
        "qpos_std": qpos_std,
        "example_qpos": qpos,
    }
    # print("action_mean_shape", stats["action_mean"].shape)
    return stats


def load_data(config: LoadDataConfig):
    dataset_dir = config.dataset_dir
    print(f"\nData from: {dataset_dir}\n")
    train_ratio = config.train_ratio
    num_episodes = config.num_episodes
    episodes_num = len(num_episodes)
    shuffled_indices = list(num_episodes)
    np.random.shuffle(shuffled_indices)
    train_indices = shuffled_indices[: int(train_ratio * episodes_num)]
    val_indices = shuffled_indices[int(train_ratio * episodes_num) :]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, config)

    # construct dataset
    camera_names = config.camera_names
    augmentors = config.augmentors
    ep_ds_config = {
        "dataset_dir": dataset_dir,
        "norm_stats": norm_stats,
        "augmentors": augmentors,
        "data_indexes": {
            "camera_names": camera_names,
            "observation": config.observation_slice,
            "action": config.action_slice,
        },
        "chunk_sizes": config.chunk_sizes,
        "action_bias": 1,
    }
    train_dataset = EpisodicDataset(train_indices, **ep_ds_config)
    val_dataset = EpisodicDataset(val_indices, **ep_ds_config)
    # construct dataloader
    batch_size_train = config.batch_size_train
    batch_size_val = config.batch_size_validate
    print("batch_size_train:", batch_size_train)
    print("batch_size_val:", batch_size_val)
    num_workers_train = config.num_workers_train
    num_workers_val = config.num_workers_validate
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers_train,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers_val,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats


def save_eval_results(
    save_dir,
    dataset_name,
    rollout_id,
    image_list,
    qpos_list,
    action_list,
    camera_names,
    dt,
    all_time_actions,
    save_time_actions=False,
    save_all=False,
):
    # saving evaluation results
    # TODO: configure what to save

    save_path = os.path.join(save_dir, dataset_name)
    if not os.path.isdir(save_dir):
        logger.info(f"Create directory for saving evaluation info: {save_dir}")
        os.makedirs(save_dir)
    save_videos(image_list, dt, video_path=f"{save_path}.mp4", decompress=False)
    if save_time_actions:
        np.save(f"{save_path}.npy", all_time_actions.cpu().numpy())
    if save_all:
        start_time = time.time()
        logger.info(f"Save all trajs to {save_path}...")

        # # save qpos
        # with open(os.path.join(save_dir, f'qpos_{rollout_id}.pkl'), 'wb') as f:
        #     pickle.dump(qpos_list, f)
        # # save actions
        # with open(os.path.join(save_dir, f'actions_{rollout_id}.pkl'), 'wb') as f:
        #     pickle.dump(action_list, f)
        # save as hdf5
        data_dict = {
            "/observations/qpos": qpos_list,
            "/action": action_list,
        }
        image_dict: Dict[str, list] = {}
        for cam_name in camera_names:
            image_dict[f"/observations/images/{cam_name}"] = []
        for frame in image_list:
            for cam_name in camera_names:
                image_dict[f"/observations/images/{cam_name}"].append(frame[cam_name])
        mid_time = time.time()
        from data_process.convert_all import (
            compress_images,
            save_dict_to_hdf5,
            Compresser,
        )
        import cv2

        compresser = Compresser("jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 50], True)
        for key, value in image_dict.items():
            image_dict[key] = compress_images(value, compresser)
        data_dict.update(image_dict)
        save_dict_to_hdf5(data_dict, f"{save_path}.hdf5", False)
        end_time = time.time()
        logger.info(
            f"{dataset_name}: construct data {mid_time - start_time} s and save data {end_time - mid_time} s"
        )


"""helper functions"""


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def replace_timestamp(input_str, time_stamp):
    """Replace the time format in the input string with the given time_stamp."""
    # 检查time_stamp是否符合格式要求
    if time_stamp == "now":
        time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    elif len(time_stamp) != 15:
        raise ValueError("The length of time_stamp should be 15.")
    # 使用正则表达式匹配时间格式的部分，并替换为当前时间
    result_str = re.sub(r"\d{8}-\d{6}", time_stamp, input_str)
    return result_str


class CAN_Tools(object):

    @staticmethod
    def check_can_status(interface):
        # 使用 ip link show 命令获取网络接口状态
        result = subprocess.run(
            ["ip", "link", "show", interface], capture_output=True, text=True
        )
        output = result.stdout.strip()

        # 检查输出中是否包含 'UP' 状态
        if "UP" in output and "LOWER_UP" in output:
            return True  # 已激活
        else:
            return False  # 未激活

    @staticmethod
    def activate_can_interface(interface, bitrate):
        # 构造要执行的命令
        command = f"sudo ip link set {interface} up type can bitrate {bitrate}"

        # 使用 Popen 执行命令并自动输入密码
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()

        # 获取命令执行结果
        return_code = proc.returncode
        if return_code == 0:
            return True, stdout.decode()
        else:
            return False, stderr.decode()


class GPUer(object):

    def __init__(self, interval=1):
        self.interval = interval
        self.gpu_info = self.get_gpu_info()
        pprint(self.gpu_info)
        gpu_num = len(self.gpu_info["name"])
        self.utilization = {
            "current": [None] * gpu_num,
            "max": [0] * gpu_num,
            "average": [None] * gpu_num,
        }
        self.update_event = Event()
        Thread(target=self.monitor, daemon=True).start()

    def update(self):
        self.update_event.set()

    def _update(self):
        self.utilization["current"] = self.get_gpu_utilization()
        for i, util in enumerate(self.utilization["current"]):
            self.utilization["max"][i] = max(self.utilization["max"][i], util)
            if self.utilization["average"][i] is None:
                self.utilization["average"][i] = util
            else:
                self.utilization["average"][i] = (
                    self.utilization["average"][i] + util
                ) / 2

    def monitor(self):
        """Monitor the GPU utilization."""
        while True:
            self.update_event.wait()
            start_time = time.time()
            self._update()
            # print(f"GPU utilization: {self.utilization}")
            sleep_time = self.interval - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.update_event.clear()

    def save_info(self, path):
        """Save the GPU utilization information to a file.
        Parameters
        ----------
        path: str
            The path to save the information.
        """
        with open(path, "w") as f:
            f.write(f"Max GPU utilization(%): {self.utilization['max']}\n")
            f.write(f"Average GPU utilization(%): {self.utilization['average']}\n")

    @staticmethod
    def get_gpu_info() -> dict:
        """Get the information of all GPUs.
        Returns
        -------
        info: list of str
            The information of all GPUs, each element is a string containing the index, name, total memory, free memory, and used memory of a GPU.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpus_info = result.stdout.strip().split("\n")
        gpus_info_dict = {
            "name": [],
            "memory.total": [],
            "memory.free": [],
            "memory.used": [],
            "memory.utilization": [],
        }
        for gpu_info in gpus_info:
            infos = gpu_info.split(",")
            gpus_info_dict["name"].append(infos[1][1:])
            gpus_info_dict["memory.total"].append(int(infos[2][1:]))
            gpus_info_dict["memory.free"].append(int(infos[3][1:]))
            gpus_info_dict["memory.used"].append(int(infos[4][1:]))
            gpus_info_dict["memory.utilization"].append(
                gpus_info_dict["memory.used"][-1]
                / gpus_info_dict["memory.total"][-1]
                * 100
            )
        return gpus_info_dict

    @staticmethod
    def get_gpu_memory_map() -> List[int]:
        """Get the current GPU memory usage.
        Returns
        -------
        usage: list of int containing the memory usage of each GPU.
        """
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_memory = [int(x) for x in result.stdout.strip().split("\n")]
        return gpu_memory

    @staticmethod
    def get_gpu_utilization() -> List[int]:
        """Get the current GPU utilization.
        Returns
        -------
        utilization: list of int containing the utilization of each GPU in percentage.
        """
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_utilization = [int(x) for x in result.stdout.strip().split("\n")]
        return gpu_utilization

    @staticmethod
    def check_all_gpus_idle(utilization_threshold=10) -> Tuple[List[int], int]:
        """Check the utilization of all GPUs and return the indices of idle GPUs.
        Parameters
        ----------
        utilization_threshold: int, optional
            The threshold of GPU utilization to determine whether a GPU is idle.
        Returns
        -------
        idle_gpus: list of int
            The indices of idle GPUs.
        total_gpus: int
            The total number of GPUs.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            gpu_usage = result.stdout.splitlines()

            idle_gpus = []
            gpu_utilizations = []
            for gpu_id, usage in enumerate(gpu_usage):
                gpu_utilization = int(usage.strip())
                gpu_utilizations.append(gpu_utilization)
                if gpu_utilization < utilization_threshold:
                    idle_gpus.append(gpu_id)

            return idle_gpus, len(gpu_usage), gpu_utilizations
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error occurred: {e}")
            return []
