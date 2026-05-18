import os
import fnmatch
import yaml
import numpy as np
import bson
import av
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import re
import pathlib
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime
import random

class EpisodeInfo:
    """Custom class to store episode path, action text, and optional length"""
    def __init__(self, path, action_text, length: Optional[int] = None):
        self.path = path
        self.action = action_text
        self.length = length  # number of steps (frames) in this episode, if known
    
    def __str__(self):
        return self.path
    
    def __repr__(self):
        return f"EpisodeInfo(path={self.path}, action={self.action[:20]}...)" if len(self.action) > 20 else f"EpisodeInfo(path={self.path}, action={self.action})"


class BsonVLADataset:
    """
    Modified BsonVLADataset to handle new directory-based episode structure with:
    - Main BSON file for arm data
    - Separate xhand_control_data.bson for dexterous hand data
    - Image file sequences from multiple cameras
    """
    def __init__(self, bson_dir: str="data/ours/final", sub_sample=1.0, 
                # normalize_mode: str=None, stats_file: str=None) -> None:
                normalize_mode: str="mean_std", stats_file: str="1113action190/dataset_statistics.json") -> None:
        """
        Initializes the BsonVLADataset.

        Args:
            bson_dir (str): The path to the dataset directory containing episode folders.
            sub_sample (float): The fraction of the dataset to use.
            normalize_mode (str): Normalization mode - 'mean_std' or 'min_max' or None.
            stats_file (str): Path to dataset_statistics.json file for normalization.
        """
        self.DATASET_NAME = "ours"
        
        # Normalization settings
        self.normalize_mode = normalize_mode
        self.stats = None
        if normalize_mode and stats_file:
            self._load_statistics(stats_file)
        
        # Camera configurations will be determined per episode
        # 3 cameras: 0,2,6 or 2,4,6 (left_hand, external, right_hand) + head in BSON
        # 4 cameras: 0,4,6,11 (head, external, right_hand, left_hand)
        # self.valid_3cam_config_1 = [0, 2, 6]  # left_hand, external, right_hand
        # self.valid_3cam_config_2 = [2, 4, 6]  # left_hand, external, right_hand
        # self.valid_4cam_config_1 = [0, 4, 6, 11]  # head, external, right_hand, left_hand
        # self.valid_4cam_config_2 = [0, 6, 8, 12]  # head, left_hand, external, right_hand
        # self.valid_4cam_config_3 = [0, 9, 13, 15] # head, left_hand, external, right_hand
        # self.valid_4cam_config_4 = [0, 8, 10, 12] # head, left_hand, external, right_hand
        # self.valid_4cam_config_5 = [0, 14, 16, 18]# head, left_hand, external, right_hand
        self.valid_4cam_config_new = ["camera_head", "camera_left_wrist", "camera_right_wrist", "camera_third_view"]


        # These will be set per episode based on actual camera configuration
        self.ext_image_keys = []
        self.ext_image_names = []
        self.image_keys = []

        print("Finding episode directories...")
        self.episode_infos: list[EpisodeInfo] = []
        
        # Find all episode directories in the new structure
        for action_item in os.listdir(bson_dir):
            if action_item != "action1":
                print("Only action1 is used")
                continue
            action_path = os.path.join(bson_dir, action_item)
            if os.path.isdir(action_path) and action_item.startswith('action'):
                # Read action.txt content
                action_txt_path = os.path.join(action_path, "instruction1.txt")
                if os.path.exists(action_txt_path):
                    with open(action_txt_path, 'r') as f:
                        action_text = f.read().strip()
                else:
                    print(f"Warning: Missing action.txt in {action_path}")
                    continue

                # Look for episode subdirectories within each action directory
                for episode_item in os.listdir(action_path):
                    episode_path = os.path.join(action_path, episode_item)
                    if os.path.isdir(episode_path) and episode_item.startswith('episode'):
                        # Check if required files exist in the episode subdirectory
                        main_bson = os.path.join(episode_path, "episode_0.bson")
                        xhand_bson = os.path.join(episode_path, "xhand_control_data.bson")
                        
                        # Validate camera configuration
                        if os.path.exists(main_bson) and os.path.exists(xhand_bson):
                            camera_config = self._validate_camera_config(episode_path)
                            if camera_config is not None:
                                # Store episode info with action text
                                episode_info = EpisodeInfo(episode_path, action_text)
                                self.episode_infos.append(episode_info)
                            else:
                                print(f"Warning: Invalid camera configuration in {episode_path}")
                        else:
                            print(f"Warning: Missing required BSON files in {episode_path}")
        
        # Sort by episode path for consistency
        self.episode_infos.sort(key=lambda x: x.path)
        print(f"Found {len(self.episode_infos)} valid episode directories.")

        # Sub-sampling logic from original
        indices = np.arange(len(self))
        np.random.seed(42)
        np.random.shuffle(indices)

        superior = sub_sample > 0.5
        if superior:
            sub_sample = 1 - sub_sample
        split = int(len(indices) * sub_sample)
        if superior:
            indices = indices[split:]
        else:
            indices = indices[:split]
        self.episode_infos = [self.episode_infos[i] for i in indices]
        print(f"Using {len(self.episode_infos)} episodes after subsampling.")

        # Load config from YAML
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISTORY_SIZE = config['common']['img_history_size']

        self._video_cache = {}
        self._image_cache = {}
        self._cache = {}
        
        # Get each episode's len to calculate sample weights
        print("Pre-calculating episode lengths for sampling...")
        episode_lens = []
        valid_episode_infos = []
        for episode_info in self.episode_infos:
            valid, res = self.parse_episode_state_only(episode_info)
            if valid:
                _len = res['state'].shape[0]
                episode_lens.append(_len)
                valid_episode_infos.append(episode_info)
                # Persist length into EpisodeInfo for later direct frame access
                episode_info.length = int(_len)
            else:
                print(f"Skipping invalid or too short episode: {episode_info}")
        
        self.episode_infos = valid_episode_infos
        if not episode_lens:
            raise ValueError("No valid episodes found in the provided directory.")

        
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
        print("Dataset initialized.")
    
    

    def _validate_camera_config(self, episode_path: str) -> Optional[Dict]:
        """
        Validates camera configuration for an episode.
        Returns camera configuration dict if valid, None otherwise.
        
        Valid configurations:
        - 3 cameras: 0,2,6 (left_hand, external, right_hand) + head in BSON
        - 4 cameras: 0,4,6,11 (head, external, right_hand, left_hand)
        """
        # Find all camera_* directories
        camera_dirs = []
        for item in os.listdir(episode_path):
            if item.startswith('camera_') and os.path.isdir(os.path.join(episode_path, item)):
                camera_dirs.append(item)
        
        camera_dirs.sort()

        if camera_dirs == self.valid_4cam_config_new:
            return {
                'type': '4cam',
                'cameras': camera_dirs,
                'ext_image_keys': camera_dirs,
                'ext_image_names': ['cam_head', 'cam_left_wrist', 'cam_right_wrist', 'cam_third_view'],
                'has_head_in_bson': False
            }
        
        # # Validate against known configurations
        # if camera_dirs == self.valid_3cam_config_1 or camera_dirs == self.valid_3cam_config_2:
        #     # 3 cameras: 0,2,6 (left_hand, external, right_hand)
        #     return {
        #         'type': '3cam',
        #         'cameras': camera_dirs,
        #         'ext_image_keys': [f'camera_{i}' for i in camera_dirs],
        #         'ext_image_names': ['cam_left_wrist', 'cam_third_view', 'cam_right_wrist'],
        #         'has_head_in_bson': True
        #     }
        # elif camera_dirs == self.valid_4cam_config_1:
        #     # 4 cameras: 0,4,6,11 (head, external, right_hand, left_hand)
        #     return {
        #         'type': '4cam',
        #         'cameras': camera_dirs,
        #         'ext_image_keys': [f'camera_{i}' for i in camera_dirs],
        #         'ext_image_names': ['cam_head', 'cam_third_view', 'cam_right_wrist', 'cam_left_wrist'],
        #         'has_head_in_bson': False
        #     }
        # elif camera_dirs == self.valid_4cam_config_2 or camera_dirs == self.valid_4cam_config_3 or camera_dirs == self.valid_4cam_config_4 or camera_dirs == self.valid_4cam_config_5:
        #     # 4 cameras: 0,6,8,12 (head, left_hand, external, right_hand)
        #     return {
        #         'type': '4cam',
        #         'cameras': camera_dirs,
        #         'ext_image_keys': [f'camera_{i}' for i in camera_dirs],
        #         'ext_image_names': ['cam_head', 'cam_left_wrist', 'cam_third_view', 'cam_right_wrist'],
        #         'has_head_in_bson': False
        #     }
        else:
            # Invalid configuration
            print(f"Invalid camera configuration in {episode_path}: found cameras {camera_dirs}")
            return None

    def __len__(self):
        return len(self.episode_infos)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int=None, state_only=False, action_only=False, frame_idx: Optional[int]=None):
        """
        Get a training sample.

        Args:
            index (int, optional): The index of the episode. If None, a random episode is chosen.
            frame_idx (int, optional): If provided and state_only/action_only are False, return the specific frame
                                       from the episode instead of sampling a random timestep.
            state_only (bool, optional): If True, returns the full state/action trajectories.
                                         Otherwise, returns a single timestep sample.
        Returns:
           A dictionary containing the training sample.
        """
        while True:
            if index is None:
                # Sample an episode based on its length
                episode_info = np.random.choice(self.episode_infos, p=self.episode_sample_weights)
            else:
                episode_info = self.episode_infos[index]

            parser = self.parse_episode if not state_only else self.parse_episode_state_only
            parser = self.parse_episode_action if action_only else parser
            if (not state_only) and (not action_only):
                valid, sample = self.parse_episode(episode_info, frame_idx=frame_idx)
            else:
                valid, sample = parser(episode_info)
            
            if valid:
                return sample
            else:
                # If randomly chosen episode was invalid, try another random one
                if index is None:
                    print(f"Warning: Invalid sample from {episode_info}, resampling...")
                    continue
                # If specific index is invalid, it's an issue with that file
                else:
                    raise RuntimeError(f"Episode at index {index} ({episode_info}) is invalid.")

    def _extract_data_from_episode(self, episode_info: EpisodeInfo) -> Optional[Dict]:
        """
        Extracts numerical data from both main BSON and xhand BSON files,
        and prepares image paths for later loading.
        """
        # Handle EpisodeInfo object or string path
        path = episode_info.path
        
        main_bson_path = os.path.join(path, "episode_0.bson")
        xhand_bson_path = os.path.join(path, "xhand_control_data.bson")
        
        # Load main BSON (arm data)
        try:
            with open(main_bson_path, 'rb') as f:
                main_bson_content = bson.decode(f.read())["data"]
        except Exception as e:
            print(f"Error reading main BSON file {main_bson_path}: {e}")
            return None

        # Load xhand BSON
        try:
            with open(xhand_bson_path, 'rb') as f:
                xhand_data = bson.decode(f.read())
        except Exception as e:
            print(f"Error reading xhand BSON file {xhand_bson_path}: {e}")
            return None

        # Define data keys for arms (same as original)
        arm_dim, eef_dim = 6, 12
        keys = {
            "left_obs_arm": "/observation/left_arm/joint_state",
            "right_obs_arm": "/observation/right_arm/joint_state",
            "left_act_arm": "/action/left_arm/joint_state",
            "right_act_arm": "/action/right_arm/joint_state",
        }

        # Check frame counts
        arm_frame_num = len(main_bson_content[keys["left_obs_arm"]])
        xhand_frame_num = len(xhand_data['frames'])
        
        if arm_frame_num == 0 or xhand_frame_num == 0:
            return None
        
        # Use minimum frame count to ensure alignment
        frame_num = min(arm_frame_num, xhand_frame_num)
        
        # Extract arm data
        state = np.zeros((frame_num, 2 * (arm_dim + eef_dim)), dtype=np.float32)
        action = np.zeros((frame_num, 2 * (arm_dim + eef_dim)), dtype=np.float32)
        
        # Check if action data has correct length (6) for arms
        use_arm_actions = True
        try:
            # Check first frame to determine if action data is valid
            left_arm_action = main_bson_content[keys["left_act_arm"]][0]["data"]["pos"]
            right_arm_action = main_bson_content[keys["right_act_arm"]][0]["data"]["pos"]
            if len(left_arm_action) != arm_dim or len(right_arm_action) != arm_dim:
                use_arm_actions = False
                # print(f"Warning: Action data has incorrect length (left: {len(left_arm_action)}, right: {len(right_arm_action)}), using observation as action")
        except (KeyError, IndexError):
            use_arm_actions = False
            print(f"Warning: Action data not available in episode {episode_info.path}, using observation as action")
        
        for i in range(frame_num):
            state[i, :] = np.concatenate([
                main_bson_content[keys["left_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["observation"]["left_hand"],
                main_bson_content[keys["right_obs_arm"]][i]["data"]["pos"],
                xhand_data['frames'][i]["observation"]["right_hand"]
            ])
            
            # Use action data if available and correct, otherwise use observation
            if use_arm_actions:
                # print("Using action data for arms")
                left_arm_data = main_bson_content[keys["left_act_arm"]][i]["data"]["pos"]
                right_arm_data = main_bson_content[keys["right_act_arm"]][i]["data"]["pos"]
            else:
                left_arm_data = main_bson_content[keys["left_obs_arm"]][i]["data"]["pos"]
                right_arm_data = main_bson_content[keys["right_obs_arm"]][i]["data"]["pos"]
                
            action[i, :] = np.concatenate([
                left_arm_data,
                xhand_data['frames'][i]["action"]["left_hand"],
                right_arm_data,
                xhand_data['frames'][i]["action"]["right_hand"]
            ])
        
        # Get camera configuration for this episode
        camera_config = self._validate_camera_config(path)
        if camera_config is None:
            print(f"Invalid camera configuration for episode {path}")
            return None
        
        # Extract image data info based on camera configuration
        images_info = {}
        
        # Head camera handling
        if camera_config['has_head_in_bson']:
            # For 3-camera config, head is in BSON
            if "/images/head_camera" in main_bson_content:
                images_info['head_camera'] = main_bson_content["/images/head_camera"]
            else:
                print(f"Warning: head_camera not found in BSON for episode {path}")
                images_info['head_camera'] = None
        
        # USB cameras from img files
        for cam in camera_config['ext_image_keys']:
            cam_path = os.path.join(path, cam)
            if os.path.exists(cam_path):
                # Get list of img files
                img_files = sorted([f for f in os.listdir(cam_path) if f.endswith('.jpg')])
                images_info[cam] = {
                    'type': 'file_sequence',
                    'path': cam_path,
                    'files': img_files[:frame_num]  # Ensure we don't exceed frame count
                }
            else:
                images_info[cam] = None
        
        return {
            "state": state,
            "action": action,
            "images_info": images_info,
            "episode_len": frame_num,
            "episode_path": path,
            "camera_config": camera_config,
        }

    def _get_decoded_video(self, episode_info, image_key: str, raw_bytes: bytes) -> np.ndarray:
        """Decodes video from raw H.264 bytes using AV, with caching."""
        cache_key = (episode_info, image_key)
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]

        frames = []
        if raw_bytes is None or len(raw_bytes) == 0:
            print(f"Warning: Empty or None raw_bytes for video decoding (episode {episode_info}, key {image_key})")
            decoded_frames = np.array([])
        else:
            try:
                in_buffer = BytesIO(raw_bytes)
                container = av.open(in_buffer)
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
                if frames:
                    decoded_frames = np.stack(frames)
                else:
                    print(f"Warning: No frames decoded from video (episode {episode_info}, key {image_key})")
                    decoded_frames = np.array([])
            except Exception as e:
                print(f"Warning: Failed to decode video (path {episode_info}, key {image_key}). Error: {e}")
                decoded_frames = np.array([])
            finally:
                container.close()
        
        #self._video_cache[cache_key] = decoded_frames
        return decoded_frames

    def _load_file_sequence(self, cam_info: Dict, start_idx: int, end_idx: int) -> np.ndarray:
        """Load a sequence of img images."""
        if cam_info is None or cam_info['type'] != 'file_sequence':
            print(f"Warning: Invalid cam_info in _load_file_sequence: {cam_info}")
            return np.array([])
        
        frames = []
        cam_path = cam_info['path']
        files = cam_info['files']
        
        for i in range(start_idx, min(end_idx, len(files))):
            cache_key = (cam_path, files[i])
            
            if cache_key in self._image_cache:
                img_array = self._image_cache[cache_key]
            else:
                img_path = os.path.join(cam_path, files[i])
                try:
                    with Image.open(img_path) as img:
                        img_array = np.array(img)
                    if img_array.ndim == 2:  # Grayscale to RGB
                        img_array = np.stack([img_array] * 3, axis=-1)
                    # if len(self._image_cache) < 20_000:
                    #     self._image_cache[cache_key] = img_array
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}. Error: {e}")
                    img_array = np.zeros((480, 640, 3), dtype=np.uint8)  # Default size
            
            frames.append(img_array)
        
        if frames:
            return np.stack(frames)
        else:
            print(f"Warning: No frames loaded in _load_file_sequence for path {cam_info.get('path', 'unknown')}")
            return np.array([])

    def parse_episode(self, episode_info, frame_idx: Optional[int] = None):
        """
        Parses an episode to generate a training sample at a random timestep,
        or at a specific frame if frame_idx is provided.
        """
        episode_data = self._extract_data_from_episode(episode_info)
        if not episode_data:
            return False, None

        qpos = episode_data["state"]
        num_steps = episode_data["episode_len"]

        if num_steps < self.CHUNK_SIZE:  # Drop too-short episodes
            return False, None
        
        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        if first_idx >= num_steps:  # case where robot doesn't move
            return False, None

        # Determine timestep: use provided frame_idx if given, otherwise random
        if frame_idx is not None:
            step_id = int(max(first_idx - 1, min(frame_idx, num_steps - 1)))
        else:
            step_id = np.random.randint(first_idx - 1, num_steps)

        # if (episode_info, step_id) in self._cache:
        #     return True, self._cache[(episode_info, step_id)]
        
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": episode_info.action
        }
        
        actions_full = episode_data["action"]
        target_qpos = actions_full[step_id : step_id + self.CHUNK_SIZE]
        
        # Parse state and action
        state = qpos[step_id:step_id+1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos**2, axis=0))
        actions = target_qpos

        if actions.shape[0] < self.CHUNK_SIZE:
            actions = np.pad(actions, ((0, self.CHUNK_SIZE - actions.shape[0]), (0, 0)), 'edge')

        state_dim = qpos.shape[1]
        state_indicator = np.ones(state_dim)

        # Parse images
        def parse_img(key):
            img_info = episode_data["images_info"].get(key)
            
            if key == 'head_camera':
                # Original video decoding for head camera
                if img_info is None:
                    print(f"Warning: head_camera not found in BSON for episode {episode_info}")
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
                
                video_frames = self._get_decoded_video(episode_info, key, img_info)
                
                if video_frames.ndim != 4:  # If decoding failed or empty
                    print(f"Warning: decoded video for {key} is empty")
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
                
                # Get image history
                start_idx = max(step_id - self.IMG_HISTORY_SIZE + 1, 0)
                imgs = video_frames[start_idx : step_id + 1]
            else:
                if img_info is None:
                    print(f"Warning: {key} not found in file for episode {episode_info}")
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
                
                start_idx = max(step_id - self.IMG_HISTORY_SIZE + 1, 0)
                imgs = self._load_file_sequence(img_info, start_idx, step_id + 1)
                
                if imgs.ndim != 4 or imgs.shape[0] == 0:  # If loading failed or empty
                    print(f"Warning: Failed to load file sequence for {key} in episode {episode_info}")
                    return np.zeros((self.IMG_HISTORY_SIZE, 480, 640, 3))
            
            # Pad images if history is not full
            if imgs.shape[0] < self.IMG_HISTORY_SIZE:
                pad_width = self.IMG_HISTORY_SIZE - imgs.shape[0]
                imgs = np.pad(imgs, ((pad_width, 0), (0,0), (0,0), (0,0)), 'edge')

            return imgs
        
        # Load head camera based on configuration
        camera_config = episode_data["camera_config"]
        if camera_config['has_head_in_bson']:
            # For 3-camera config, head is in BSON
            cam_high = parse_img('head_camera')
        else:
            # For 4-camera config, head is camera_0
            # cam_high = parse_img('camera_0')
            cam_high = parse_img('camera_head')
        
        # Create masks
        valid_len = min(step_id - (first_idx - 1) + 1, self.IMG_HISTORY_SIZE)
        cam_mask = np.array(
            [False] * (self.IMG_HISTORY_SIZE - valid_len) + [True] * valid_len
        )

        # Apply normalization if enabled
        if self.normalize_mode:
            state = self._normalize_data(state, 'state')
            actions = self._normalize_data(actions, 'action')
        
        sample = {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_mask,
        }
        
        # Add external cameras based on episode camera configuration
        camera_config = episode_data["camera_config"]
        for key, name in zip(camera_config['ext_image_keys'], camera_config['ext_image_names']):
            # Skip camera_0 for 4-camera config since it's already processed as cam_high
            if not camera_config['has_head_in_bson'] and key == 'camera_0':
                continue
            sample[name] = parse_img(key)
            sample[name + '_mask'] = cam_mask
        
        # self._cache[(episode_info, step_id)] = sample
        return True, sample
    
    def parse_episode_action(self, episode_info):
        """only return a random action chunk"""
        episode_data = self._extract_data_from_episode(episode_info)
        if not episode_data:
            return False, None

        qpos = episode_data["state"]
        num_steps = episode_data["episode_len"]

        if num_steps < self.CHUNK_SIZE:  # Drop too-short episodes
            return False, None
        
        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        if first_idx >= num_steps:  # case where robot doesn't move
            return False, None

        # Randomly sample a timestep
        step_id = np.random.randint(first_idx - 1, num_steps)
        
        actions_full = episode_data["action"]
        target_qpos = actions_full[step_id : step_id + self.CHUNK_SIZE]
        actions = target_qpos

        if actions.shape[0] < self.CHUNK_SIZE:
            actions = np.pad(actions, ((0, self.CHUNK_SIZE - actions.shape[0]), (0, 0)), 'edge')

        sample = {
            "actions": actions,
        }
        
        return True, sample

    def parse_episode_state_only(self, episode_info):
        """
        Parses an episode to generate full state and action trajectories.
        """
        episode_data = self._extract_data_from_episode(episode_info)
        if not episode_data:
            return False, None
        
        qpos = episode_data["state"]
        actions = episode_data["action"]
        num_steps = episode_data["episode_len"]

        if num_steps < self.CHUNK_SIZE:  # Drop too-short episodes
            return False, None
        
        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        first_idx = indices[0] if len(indices) > 0 else 1
        
        if first_idx >= num_steps:
            return False, None

        state_traj = qpos[first_idx-1:]
        action_traj = actions[first_idx-1:]
        
        # Apply normalization if enabled
        if self.normalize_mode:
            state_traj = self._normalize_data(state_traj, 'state')
            action_traj = self._normalize_data(action_traj, 'action')
        return True, {
            "state": state_traj,
            "action": action_traj
        }
    
    def _load_statistics(self, stats_file: str):
        """
        Load statistics from JSON file for normalization.
        
        Args:
            stats_file (str): Path to dataset_statistics.json file
        """
        try:
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded statistics from {stats_file}")
            print(f"State dim: {len(self.stats['state']['mean'])}, Action dim: {len(self.stats['action']['mean'])}")
        except Exception as e:
            print(f"Warning: Failed to load statistics from {stats_file}: {e}")
            self.stats = None
    
    def _normalize_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Normalize data using loaded statistics.
        
        Args:
            data (np.ndarray): Data to normalize (state or action)
            data_type (str): 'state' or 'action'
            
        Returns:
            np.ndarray: Normalized data
        """
        if self.stats is None or self.normalize_mode is None:
            return data
            
        if data_type not in self.stats:
            print(f"Warning: No statistics found for {data_type}")
            return data
            
        stats_data = self.stats[data_type]
        
        if self.normalize_mode == 'mean_std':
            # Normalize using mean and standard deviation: (x - mean) / std
            mean = np.array(stats_data['mean'])
            std = np.array(stats_data['std'])
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            normalized = (data - mean) / std
            normalized = np.clip(normalized, -2.0, 2.0)  
            
        elif self.normalize_mode == 'min_max':
            # Normalize using percentiles as min/max: (x - min) / (max - min)
            min_val = np.array(stats_data['percentile_1'])
            max_val = np.array(stats_data['percentile_99'])
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val
            
        else:
            print(f"Warning: Unknown normalization mode {self.normalize_mode}")
            return data
            
        return normalized

def collect_statistics(dataset: BsonVLADataset, num_samples=10000):
    """
    Collect statistics for state and action dimensions by iterating through episodes.
    
    Args:
        dataset: BsonVLADataset instance
        num_samples: Maximum number of samples to collect statistics from
        
    Returns:
        dict: Statistics containing mean, std, 1st and 99th percentiles
    """
    print(f"Collecting statistics from {len(dataset)} episodes (target: {num_samples} samples)...")
    
    states = []
    actions = []
    total_samples = 0
    
    # Iterate through all episodes
    episode_indices = list(range(len(dataset)))
    random.shuffle(episode_indices)
    for ep_idx in episode_indices:
        if total_samples >= num_samples:
            break
            
        if ep_idx % 50 == 0:
            print(f"Processing episode {episode_indices.index(ep_idx)}/{len(dataset)}, collected {total_samples} samples")
        
        # Get full trajectory for this episode
        sample = dataset.get_item(index=ep_idx, state_only=True)
        
        if sample is None:
            print(f"Episode {ep_idx} is invalid, skipping...")
            continue
            
        episode_states = sample['state']  # Shape: (traj_len, state_dim)
        episode_actions = sample['action']  # Shape: (traj_len, action_dim)
        # print(f"Episode {ep_idx} has {len(episode_states)} timesteps.")
        
        # Add all timesteps from this episode
        for t in range(len(episode_states)):
            if total_samples >= num_samples:
                break
                
            states.append(episode_states[t])
            actions.append(episode_actions[t])
            total_samples += 1
                
    if not states or not actions:
        raise ValueError("No valid samples collected")
    
    # Convert to numpy arrays
    states = np.array(states)  # Shape: (num_samples, state_dim)
    actions = np.array(actions)  # Shape: (num_samples, action_dim)
    
    print(f"Collected {len(states)} valid samples from {episode_indices.index(ep_idx) + 1} episodes")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    
    # Calculate statistics
    stats = {
        'state': {
            'mean': np.mean(states, axis=0).tolist(),
            'std': np.std(states, axis=0).tolist(),
            'percentile_1': np.percentile(states, 1, axis=0).tolist(),
            'percentile_99': np.percentile(states, 99, axis=0).tolist()
        },
        'action': {
            'mean': np.mean(actions, axis=0).tolist(),
            'std': np.std(actions, axis=0).tolist(),
            'percentile_1': np.percentile(actions, 1, axis=0).tolist(),
            'percentile_99': np.percentile(actions, 99, axis=0).tolist()
        },
        'metadata': {
            'num_samples': len(states),
            'state_dim': states.shape[1],
            'action_dim': actions.shape[1],
            'num_episodes_processed': ep_idx + 1,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return stats, states, actions

def plot_distributions(states, actions, output_dir="."):
    """
    Plot distribution histograms for state and action dimensions with statistics overlay.
    
    Args:
        states: numpy array of states (num_samples, state_dim)
        actions: numpy array of actions (num_samples, action_dim)
        output_dir: Directory to save plots
    """
    print("Generating distribution plots with statistics...")
    
    # Calculate statistics for plotting
    state_means = np.mean(states, axis=0)
    state_stds = np.std(states, axis=0)
    state_p1 = np.percentile(states, 1, axis=0)
    state_p99 = np.percentile(states, 99, axis=0)
    
    action_means = np.mean(actions, axis=0)
    action_stds = np.std(actions, axis=0)
    action_p1 = np.percentile(actions, 1, axis=0)
    action_p99 = np.percentile(actions, 99, axis=0)
    
    # Plot state distributions
    state_dim = states.shape[1]
    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    fig.suptitle('State Dimensions Distribution with Statistics', fontsize=16)
    
    for i in range(min(36, state_dim)):
        row = i // 6
        col = i % 6
        if row < 6 and col < 6:
            ax = axes[row, col]
            
            # Plot histogram
            ax.hist(states[:, i], bins=50, alpha=0.7, edgecolor='black', color='skyblue')
            
            # Add statistical lines
            ax.axvline(state_means[i], color='red', linestyle='-', linewidth=2, label=f'Mean: {state_means[i]:.3f}')
            ax.axvline(state_means[i] - state_stds[i], color='orange', linestyle='--', linewidth=1.5, label=f'Mean-Std: {state_means[i]-state_stds[i]:.3f}')
            ax.axvline(state_means[i] + state_stds[i], color='orange', linestyle='--', linewidth=1.5, label=f'Mean+Std: {state_means[i]+state_stds[i]:.3f}')
            ax.axvline(state_p1[i], color='green', linestyle=':', linewidth=1.5, label=f'P1: {state_p1[i]:.3f}')
            ax.axvline(state_p99[i], color='green', linestyle=':', linewidth=1.5, label=f'P99: {state_p99[i]:.3f}')
            
            ax.set_title(f'State Dim {i}\nμ={state_means[i]:.3f}, σ={state_stds[i]:.3f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='upper right')
    
    # Hide unused subplots
    for i in range(state_dim, 36):
        row = i // 6
        col = i % 6
        if row < 6 and col < 6:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    state_plot_path = os.path.join(output_dir, 'state_distributions.png')
    plt.savefig(state_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"State distribution plot saved to: {state_plot_path}")
    
    # Plot action distributions
    action_dim = actions.shape[1]
    fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    fig.suptitle('Action Dimensions Distribution with Statistics', fontsize=16)
    
    for i in range(min(36, action_dim)):
        row = i // 6
        col = i % 6
        if row < 6 and col < 6:
            ax = axes[row, col]
            
            # Plot histogram
            ax.hist(actions[:, i], bins=50, alpha=0.7, edgecolor='black', color='lightcoral')
            
            # Add statistical lines
            ax.axvline(action_means[i], color='red', linestyle='-', linewidth=2, label=f'Mean: {action_means[i]:.3f}')
            ax.axvline(action_means[i] - action_stds[i], color='orange', linestyle='--', linewidth=1.5, label=f'Mean-Std: {action_means[i]-action_stds[i]:.3f}')
            ax.axvline(action_means[i] + action_stds[i], color='orange', linestyle='--', linewidth=1.5, label=f'Mean+Std: {action_means[i]+action_stds[i]:.3f}')
            ax.axvline(action_p1[i], color='green', linestyle=':', linewidth=1.5, label=f'P1: {action_p1[i]:.3f}')
            ax.axvline(action_p99[i], color='green', linestyle=':', linewidth=1.5, label=f'P99: {action_p99[i]:.3f}')
            
            ax.set_title(f'Action Dim {i}\nμ={action_means[i]:.3f}, σ={action_stds[i]:.3f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='upper right')
    
    # Hide unused subplots
    for i in range(action_dim, 36):
        row = i // 6
        col = i % 6
        if row < 6 and col < 6:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    action_plot_path = os.path.join(output_dir, 'action_distributions.png')
    plt.savefig(action_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Action distribution plot saved to: {action_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BSON VLA Dataset Testing and Statistics')
    parser.add_argument('--stat', action='store_true', help='Collect statistics from 10k samples')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples for statistics (default: 10000)')
    parser.add_argument('--output_dir', type=str, default='bson_stats', help='Output directory for statistics and plots')
    parser.add_argument('--normalize', choices=['mean_std', 'min_max'], help='Normalization mode')
    parser.add_argument('--stats_file', type=str, help='Path to dataset_statistics.json for normalization')
    
    args = parser.parse_args()
    
    # --- Dataset Initialization ---
    ds = BsonVLADataset(normalize_mode=args.normalize, stats_file=args.stats_file)
    
    if len(ds) == 0:
        print("\nDataset initialized but contains no valid episodes.")
        exit(1)
    
    if args.stat:
        stats, states, actions = collect_statistics(ds, args.num_samples)
        
        # Save statistics to JSON
        os.makedirs(args.output_dir, exist_ok=True)
        stats_file = os.path.join(args.output_dir, 'dataset_statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_file}")
        
        # Generate distribution plots
        plot_distributions(states, actions, args.output_dir)
        
        # Print summary
        print("\n=== Statistics Summary ===")
        print(f"State dimensions: {stats['metadata']['state_dim']}")
        print(f"Action dimensions: {stats['metadata']['action_dim']}")
        print(f"Samples collected: {stats['metadata']['num_samples']}")
        print(f"\nState statistics:")
        print(f"  Mean range: [{np.min(stats['state']['mean']):.4f}, {np.max(stats['state']['mean']):.4f}]")
        print(f"  Std range: [{np.min(stats['state']['std']):.4f}, {np.max(stats['state']['std']):.4f}]")
        print(f"\nAction statistics:")
        print(f"  Mean range: [{np.min(stats['action']['mean']):.4f}, {np.max(stats['action']['mean']):.4f}]")
        print(f"  Std range: [{np.min(stats['action']['std']):.4f}, {np.max(stats['action']['std']):.4f}]")
            
    else:
        # --- Example Usage ---
        print(f"\n--- Testing get_item (state_only=False) for one item ---")
        sample = ds.get_item()
        print("Sample keys:", sample.keys())
        print("Meta:", sample['meta'])
        print("State shape:", sample['state'].shape)
        print("State indicator shape:", sample['state_indicator'].shape)
        print("State mean shape:", sample['state_mean'].shape)
        print("Actions shape:", sample['actions'].shape)
        print("Cam High shape:", sample['cam_high'].shape)
        # Print camera info based on what's actually in the sample
        camera_keys = [k for k in sample.keys() if k.startswith('cam_') and not k.endswith('_mask')]
        for cam_key in camera_keys:
            print(f"Cam {cam_key} shape:", sample[cam_key].shape)
        print("Cam masks:", sample['cam_high_mask'])

        print(f"\n--- Testing get_item (state_only=True) for one item ---")
        state_sample = ds.get_item(state_only=True)
        print("State sample keys:", state_sample.keys())
        print("Full state trajectory shape:", state_sample['state'].shape)
        print("Full action trajectory shape:", state_sample['action'].shape)

        print("First state:", state_sample['state'][0])

        print("\n--- Testing get_item for 100 items ---")
        for _ in range(100):
            ds.get_item()