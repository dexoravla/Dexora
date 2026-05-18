from dataclasses import dataclass, field, replace
from habitats.common.robot_devices.cameras.utils import Camera
from typing import Dict, Optional, List, Union
from robots.airbots.airbot_play.airbot_play_2 import AIRBOTPlay, AIRBOTPlayConfig
from robots.airbots.airbot_play.airbot_replay_remote import (
    AIRBOTReplay,
    AIRBOTReplayConfig,
)
from threading import Thread, Event
import time
import numpy as np


@dataclass
class AIRBOTPlayDemonstrationConfig(object):
    groups: Dict[str, Dict[str, Union[AIRBOTPlayConfig, AIRBOTReplayConfig]]] = field(
        default_factory=lambda: {}
    )
    cameras: Dict[str, Camera] = field(default_factory=lambda: {})


class AIRBOTPlayDemonstration(object):
    def __init__(
        self, config: Optional[AIRBOTPlayDemonstrationConfig] = None, **kwargs
    ):
        if config is None:
            config = AIRBOTPlayDemonstrationConfig()
        # Overwrite config arguments using kwargs (used for yaml config)
        self.config = replace(config, **kwargs)
        # TODO: add cameras for each robot?
        self.cameras = self.config.cameras
        self._state_mode = "active"
        self.logs = {}
        self.leaders: Dict[str, Union[AIRBOTPlay, AIRBOTReplay]] = {}
        self.followers: Dict[str, List[AIRBOTPlay]] = {}

        self.is_replay = {}
        for g_name, g_value in self.config.groups.items():
            leader_cfg: dict = g_value["leader"]
            followers_cfg = g_value["followers"]
            if leader_cfg.get("joint_states_topic", None) is None:
                self.leaders[g_name] = AIRBOTPlay(**leader_cfg)
                self.is_replay[g_name] = (
                    leader_cfg["forearm_type"],
                    leader_cfg["bigarm_type"],
                ) == ("encoder", "encoder")
            else:
                self.leaders[g_name] = AIRBOTReplay(**leader_cfg)
                self.is_replay[g_name] = True
            self.followers[g_name] = []
            for f_cfg in followers_cfg:
                self.followers[g_name].append(AIRBOTPlay(**f_cfg))
        for name in self.cameras:
            self.cameras[name].connect()
        self._is_running = True
        self._reseting = Event()
        self.__sync_thread = Thread(target=self.__sync, daemon=True)
        self.__sync_thread.start()
        self.reset()

    def __sync(self):
        duration = 0.001
        while self._is_running:
            self._reseting.wait()
            for g_name in self.config.groups.keys():
                l_pos = self.leaders[g_name].get_current_joint_positions()
                for follower in self.followers[g_name]:
                    follower.set_joint_position_target(l_pos, [6.0])
            time.sleep(duration)

    def reset(self):
        self._reseting.clear()
        time.sleep(0.1)
        leaders = list(self.leaders.values())
        is_replay = list(self.is_replay.values())
        for index, followers in enumerate(self.followers.values()):
            default_action = leaders[index].config.default_action
            if (default_action is not None) and not is_replay[index]:
                for follower in followers:
                    follower.enter_active_mode()
                    follower.set_joint_position_target(default_action, [0.2], True)
        for index, leader in enumerate(leaders):
            default_action = leader.config.default_action
            if (default_action is not None) and not is_replay[index]:
                if leader.enter_active_mode():
                    leader.set_joint_position_target(default_action, [0.2], True)
        self._state_mode = "active"
        self._reseting.set()

    def enter_active_mode(self):
        for leader in self.leaders.values():
            leader.enter_active_mode()
        self._state_mode = "active"

    def enter_passive_mode(self):
        for leader in self.leaders.values():
            leader.enter_passive_mode()
        self._state_mode = "passive"

    def get_low_dim_data(self) -> Dict[str, list]:
        data = {}
        data["/time"] = time.time()

        action_arm_jq = []
        action_eef_jq = []
        action_eef_pose = []
        for leader in self.leaders.values():
            jq = leader.get_current_joint_positions()
            action_arm_jq.extend(jq[:6])
            action_eef_jq.append(jq[6])
            pose = leader.get_current_pose()
            action_eef_pose.extend(pose[0] + pose[1])  # xyz + quat(xyzw)
        data["action/arm/joint_position"] = action_arm_jq
        data["action/eef/joint_position"] = action_eef_jq
        data["action/eef/pose"] = action_eef_pose

        obs_arm_jq = []
        obs_eef_jq = []
        obs_eef_pose = []
        for followers in self.followers.values():
            for follower in followers:
                jq = follower.get_current_joint_positions()
                obs_arm_jq.extend(jq[:6])
                obs_eef_jq.append(jq[6])
                pose = follower.get_current_pose()
                obs_eef_pose.extend(pose[0] + pose[1])  # xyz + quat(xyzw)
        data["observation/arm/joint_position"] = obs_arm_jq
        data["observation/eef/joint_position"] = obs_eef_jq
        data["observation/eef/pose"] = obs_eef_pose

        return data

    def capture_observation(self) -> Dict[str, Union[dict, np.ndarray]]:
        """The returned observations do not have a batch dimension."""
        obs_act_dict = {}
        # Capture images from cameras
        images = {}
        depths = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            cam_data = self.cameras[name].async_read()
            if len(cam_data) == 2:
                images[name], depths[name] = cam_data
            else:
                images[name] = cam_data
            # images[name] = torch.from_numpy(images[name])
            obs_act_dict[f"/time/{name}"] = time.time()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        obs_act_dict["low_dim"] = self.get_low_dim_data()
        for name in images:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        for name in depths:
            obs_act_dict[f"observation.depths.{name}"] = depths[name]
        return obs_act_dict

    def exit(self):
        for name in self.cameras:
            self.cameras[name].disconnect()
        self._is_running = False
        print("Waiting for sync thread to finish")
        self.__sync_thread.join()
        print("deleting leaders")
        for g_name, leader in self.leaders.items():
            if not self.is_replay[g_name]:
                del leader
        print("deleting followers")
        del self.followers
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode
