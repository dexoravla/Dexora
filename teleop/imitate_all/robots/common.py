"""A robot is a physical instance that has its proprioception state and can interact with the environment by subjective initiative actions. The robot's state is the information that can be obtained from the robot's body sensors while the actions are the control commands that can be sent to the robot's actuators. Vision, touch and the other external sensation (obtained by tactile sensors, cameras, lidar, radar, ultrasonic sensors, etc.) are not included in the robot's state, but in the environment. However, in addition to being related to the external environment, external observation also depends on the robot's state and the position and posture of the corresponding sensors. So the robot instance should have the full information and configurations of its external sensors to let the environment obtaining correct observations."""

from typing import Protocol, Dict, List, Optional, Union
from dataclasses import dataclass, field, replace
from habitats.common.robot_devices.cameras.utils import Camera
import time
import numpy as np


# class Robot(Protocol):
#     """Assume the __init__ method of the robot class is the same as the reset method.
#     So you can inherit this class to save writing the initialization function."""

#     def __init__(self, config) -> None:
#         self.reset(config)

#     def reset(self, config):
#         raise NotImplementedError

#     def step(self, action):
#         raise NotImplementedError

#     def exit(self):
#         raise NotImplementedError

#     @property
#     def state(self):
#         raise NotImplementedError


class Robot(Protocol):
    def init_teleop(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def enter_active_mode(self): ...
    def enter_passive_mode(self): ...
    def enter_servo_mode(self): ...
    def enter_traj_mode(self): ...
    def get_low_dim_data(self): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def reset(self): ...
    def exit(self): ...
    def get_state_mode(self): ...
    def low_dim_to_action(self, low_dim, step): ...


@dataclass
class FakeRobotConfig(object):
    cameras: Dict[str, Camera] = field(default_factory=lambda: {})


class FakeRobot(object):
    def __init__(self, config: Optional[FakeRobotConfig] = None, **kwargs) -> None:
        if config is None:
            config = FakeRobotConfig()
        self.config = replace(config, **kwargs)
        self.cameras = self.config.cameras
        self.logs = {}
        self.state = [0] * 6
        self.__init()

    def __init(self):
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
        self.reset()

    def reset(self):
        self.state = [0] * 6
        self._state_mode = "active"
        print("Fake robot reset")

    def enter_active_mode(self):
        self._state_mode = "active"
        print("Fake robot entered active mode")

    def enter_passive_mode(self):
        self._state_mode = "passive"
        print("Fake robot entered passive mode")

    def get_low_dim_data(self):
        pose = [0.5, 0.5, 0.5, 0, 0, 0, 1]
        # pose = [0, 0, 0, 0, 0, 0, 1]
        low_dim = {
            "observation/arm/joint_position": self.state,
            "observation/eef/joint_position": [0],
            "observation/eef/pose": pose,
        }
        return low_dim

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

    def init_teleop(self):
        print("init_teleop")

    def run_calibration(self):
        print("run_calibration`")

    def teleop_step(self, record_data=False):
        print("teleop_step")

    def send_action(self, action):
        # print(f"send_action:{action}")
        self.state = action

    def exit(self):
        try:
            for name in self.cameras:
                self.cameras[name].disconnect()
        except Exception as e:
            pass
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode


def make_robot(config) -> Robot:
    if isinstance(config, str):
        raise NotImplementedError("config should be a dict or a dataclass object.")
    return Robot(config)


def make_robot_from_hydra_config(cfg) -> Robot:
    from omegaconf import DictConfig
    import hydra

    cfg: DictConfig
    robot = hydra.utils.instantiate(cfg)
    return robot


def make_robot_from_yaml(
    robot_path: str, robot_overrides: Optional[List[str]] = None
) -> Robot:
    from habitats.common.utils.utils import init_hydra_config

    robot_cfg = init_hydra_config(robot_path, robot_overrides)
    robot = make_robot_from_hydra_config(robot_cfg)

    return robot
