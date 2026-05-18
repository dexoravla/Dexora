from robots.airbots.airbot_mmk.airbot_mmk2 import AIRBOTMMK2
from robots.common import make_robot_from_yaml
import time
import collections
import dm_env
import numpy as np


class AIRBOTMMK2Env(object):
    def __init__(self, config_path: str):
        self.robot: AIRBOTMMK2 = make_robot_from_yaml(config_path)
        assert isinstance(self.robot, AIRBOTMMK2)
        self._all_joints_num = self.robot.joint_num

    def set_reset_position(self, reset_position):
        assert (
            len(reset_position) == self._all_joints_num
        ), f"Expected {self._all_joints_num} joints, got {len(reset_position)}"
        self.robot.config.default_action = reset_position

    def reset(self, sleep_time=0):
        self.robot.reset(sleep_time)
        return self._get_obs()

    def _get_obs(self):
        obs = collections.OrderedDict()
        obs["qpos"] = []
        obs["images"] = {}
        raw_obs = self.robot.capture_observation()
        low_dim = raw_obs["low_dim"]
        for comp in self.robot.components:
            obs["qpos"].extend(low_dim[f"observation/{comp.value}/joint_position"])
        for camera in self.robot.cameras:
            assert camera not in obs["images"], f"Duplicate camera name: {camera}"
            obs["images"][camera.value] = raw_obs[f"observation.images.{camera.value}"]

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        sleep_time=0,
        get_obs=True,
    ):
        joint_limits = (
            (-3.09, 2.04),
            (-2.92, 0.12),
            (-0.04, 3.09),
            (-3.1, 3.1),  # (-2.95, 2.95),
            (-1.9, 1.9),  # (-1.08, 1.08),
            (-3.0, 3.0),  # (-2.90, 2.90),
            (0, 1),
        )

        jn = len(joint_limits)
        for i in range(2):
            for j in range(jn):
                index = j + jn * i
                action[index] = np.clip(action[index], *joint_limits[j])

        self.robot.send_action(action)
        time.sleep(sleep_time)
        obs = self._get_obs() if get_obs else None
        return obs


def make_env(config):

    env = AIRBOTMMK2Env(config["env_config_path"])
    env.set_reset_position(config["start_joint"])

    return env
