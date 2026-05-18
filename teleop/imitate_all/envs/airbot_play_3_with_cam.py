import time
import collections
import dm_env
from typing import List, Union
from robots.airbots.airbot_play.airbot_play_3 import AIRBOTPlay
from robots.common import make_robot_from_yaml


class AIRBOTPlayWithCameraEnv(object):

    def __init__(
        self,
        config_paths: Union[List[str], str],
    ):
        self.robots: List[AIRBOTPlay] = []
        if isinstance(config_paths, str):
            config_paths = [config_paths]
        for cfg in config_paths:
            robot = make_robot_from_yaml(cfg)
            assert isinstance(robot, AIRBOTPlay)
            self.robots.append(robot)
        self._all_joints_num = [7] * len(self.robots)
        print("robot number:", len(self.robots))

    def set_reset_position(self, reset_position):
        self._all_joints_num = []
        for i, robot in enumerate(self.robots):
            robot.config.start_arm_joint_position = reset_position[i * 6 : (i + 1) * 6]
            robot.config.start_eef_joint_position = reset_position[(i + 1) * 6]
            self._all_joints_num.append(7)

    def _get_obs(self):
        obs = collections.OrderedDict()
        obs["qpos"] = []
        obs["images"] = {}
        for robot in self.robots:
            raw_obs = robot.capture_observation()
            low_dim = raw_obs["low_dim"]
            obs["qpos"].extend(
                low_dim["observation/arm/joint_position"]
                + low_dim["observation/eef/joint_position"]
            )
            for name in robot.cameras:
                assert name not in obs["images"], f"Duplicate camera name: {name}"
                obs["images"][name] = raw_obs[f"observation.images.{name}"]

        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=0,
            discount=None,
            observation=obs,
        )

    def reset(self, sleep_time=0):
        for robot in self.robots:
            robot.reset()
            robot.enter_servo_mode()
        time.sleep(sleep_time)
        return self._get_obs()

    def step(
        self,
        action,
        sleep_time=0,
        get_obs=True,
    ):
        for index, jn in enumerate(self._all_joints_num):
            one_action = action[jn * index : jn * (index + 1)]
            self.robots[index].send_action(one_action)
        time.sleep(sleep_time)
        obs = self._get_obs() if get_obs else None
        return obs
