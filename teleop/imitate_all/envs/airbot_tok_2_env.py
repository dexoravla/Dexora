from robots.airbots.airbot_tok.airbot_tok_2 import AIRBOTTOK
from robots.common import make_robot_from_yaml
import time
import collections
import dm_env


class AIRBOTTOKEnv(object):
    def __init__(self, config_path: str):
        self.robot: AIRBOTTOK = make_robot_from_yaml(config_path)
        assert isinstance(self.robot, AIRBOTTOK)
        self._all_joints_num = 14

    def set_reset_position(self, reset_position):
        reset_position = list(reset_position)
        assert (
            len(reset_position) == self._all_joints_num
        ), f"Expected {self._all_joints_num} joints, got {len(reset_position)}"
        if self.robot.config.data_style == 3.0:
            arm_targets = (
                reset_position[:6] + reset_position[12:13],
                reset_position[6:12] + reset_position[13:14],
            )
        else:
            arm_targets = (reset_position[:7], reset_position[7:14])
        for i, arm in enumerate(self.robot.arms.values()):
            arm.config.default_action = arm_targets[i]

    def reset(self, sleep_time=0):
        self.robot.reset()
        time.sleep(sleep_time)
        return self._get_obs()

    def _get_obs(self):
        obs = collections.OrderedDict()
        obs["images"] = {}
        raw_obs = self.robot.capture_observation()
        low_dim = raw_obs["low_dim"]
        if self.robot.config.data_style == 3.0:
            qpos = [0] * self._all_joints_num
            arm_joint_num = len(self.robot.arms) * 6
            for index, arm_name in enumerate(self.robot.arms.keys()):
                arm_eef = low_dim[f"observation/{arm_name}/joint_position"]
                qpos[index * 6 : (index + 1) * 6] = arm_eef[:6]
                qpos[arm_joint_num + index] = arm_eef[6]
            obs["qpos"] = qpos
        else:
            obs["qpos"] = []
            for arm_name in self.robot.arms:
                obs["qpos"].extend(low_dim[f"observation/{arm_name}/joint_position"])

        for name in self.robot.cameras:
            assert name not in obs["images"], f"Duplicate camera name: {name}"
            obs["images"][name] = raw_obs[f"observation.images.{name}"]

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
        self.robot.send_action(action)
        time.sleep(sleep_time)
        obs = self._get_obs() if get_obs else None
        return obs


def make_env(config):

    env = AIRBOTTOKEnv(config["env_config_path"])
    env.set_reset_position(config["start_joint"])

    return env
