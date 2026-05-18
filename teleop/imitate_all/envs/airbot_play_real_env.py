import time
import numpy as np
import collections
import dm_env
from typing import List
from robots.common_robot import AssembledRobot, AssembledFakeRobot
from envs.common_env import move_robots


class RealEnv:
    """
    Environment for real robot one-manual manipulation
    Action space:      [arm_qpos (6),             # absolute joint position
                        gripper_positions (1),]    # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ arm_qpos (6),          # absolute joint position
                                        gripper_position (1),]  # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ arm_qvel (6),         # absolute joint velocity (rad)
                                        gripper_velocity (1),]  # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"0": (480x640x3),        # h, w, c, dtype='uint8'
                                   "1": (480x640x3),         # h, w, c, dtype='uint8'
                                   "2": (480x640x3),}  # h, w, c, dtype='uint8'
    """

    def __init__(
        self,
        record_images=True,
        robot_instances: List[AssembledRobot] = None,
        cameras=None,
    ):
        """
        当指定多个机器人时，将会对step输入的action按照机器人数量均分并按顺序依次分配执行，获取观测时同理。
        :param record_images: 是否记录图像
        :param robot_instances: 机器人实例列表
        :param cameras: 相机实例字典
        环境将根据机器人的外部传感器配置信息自动启动相应的接口来获取数据，例如USB相机和RealSense等。
        """
        self.airbot_players = robot_instances
        self.robot_num = len(self.airbot_players)
        self.use_base = False
        use_fake_robot = isinstance(self.airbot_players[0], AssembledFakeRobot)
        if record_images:
            if use_fake_robot and not AssembledFakeRobot.real_camera:
                from robot_utils import ImageRecorderFake as ImageRecorder
            elif isinstance(cameras, dict):
                from robot_utils import ImageRecorderVideo as ImageRecorder
            else:
                from robot_tools.recorder import ImageRecorderRos as ImageRecorder
                import rospy

                if rospy.get_name() == "/unnamed":
                    rospy.init_node("real_env", anonymous=True)
            self.image_recorder = ImageRecorder(cameras)
        self.reset_position = None
        self.joints_num = self.airbot_players[0].joints_num
        if use_fake_robot:
            print("AIRBOT Play Fake Env Created.")
        else:
            print("AIRBOT Play Real Env Created.")

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position

    def get_qpos(self, normalize_gripper=False):
        """7 dof: 6 arm joints + 1 gripper joint"""
        qpos = []
        for airbot in self.airbot_players:
            qpos.append(airbot.get_current_joint_positions())
        return np.hstack(qpos)

    def get_qvel(self):
        qvel = []
        for airbot in self.airbot_players:
            qvel.append(airbot.get_current_joint_velocities())
        return np.hstack(qvel)

    def get_effort(self):
        effort = []
        for airbot in self.airbot_players:
            effort.append(airbot.get_current_joint_efforts())
        return np.hstack(effort)

    def get_images(self):
        return self.image_recorder.get_images()

    def _reset_joints(self):
        assert self.reset_position is not None, "Reset position is not set."
        arms_reset_position = []
        all_n = self.joints_num
        for i in range(self.robot_num):
            start = i * all_n
            end = start + all_n
            arms_reset_position.append(self.reset_position[start:end])
        move_robots(self.airbot_players, arms_reset_position, move_time=1)

    def _get_observation(self):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["images"] = self.get_images()
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        if not fake:
            self._reset_joints()
            time.sleep(sleep_time)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self._get_observation(),
        )

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
        arm_vel=0,
    ):
        use_planning = False
        for index, robot in enumerate(self.airbot_players):
            jn = robot.joints_num
            robot.set_joint_position_target(
                action[jn * index : jn * (index + 1)], [arm_vel], use_planning,
            )
        time.sleep(sleep_time)
        if get_obs:
            obs = self._get_observation()
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

def make_env(
    record_images=True,
    robot_instance=None,
    cameras=None,
):
    env = RealEnv(record_images, robot_instance, cameras)
    return env
