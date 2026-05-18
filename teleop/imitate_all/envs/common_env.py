import time
import numpy as np
import dm_env
import torch
from typing import List
from einops import rearrange
from robots.common_robot import AssembledRobot


class CommonEnvConfig(object):
    def __init__(self) -> None:
        self.robots = []


class CommonEnv:
    """
    An environment is a combination of robots, scenes and objects. It should be able to reset and step.
    The environment will return observations based on the state of the robot, the position of the sensors, and the current scene and object conditions. And for RL and data collection, it should also send rewards and done signals.
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def reset(self) -> dm_env.TimeStep:
        raise NotImplementedError

    def step(self, action) -> dm_env.TimeStep:
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError


def get_image(ts: dm_env.TimeStep, camera_names, mode=0):
    # TODO: remove this function
    images: dict = ts.observation["images"]
    if mode == 0:  # 输出拼接之后的张量图
        curr_images = []
        for cam_name in camera_names:
            assert (
                cam_name in images
            ), f"Camera {cam_name} not found in images {images.keys()}"
            curr_image = rearrange(images[cam_name], "h w c -> c h w")
            curr_images.append(curr_image)
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    else:  # 输出独立的张量图（且每个是多维的）  # TODO: 修改为每个是一维的
        curr_image = {}
        for cam_name in camera_names:
            raw_img = images[cam_name]
            # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            # cv2.imshow(cam_name, raw_img.astype(np.uint8))
            # cv2.waitKey(0)
            curr_image[cam_name] = torch.from_numpy(
                np.copy(raw_img[np.newaxis, :])
            ).float()
    return curr_image


def move_robots(bot_list: List[AssembledRobot], target_pose_list, move_time=1):
    DT = max([bot.dt for bot in bot_list])  # TODO: change dt to arg as dt_list
    num_steps = int(move_time / DT)
    curr_pose_list = [bot.get_current_joint_positions() for bot in bot_list]
    # 进行关节插值，保证平稳运动
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            # blocking为False用于多台臂可以同时移动
            bot.set_joint_position_target(traj_list[bot_id][t], [6], blocking=False)
        time.sleep(DT)
