import time
import collections
import dm_env
import importlib
from discoverse.envs.airbot_play_base import AirbotPlayCfg
from discoverse.task_base import AirbotPlayTaskBase


class MujocoEnv(object):
    """
    Mujoco environment for airbot_play
    path: path to the script containing the SimNode class and config (mainly for data collection)
    """

    def __init__(self, path: str):
        module = importlib.import_module(path.replace("/", ".").replace(".py", ""))
        node_cls = getattr(module, "SimNode")
        cfg: AirbotPlayCfg = getattr(module, "cfg")
        cfg.headless = False
        self.exec_node: AirbotPlayTaskBase = node_cls(cfg)
        # self.exec_node.cam_id = self.exec_node.config.obs_camera_id
        self.reset_position = None
        print("MujocoEnv initialized")

    def set_reset_position(self, reset_position):
        self.reset_position = reset_position
        print("Resetting to the given position: ", self.reset_position)

    def get_reward(self):
        return 0

    def reset(self, fake=False, sleep_time=0):
        self.exec_node.domain_randomization()
        raw_obs = self.exec_node.reset()
        # if self.reset_position is not None:
        #     # self.reset_position[-1] = 0.96
        #     # print("Resetting to the given position: ", self.reset_position)
        #     self.reset_position[-1] = 0.04  # undo the normalization
        #     self.exec_node.mj_data.ctrl[:7] = self.reset_position
        #     self.exec_node.mj_data.ctrl[7] = -self.exec_node.mj_data.ctrl[6]
        #     self.exec_node.mj_data.qpos[:7] = self.reset_position
        #     self.exec_node.mj_data.qpos[7] = -self.exec_node.mj_data.qpos[6]
        #     raw_obs, pri_obs, rew, ter, info = self.exec_node.step(self.reset_position)
        time.sleep(sleep_time)
        obs = collections.OrderedDict()
        obs["qpos"] = list(raw_obs["jq"])
        # print("obs gripper", raw_obs["jq"][-1])
        # print("pre_obs", obs["qpos"])
        # obs["qpos"][-1] *= 25  # undo the normalization
        # print("post_obs", obs["qpos"])
        # obs["qvel"] = raw_obs["jv"]
        obs["images"] = {}
        for id in self.exec_node.config.obs_rgb_cam_id:
            obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        get_obs=True,
        sleep_time=0,
    ):
        raw_obs, pri_obs, rew, ter, info = self.exec_node.step(action)
        time.sleep(sleep_time)

        if get_obs:
            obs = collections.OrderedDict()
            obs["qpos"] = list(raw_obs["jq"])
            obs["images"] = {}
            for id in self.exec_node.config.obs_rgb_cam_id:
                obs["images"][f"{id}"] = raw_obs["img"][id][:, :, ::-1]
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )


def make_env(path):
    env = MujocoEnv(path)
    return env
