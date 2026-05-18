from mmk2_sdk.mmk2_client import AIRBOTMMK2 as AIRBOTMMK2Client
from mmk2_types.types import (
    MMK2Components,
    JointNames,
    ComponentTypes,
    TopicNames,
    MMK2ComponentsGroup,
    ImageTypes,
    ControllerTypes,
)
from mmk2_types.grpc_msgs import (
    Time,
    JointState,
    TrajectoryParams,
    MoveServoParams,
    ForwardPositionParams,
)
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, replace, field
import time
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIRBOTMMK2Config(object):
    name: str = "mmk2"
    domain_id: int = -1
    ip: str = "192.168.11.200"
    port: int = 50055
    default_action: Optional[List[float]] = None
    cameras: Dict[str, str] = field(default_factory=lambda: {})
    components: List[str] = field(
        default_factory=lambda: [
            MMK2Components.LEFT_ARM.value,
            MMK2Components.LEFT_ARM_EEF.value,
            MMK2Components.RIGHT_ARM.value,
            MMK2Components.RIGHT_ARM_EEF.value,
        ]
    )
    demonstrate: bool = False


class AIRBOTMMK2(object):
    def __init__(self, config: Optional[AIRBOTMMK2Config] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTMMK2Config()
        self.config = replace(config, **kwargs)
        self.robot = AIRBOTMMK2Client(
            self.config.ip,
            self.config.port,
            self.config.name,
            self.config.domain_id,
        )
        self.joint_names = {}
        self.cameras: Dict[MMK2Components, str] = {}
        self.components: Dict[MMK2Components, ComponentTypes] = {}
        all_joint_names = JointNames()
        self.joint_num = 0
        for k, v in self.config.cameras.items():
            self.cameras[MMK2Components(k)] = ImageTypes(v)
        for comp_str in self.config.components:
            comp = MMK2Components(comp_str)
            # TODO: get the type info from SDK
            self.components[comp] = ComponentTypes.UNKNOWN
            names = all_joint_names.__dict__[comp_str]
            self.joint_names[comp] = names
            self.joint_num += len(names)
        logger.info(f"Components: {self.components}")
        logger.info(f"Joint numbers: {self.joint_num}")
        self.robot.enable_resources(
            {
                comp: {
                    "rgb_camera.color_profile": "640,480,30",
                    "enable_depth": "false",
                }
                for comp in self.cameras
            }
        )
        # use stream to get images
        # self.robot.enable_stream(self.robot.get_image, self.cameras)
        if self.config.demonstrate:
            comp_action_topic = {
                comp: TopicNames.tracking.format(component=comp.value)
                for comp in MMK2ComponentsGroup.ARMS
            }
            comp_action_topic.update(
                {
                    comp: TopicNames.controller_command.format(
                        component=comp.value,
                        controller=ControllerTypes.FORWARD_POSITION.value,
                    )
                    for comp in MMK2ComponentsGroup.HEAD_SPINE
                }
            )
            self.robot.listen_to(list(comp_action_topic.values()))
            self._comp_action_topic = comp_action_topic
        self.logs = {}
        self.enter_active_mode = lambda: self._set_mode("active")
        self.enter_passive_mode = lambda: self._set_mode("passive")
        self.get_state_mode = lambda: self._state_mode
        self.exit = lambda: None
        # logger.info("Warm up the robot")
        # for _ in range(5):
        #     self._capture_images()
        # logger.info("AIRBOTMMK2 is ready")
        self.reset()

    def _move_by_traj(self, goal: dict):
        # goal.update(
        #     {
        #         MMK2Components.HEAD: JointState(position=[0, -1.0]),
        #         MMK2Components.SPINE: JointState(position=[0.15]),
        #     }
        # )
        if self.config.demonstrate:
            # TODO: since the arms and eefs are controlled by the teleop bag
            for comp in MMK2ComponentsGroup.ARMS_EEFS:
                goal.pop(comp)
        if goal:
            self.robot.set_goal(goal, TrajectoryParams())
            self.robot.set_goal(goal, ForwardPositionParams())

        return goal

    def reset(self, sleep_time=0):
        if self.config.default_action is not None:
            goal = self._action_to_goal(self.config.default_action)
            # logger.info(f"Reset to default action: {self.config.default_action}")
            # logger.info(f"Reset to default goal: {goal}")
            # TODO: hard code for spine&head control
            self._move_by_traj(goal)
        else:
            logger.warning("No default action is set.")
        time.sleep(sleep_time)
        self.enter_servo_mode()

    def send_action(self, action, wait=False):
        goal = self._action_to_goal(action)
        # logger.info(f"Send action: {action}")
        # logger.info(f"Send goal: {goal}")

        # param = MoveServoParams(header=self.robot.get_header())
        if self.traj_mode:
            self._move_by_traj(goal)
        else:
            param = ForwardPositionParams()
            # param = MoveServoParams(header=self.robot.get_header())
            # param = {
            #     MMK2Components.LEFT_ARM: MoveServoParams(header=self.robot.get_header()),
            #     MMK2Components.RIGHT_ARM: MoveServoParams(header=self.robot.get_header()),
            #     MMK2Components.LEFT_ARM_EEF: TrajectoryParams(),
            #     MMK2Components.RIGHT_ARM_EEF: TrajectoryParams(),
            #     MMK2Components.HEAD: ForwardPositionParams(),
            #     MMK2Components.SPINE: ForwardPositionParams(),
            # }
            self.robot.set_goal(goal, param)

    def get_low_dim_data(self):
        data = {}
        all_joints = self.robot.get_robot_state().joint_state
        # logger.info(f"joint_stamp: {all_joints.header.stamp}")
        for comp in self.components:
            joint_states = self.robot.get_joint_values_by_names(
                all_joints, self.joint_names[comp]
            )
            data[f"observation/{comp.value}/joint_position"] = joint_states
            if self.config.demonstrate:
                if comp in MMK2ComponentsGroup.ARMS:
                    arm_jn = JointNames().__dict__[comp.value]
                    comp_eef = comp.value + "_eef"
                    eef_jn = JointNames().__dict__[comp_eef]
                    js = self.robot.get_listened(self._comp_action_topic[comp])
                    jq = self.robot.get_joint_values_by_names(js, arm_jn + eef_jn)
                    data[f"action/{comp.value}/joint_position"] = jq[:-1]
                    # the eef joint is in arms
                    data[f"action/{comp_eef}/joint_position"] = jq[-1:]
                elif comp in MMK2ComponentsGroup.HEAD_SPINE:
                    jq = list(
                        self.robot.get_listened(self._comp_action_topic[comp]).data
                    )
                    data[f"action/{comp.value}/joint_position"] = jq
        return data

    def _capture_images(self) -> Tuple[Dict[str, bytes], Dict[str, Time]]:
        images = {}
        img_stamps: Dict[MMK2Components, Time] = {}
        before_camread_t = time.perf_counter()
        comp_images = self.robot.get_image(self.cameras)
        for comp, image in comp_images.items():
            # TODO: now only support for color image
            kind = self.cameras[comp]
            if kind == ImageTypes.RGB:
                images[comp.value] = image.color
            elif kind == ImageTypes.DEPTH:
                images[comp.value] = image.depth
            elif kind == ImageTypes.RGBD:
                images[comp.value + "_color"] = image.color
                images[comp.value + "_depth"] = image.depth
            img_stamps[comp.value] = image.stamp

        name = "cameras"
        # self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
        #     "delta_timestamp_s"
        # ]
        self.logs[f"async_read_camera_{name}_dt_s"] = (
            time.perf_counter() - before_camread_t
        )
        return images, img_stamps

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        # Capture images from cameras
        images, img_stamps = self._capture_images()
        low_dim_data = self.get_low_dim_data()

        obs_act_dict = {}
        for comp, stamp in img_stamps.items():
            obs_act_dict[f"/time/{comp}"] = stamp.sec + stamp.nanosec * 1e-9
        # Populate output dictionnaries and format to pytorch
        obs_act_dict["low_dim"] = low_dim_data
        for name in images:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict

    def low_dim_to_action(self, low_dim: dict, step: int) -> list:
        action = []
        # logger.info(low_dim.keys())
        for comp in self.components:
            # action.extend(low_dim[f"action/{comp.value}/joint_position"][step])
            # old version
            if comp in MMK2ComponentsGroup.ARMS_EEFS:
                pos_comp = comp.value.split("_")
                key = f"{pos_comp[1]}/{pos_comp[0]}"
            else:
                key = comp.value
            action.extend(low_dim[f"action/{key}/joint_position"][step])
        return action

    def _set_mode(self, mode):
        self._state_mode = mode

    def _action_check(self, action):
        assert (
            len(action) == self.joint_num
        ), f"Invalid action {action} with length: {len(action)}"

    def _action_to_goal(self, action) -> Dict[MMK2Components, JointState]:
        self._action_check(action)
        goal = {}
        j_cnt = 0
        for comp in self.components:
            end = j_cnt + len(self.joint_names[comp])
            goal[comp] = JointState(position=action[j_cnt:end])
            j_cnt = end
        return goal

    def enter_traj_mode(self):
        self.traj_mode = True

    def enter_servo_mode(self):
        self.traj_mode = False


def main():
    robot = AIRBOTMMK2()
    robot.reset()


if __name__ == "__main__":
    main()
