from mmk2_types.types import (
    JointNames,
    MMK2Components,
    MMK2ComponentsGroup,
)
from mmk2_types.grpc_msgs import Time
from typing import Optional, Dict
import logging
import numpy as np
from robots.airbots.airbot_mmk.airbot_com_mmk2 import AIRBOTMMK2Config
from robots.airbots.airbot_mmk.airbot_com_mmk2 import AIRBOTMMK2 as AIRBOTMMK2_BASE


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIRBOTMMK2(AIRBOTMMK2_BASE):

    def __init__(self, config: Optional[AIRBOTMMK2Config] = None, **kwargs):
        self.images_ts: Dict[str, int] = {}
        super().__init__(config, **kwargs)

    def _get_joint_state(
        self, ns: str, comp: str, stamp: Time, pos=None, vel=None, eff=None
    ) -> dict:
        data = {}
        handle = "joint_state"
        data[f"/{ns}/{comp}/{handle}"] = {
            "t": int((stamp.sec + stamp.nanosec / 1e9) * 1e3),
            "data": {
                "pos": pos,
                "vel": vel,
                "eff": eff,
            },
        }
        return data

    def _get_pose(self, ns: str, comp: str, stamp: Time, t: list, r: list) -> dict:
        data = {}
        handle = "pose"
        data[f"/{ns}/{comp}/{handle}"] = {
            "t": int((stamp.sec + stamp.nanosec / 1e9) * 1e3),
            "data": {
                "t": t,
                "r": r,
            },
        }
        return data

    def _get_image(self, comp: str, stamp: Time, image: np.ndarray) -> dict:
        data = {}
        data[f"/images/{comp}"] = {
            "t": int((stamp.sec + stamp.nanosec / 1e9) * 1e3),
            "data": image,
        }
        if (
            self.images_ts.get(f"/images/{comp}", None) is not None
            and data[f"/images/{comp}"]["t"] <= self.images_ts[f"/images/{comp}"]
        ):
            data[f"/images/{comp}"]["t"] = self.images_ts[f"/images/{comp}"] + 1
        self.images_ts[f"/images/{comp}"] = data[f"/images/{comp}"]["t"]
        return data

    def get_low_dim_data(self):
        data = {}
        robot_state = self.robot.get_robot_state()
        all_joints = robot_state.joint_state
        for comp in self.components:
            stamp = all_joints.header.stamp
            if comp != MMK2Components.BASE:
                # TODO: hand has no joint states
                names = self.joint_names[comp]
                if set(names) - set(all_joints.name):
                    joint_pos = [0.0] * len(names)
                    joint_vel = [0.0] * len(names)
                    joint_eff = [0.0] * len(names)
                else:
                    joint_pos = self.robot.get_joint_values_by_names(
                        all_joints, names, "position"
                    )
                    joint_vel = self.robot.get_joint_values_by_names(
                        all_joints, names, "velocity"
                    )
                    joint_eff = self.robot.get_joint_values_by_names(
                        all_joints, names, "effort"
                    )
                # TODO: configure has-pose components
                if comp in MMK2ComponentsGroup.ARMS:
                    poses = robot_state.robot_pose.robot_pose[comp.value]
                    t = poses.position.x, poses.position.y, poses.position.z
                    r = (
                        poses.orientation.x,
                        poses.orientation.y,
                        poses.orientation.z,
                        poses.orientation.w,
                    )
                    data.update(self._get_pose("observation", comp.value, stamp, t, r))
            else:
                base_pose = robot_state.base_state.pose
                base_vel = robot_state.base_state.velocity
                # TODO: set used states in joint_pos
                joint_pos = [base_pose.x, base_pose.y, base_pose.theta]
                joint_vel = [base_vel.x, base_vel.y, base_vel.omega]
                joint_eff = [0.0, 0.0, 0.0]
                data.update(
                    self._get_joint_state(
                        "action", comp.value, stamp, joint_pos, joint_vel, joint_eff
                    )
                )
            data.update(
                self._get_joint_state(
                    "observation", comp.value, stamp, joint_pos, joint_vel, joint_eff
                )
            )
            if self.config.demonstrate:
                if comp in MMK2ComponentsGroup.ARMS:
                    arm_jn = JointNames().__dict__[comp.value]
                    comp_eef = comp.value + "_eef"
                    eef_jn = JointNames().__dict__[comp_eef]
                    js = self.robot.get_listened(self._comp_action_topic[comp])
                    assert (
                        js is not None
                    ), "The AIRBOT MMK2 should be in bag teleopration mode."
                    jq = self.robot.get_joint_values_by_names(js, arm_jn)
                    data.update(
                        self._get_joint_state(
                            "action", comp.value, js.header.stamp, jq[:len(arm_jn)]
                        )
                    )
                    # data.update(
                    #     self._get_joint_state(
                    #         "action", comp_eef, js.header.stamp, jq[len(arm_jn):]
                    #     )
                    # )
                elif comp in MMK2ComponentsGroup.HEAD_SPINE:
                    result = self.robot.get_listened(self._comp_action_topic[comp])
                    assert (
                        result is not None
                    ), "The AIRBOT MMK2 should be in bag teleopration sync mode."
                    jq = list(result.data)
                    data.update(
                        self._get_joint_state("action", comp.value, result.stamp, jq)
                    )
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        # Capture images from cameras
        images, img_stamps = self._capture_images()
        data = self.get_low_dim_data()
        # logger.info(f"Low dim data: {data.keys()}")
        # from pprint import pprint
        # pprint(data)
        for name, img in images.items():
            data.update(self._get_image(name, img_stamps[name], img))
        # logger.info(f"Images: {images.keys()}")
        return data


def main():
    robot = AIRBOTMMK2()
    robot.reset()


if __name__ == "__main__":
    main()
