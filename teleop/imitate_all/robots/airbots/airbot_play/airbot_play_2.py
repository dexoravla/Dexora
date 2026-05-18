import airbot
from robots.common_robot import Configer
from typing import Optional, List
from dataclasses import dataclass, replace


@dataclass
class AIRBOTPlayConfig(object):
    model_path: str = "/usr/share/airbot_models/airbot_play_with_gripper.urdf"
    gravity_mode: str = "down"
    can_bus: str = "can0"
    vel: float = 2.0
    eef_mode: str = "none"
    bigarm_type: str = "OD"
    forearm_type: str = "DM"
    # other
    joint_vel: float = 6.0
    default_action: Optional[List[float]] = None


class AIRBOTPlay(object):
    def __init__(self, config: Optional[AIRBOTPlayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTPlayConfig()
        # Overwrite config arguments using kwargs (used for yaml config)
        self.config = replace(config, **kwargs)
        cfg = Configer.config2tuple(self.config)[:7]
        print("cfg", cfg)
        self.robot = airbot.create_agent(*cfg)
        self._arm_joints_num = 6
        self._joints_num = 7
        self.end_effector_open = 1
        self.end_effector_close = 0

    def _set_eef(self, target, ctrl_type):
        if len(target) == 1:
            target = target[0]
        if ctrl_type == "pos":
            self.robot.set_target_end(target)
        elif ctrl_type == "vel":
            self.robot.set_target_end_v(target)
        elif ctrl_type == "eff":
            self.robot.set_target_end_t(target)
        else:
            raise ValueError(f"Invalid type: {ctrl_type}")

    def get_current_joint_positions(self) -> List[float]:
        joints = self.robot.get_current_joint_q()
        if self.config.eef_mode in ["gripper", "teacherv2", "encoder"]:
            joints += [self.robot.get_current_end()]
        return joints

    def get_current_joint_velocities(self):
        joints = self.robot.get_current_joint_v()
        if self.config.eef_mode in ["gripper", "teacherv2"]:
            joints += [self.robot.get_current_end_v()]
        return joints

    def get_current_joint_efforts(self):
        joints = self.robot.get_current_joint_t()
        if self.config.eef_mode in ["gripper", "teacherv2"]:
            joints += [self.robot.get_current_end_t()]
        return joints

    def get_current_pose(self):
        return self.robot.get_current_pose()

    def set_joint_position_target(
        self, qpos, qvel=None, blocking=False, use_planning=None
    ):
        if qvel is None:
            qvel = [self.config.joint_vel]
        use_planning = blocking if use_planning is None else use_planning
        self.robot.set_target_joint_q(
            qpos[: self._arm_joints_num], use_planning, qvel[0], blocking
        )
        if len(qpos) - self._arm_joints_num > 0:
            self._set_eef(qpos[self._arm_joints_num :], "pos")

    def set_joint_velocity_target(self, qvel, blocking=False):
        self.robot.set_target_joint_v(qvel[: self._arm_joints_num])
        if len(qvel) - self._arm_joints_num > 0:
            self._set_eef(qvel[self._arm_joints_num :], "vel")

    def set_joint_effort_target(self, qeffort, blocking=False):
        self.robot.set_target_joint_t(qeffort[: self._arm_joints_num])
        if len(qeffort) - self._arm_joints_num > 0:
            self._set_eef(qeffort[self._arm_joints_num :], "eff")

    def enter_passive_mode(self):
        self.robot.manual_mode()

    def enter_active_mode(self) -> bool:
        if (
            self.config.bigarm_type == "encoder"
            or self.config.forearm_type == "encoder"
        ):
            return False
        else:
            self.robot.online_mode()
            return True
