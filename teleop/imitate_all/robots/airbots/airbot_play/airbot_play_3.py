from dataclasses import dataclass, field, replace
import time
from habitats.common.robot_devices.cameras.utils import Camera
from typing import Dict, Optional, List
from airbot_python_sdk.airbot_client import Robot


@dataclass
class AIRBOTPlayConfig(object):
    arm_type: str = "play_long"
    end_effector: str = "E2B"
    can_interface: str = "can0"
    domain_id: int = 77
    start_arm_joint_position: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    start_eef_joint_position: float = 0.0
    # ONLINE_TRAJ, ONLINE_IDLE, ONLINE_SERVO, DEMONSTRATE_PREP
    default_robot_mode: str = "ONLINE_IDLE"
    cameras: dict = field(default_factory=lambda: {})
    display: bool = False

    def __post_init__(self):
        assert self.default_robot_mode in [
            "ONLINE_TRAJ",
            "ONLINE_IDLE",
            "ONLINE_SERVO",
            "DEMONSTRATE_PREP",
        ]


class AIRBOTPlay(object):
    def __init__(self, config: Optional[AIRBOTPlayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTPlayConfig()
        self.config = replace(config, **kwargs)
        self.cameras: Dict[str, Camera] = self.config.cameras
        self.logs = {}
        self.__init()
        self._state_mode = "active"
        self._exited = False

    def __init(self):
        args = self.config
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
        # Connect the robot
        self.robot = Robot(
            arm_type=args.arm_type,
            end_effector=args.end_effector,
            can_interface=args.can_interface,
            domain_id=args.domain_id,
        )
        time.sleep(0.3)
        self.reset()

    def reset(self):
        args = self.config
        robot = self.robot
        # set to traj mode
        if args.arm_type != "replay" and robot.get_current_state() != "ONLINE_TRAJ":
            assert robot.online_idle_mode(), "online idle mode failed"
            assert robot.online_traj_mode(), "online traj mode failed"
        time.sleep(0.3)
        # go to start position
        if args.arm_type != "replay":
            if args.start_arm_joint_position is not None:
                assert robot.set_target_joint_q(
                    args.start_arm_joint_position, wait=True
                ), "set target joint q failed"
            if args.start_eef_joint_position is not None:
                assert robot.set_target_end(
                    args.start_eef_joint_position, wait=False
                ), "set target end failed"
            # enter default mode
            if args.default_robot_mode == "ONLINE_TRAJ":
                self.enter_traj_mode()
            elif args.default_robot_mode == "ONLINE_IDLE":
                self.enter_active_mode()
            elif args.default_robot_mode == "ONLINE_SERVO":
                self.enter_servo_mode()
            else:
                raise ValueError(
                    f"Invalid default robot mode: {args.default_robot_mode}"
                )

        self._state_mode = "active"

    def enter_traj_mode(self):
        self.enter_active_mode()
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.online_traj_mode(), "online traj mode failed"
        time.sleep(0.5)
        self._state_mode = "active"

    def enter_active_mode(self):
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.online_idle_mode(), "online idle mode failed"
        self._state_mode = "active"

    def enter_passive_mode(self):
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.demonstrate_prep_mode(), "demonstrate start mode failed"
        self._state_mode = "passive"

    def enter_servo_mode(self):
        self.enter_active_mode()
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.online_servo_mode(), "online_servo_mode mode failed"
        self._state_mode = "active"

    def send_action(self, action, wait=False):
        assert self._state_mode == "active", "Robot is not in active mode"
        if self.config.arm_type == "replay":
            return
        else:
            assert self.robot.set_target_joint_q(
                action[:6], wait
            ), "set target joint q failed"
            if self.config.end_effector not in ["none", "E2B"]:
                assert self.robot.set_target_end(
                    action[6], wait
                ), "set target end failed"
        return action

    def get_low_dim_data(self):
        data = {}
        data["/time"] = time.time()
        pose = self.robot.get_current_pose()
        data["observation/arm/joint_position"] = list(self.robot.get_current_joint_q())
        data["observation/eef/joint_position"] = [self.robot.get_current_end()]
        data["observation/eef/pose"] = pose[0] + pose[1]  # xyz + quat(xyzw)
        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        obs_act_dict = {}
        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            # images[name] = torch.from_numpy(images[name])
            obs_act_dict[f"/time/{name}"] = time.time()
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs[
                "delta_timestamp_s"
            ]
            self.logs[f"async_read_camera_{name}_dt_s"] = (
                time.perf_counter() - before_camread_t
            )

        low_dim_data = self.get_low_dim_data()

        # Populate output dictionnaries and format to pytorch
        obs_act_dict["low_dim"] = low_dim_data
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict

    def exit(self):
        assert not self._exited, "Robot already exited"
        for name in self.cameras:
            self.cameras[name].disconnect()
        self._exited = True
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode
