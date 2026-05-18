from dataclasses import dataclass, field, replace
import time
from habitats.common.robot_devices.cameras.utils import Camera
from typing import Dict, Optional, List
from airbot_python_sdk.airbot_client import Robot


@dataclass
class AIRBOTPlayConfig(object):

    leader_number: int = 1
    follower_number: int = 1
    leader_arm_type: List[str] = field(default_factory=lambda: ["play_long"])
    follower_arm_type: List[str] = field(default_factory=lambda: ["play_long"])
    leader_end_effector: List[str] = field(default_factory=lambda: ["E2B"])
    follower_end_effector: List[str] = field(default_factory=lambda: ["G2"])
    leader_can_interface: List[str] = field(default_factory=lambda: ["can0"])
    follower_can_interface: List[str] = field(default_factory=lambda: ["can1"])
    leader_domain_id: List[int] = field(default_factory=lambda: [90])
    follower_domain_id: List[int] = field(default_factory=lambda: [88])
    start_arm_joint_position: List[List[float]] = field(
        default_factory=lambda: [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    start_eef_joint_position: List[float] = field(default_factory=lambda: [0.0])

    cameras: Dict[str, Camera] = field(default_factory=lambda: {})

    def __post_init__(self):
        assert len(self.leader_arm_type) == self.leader_number
        assert len(self.follower_arm_type) == self.follower_number
        assert len(self.leader_end_effector) == self.leader_number
        assert len(self.follower_end_effector) == self.follower_number
        assert len(self.leader_can_interface) == self.leader_number
        assert len(self.follower_can_interface) == self.follower_number
        assert len(self.leader_domain_id) == self.leader_number
        assert len(self.follower_domain_id) == self.follower_number


class AIRBOTPlay(object):
    def __init__(self, config: Optional[AIRBOTPlayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTPlayConfig()
        # Overwrite config arguments using kwargs (used for yaml config)
        self.config = replace(config, **kwargs)
        self.cameras = self.config.cameras
        self.logs = {}
        self.__init()

    def __init(self):
        args = self.config
        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()
        # Connect the robot
        leader_robot = []
        follower_robot = []
        for i in range(args.leader_number):
            leader_robot.append(
                Robot(
                    arm_type=args.leader_arm_type[i],
                    end_effector=args.leader_end_effector[i],
                    can_interface=args.leader_can_interface[i],
                    domain_id=args.leader_domain_id[i],
                )
            )
        for i in range(args.follower_number):
            follower_robot.append(
                Robot(
                    arm_type=args.follower_arm_type[i],
                    end_effector=args.follower_end_effector[i],
                    can_interface=args.follower_can_interface[i],
                    domain_id=args.follower_domain_id[i],
                )
            )
        self.leader_robot = leader_robot
        self.follower_robot = follower_robot
        time.sleep(0.3)
        self.reset()

    def reset(self):
        args = self.config
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        wait_time = 0.3
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            if leader_robot[i].get_current_state() != "ONLINE_TRAJ":
                assert leader_robot[i].online_idle_mode(), (
                    "Leader robot %d online idle mode failed" % i
                )
                assert leader_robot[i].online_traj_mode(), (
                    "Leader robot %d online traj mode failed" % i
                )
            time.sleep(wait_time)
        for i in range(args.follower_number):
            if follower_robot[i].get_current_state() != "SLAVE_MOVING":
                assert follower_robot[i].online_idle_mode(), (
                    "Follower robot %d online idle mode failed" % i
                )
                assert follower_robot[i].slave_waiting_mode(args.leader_domain_id[i]), (
                    "Follower robot %d slave waiting mode failed" % i
                )
                time.sleep(wait_time)
                assert follower_robot[i].slave_reaching_mode(), (
                    "Follower robot %d slave reaching mode failed" % i
                )
                while follower_robot[i].get_current_state() != "SLAVE_REACHED":
                    time.sleep(wait_time)
                assert follower_robot[i].slave_moving_mode(), (
                    "Follower robot %d slave moving mode failed" % i
                )
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            assert leader_robot[i].set_target_joint_q(args.start_arm_joint_position[i]), (
                "Leader robot %d set target joint q failed" % i
            )
            if args.leader_end_effector[i] not in ["E2B", "none"]:
                assert leader_robot[i].set_target_end(args.start_eef_joint_position[i]), (
                    "Leader robot %d set target end failed" % i
                )
            assert leader_robot[i].online_idle_mode(), (
                "Leader robot %d online idle mode failed" % i
            )
        self._state_mode = "active"

    def enter_active_mode(self):
        args = self.config
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            assert self.leader_robot[i].online_idle_mode(), (
                "Leader robot %d online idle mode failed" % i
            )
        self._state_mode = "active"

    def enter_passive_mode(self):
        args = self.config
        for i in range(args.leader_number):
            if args.leader_arm_type[i] == "replay":
                continue
            assert self.leader_robot[i].demonstrate_prep_mode(), (
                "Leader robot %d demonstrate start mode failed" % i
            )
        self._state_mode = "passive"

    def clear_boundary_error(self):
        self.enter_passive_mode()

    def get_low_dim_data(self):
        args = self.config
        leader_robot = self.leader_robot
        follower_robot = self.follower_robot
        data = {}
        data["/time"] = time.time()

        action_arm_jq = []
        action_eef_jq = []
        action_eef_pose = []
        for i in range(args.leader_number):
            action_arm_jq.extend(leader_robot[i].get_current_joint_q())
            action_eef_jq.append(leader_robot[i].get_current_end())
            pose = leader_robot[i].get_current_pose()
            action_eef_pose.extend(pose[0] + pose[1])  # xyz + quat(xyzw)
        data["action/arm/joint_position"] = action_arm_jq
        data["action/eef/joint_position"] = action_eef_jq
        data["action/eef/pose"] = action_eef_pose

        obs_arm_jq = []
        obs_eef_jq = []
        obs_eef_pose = []
        for i in range(args.follower_number):
            obs_arm_jq.extend(follower_robot[i].get_current_joint_q())
            obs_eef_jq.append(follower_robot[i].get_current_end())
            pose = follower_robot[i].get_current_pose()
            obs_eef_pose.extend(pose[0] + pose[1])  # xyz + quat(xyzw)
        data["observation/arm/joint_position"] = obs_arm_jq
        data["observation/eef/joint_position"] = obs_eef_jq
        data["observation/eef/pose"] = obs_eef_pose

        return data

    def capture_observation(self):
        """The returned observations do not have a batch dimension."""
        # if not self.is_connected:
        #     raise RobotDeviceNotConnectedError(
        #         "ManipulatorRobot is not connected. You need to run `robot.connect()`."
        #     )
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

        # Populate output dictionnaries
        obs_act_dict["low_dim"] = low_dim_data
        for name in self.cameras:
            obs_act_dict[f"observation.images.{name}"] = images[name]
        return obs_act_dict

    def exit(self):
        try:
            for name in self.cameras:
                self.cameras[name].disconnect()
        except Exception as e:
            pass
        print("Robot exited")

    def get_state_mode(self):
        return self._state_mode
