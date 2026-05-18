try:
    import rospy
    from sensor_msgs.msg import JointState
except Exception as e:
    print(e)
    print("ROS not installed. This is expected if you are running this code on your local machine.")
    print("This code is meant to be run on the robot.")
    from typing import Any

    rospy = Any
    JointState = Any
from typing import Optional, List
from dataclasses import dataclass, replace


@dataclass
class AIRBOTReplayConfig(object):
    joint_states_topic: str = "/remote_states/arms/{pos}/joint_states"


class AIRBOTReplay(object):
    def __init__(self, config: Optional[AIRBOTReplayConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTReplayConfig()
        # Overwrite config arguments using kwargs (used for yaml config)
        self.config = replace(config, **kwargs)
        self._joint_states = None
        if rospy.get_name() == "/unnamed":
            rospy.loginfo("Initializing airbot replay remote node")
            rospy.init_node("airbot_replay_remote", anonymous=True)
        rospy.loginfo("Waiting for joint states topic")
        rospy.wait_for_message(self.config.joint_states_topic, JointState, timeout=1)
        self._joint_states_sub = rospy.Subscriber(
            self.config.joint_states_topic, JointState, self._joint_states_callback
        )

    def _joint_states_callback(self, msg):
        self._joint_states = msg

    def get_current_joint_positions(self) -> List[float]:
        if self._joint_states is None:
            return [0] * 7
        return self._joint_states.position

    def get_current_joint_velocities(self):
        if self._joint_states is None:
            return [0] * 7
        return self._joint_states.velocity

    def get_current_joint_efforts(self):
        if self._joint_states is None:
            return [0] * 7
        return self._joint_states.effort

    def set_joint_position_target(
        self, qpos, qvel=None, blocking=False, use_planning=None
    ):
        pass

    def set_joint_velocity_target(self, qvel, blocking=False):
        pass

    def set_joint_effort_target(self, qeffort, blocking=False):
        pass

    def get_current_pose(self):
        raise NotImplementedError

    def enter_passive_mode(self):
        pass

    def enter_active_mode(self) -> bool:
        pass
