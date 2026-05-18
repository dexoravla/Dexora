from typing import Protocol


# class Robot(Protocol):
#     def init_teleop(self): ...
#     def run_calibration(self): ...
#     def teleop_step(self, record_data=False): ...
#     def capture_observation(self): ...
#     def send_action(self, action): ...

"""A robot is a physical instance that has its proprioception state and can interact with the environment by subjective initiative actions. The robot's state is the information that can be obtained from the robot's body sensors while the actions are the control commands that can be sent to the robot's actuators. Vision, touch and the other external sensation (obtained by tactile sensors, cameras, lidar, radar, ultrasonic sensors, etc.) are not included in the robot's state, but in the environment. However, in addition to being related to the external environment, external observation also depends on the robot's state and the position and posture of the corresponding sensors. So the robot instance should have the full information and configurations of its external sensors to let the environment obtaining correct observations."""


class Robot(Protocol):
    """Assume the __init__ method of the robot class is the same as the reset method.
    So you can inherit this class to save writing the __init__ function."""

    def __init__(self, config) -> None:
        self.reset(config)

    def reset(self, config):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def exit(self):
        raise NotImplementedError

    @property
    def state(self):
        raise NotImplementedError
