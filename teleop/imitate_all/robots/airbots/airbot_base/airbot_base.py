try:
    import airbase_py
except ImportError as e:
    print("Warning: airbase_py not found.")
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AIRBOTBaseConfig(object):
    ip: str = "192.168.11.1"
    velocity: str = "high"


class AIRBOTBase(object):

    def __init__(self, config: Optional[AIRBOTBaseConfig] = None) -> None:
        self.airbase = airbase_py.AirBase(config.ip, config.velocity)
        self.forward = airbase_py.Direction(airbase_py.ACTION_DIRECTION.FORWARD)
        self.backward = airbase_py.Direction(airbase_py.ACTION_DIRECTION.BACKWARD)
        self.turn_left = airbase_py.Direction(airbase_py.ACTION_DIRECTION.TURNLEFT)
        self.turn_right = airbase_py.Direction(airbase_py.ACTION_DIRECTION.TURNRIGHT)

    def lock_change(self):
        self.airbase.set_base_lock_state(not self.airbase.get_base_lock_state())

    def lock(self, lock_state):
        self.airbase.set_base_lock_state(lock_state)

    def get_base_lock_state(self):
        return self.airbase.get_base_lock_state()

    def build_map(self):
        self.airbase.build_stcm_map("map.stcm")

    def load_map(self):
        self.airbase.load_stcm_map("map.stcm")

    def move_to_origin(self):
        self.airbase.move_to_origin()

    def move_by_key(self, key):
        if key == "w":
            self.airbase.platform.move_by(self.forward)
        elif key == "s":
            self.airbase.platform.move_by(self.backward)
        elif key == "a":
            self.airbase.platform.move_by(self.turn_left)
        elif key == "d":
            self.airbase.platform.move_by(self.turn_right)

    def move_at_velocity(self, velocity: Tuple[float, float, float]):
        self.airbase.move_at_velocity(*velocity)

    def move_at_velocity2D(self, velocity: Tuple[float, float]):
        self.airbase.move_at_velocity(velocity[0], 0.0, velocity[1])

    def stop(self):
        self.airbase.stop()

    def record_pose_once(self):
        self.airbase.record_pose_once()

    def save_pose(self, data_dir):
        self.airbase.save_pose(data_dir)

    def get_current_pose(self):
        return self.airbase.get_current_pose()

    def get_current_velocity(self) -> Tuple[float, float, float]:
        return self.airbase.get_current_velocity()

    def get_current_velocity2D(self) -> Tuple[float, float]:
        vel = self.get_current_velocity()
        return vel[0], vel[2]

    def init_behavior(self, track, data_path):
        return self.airbase.init_behavior(track, data_path)

    def replay_pose_once(self, index, track=True, wait=True):
        self.airbase.replay_pose_once(index, track, wait)

    def move_to_pose(self, x, y, angle, wait=True, backward=False):
        self.airbase.move_to_pose(x, y, angle, wait, backward)

    def move_to_position(self, x, y, wait=True, backward=False):
        self.airbase.move_to_position(x, y, wait, backward)

    def move_by_angle(self, angle, wait=True):
        self.airbase.move_by_angle(angle, wait)

    def move_by_line(self, x, y, wait=True, backward=False):
        self.airbase.move_by_line(x, y, wait, backward)

    def move_by_points(self, points, wait=True, backward=False):
        self.airbase.move_by_points(points, wait, backward)


if __name__ == "__main__":

    import time

    airbase = AIRBOTBase()
    for _ in range(10):
        airbase.move_at_velocity((0.0, 0.0, -0.5))
        time.sleep(0.5)

    print(airbase.get_current_velocity())
    airbase.stop()
