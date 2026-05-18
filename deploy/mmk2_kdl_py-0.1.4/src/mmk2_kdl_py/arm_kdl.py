"""Module for Arm kinematics and dynamics."""

# Standard imports
import os
from pathlib import Path
from dataclasses import dataclass

# Third-party imports
import numpy as np
import typing

# from xacrodoc import XacroDoc
from .utils import timed

# HAS_PINOCCHIO = True
# try:
#     import pinocchio as pin
# except ImportError:
#     HAS_PINOCCHIO = False


def _cos(q: float) -> float:
    """
    Calculate the cosine of an angle in radians.

    Args:
        q (float): angle in radians

    Returns:
        float: cosine of the angle
    """
    while q < -np.pi:
        q += 2 * np.pi
    while q > np.pi:
        q -= 2 * np.pi
    if q < 0:
        return _cos(-q)
    if q > np.pi / 2:
        return -_cos(np.pi - q)
    if abs(q) < 1e-6:
        return 1
    if abs(q - np.pi / 2) < 1e-6:
        return 0
    return np.cos(q)


def _sin(q: float) -> float:
    """
    Calculate the sine of an angle in radians.

    Args:
        q (float): angle in radians

    Returns:
        float: sine of the angle
    """
    while q < -np.pi:
        q += 2 * np.pi
    while q > np.pi:
        q -= 2 * np.pi
    if q < 0:
        return -_sin(-q)
    if q > np.pi / 2:
        return _sin(np.pi - q)
    if abs(q) < 1e-6:
        return 0
    if abs(q - np.pi / 2) < 1e-6:
        return 1
    return np.sin(q)


@dataclass
class ArmDH:
    """Arm Denavit-Hartenberg parameters class"""

    alpha: np.ndarray  # alpha[i]
    a: np.ndarray  # a[i]
    d: np.ndarray  # d[i+1]
    theta_offset: np.ndarray  # theta_offset[i+1]
    end_convert_matrix: np.ndarray
    joints_limit: np.ndarray

    def adjacent_transform(self, q: float, index: int) -> np.ndarray:
        """
        Get the transformation matrix of the i-th joint. (1-7)

        Args:
            q (float): joint angle
            index (int): joint index

        Returns:
            np.ndarray: transformation matrix
        """
        # fmt: off
        # pylint: disable=line-too-long
        res =  np.array([
            [                            _cos(q + self.theta_offset[index-1]),                            -_sin(q + self.theta_offset[index-1]),                          0,                              self.a[index-1]],
            [_sin(q + self.theta_offset[index-1]) * _cos(self.alpha[index-1]), _cos(q + self.theta_offset[index-1]) * _cos(self.alpha[index-1]), -_sin(self.alpha[index-1]), -self.d[index-1] * _sin(self.alpha[index-1])],
            [_sin(q + self.theta_offset[index-1]) * _sin(self.alpha[index-1]), _cos(q + self.theta_offset[index-1]) * _sin(self.alpha[index-1]),  _cos(self.alpha[index-1]),  self.d[index-1] * _cos(self.alpha[index-1])],
            [                                                               0,                                                                0,                          0,                                            1]
        ])
        # pylint: enable=line-too-long
        # fmt: on
        # print("adjacent_transform", index)
        # print(res)
        return res

    def adjacent_rotation(self, q: float, index: int) -> np.ndarray:
        """
        Get the rotation matrix of the i-th joint. (1-7)

        Args:
            q (float): joint angle
            index (int): joint index

        Returns:
            np.ndarray: rotation matrix
        """
        # fmt: off
        # pylint: disable=line-too-long
        res = np.array([
            [                            _cos(q + self.theta_offset[index-1]),                            -_sin(q + self.theta_offset[index-1]),                          0],
            [_sin(q + self.theta_offset[index-1]) * _cos(self.alpha[index-1]), _cos(q + self.theta_offset[index-1]) * _cos(self.alpha[index-1]), -_sin(self.alpha[index-1])],
            [_sin(q + self.theta_offset[index-1]) * _sin(self.alpha[index-1]), _cos(q + self.theta_offset[index-1]) * _sin(self.alpha[index-1]),  _cos(self.alpha[index-1])],
        ])
        # pylint: enable=line-too-long
        # fmt: on
        return res

    def valid_joints(self, q: np.ndarray) -> bool:
        """
        Check if the joint angles are within the limits.

        Args:
            q (np.ndarray): joint angles

        Returns:
            bool: True if valid, False otherwise
        """
        for i in range(len(q)):
            if q[i] < self.joints_limit[i][0] or q[i] > self.joints_limit[i][1]:
                return False
        return True

    def limit_joints(
        self, q: np.ndarray, force_calculate: bool = False
    ) -> typing.Optional[np.ndarray]:
        """
        Limit the joint angles to the limits.

        Args:
            q (np.ndarray): joint angles
            force_calculate (bool): whether to force calculation even if limits are exceeded

        Returns:
            np.ndarray: limited joint angles
        """
        for i in range(len(q)):
            if q[i] >= self.joints_limit[i][0] and q[i] <= self.joints_limit[i][1]:
                continue
            if q[i] < self.joints_limit[i][0]:
                t1 = np.floor((self.joints_limit[i][0] - q[i]) / (2 * np.pi))
                q_t1 = t1 * 2 * np.pi + q[i]
                t2 = np.ceil((self.joints_limit[i][0] - q[i]) / (2 * np.pi))
                q_t2 = t2 * 2 * np.pi + q[i]
                if q_t2 <= self.joints_limit[i][1]:
                    q[i] = q_t2
                else:
                    if force_calculate:
                        q[i] = (
                            q_t1
                            if (self.joints_limit[i][0] - q_t1)
                            < (q_t2 - self.joints_limit[i][1])
                            else q_t2
                        )
                    else:
                        return None
            elif q[i] > self.joints_limit[i][1]:
                t1 = np.floor((q[i] - self.joints_limit[i][1]) / (2 * np.pi))
                q_t1 = q[i] - t1 * 2 * np.pi
                t2 = np.ceil((q[i] - self.joints_limit[i][1]) / (2 * np.pi))
                q_t2 = q[i] - t2 * 2 * np.pi
                if q_t2 >= self.joints_limit[i][0]:
                    q[i] = q_t2
                else:
                    if force_calculate:
                        q[i] = (
                            q_t1
                            if (q_t1 - self.joints_limit[i][1])
                            < (self.joints_limit[i][0] - q_t2)
                            else q_t2
                        )
                    else:
                        return None
        return q

    def _cal_limit_bias(self, q: np.ndarray) -> float:
        """
        Calculate the limit bias of the joint angles.

        Args:
            q (np.ndarray): joint angles

        Returns:
            float: limit bias
        """
        bias = 0.0
        for i in range(len(q)):
            if q[i] < self.joints_limit[i][0]:
                bias += self.joints_limit[i][0] - q[i]
            elif q[i] > self.joints_limit[i][1]:
                bias += q[i] - self.joints_limit[i][1]
        return bias

    def _clip_joints(self, q: np.ndarray) -> np.ndarray:
        """
        Clip the joint angles to the limits.

        Args:
            q (np.ndarray): joint angles

        Returns:
            np.ndarray: clipped joint angles
        """
        for i in range(len(q)):
            if q[i] < self.joints_limit[i][0]:
                q[i] = self.joints_limit[i][0]
            elif q[i] > self.joints_limit[i][1]:
                q[i] = self.joints_limit[i][1]
        return q


# fmt: off
# pylint: disable=line-too-long

DEFAULT_ALPHA = np.array([0, np.pi / 2, 0, -np.pi / 2, np.pi / 2, -np.pi / 2, 0])

DEFAULT_THETA_OFFSET = np.array([0, np.pi - 21.93 / 180 * np.pi, np.pi / 2 + 21.93 / 180 * np.pi, -np.pi / 2, 0, 0, 0])

DEFAULT_END_CONVERT_MATRIX = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1]
])

DEFAULT_JOINTS_LIMIT_PLAY = np.array([
    [-3.151, 2.080],  # joint 1
    [-2.963, 0.181],  # joint 2
    [-0.094, 3.161],  # joint 3
    [-3.012, 3.012],  # joint 4
    [-1.859, 1.859],  # joint 5
    [-3.017, 3.017],  # joint 6
])

DEFAULT_DH_MAP = {
    "play_short": {
        "none": ArmDH(
            alpha        = DEFAULT_ALPHA,
            a            = np.array([     0, 0, 0.270092,        0, 0, 0,         0]),
            d            = np.array([0.1127, 0,        0, 0.290149, 0, 0, 0.0864995]),
            theta_offset = DEFAULT_THETA_OFFSET,
            end_convert_matrix = DEFAULT_END_CONVERT_MATRIX,
            joints_limit = DEFAULT_JOINTS_LIMIT_PLAY,
        ),
        "G2": ArmDH(
            alpha        = DEFAULT_ALPHA,
            a            = np.array([     0, 0, 0.270092,        0, 0, 0,         0]),
            d            = np.array([0.1127, 0,        0, 0.290149, 0, 0, 0.2466995]),
            theta_offset = DEFAULT_THETA_OFFSET,
            end_convert_matrix = DEFAULT_END_CONVERT_MATRIX,
            joints_limit = DEFAULT_JOINTS_LIMIT_PLAY,
        ),
        "E2B": ArmDH(
            alpha        = DEFAULT_ALPHA,
            a            = np.array([     0, 0, 0.270092,        0, 0, 0,         0]),
            d            = np.array([0.1127, 0,        0, 0.290149, 0, 0, 0.1488995]),
            theta_offset = DEFAULT_THETA_OFFSET,
            end_convert_matrix = DEFAULT_END_CONVERT_MATRIX,
            joints_limit = DEFAULT_JOINTS_LIMIT_PLAY,
        ),
        "INSFTP56L": ArmDH(
            alpha        = DEFAULT_ALPHA,
            a            = np.array([     0, 0, 0.270092,        0, 0, 0, 0]),
            d            = np.array([0.1127, 0,        0, 0.290149, 0, 0, 0]),
            theta_offset = DEFAULT_THETA_OFFSET,
            end_convert_matrix = np.array([
                [ 0.8414710, -0.5403023,  0,      0.08],
                [         0,          0, -1, -0.038225],
                [ 0.5403023,  0.8414710,  0, 0.2814995],
                [         0,          0,  0,         1]
            ]),
            joints_limit = DEFAULT_JOINTS_LIMIT_PLAY,
        ),
        "INSFTP56R": ArmDH(
            alpha        = DEFAULT_ALPHA,
            a            = np.array([     0, 0, 0.270092,        0, 0, 0, 0]),
            d            = np.array([0.1127, 0,        0, 0.290149, 0, 0, 0]),
            theta_offset = DEFAULT_THETA_OFFSET,
            end_convert_matrix = np.array([
                [ -0.8414710, -0.5403023,  0,     -0.08],
                [          0,          0, -1, -0.038225],
                [  0.5403023, -0.8414710,  0, 0.2814995],
                [          0,          0,  0,         1]
            ]),
            joints_limit = DEFAULT_JOINTS_LIMIT_PLAY,
        ),
    },

}
# pylint: enable=line-too-long
# fmt: on


class ArmKdl:
    """Arm kinematics and dynamics class"""

    def __init__(
        self,
        arm_type: str = "play_short",
        eef_type: str = "G2",
        punish: typing.Optional[typing.List[float]] = None,
    ):
        if eef_type == "old_G2":
            eef_type = "G2"
        self.arm_type = arm_type
        self.eef_type = eef_type
        self.dh = DEFAULT_DH_MAP[arm_type][eef_type]
        self.punish = punish if punish is not None else [1.0] * 6

        # if HAS_PINOCCHIO:
        # doc = XacroDoc.from_file(
        #         os.path.join(
        #             str(Path(__file__).resolve().parent),
        #             "./models/urdf/airbot_" + arm_type + ".xacro",
        #         ),
        #         subargs={"eef_type": eef_type, "disable_ros2_control": "true"},
        #     )

        #     urdf_str = doc.to_urdf_string()

        #     self._model = pin.buildModelFromXML(urdf_str)
        #     print(self._model)
        #     self._data = self._model.createData()
        #     self.size = self._model.nv

    def _cal_punish(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate the punishment of the joint angles"""
        assert len(q1) == len(q2), "q1 and q2 should have the same length"
        assert len(self.punish) == len(q1), "punish should have the same length as q1"
        return np.sum(self.punish * np.abs(q1 - q2))

    def update_punish(self, punish: typing.List[float]) -> None:
        """
        Update the punishment of the joint angles.

        Args:
            punish (typing.List[float]): new punishment values for each joint
        """
        assert len(punish) == 6, "punish should have 6 elements"
        self.punish = punish

    # def inverse_dynamics(
    #     self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    # ) -> np.ndarray:
    #     """
    #     Calculate the inverse dynamics of the arm.

    #     Args:
    #         q (np.ndarray): current joint positions
    #         v (np.ndarray): current joint velocities
    #         a (np.ndarray): current joint accelerations

    #     Returns:
    #         np.ndarray: joint torques
    #     """
    #     if not HAS_PINOCCHIO:
    #         raise ImportError(
    #             "Pinocchio is not installed. Please install it to use this method."
    #         )
    #     if len(q) < self.size:
    #         q = np.append(q, pin.utils.zero(self.size - len(q)))
    #     if len(v) < self.size:
    #         v = np.append(v, pin.utils.zero(self.size - len(v)))
    #     if len(a) < self.size:
    #         a = np.append(a, pin.utils.zero(self.size - len(a)))
    #     return pin.rnea(self._model, self._data, q, v, a)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian matrix of the arm.

        Args:
            q (np.ndarray): current joint positions (6 elements)

        Returns:
            np.ndarray: 6x6 Jacobian matrix [J_v; J_w] where J_v is linear velocity
                    and J_w is angular velocity
        """
        assert len(q) == 6, "q must be 6 elements"

        J = np.zeros((6, 7))
        T = [np.eye(4)]  # T[0] is base frame

        # Calculate T0_1 to T0_6
        for i in range(6):
            T_i = np.dot(T[-1], self.dh.adjacent_transform(q[i], i + 1))
            T.append(T_i)

        T_7 = np.dot(T[-1], self.dh.adjacent_transform(0, 7))
        T_7 = np.dot(T_7, self.dh.end_convert_matrix)
        T.append(T_7)

        o_n = T_7[0:3, 3]

        # Calculate Jacobian columns for each joint
        for i in range(7):
            z_i = T[i + 1][0:3, 2]  # z-axis
            o_i = T[i + 1][0:3, 3]  # origin position

            # Linear velocity part: J_v = z_i × (o_n - o_i)
            J[0:3, i] = np.cross(z_i, o_n - o_i)

            # Angular velocity part: J_w = z_i
            J[3:6, i] = z_i

        return np.round(J[:, :6], decimals=9)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate the forward kinematics of the arm.

        Args:
            q (np.ndarray): current joint positions

        Returns:
            np.ndarray: end effector position
        """
        assert len(q) == 6, "q must be 6 elements"

        res = np.eye(4)
        for i in range(6):
            res = np.dot(res, self.dh.adjacent_transform(q[i], i + 1))
            # print(i+1)
            # print(res)
        res = np.dot(res, self.dh.adjacent_transform(0, 7))
        # print(7)
        # print(res)
        res = np.dot(res, self.dh.end_convert_matrix)
        # print("res")
        return np.round(res, decimals=9)

    # @timed
    def inverse_kinematics(
        self,
        pose: np.ndarray,
        ref_pos: typing.Optional[np.ndarray] = None,
        force_calculate: bool = False,
        use_clip: bool = False,
    ) -> typing.Optional[np.ndarray]:
        """
        Calculate the inverse kinematics of the arm.

        Args:
            pose (np.ndarray): end effector pose
            ref_pos (np.ndarray): reference position
            force_calculate (bool): whether to force the calculation
            use_clip (bool): whether to clip the joint angles to the limits (only if force_calculate is True)

        Returns:
            Optional[np.ndarray]: joint angles or None if not solvable
        """
        assert len(pose) == 4 and len(pose[0]) == 4, "pose must be 4x4 matrix"
        if ref_pos is not None:
            assert len(ref_pos) == 6, "ref_pos must be 6 elements"

        res = []

        arm_pose = self._cut_end(pose)

        # print("arm_pose")
        # print(arm_pose)

        wrist_pos = arm_pose[0:3, 3]
        s3 = -(
            wrist_pos[0] ** 2
            + wrist_pos[1] ** 2
            + (wrist_pos[2] - self.dh.d[0]) ** 2
            - self.dh.d[3] ** 2
            - self.dh.a[2] ** 2
        ) / (2 * self.dh.d[3] * self.dh.a[2])

        if s3 < -1 or s3 > 1:
            if force_calculate:
                # If s3 is out of range, return the closest solution
                s3 = np.clip(s3, -1, 1)
            else:
                return []

        for i1 in [1, -1]:
            if abs(wrist_pos[0]) < 1e-3 and abs(wrist_pos[1]) < 1e-3:
                theta1 = ref_pos[0] if ref_pos is not None else 0.0
            else:
                theta1 = (
                    np.arctan2(i1 * wrist_pos[1], i1 * wrist_pos[0])
                    - self.dh.theta_offset[0]
                )
            for i2 in [1, -1]:
                c3 = i2 * np.sqrt(1 - s3**2)
                theta3 = np.arctan2(s3, c3) - self.dh.theta_offset[2]

                k1 = self.dh.a[2] - self.dh.d[3] * s3
                k2 = self.dh.d[3] * c3
                k3 = np.sqrt(wrist_pos[0] ** 2 + wrist_pos[1] ** 2)
                k4 = wrist_pos[2] - self.dh.d[0]

                theta2 = (
                    np.arctan2(-i1 * k2 * k3 + k1 * k4, i1 * k1 * k3 + k2 * k4)
                    - self.dh.theta_offset[1]
                )

                t_3_0 = np.dot(
                    self.dh.adjacent_transform(theta1, 1),
                    np.dot(
                        self.dh.adjacent_transform(theta2, 2),
                        self.dh.adjacent_transform(theta3, 3),
                    ),
                )
                t_6_3 = np.dot(
                    np.linalg.inv(t_3_0),
                    arm_pose,
                )
                # print("t_6_3")
                # print(t_6_3)
                for i5 in [1, -1]:
                    theta5 = (
                        np.arctan2(
                            i5 * np.sqrt(t_6_3[1, 0] ** 2 + t_6_3[1, 1] ** 2),
                            t_6_3[1, 2],
                        )
                        - self.dh.theta_offset[4]
                    )
                    if abs(theta5) > 1e-3:
                        theta4 = (
                            np.arctan2(i5 * t_6_3[2, 2], -i5 * t_6_3[0, 2])
                            - self.dh.theta_offset[3]
                        )
                        theta6 = (
                            np.arctan2(-i5 * t_6_3[1, 1], i5 * t_6_3[1, 0])
                            - self.dh.theta_offset[5]
                        )
                    else:
                        # if theta5 is too small, which means that motor 3, 4, 5, 6 are on the same line, calculate the total rotation angle of motor 4 and 6 through the rotation of t_6_3
                        rot_angle = np.arctan2(-t_6_3[2, 0], t_6_3[0, 0])
                        if ref_pos is not None:
                            rot_angle = rot_angle - (
                                ref_pos[3]
                                + self.dh.theta_offset[3]
                                + ref_pos[5]
                                + self.dh.theta_offset[5]
                            )
                            min_angle = (
                                self.dh.joints_limit[3][0]
                                - ref_pos[3]
                                + self.dh.joints_limit[5][0]
                                - ref_pos[5]
                            )
                            max_angle = (
                                self.dh.joints_limit[3][1]
                                - ref_pos[3]
                                + self.dh.joints_limit[5][1]
                                - ref_pos[5]
                            )
                            while rot_angle < min_angle:
                                rot_angle += 2 * np.pi
                            while rot_angle > max_angle:
                                rot_angle -= 2 * np.pi
                            while True:
                                if rot_angle + 2 * np.pi < max_angle and abs(
                                    rot_angle + 2 * np.pi
                                ) < abs(rot_angle):
                                    rot_angle += 2 * np.pi
                                elif rot_angle - 2 * np.pi > min_angle and abs(
                                    rot_angle - 2 * np.pi
                                ) < abs(rot_angle):
                                    rot_angle -= 2 * np.pi
                                else:
                                    break
                            if rot_angle / 2 + ref_pos[3] > self.dh.joints_limit[3][1]:
                                theta4 = self.dh.joints_limit[3][1]
                                theta6 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[3][1]
                                )
                            elif (
                                rot_angle / 2 + ref_pos[3] < self.dh.joints_limit[3][0]
                            ):
                                theta4 = self.dh.joints_limit[3][0]
                                theta6 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[3][0]
                                )
                            elif (
                                rot_angle / 2 + ref_pos[5] > self.dh.joints_limit[5][1]
                            ):
                                theta6 = self.dh.joints_limit[5][1]
                                theta4 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[5][1]
                                )
                            elif (
                                rot_angle / 2 + ref_pos[5] < self.dh.joints_limit[5][0]
                            ):
                                theta6 = self.dh.joints_limit[5][0]
                                theta4 = (
                                    ref_pos[3]
                                    + ref_pos[5]
                                    + rot_angle
                                    - self.dh.joints_limit[5][0]
                                )
                            else:
                                theta4 = rot_angle / 2 + ref_pos[3]
                                theta6 = rot_angle / 2 + ref_pos[5]
                        else:
                            theta4 = rot_angle / 2 - self.dh.theta_offset[3]
                            theta6 = rot_angle / 2 - self.dh.theta_offset[5]

                    # print("i5 ", i5, "t_6_3[1,1]", t_6_3[1, 1], "t_6_3[1,0]", t_6_3[1, 0])
                    # print(-i5 * t_6_3[1, 1], i5 * t_6_3[1, 0], np.arctan2(-i5 * t_6_3[1, 1], i5 * t_6_3[1, 0]), self.dh.theta_offset[5])
                    # print("theta6", theta6)
                    # print("theta1,2,3,4,5,6")
                    # print(theta1, theta2, theta3, theta4, theta5, theta6)
                    joints = self.dh.limit_joints(
                        np.array(
                            [
                                theta1,
                                theta2,
                                theta3,
                                theta4,
                                theta5,
                                theta6,
                            ]
                        ),
                        force_calculate=force_calculate,
                    )
                    if joints is not None:
                        res.append(joints)

        limit_bias = []
        ref_bias = []

        for i in range(len(res)):
            limit_bias.append(self.dh._cal_limit_bias(res[i]))
            ref_bias.append(
                self._cal_punish(res[i], ref_pos) if ref_pos is not None else 0
            )

        combined = list(zip(res, limit_bias, ref_bias))
        combined.sort(key=lambda x: (x[1], x[2]))
        res = [item[0] for item in combined]
        if use_clip:
            res = [self.dh._clip_joints(q) for q in res]

        return res

    def _cut_end(self, pose: np.ndarray) -> np.ndarray:
        """
        Cut the end effector position to joint4(,5, 6).

        Args:
            pose (np.ndarray): end effector pose

        Returns:
            np.ndarray: joint4(,5, 6) pose
        """
        assert len(pose) == 4 and len(pose[0]) == 4, "pose must be 4x4 matrix"

        pose = np.dot(pose, np.linalg.inv(self.dh.end_convert_matrix))

        pos = pose[0:3, 3]
        rot = pose[0:3, 0:3]

        res_pos = pos - rot[:, 2] * self.dh.d[6]
        res_rot = rot
        res = np.eye(4)
        res[0:3, 0:3] = res_rot
        res[0:3, 3] = res_pos
        res[3, 3] = 1.0

        return res


def validate_jacobian(arm_kdl, q, epsilon=1e-4):
    """完整验证几何雅可比矩阵"""
    J_geo = np.round(arm_kdl.jacobian(q), 3)
    # 验证线速度部分（前3行）
    J_v_numerical = np.zeros((3, len(q)))

    # 验证角速度部分（后3行）- 这部分更复杂
    J_w_numerical = np.zeros((3, len(q)))

    # 获取当前位姿
    T_current = arm_kdl.forward_kinematics(q)
    p_current = T_current[:3, 3]
    R_current = T_current[:3, :3]

    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = epsilon

        # 计算位置变化 - 线速度验证
        T_plus = arm_kdl.forward_kinematics(q + dq)
        T_minus = arm_kdl.forward_kinematics(q - dq)

        p_plus = T_plus[:3, 3]
        p_minus = T_minus[:3, 3]
        J_v_numerical[:, i] = (p_plus - p_minus) / (2 * epsilon)

        # 计算角速度 - 使用旋转矩阵的变化
        R_plus = T_plus[:3, :3]
        R_minus = T_minus[:3, :3]

        # 角速度的近似：ω ≈ (R_plus - R_minus) * R_current^T / (2*epsilon)的反对称部分
        dR = (R_plus - R_minus) / (2 * epsilon)
        omega_skew = dR @ R_current.T

        # 提取反对称矩阵的向量形式
        J_w_numerical[0, i] = omega_skew[2, 1]  # ω_x
        J_w_numerical[1, i] = omega_skew[0, 2]  # ω_y
        J_w_numerical[2, i] = omega_skew[1, 0]  # ω_z

    print("数值解jacobian：\n", np.round(np.concatenate([J_v_numerical, J_w_numerical]), 3))

    # 比较结果
    print("=== 线速度部分验证 ===")
    print("数值计算:", J_v_numerical)
    print("解析计算:", J_geo[:3, :])
    print("最大误差:", np.max(np.abs(J_geo[:3, :] - J_v_numerical)))

    print("\n=== 角速度部分验证 ===")
    print("数值计算:", np.round(J_w_numerical, 3))
    print("解析计算:", J_geo[3:, :])
    print("最大误差:", np.max(np.abs(J_geo[3:, :] - J_w_numerical)))

    return (
        np.max(np.abs(J_geo[:3, :] - J_v_numerical)) < 1e-4
        and np.max(np.abs(J_geo[3:, :] - J_w_numerical)) < 1e-4
    )


def main():
    """
    Example usage of the ArmKdl class.
    """
    arm_kdl = ArmKdl(eef_type="none")
    end_pose = arm_kdl.forward_kinematics(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    print("end_pose")
    print(end_pose)
    # 0.250000 0.200000 0.450000 0.000000 0.000000 -0.000000 1.000000
    # end_pose = np.array(
    #     [
    #         [0.0000000, 0.0000000, 1.0000000, 0.31657160038479626],
    #         [0.9999983, 0.0018234, -0.0000000, 0.07548884871169535],
    #         [-0.0018234, 0.9999983, -0.0000000, 0.0],
    #         [0.0, 0.0, 0.0, 1.0],
    #     ]
    # )
    joints = arm_kdl.inverse_kinematics(
        end_pose,
        ref_pos=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        # ref_pos=np.array(
        #     [
        #         0.016594186425209045,
        #         -0.579270601272583,
        #         1.1369879245758057,
        #         -1.575303316116333,
        #         1.71644926071167,
        #         1.5714884996414185,
        #     ]
        # ),
    )
    print("joints")
    print(joints)
    # for j in joints:
    #     print(j)
    #     print("forward_kinematics")
    #     print(arm_kdl.forward_kinematics(j))
    #     print()
    #     print()
    #     print()
    jacobian = np.round(arm_kdl.jacobian(joints[0]), 3)
    print("jacobian")
    print(jacobian)
    validate_jacobian(arm_kdl, joints[0])


if __name__ == "__main__":
    main()
