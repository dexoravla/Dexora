"""Python SDK for MMK2 kinematics and dynamics."""

# Standard imports
from dataclasses import dataclass

# Third-party imports
import numpy as np
import typing

# Local imports
from .arm_kdl import ArmKdl, ArmDH
from .utils import timed


@dataclass
class HeadKdl:
    """Head kinematics class."""

    dh: ArmDH

    def forward_kinematics_rotation(self, q: np.ndarray) -> np.ndarray:
        """
        Calculate the forward kinematics rotation matrix for the head
        Args:
            q: joint angles
        Returns:
            rotation matrix of the head
        """
        assert len(q) == 2, "q should be a 2-element array"

        res = np.eye(3)
        for i in range(2):
            res = np.dot(res, self.dh.adjacent_rotation(q[i], i + 1))

        res = np.dot(res, self.dh.end_convert_matrix[:3, :3])

        return np.round(res, decimals=9)

    def inverse_kinematics_rotation(
        self,
        rotation: np.ndarray,
        force_calculate: bool = False,
        use_clip: bool = False,
    ) -> typing.Optional[np.ndarray]:
        """
        Calculate the inverse kinematics rotation matrix for the head
        Args:
            rotation: rotation matrix
            force_calculate: whether to force the calculation
            use_clip: whether to clip the joint angles to the limits (only used if force_calculate is True)
        Returns:
            joint angles of the head
        """
        assert rotation.shape == (3, 3), "rotation should be a 3x3 matrix"
        rotation = np.dot(rotation, np.linalg.inv(self.dh.end_convert_matrix[:3, :3]))
        theta1 = np.arctan2(rotation[0, 2], -rotation[1, 2]) - self.dh.theta_offset[0]
        theta2 = np.arctan2(rotation[2, 0], rotation[2, 1]) - self.dh.theta_offset[1]
        res = self.dh.limit_joints(
            np.array([theta1, theta2]), force_calculate=force_calculate
        )
        if use_clip and res is not None:
            res = self.dh._clip_joints(res)
        return res


@dataclass
class SpineKdl:
    dx: float
    dz: float
    joint_limits: np.ndarray

    def get_transformation_matrix(self, q: float) -> np.ndarray:
        """Get the transformation matrix of the spine"""
        # fmt: off
        # pylint: disable=line-too-long
        return np.array([[0, 1,  0,     self.dx],
                         [1, 0,  0,           0],
                         [0, 0, -1, self.dz - q],
                         [0, 0,  0,           1]])
        # pylint: enable=line-too-long
        # fmt: on


@dataclass
class Spine2ArmKdl:
    dx: float
    dy: float
    dz: float

    def get_transformation_matrix(self, index: str = "left") -> np.ndarray:
        """Get the transformation matrix of the spine to arm"""
        coef = 1 if index == "left" else -1
        # fmt: off
        # pylint: disable=line-too-long
        return np.array([[-coef * np.sqrt(2)/2,        0, coef * np.sqrt(2)/2, coef * self.dx],
                         [        np.sqrt(2)/2,        0,        np.sqrt(2)/2,        self.dy],
                         [                   0, coef * 1,                   0,        self.dz],
                         [                   0,        0,                   0,              1]])
        # pylint: enable=line-too-long
        # fmt: on


class MMK2Kdl:
    """Kinematics and dynamics of MMK2."""

    def __init__(
        self, punish: typing.Optional[typing.List[float]] = None, iteration: int = 100
    ):
        self.head = HeadKdl(
            dh=ArmDH(
                alpha=np.array([0, np.pi / 2]),
                a=np.array([0, 0]),
                d=np.array([0, 0]),
                theta_offset=np.array([0, np.pi / 2]),
                end_convert_matrix=np.array(
                    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
                ),
                joints_limit=np.array([[-0.5, 0.5], [-1.18, 0.16]]),
            )
        )
        self.left_arm = ArmKdl()
        self.right_arm = ArmKdl()
        self.spine = SpineKdl(
            dx=0.033942, dz=1.406, joint_limits=np.array([-0.04, 0.87])
        )
        self.spine2arm = Spine2ArmKdl(dx=0.10704, dy=0.02283, dz=0.09475)
        if punish is not None:
            self.punish = punish
            self.left_arm.update_punish(punish[1:7])
            self.right_arm.update_punish(punish[7:])
        else:
            self.punish = [
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        self.iteration = iteration
        self.index = None

    def _cal_punish(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Calculate the punishment of the joint angles"""
        assert len(q1) == len(q2), "q1 and q2 should have the same length"
        assert len(self.punish) == len(q1), "punish should have the same length as q1"
        return np.sum(self.punish * np.abs(q1 - q2))

    def update_punish(self, punish: typing.List[float]) -> None:
        """
        Update the punishment of the joint angles.

        Args:
            punish (list[float]): new punishment values for each joint
        """
        assert len(punish) == 13, "punish should be a 13-element array"
        self.punish = punish
        self.left_arm.update_punish(punish[1:7])
        self.right_arm.update_punish(punish[7:13])

    def forward_kinematics(
        self, q: np.ndarray, index: typing.Optional[str] = None
    ) -> typing.Tuple[typing.Optional[np.ndarray], typing.Optional[np.ndarray]]:
        """
        Get the forward kinematics of mmk2

        Args:
            q: joint angles
            index: index of the arm, "left" or "right", if None, return both arms
        Returns:
            T_left: transformation matrix of the left arm
            T_right: transformation matrix of the right arm
        """

        if index is None:
            assert len(q) == 13, "q should be a 13-element array"
        else:
            assert len(q) == 7, "q should be a 7-element array"
            assert index in ["left", "right"], "index should be 'left' or 'right'"
        T_spine = self.spine.get_transformation_matrix(q[0])
        T_spine2arm_left = self.spine2arm.get_transformation_matrix("left")
        T_spine2arm_right = self.spine2arm.get_transformation_matrix("right")

        if index is None:
            T_left_arm = self.left_arm.forward_kinematics(q[1:7])
            T_right_arm = self.right_arm.forward_kinematics(q[7:13])

            T_left = T_spine @ T_spine2arm_left @ T_left_arm
            T_right = T_spine @ T_spine2arm_right @ T_right_arm
            return T_left, T_right

        if index == "left":
            T_left_arm = self.left_arm.forward_kinematics(q[1:7])
            T_left = T_spine @ T_spine2arm_left @ T_left_arm
            return T_left, None

        if index == "right":
            T_right_arm = self.right_arm.forward_kinematics(q[1:7])
            T_right = T_spine @ T_spine2arm_right @ T_right_arm
            return None, T_right

        # This should never be reached due to the assertions above
        return None, None

    def _cal_limit_bias(self, q: np.ndarray) -> float:
        """Calculate the limit bias of the joint angles"""
        assert len(q) == 13, "q should be a 13-element array"
        return self.left_arm.dh._cal_limit_bias(
            q[1:7]
        ) + self.right_arm.dh._cal_limit_bias(q[7:13])

    def _inverse_kinematics(
        self,
        T_left: typing.Optional[np.ndarray],
        T_right: typing.Optional[np.ndarray],
        target_height: float,
        ref_arm_pos: typing.Optional[np.ndarray] = None,
        force_calculate: bool = False,
    ) -> typing.List[np.ndarray]:
        """Get the inverse kinematics of mmk2"""

        assert T_left is not None or T_right is not None

        if T_left is not None:
            assert T_left.shape == (4, 4), "T_left should be a 4x4 matrix"
        if T_right is not None:
            assert T_right.shape == (4, 4), "T_right should be a 4x4 matrix"

        if ref_arm_pos is not None:
            if self.index == "both":
                assert (
                    len(ref_arm_pos) == 12
                ), "ref_arm_pos should be a 12-element array"
            else:
                assert (
                    len(ref_arm_pos) == 6 or len(ref_arm_pos) == 12
                ), "ref_arm_pos should be a 6-element array or a 12-element array"

        T_spine = self.spine.get_transformation_matrix(target_height)

        if self.index == "both":
            assert (
                T_left is not None and T_right is not None
            ), "T_left and T_right should not be None"
            T_spine2arm_left = self.spine2arm.get_transformation_matrix("left")
            T_spine2arm_right = self.spine2arm.get_transformation_matrix("right")
            T_left_arm = np.linalg.inv(T_spine @ T_spine2arm_left) @ T_left
            T_right_arm = np.linalg.inv(T_spine @ T_spine2arm_right) @ T_right
            left_arm_joints = self.left_arm.inverse_kinematics(
                T_left_arm,
                ref_arm_pos[:6] if ref_arm_pos is not None else None,
                force_calculate=force_calculate,
            )
            right_arm_joints = self.right_arm.inverse_kinematics(
                T_right_arm,
                ref_arm_pos[6:] if ref_arm_pos is not None else None,
                force_calculate=force_calculate,
            )
            if len(left_arm_joints) == 0 or len(right_arm_joints) == 0:
                return []
            res = []
            for joints in left_arm_joints:
                for joints2 in right_arm_joints:
                    res.append(
                        np.array(
                            [
                                target_height,
                                joints[0],
                                joints[1],
                                joints[2],
                                joints[3],
                                joints[4],
                                joints[5],
                                joints2[0],
                                joints2[1],
                                joints2[2],
                                joints2[3],
                                joints2[4],
                                joints2[5],
                            ]
                        )
                    )

            limit_bias = []
            ref_bias = []
            ref_pos = (
                np.array([target_height] + list(ref_arm_pos))
                if ref_arm_pos is not None
                else None
            )
            for joints in res:
                limit_bias.append(self._cal_limit_bias(joints))
                ref_bias.append(
                    self._cal_punish(ref_pos, joints) if ref_pos is not None else 0.0
                )

            combined = list(zip(res, limit_bias, ref_bias))
            combined.sort(key=lambda x: (x[1], x[2]))
            res = [item[0] for item in combined]
            return res

        if self.index == "left":
            assert T_left is not None, "T_left should not be None"
            T_spine2arm = self.spine2arm.get_transformation_matrix("left")
            T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T_left
            if ref_arm_pos is not None and len(ref_arm_pos) == 12:
                ref_arm_pos = ref_arm_pos[:6]
            arm_joints = self.left_arm.inverse_kinematics(
                T_arm, ref_arm_pos, force_calculate=force_calculate
            )
            if len(arm_joints) == 0:
                return []
            res = []
            for joints in arm_joints:
                res.append(
                    np.array(
                        [
                            target_height,
                            joints[0],
                            joints[1],
                            joints[2],
                            joints[3],
                            joints[4],
                            joints[5],
                        ]
                    )
                )
            return res

        if self.index == "right":
            assert T_right is not None, "T_right should not be None"
            T_spine2arm = self.spine2arm.get_transformation_matrix("right")
            T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T_right
            if ref_arm_pos is not None and len(ref_arm_pos) == 12:
                ref_arm_pos = ref_arm_pos[6:]
            arm_joints = self.right_arm.inverse_kinematics(
                T_arm, ref_arm_pos, force_calculate=force_calculate
            )
            if len(arm_joints) == 0:
                return []
            res = []
            for joints in arm_joints:
                res.append(
                    np.array(
                        [
                            target_height,
                            joints[0],
                            joints[1],
                            joints[2],
                            joints[3],
                            joints[4],
                            joints[5],
                        ]
                    )
                )
            return res

        return []  # This should never be reached due to the assertions above

    def _validate_ik_inputs(
        self,
        T_left: typing.Optional[np.ndarray],
        T_right: typing.Optional[np.ndarray],
        ref_pos: typing.Optional[np.ndarray] = None,
        target_height: typing.Optional[float] = None,
    ) -> str:
        """Validate inputs for inverse kinematics."""
        assert (
            T_left is not None or T_right is not None
        ), "T_left and T_right should not be None at the same time"
        assert (
            ref_pos is not None or target_height is not None
        ), "ref_pos and target_height should not be None at the same time"

        index = (
            "both"
            if (T_left is not None and T_right is not None)
            else ("left" if T_left is not None else "right")
        )

        if T_left is not None:
            assert T_left.shape == (4, 4), "T_left should be a 4x4 matrix"
        if T_right is not None:
            assert T_right.shape == (4, 4), "T_right should be a 4x4 matrix"

        if target_height is not None:
            assert (
                target_height >= self.spine.joint_limits[0]
                and target_height <= self.spine.joint_limits[1]
            ), "target_height should be within the spine joint limits"

        if ref_pos is not None:
            if index == "both" and target_height is not None:
                assert len(ref_pos) == 13, "ref_pos should be a 13-element array"
            else:
                assert (
                    len(ref_pos) == 7 or len(ref_pos) == 13
                ), "ref_pos should be a 7-element array or a 13-element array"

        return index

    def inverse_kinematics(
        self,
        T_left: typing.Optional[np.ndarray] = None,
        T_right: typing.Optional[np.ndarray] = None,
        ref_pos: typing.Optional[np.ndarray] = None,
        target_height: typing.Optional[float] = None,
        force_calculate: bool = False,
        use_clip: bool = False,
    ) -> typing.List[np.ndarray]:
        """
        Get the inverse kinematics of mmk2, T_left and T_right should not be None at the same time, ref_pos should not be None if target_height is None

        Args:
            T_left: transformation matrix of the left arm
            T_right: transformation matrix of the right arm
            ref_pos: reference joint angles, 0： spine 1-7： left arm joints 8-13： right arm joints
            target_height: target height of the spine
            force_calculate: whether to force the calculation
            use_clip: whether to clip the joint angles to the limits (only used if force_calculate is True)

        Returns:
            res: joint angles of the arm
        """

        self.index = self._validate_ik_inputs(T_left, T_right, ref_pos, target_height)
        res: typing.List[np.ndarray]

        if target_height is not None:
            ref_arm_pos = None
            if ref_pos is not None:
                ref_arm_pos = ref_pos[1:]
            res = self._inverse_kinematics(
                T_left,
                T_right,
                target_height,
                ref_arm_pos,
                force_calculate=force_calculate,
            )
            if len(res) == 0:
                return []
            if use_clip:
                clip_res = res[0].copy()
                if len(clip_res) == 13:
                    clip_res[1:7] = self.left_arm.dh._clip_joints(clip_res[1:7])
                    clip_res[7:13] = self.right_arm.dh._clip_joints(clip_res[7:13])
                else:
                    if self.index == "left":
                        clip_res[1:7] = self.left_arm.dh._clip_joints(clip_res[1:7])
                    elif self.index == "right":
                        clip_res[1:7] = self.right_arm.dh._clip_joints(clip_res[1:7])
                res[0] = clip_res
            return [res[0]]

        assert (
            ref_pos is not None
        ), "ref_pos should not be None if target_height is None"
        sum_punish: float = np.inf
        target_spine_height = ref_pos[0]

        ref_arm_pos = ref_pos[1:]
        res = self._inverse_kinematics(
            T_left,
            T_right,
            target_spine_height,
            ref_arm_pos,
            force_calculate=force_calculate,
        )
        if len(res) > 0:
            sum_punish = self._cal_punish(ref_pos, res[0])

        for _ in range(self.iteration):
            target_spine_height = np.random.uniform(
                self.spine.joint_limits[0], self.spine.joint_limits[1]
            )
            ref_arm_pos = None
            ref_arm_pos = ref_pos[1:]
            temp_res = self._inverse_kinematics(
                T_left,
                T_right,
                target_spine_height,
                ref_arm_pos,
                force_calculate=force_calculate,
            )
            if len(temp_res) == 0:
                continue
            temp_punish = self._cal_punish(ref_pos, temp_res[0])
            if temp_punish < sum_punish:
                sum_punish = temp_punish
                res = temp_res

        if len(res) == 0:
            return []
        if use_clip:
            clip_res = res[0].copy()
            if len(clip_res) == 13:
                clip_res[1:7] = self.left_arm.dh._clip_joints(clip_res[1:7])
                clip_res[7:13] = self.right_arm.dh._clip_joints(clip_res[7:13])
            else:
                if index == "left":
                    clip_res[1:7] = self.left_arm.dh._clip_joints(clip_res[1:7])
                elif index == "right":
                    clip_res[1:7] = self.right_arm.dh._clip_joints(clip_res[1:7])
            res[0] = clip_res
        return [res[0]]


def main():
    """Example usage of the MMK2Kdl class."""
    mmk2 = MMK2Kdl()
    # q是当前的关节角度，0： spine 1-7： left arm joints 8-13： right arm joints
    q = np.array(
        [
            0.571,
            0.838,
            0.009,
            0.923,
            -1.675,
            -1.604,
            -0.009,
            -2.325,
            -1.868,
            3.077,
            1.696,
            1.437,
            2.596,
        ]
    )
    T_left, T_right = mmk2.forward_kinematics(q)
    print("Left Arm Transformation Matrix:\n", T_left)
    print("Right Arm Transformation Matrix:\n", T_right)

    joints_left = mmk2.inverse_kinematics(T_left, None, target_height=q[0], ref_pos=q)
    print("Left Arm Joints:\n", joints_left)

    joints_right = mmk2.inverse_kinematics(None, T_right, target_height=q[0])
    print("Right Arm Joints:\n", joints_right)

    joints = mmk2.inverse_kinematics(T_left, T_right, ref_pos=q)
    print("Joints with ref_pos:\n", joints)

    T_left, T_right = mmk2.forward_kinematics(joints[0])

    print("Left Arm Transformation Matrix from Inverse Kinematics:\n", T_left)
    print("Right Arm Transformation Matrix from Inverse Kinematics:\n", T_right)

    # q = np.array([0, 0])
    # rotation = mmk2.head.forward_kinematics_rotation(q)
    # print("Forward Kinematics Rotation Matrix:\n", rotation)
    # q_inv = mmk2.head.inverse_kinematics_rotation(rotation)
    # print("Inverse Kinematics Joint Angles:\n", q_inv)


if __name__ == "__main__":
    main()
