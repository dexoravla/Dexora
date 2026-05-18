from __future__ import annotations

"""
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-05-13 12:02:00
LastEditTime: 2025-08-25 14:38:05
"""

import numpy as np
from dataclasses import dataclass

from .arm_kdl_ops import ArmKdlNumerical
from .mmk2_kdl import MMK2Kdl  # Add this import
from .utils import timed


class MMK2KdlNumerical(MMK2Kdl):  # Inherit from MMK2Kdl
    """Kinematics and dynamics of mmk2"""

    def __init__(self, punish: list[float] = None, iteration: int = 10):
        super().__init__(punish, iteration)
        self.left_arm = ArmKdlNumerical()
        self.right_arm = ArmKdlNumerical()

    def _inverse_kinematics(
        self,
        T_left: np.array,
        T_right: np.array,
        target_height: float,
        ref_arm_pos: np.array = None,
    ) -> list[np.array] | None:  # Adjust return type to match parent's list format
        """Get the inverse kinematics of mmk2 using numerical methods"""

        T_spine = self.spine.get_transformation_matrix(target_height)

        if self.index == "both":
            T_spine2arm_left = self.spine2arm.get_transformation_matrix("left")
            T_spine2arm_right = self.spine2arm.get_transformation_matrix("right")
            T_left_arm = np.linalg.inv(T_spine @ T_spine2arm_left) @ T_left
            T_right_arm = np.linalg.inv(T_spine @ T_spine2arm_right) @ T_right

            current_q_left = ref_arm_pos[:6] if ref_arm_pos is not None else None
            current_q_right = ref_arm_pos[6:] if ref_arm_pos is not None else None

            left_arm_joints = self.left_arm.solve_ik(
                T_left_arm, method="numerical", current_q=current_q_left
            )
            right_arm_joints = self.right_arm.solve_ik(
                T_right_arm, method="numerical", current_q=current_q_right
            )

            if left_arm_joints is None or right_arm_joints is None:
                return []

            return [
                np.array(
                    [
                        target_height,
                        *left_arm_joints,
                        *right_arm_joints,
                    ]
                )
            ]

        elif self.index == "left":
            T_spine2arm = self.spine2arm.get_transformation_matrix("left")
            T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T_left

            current_q = (
                ref_arm_pos[:6]
                if ref_arm_pos is not None
                else (
                    ref_arm_pos
                    if ref_arm_pos is not None and len(ref_arm_pos) == 6
                    else None
                )
            )

            arm_joints = self.left_arm.solve_ik(
                T_arm, method="numerical", current_q=current_q
            )

            if arm_joints is None:
                return []

            return [np.array([target_height, *arm_joints])]

        elif self.index == "right":
            T_spine2arm = self.spine2arm.get_transformation_matrix("right")
            T_arm = np.linalg.inv(T_spine @ T_spine2arm) @ T_right

            current_q = (
                ref_arm_pos[6:]
                if ref_arm_pos is not None and len(ref_arm_pos) == 12
                else (ref_arm_pos if ref_arm_pos is not None else None)
            )

            arm_joints = self.right_arm.solve_ik(
                T_arm, method="numerical", current_q=current_q
            )

            if arm_joints is None:
                return []

            return [np.array([target_height, *arm_joints])]

        return []

    # @timed
    def inverse_kinematics(
        self,
        T_left: np.array = None,
        T_right: np.array = None,
        ref_pos: np.array = None,
        target_height: float = None,
    ) -> list[np.array]:

        self.index = self._validate_ik_inputs(T_left, T_right, ref_pos, target_height)

        if target_height is not None:
            ref_arm_pos = ref_pos[1:] if ref_pos is not None else None
            res = self._inverse_kinematics(T_left, T_right, target_height, ref_arm_pos)
            return res if res else []

        # If no target_height, use random sampling as before
        assert ref_pos is not None
        best_res = []
        min_punish = np.inf

        # Try reference height first
        ref_height = ref_pos[0]
        ref_arm_pos = ref_pos[1:]
        temp_res = self._inverse_kinematics(T_left, T_right, ref_height, ref_arm_pos)
        if temp_res:
            punish = self._cal_punish(ref_pos, temp_res[0])
            if punish < min_punish:
                min_punish = punish
                best_res = temp_res

        # TODO: remove random sampling
        for _ in range(self.iteration):
            rand_height = np.random.uniform(
                self.spine.joint_limits[0], self.spine.joint_limits[1]
            )
            temp_res = self._inverse_kinematics(
                T_left, T_right, rand_height, ref_pos[1:]
            )
            if temp_res:
                punish = self._cal_punish(ref_pos, temp_res[0])
                if punish < min_punish:
                    min_punish = punish
                    best_res = temp_res

        return best_res


def main():
    """Example usage of the MMK2Kdl class."""
    mmk2 = MMK2KdlNumerical()
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


def test():
    mmk2 = MMK2KdlNumerical()

    T_left = np.array(
        [
            [0.0, 0.0, 1.0, 0.364],
            [0.0, 1.0, 0.0, 0.149],
            [-1.0, 0.0, 0.0, 1.069],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ref_pose = np.zeros(7)
    ref_pose[0] = 0.0
    ref_pose[1:] = [0.03, -0.887, 0.442, 0.013, 1.543, 0.339]

    joints_right = mmk2.inverse_kinematics(
        T_left, None, target_height=ref_pose[0], ref_pos=ref_pose
    )
    print("Right Arm Joints:\n", joints_right)


if __name__ == "__main__":
    main()
