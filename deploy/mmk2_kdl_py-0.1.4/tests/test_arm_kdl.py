"""
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-08-27 11:46:34
LastEditTime: 2025-08-27 18:17:41
"""
import numpy as np
import pytest

from mmk2_kdl_py.arm_kdl import ArmKdl
from mmk2_kdl_py.arm_kdl_ops import ArmKdlNumerical


@pytest.fixture
def arm_kdl():
    return ArmKdl(eef_type="G2")


@pytest.fixture
def arm_kdl_numerical():
    return ArmKdlNumerical(eef_type="G2")


@pytest.fixture
def sample_q():
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_forward_kinematics(arm_kdl, sample_q):
    pose = arm_kdl.forward_kinematics(sample_q)
    assert pose.shape == (4, 4)
    # Expected pose for zero configuration (based on DH parameters)
    expected_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.2863],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.213572],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(pose, expected_pose, atol=1e-6)


def test_inverse_kinematics(arm_kdl, sample_q):
    pose = arm_kdl.forward_kinematics(sample_q)
    solutions = arm_kdl.inverse_kinematics(pose, ref_pos=sample_q)
    assert len(solutions) > 0
    assert solutions[0].shape == (6,)
    np.testing.assert_allclose(solutions[0], sample_q, atol=1e-6)


def test_round_trip(arm_kdl, sample_q):
    pose = arm_kdl.forward_kinematics(sample_q)
    solutions = arm_kdl.inverse_kinematics(pose, ref_pos=sample_q)
    new_pose = arm_kdl.forward_kinematics(solutions[0])
    np.testing.assert_allclose(pose, new_pose, atol=1e-6)


def test_jacobian(arm_kdl, sample_q):
    J = arm_kdl.jacobian(sample_q)
    assert J.shape == (6, 6)
    # For zero configuration, Jacobian should have specific structure
    # This is a basic check; more comprehensive validation in validate_jacobian


def test_forward_kinematics_numerical(arm_kdl_numerical, sample_q):
    pose = arm_kdl_numerical.forward_kinematics(sample_q)
    assert pose.shape == (4, 4)


def test_solve_ik_numerical(arm_kdl_numerical, sample_q):
    pose = arm_kdl_numerical.forward_kinematics(sample_q)
    q_sol = arm_kdl_numerical.solve_ik(pose, method="numerical", current_q=sample_q)
    assert q_sol.shape == (6,)
    np.testing.assert_allclose(
        q_sol, sample_q, atol=0.1
    )  # Numerical method may have slight differences


def test_round_trip_numerical(arm_kdl_numerical, sample_q):
    pose = arm_kdl_numerical.forward_kinematics(sample_q)
    q_sol = arm_kdl_numerical.solve_ik(pose, method="numerical", current_q=sample_q)
    new_pose = arm_kdl_numerical.forward_kinematics(q_sol)
    np.testing.assert_allclose(pose, new_pose, atol=0.1)
