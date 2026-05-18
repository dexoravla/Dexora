"""
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-08-25 12:41:46
LastEditTime: 2025-08-25 19:47:12
"""
import numpy as np
import pytest

from mmk2_kdl_py.mmk2_kdl import MMK2Kdl
from mmk2_kdl_py.mmk2_kdl_ops import MMK2KdlNumerical


@pytest.fixture
def mmk2():
    return MMK2Kdl()


@pytest.fixture
def mmk2_numerical():
    return MMK2KdlNumerical()


@pytest.fixture
def sample_q():
    return np.array(
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


def test_forward_kinematics(mmk2, sample_q):
    T_left, T_right = mmk2.forward_kinematics(sample_q)
    assert T_left.shape == (4, 4)
    assert T_right.shape == (4, 4)


def test_inverse_kinematics(mmk2, sample_q):
    T_left, T_right = mmk2.forward_kinematics(sample_q)
    joints = mmk2.inverse_kinematics(T_left, T_right, ref_pos=sample_q)
    assert len(joints) > 0
    assert joints[0].shape == (13,)


def test_round_trip(mmk2, sample_q):
    T_left, T_right = mmk2.forward_kinematics(sample_q)
    joints = mmk2.inverse_kinematics(T_left, T_right, ref_pos=sample_q)
    T_left_new, T_right_new = mmk2.forward_kinematics(joints[0])
    np.testing.assert_allclose(T_left, T_left_new, atol=1e-6)
    np.testing.assert_allclose(T_right, T_right_new, atol=1e-6)


def test_forward_kinematics_numerical(mmk2_numerical, sample_q):
    T_left, T_right = mmk2_numerical.forward_kinematics(sample_q)
    assert T_left.shape == (4, 4)
    assert T_right.shape == (4, 4)


def test_inverse_kinematics_numerical(mmk2_numerical, sample_q):
    T_left, T_right = mmk2_numerical.forward_kinematics(sample_q)
    joints = mmk2_numerical.inverse_kinematics(T_left, T_right, ref_pos=sample_q)
    assert len(joints) > 0
    assert joints[0].shape == (13,)


def test_round_trip_numerical(mmk2_numerical, sample_q):
    T_left, T_right = mmk2_numerical.forward_kinematics(sample_q)
    joints = mmk2_numerical.inverse_kinematics(T_left, T_right, ref_pos=sample_q)
    T_left_new, T_right_new = mmk2_numerical.forward_kinematics(joints[0])
    np.testing.assert_allclose(T_left, T_left_new, atol=0.5)
    np.testing.assert_allclose(T_right, T_right_new, atol=0.5)
