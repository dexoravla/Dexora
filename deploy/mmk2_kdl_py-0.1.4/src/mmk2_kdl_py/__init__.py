"""
Copyright: qiuzhi.tech
Author: hanyang
Date: 2025-08-25 11:37:56
LastEditTime: 2025-08-25 14:38:24
"""
"""Python SDK for MMK2 kinematics and dynamics."""

from .arm_kdl import ArmKdl
from .arm_kdl_ops import ArmKdlNumerical
from .mmk2_kdl import MMK2Kdl
from .mmk2_kdl_ops import MMK2KdlNumerical

try:
    from mmk2_kdl_py._version import version as __version__
except ImportError:
    from importlib.metadata import version

    __version__ = version("mmk2_kdl_py")
