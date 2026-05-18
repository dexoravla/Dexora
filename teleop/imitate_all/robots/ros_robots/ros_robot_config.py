import numpy as np
from functools import partial
from std_msgs.msg import Float32MultiArray, Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from ros_tools import Lister


# state and action types
JOINT = "joint"
JOINT_MIT = "joint_mit"  # joint position and velocity
JOINT_POSITION = "joint_position"
JOINT_VELOCITY = "joint_velocity"
JOINT_EFFORT = "joint_effort"
EEF_POSE = "pose"
EEF_POSE_POSITION = "pose_position"
EEF_POSE_ORIENTATION = "pose_orientation"
EEF_VELOCITY = "eef_velocity"
EEF_EFFORT = "eef_effort"

SIMPLE_KEY_MAP = {
    JOINT: "a",
    JOINT_MIT: "pv",
    JOINT_POSITION: "p",
    JOINT_VELOCITY: "v",
    JOINT_EFFORT: "e",
}

# Initiate All Available Configs for Actions and Observations
ACTIONS_TOPIC_CONFIG = {
    "arm": {
        "left": {
            JOINT_POSITION: ("/control/arms/left/joint_states", JointState),
            EEF_POSE: ("/control/arms/left/pose", Pose),
        },
        "right": {
            JOINT_POSITION: ("/control/arms/right/joint_states", JointState),
            EEF_POSE: ("/control/arms/left/pose", Pose),
        },
    },
    "head": {
        JOINT_POSITION: ("/control/head/move_to", Float32MultiArray),
        JOINT_VELOCITY: ("/control/head/move", Float32MultiArray),
    },
    "spine": {
        JOINT_POSITION: ("/control/spine/move_to", Int32),
        JOINT_VELOCITY: ("/control/spine/move", Int32),
    },
}
STATES_TOPIC_CONFIG = {
    "arm": {
        "left": {
            JOINT_POSITION: (
                "/follower_arms/left/joint_states",
                JointState,
            ),
            EEF_POSE: ("/follower_arms/left/current_pose", Pose),
        },
        "right": {
            JOINT_POSITION: (
                "/follower_arms/right/joint_states",
                JointState,
            ),
            EEF_POSE: ("/follower_arms/right/current_pose", Pose),
        },
    },
    "head": {
        JOINT_POSITION: (
            "/state/head/pos",
            Float32MultiArray,
        )
    },
    "spine": {
        JOINT_POSITION: (
            "/state/spine/pos",
            Int32,
        )
    },
}

EXAMPLE_CONFIG = {
    # only can have one type for each obs, action and reset
    "state": {
        "arm": {"left": JOINT_POSITION, "right": JOINT_POSITION},
        "head": JOINT_POSITION,
        "spine": JOINT_POSITION,
    },
    "action": {
        "arm": {"left": JOINT_POSITION, "right": JOINT_POSITION},
        "head": JOINT_VELOCITY,
        "spine": JOINT_VELOCITY,
    },
    "reset": {  # will also be used to get the dim of each action
        # the reset actions type can be different from those of step
        "arm": {
            "left": {JOINT_POSITION: JointState(position=[0] * 7)},
            "right": {JOINT_POSITION: JointState(position=[0] * 7)},
        },
        "head": {JOINT_POSITION: Float32MultiArray(data=[0.5541, 0.0498])},
        "spine": {JOINT_POSITION: Int32(data=-97443)},
    },
    # can have multi types for each action and state but only one will be used for preprocess and lister
    "preprocess": {
        # spine pos to float[-2π, 0]
        "state": {
            "spine": {
                JOINT_POSITION: lambda x: (np.array(x) / 120000.0 * 2 * np.pi).tolist()
            }
        },
        "action": {
            "spine": {
                # pos to int[-120000, 0]
                JOINT_POSITION: lambda x: (np.array(x) * 120000.0 / 2 / np.pi).tolist(),
                # vel to int[-1, 1]
                JOINT_VELOCITY: lambda x: [np.round(min(max(v, -1), 1)) for v in x],
            },
            # elements of vel tuple to float[-1, 1]
            "head": {JOINT_VELOCITY: lambda x: [min(max(v, -1), 1) for v in list(x)]},
        },
    },
    "lister": {
        "state": {  # convert all current data to list
            "arm": {
                "left": {JOINT_POSITION: Lister.joint_state_to_list},
                "right": {JOINT_POSITION: Lister.joint_state_to_list},
            },
            "head": {
                JOINT_POSITION: Lister.data_field_to_list,
                JOINT_VELOCITY: Lister.data_field_to_list,
            },
            "spine": {
                JOINT_POSITION: Lister.data_field_to_list,
                JOINT_VELOCITY: Lister.data_field_to_list,
            },
        },
        "action": {  # convert action list to target data (type) for publishing
            "arm": {
                "left": {
                    JOINT_POSITION: partial(
                        Lister.list_to_joint_state, SIMPLE_KEY_MAP[JOINT_POSITION]
                    ),
                    EEF_POSE: Lister.list_to_pose,
                },
                "right": {
                    JOINT_POSITION: partial(
                        Lister.list_to_joint_state, SIMPLE_KEY_MAP[JOINT_POSITION]
                    ),
                    EEF_POSE: Lister.list_to_pose,
                },
            },
            "head": {
                JOINT_POSITION: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["head"][JOINT_POSITION][1],
                ),
                JOINT_VELOCITY: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["head"][JOINT_VELOCITY][1],
                ),
            },
            "spine": {
                JOINT_POSITION: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["spine"][JOINT_POSITION][1],
                ),
                JOINT_VELOCITY: partial(
                    Lister.list_to_given_data_field,
                    ACTIONS_TOPIC_CONFIG["spine"][JOINT_VELOCITY][1],
                ),
            },
        },
    },
    "param": {
        "control_freq": 200,
        # the dim of each action must be set since the interface type and dims may be different
        # from the obersevations and reset actions. These dims will be used to convert the action
        # list to target data for publishing
        "actions_dim": {
            "arm": {
                "left": {JOINT_POSITION: 7, EEF_POSE: 7},
                "right": {JOINT_POSITION: 7, EEF_POSE: 7},
            },
            "head": {JOINT_POSITION: 2, JOINT_VELOCITY: 2},
            "spine": {JOINT_POSITION: 1, JOINT_VELOCITY: 1},
        },
        # the tolerance for reaching the target of each action
        # if None, will publish the target for 1s at control_freq
        # if int or float, will be applied to all the dims
        # if list, will be applied to each dim
        "reach_tolerance": None,
    },
}
