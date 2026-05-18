TASK_NAME = "template"  # usually the name of the this file's name

# TODO: change to a class

"""
The default configs for the template task:
- The task name is "template" which is the file name.
- The TASK_NAME is used to easily define the paths since the path usually contains the task name.
- The default data and ckpt paths are "./data/hdf5/template" and "./my_ckpt/template/ckpt"
- The default policy config is for ACT (POLICY_CONFIG_ACT_DEFAULT) used in the common config
- The default "common" config is COMMON_CONFIG_DEFAULT, which should be modified for both training and evaluating.
- The default "train" config is TRAIN_CONFIG_DEFAULT, which should be modified for training.
- The default "eval" config is EVAL_CONFIG_DEFAULT, which should be modified for evaluating.
The default task config is TASK_CONFIG_DEFAULT, which is a dict containing the common, train, and eval configs.
The final config is TASK_CONFIG, which is used by the training, evaluating and so on.
"""


def get_task_name(cur_file: str):
    """Get the task name from the current file __file__ arg."""
    return cur_file.split("/")[-1].replace(".py", "")


def activator(flag: bool):
    """A decorator to add a activated flag to the function."""
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def func_add_flag(*args, **kwargs):
            return func(*args, **kwargs)

        func_add_flag.__dict__["activated"] = flag
        return func_add_flag

    return decorator


def policy_maker(config: dict, stage=None):
    """
    Make the policy instance.
    Arg:
    - config: the config for the policy containing the policy class and configs
    - stage: the stage of the policy, e.g. "train", "eval". None means both the same
    Return:
    - the policy (nn.Module or any instance and funcitonable which has one input param corresponding to the envrionment step output and returns the loss dict for trainning or the action tensor for inferencing)
    """
    # Note: you should not use the "policy_maker" in the config in this function
    # since it will cause a recursive call
    print("not use custom policy maker")
    print("policy_config:", config)
    print("stage", stage)
    return None


def environment_maker(config: dict, stage=None):
    """
    Make the environment instance. A environment is the combination of the habitat and robot.
    Arg:
    - config: the config for the habitat containing the env and robot configs
    - stage: the stage of the environment, e.g. "train", "eval". None means both the same
    Return:
    - the environment instance (must have the reset and step methods)
    """
    # Note: you should not use the "habitat_maker" in the config in this function
    # since it will cause a recursive call
    print("not use custom environment maker")
    print("environment_config:", config)
    print("stage", stage)
    return None


@activator(False)  # set to True to use augment_images function
def augment_images(image):
    """Augment the images."""
    # this shows no augmentation
    return image


def policy_ensembler(grouped_out: dict):
    """Ensemble the output of all policies and return one to execute."""
    # this shows equal-weight average ensemble for all groups
    # TODO: move to the wrapper class
    outputs = tuple(grouped_out.values())
    group_num = len(outputs)
    grouped_ave = [sum(x) / len(x) for x in outputs]
    weights = [1 / group_num] * group_num
    return sum([x * w for x, w in zip(grouped_ave, weights)])


# relative to the root of the project where you run the train/eval script
DATA_DIR_DEFAULT = "./data/hdf5"
TRAIN_DIR_DEFAULT = "./my_ckpt"  # when training to save and when evaluating to load
EVAL_DIR_DEFAULT = "./eval_results"

POLICY_CONFIG_ACT_DEFAULT = {
    "kl_weight": 10,
    "chunk_size": 40,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "temporal_agg": False,
    "num_queries": 40,  # the same as the chunk_size
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "backbone": "resnet18",
    "lr_backbone": 1e-5,
    "policy_class": "ACT",  # TODO:remove this
    "policy_maker": policy_maker,
}

ENV_EVAL_CONFIG_DEFAULT = {
    # the habitat and robot configs for evaluating
    # e.g. "habitats": ["habitat1", "habitat2"], "robots": [["robot1", "robot2"], "robot2"]
    "habitats": [],
    "robots": [],
    "environment_maker": environment_maker,
}
ENV_TRAIN_CONFIG_DEFAULT = {
    # TODO:the habitat and robot configs for training
    "habitats": ["CKPT_PATH"],
    "robots": ["DataLoader"],
    "environment_maker": None,
}

# TODO: use robot_config class instead of robot_num and joint_num
COMMON_CONFIG_DEFAULT = {
    "state_dim": 7,  # the dimension of the state space
    "action_dim": 7,  # the dimension of the action space
    "camera_names": [
        "0",
        "1",
        "2",
    ],
    # the state_dim and action_dim are used in policy_config
    # TODO: the policy_config should be used by others so that
    # state_dim and action_dim should be moved to policy_config
    # and camera_names should be used by the dataloader env or the real/sim env
    "policy_config": POLICY_CONFIG_ACT_DEFAULT,
    "ckpt_dir": TRAIN_DIR_DEFAULT + f"/{TASK_NAME}/ckpt",
    # stats_path所在路径包含了统计信息、最优/后权重数据等核心文件
    "stats_path": "",  # "" if not use, TRAIN_DIR_DEFAULT + f"/{TASK_NAME}/{TASK_NAME}",
}

TRAIN_CONFIG_DEFAULT = {
    "seed": 1,
    "learning_rate": 2e-5,
    "lr_backbone": 1e-5,
    "num_epochs": 7000,
    "pretrain_ckpt_path": "",  # "" if not use
    # "AUTO" if set according to the pretrain_ckpt_path (last/best from 0 and others from the epoch number in the path); any uint number if set manually
    "pretrain_epoch_base": "AUTO",
    # not used for now
    "eval_every": 0,  # 0 if not use
    "validate_every": 500,
    "save_every": 500,
    "skip_mirrored_data": False,
    # for cotraining (not used for now)
    "sample_weights": [7.5, 2.5],
    "cotrain_dir": "",
    "parallel": None,  # {"mode":str, "device_ids":list}, mode: "DP" or "DDP"; device_ids: e.g. [0, 1] or None for all
    "environments": ENV_TRAIN_CONFIG_DEFAULT,
    # directory containing the hdf5 files
    "load_data": {
        "dataset_dir": DATA_DIR_DEFAULT + f"/{TASK_NAME}",
        # 0/"ALL" if use all episodes or a number to use the first n episodes
        # or a tuple of (start, end) to use the episodes from start to end (not included the end), e.g. (50, 100)
        # or a tuple of (start, end, postfix) to use the episodes from start to end with the postfix,
        # e.g. (50, 100, "augmented")
        # or a list(not tuple!) of multi tuples e.g. [(0, 50), (100, 200)]
        # TODO: not implemented; support custom name postfix, e.g. "episode_0_augmented"
        "num_episodes": "ALL",
        "batch_size_train": 16,
        "batch_size_validate": 16,
        # ratio of train data from the first dataset_dir, the rest is validation data
        "train_ratio": 0.8,
        "augmentors": {
            "image": None,  # augment_images,
        },
        "check_episodes": True,  # check the existence of all episodes
        "num_workers_train": 1,
        "num_workers_validate": 1,
        "observation_slice": None,
        "action_slice": None,
    },
}

EVAL_CONFIG_DEFAULT = {
    # robot and env conigurations
    # TODO: since there is online training, these should be moved to the common config?
    # or use a maker to make the robot and env instances separately for training and evaluation
    "seed": 1000,
    "robot_name": "airbot_play_v3",
    "robot_description": "<path/to/your/robot_description>",
    "habitat": "real",  # TODO:habitat instance or "real", "mujoco" to use the corresponding env
    # "AUTO" will try to get from the key_info.pkl, if failed, use all zero
    "robot_num": 1,  # the number of (follower) robots evoloved in the task
    "joint_num": 7,  # the number of joints of one robot (e.g. arm + end effector)
    "start_joint": [0.0] * 7,  # the start joint angles of the robot
    "max_timesteps": 400,  # 一般可以设置跟数据采集时的episode_len相等
    "num_rollouts": 50,  # 程序一次运行最大的推理次数
    "save_dir": EVAL_DIR_DEFAULT,  # "" if not save the evaluation results, AUTO means the same as the ckpt_dir
    "time_stamp": None,  # None if not used
    "fps": 25,  # the frequency for executing actions (should > inference frequency)
    "ckpt_names": [
        "policy_best.ckpt"
    ],  # policy_last or any other ckpts to evaluate 'num_rollouts' times in sequence
    "ensemble": {  # will replace the corresponding configs in the other configs; None if not used
        "Group1": {  # the name of the ensemble group
            "policies": [POLICY_CONFIG_ACT_DEFAULT],
            "ckpt_dirs": [TRAIN_DIR_DEFAULT + f"/{TASK_NAME}/ckpt"],
            "ckpt_names": [("policy_best.ckpt",)],
            "fps": [25],
        },
        "ensembler": policy_ensembler,
    },
    "environments": ENV_EVAL_CONFIG_DEFAULT,
    "info_records": {
        "save_dir": EVAL_DIR_DEFAULT,
        "info_types": {
            "color_image": "mp4",
            "state": "npy",
            "action": "npy",
            "trajectory": "hdf5",
            "config": "pkl",
        },
    },
}

TASK_CONFIG_DEFAULT = {
    "common": COMMON_CONFIG_DEFAULT,
    "train": TRAIN_CONFIG_DEFAULT,
    "eval": EVAL_CONFIG_DEFAULT,
}


def get_time_stamp():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def replace_task_name(name: str, stats_name="dataset_stats.pkl", time_stamp="now"):
    """Replace the task name in the default dirs/paths."""
    global TASK_NAME
    TASK_NAME = name
    if time_stamp == "now":
        ts = get_time_stamp() + "/"
    elif time_stamp is not None:
        ts = time_stamp + "/"
    else:
        ts = ""
    TRAIN_CONFIG_DEFAULT["load_data"]["dataset_dir"] = (
        DATA_DIR_DEFAULT + f"/{TASK_NAME}"
    )
    COMMON_CONFIG_DEFAULT["ckpt_dir"] = TRAIN_DIR_DEFAULT + f"/{TASK_NAME}/{ts}ckpt"
    EVAL_CONFIG_DEFAULT["save_dir"] = EVAL_DIR_DEFAULT + f"/{TASK_NAME}/{ts}"
    if stats_name not in ["", None]:
        COMMON_CONFIG_DEFAULT["stats_path"] = (
            TRAIN_DIR_DEFAULT + f"/{TASK_NAME}/{ts}{TASK_NAME}/{ts}{stats_name}"
        )


def set_paths(
    data_dir: str, ckpt_dir: str, stats_path: str = "", save_dir: str = "AUTO"
):
    """Replace the default data, ckpt, stats and eval dirs/paths."""
    TRAIN_CONFIG_DEFAULT["load_data"]["dataset_dir"] = data_dir
    COMMON_CONFIG_DEFAULT["ckpt_dir"] = ckpt_dir
    COMMON_CONFIG_DEFAULT["stats_path"] = stats_path
    EVAL_CONFIG_DEFAULT["save_dir"] = save_dir


def is_valid_module_name(name):
    import re

    # 只允许字母、数字、下划线，且必须以字母或下划线开头
    pattern = r"^[a-z_][a-z0-9_]*$"
    return bool(re.match(pattern, name))


# finally choose one of the data and model configs
replace_task_name(TASK_NAME)
TASK_CONFIG = TASK_CONFIG_DEFAULT
