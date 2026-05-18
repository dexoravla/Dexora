import os
import importlib
from utils.utils import (
    get_init_states,
    replace_timestamp,
)
import argparse


def basic_parser():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument(
        "-tn", "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument("--seed", action="store", type=int, help="seed", required=False)
    return parser


def load_task_config(path: str):
    """Load task config from task config python file"""
    # TODO: support for yaml config file?
    try:
        module = importlib.import_module(path)
    except Exception as e:
        print(f"Error: {e}")
        raise ImportError(
            "Is your configuration file name the same as the task name? Or have you added the configuration file to the configurations/task_configs folder?"
        )

    TASK_CONFIG = getattr(module, "TASK_CONFIG")
    image_augmentor = getattr(module, "augment_images")
    assert TASK_CONFIG != {}, f"No task config found for {path}"
    config_sys_path = module.__file__
    task_funcs = {
        "image_augmentor": image_augmentor,
    }
    return TASK_CONFIG, task_funcs, config_sys_path


def remove_none(args: dict):
    """Remove keys with None values from dict
    This is used to remove None key-values from CLI args,
    which avoids passing None values to override the actual config from config file.
    """
    return {k: v for k, v in args.items() if v is not None}


def config_policy(args: dict):
    # TODO: remove this function and use a config class instead
    camera_names = args["camera_names"]
    policy_args = args["policy_config"]
    policy_class = policy_args["policy_class"]
    state_dim = args["state_dim"]
    action_dim = args["action_dim"]
    if policy_class == "ACT":
        policy_config = {
            # TODO: should lr in policy config here?
            # TODO: build_backbone function will use lr_backbone > 0 means train_backbone=True
            # TODO: build_optimizer function will use lr_backbone to build an optimizer
            "lr_backbone": policy_args["lr_backbone"],
            "lr": args["learning_rate"],
            "camera_names": camera_names,
        }
    elif policy_class == "CNNMLP":
        backbone = "resnet18"
        policy_config = {
            "lr": args["learning_rate"],
            "lr_backbone": args["lr_backbone"],
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        policy_config = {}
    # update custom policy configs
    policy_config.update(policy_args)
    # add state_dim and action_dim into policy_config
    policy_config["state_dim"] = state_dim
    policy_config["action_dim"] = action_dim
    return policy_config


def get_stats_path(stats_path_config: str, task_name: str):
    # task_name = task_name.split("/")[-1]
    dir_level = stats_path_config.count(task_name)
    stats_name = os.path.basename(stats_path_config)
    stats_dir: str = os.path.dirname(stats_path_config)
    stats_path = os.path.join(stats_dir, stats_name)
    print(f"stats_dir={stats_dir}, stats_name={stats_name}")
    if not os.path.exists(stats_dir):
        if dir_level == 2:
            # 降级一级再检查
            index = stats_dir.rfind(task_name)
            stats_dir = stats_dir[:index]
            print(f"Warning: stats_dir {stats_dir} not found, try {stats_dir}")
            if not os.path.exists(stats_dir):
                raise FileNotFoundError(f"stats_dir {stats_dir} also not found")
            else:
                stats_path = os.path.join(stats_dir, stats_name)
        else:
            raise FileNotFoundError(f"stats_dir {stats_dir} not found")
    return stats_dir, stats_path


def get_all_config(args: dict, stage: str):
    """
    Get all config for train or eval stage from args directly and task config file.
    - args: command line args dict
    - stage: train, eval
    # TODO: import a config class instead of dict and change it to dict(all members or just properties)
    """
    assert stage in ["train", "eval"], f"Invalid stage: {stage}"
    # import config script according to task_name
    config_rela_path: str = args.get("config_path", None)
    if config_rela_path is None:
        config_rela_path = (
            f"configurations.task_configs.{args['task_name'].replace('/','.')}"
        )
    else:
        # e.g. jimu/dmil or jimu.dmil
        config_rela_path = "configurations.task_configs." + config_rela_path.replace(
            "/", "."
        )
    TASK_CONFIG, task_funcs, config_sys_path = load_task_config(config_rela_path)
    print(f"config_file_sys_path={config_sys_path}")
    # assert os.path.exists(config_sys_path), f"config file {config_sys_path} not found"
    # merge configs to all_config
    all_config = {"config_file_sys_path": config_sys_path}
    all_config.update(TASK_CONFIG["common"])
    all_config.update(TASK_CONFIG[stage])
    all_config.update(remove_none(args))  # args优先级最高
    all_config["task_name"] = all_config["task_name"].split("/")[-1]
    print(f"task_name={all_config['task_name']}")
    # common path
    all_config["ckpt_dir"] = os.path.abspath(all_config["ckpt_dir"])
    # stats_path可以为""，表示不使用统计数据
    if "/" in all_config["stats_path"]:
        use_stats = True
        all_config["stats_path"] = os.path.abspath(all_config["stats_path"])
    else:
        use_stats = False
    if stage == "train":
        assert use_stats, "now training must use stats"
        # set start joint
        if all_config.get("start_joint", None) is None:
            init_states = get_init_states(all_config["load_data"]["dataset_dir"], 0)
            all_config["start_action"] = init_states[0]
            all_config["start_joint"] = init_states[1]
        # set augmentors
        all_config["load_data"]["augmentors"]["image"] = task_funcs["image_augmentor"]
        all_config["augmentors_flag"] = {
            "image": task_funcs["image_augmentor"].activated
        }
    elif stage == "eval":
        # 评估时需要将自动根据当前时间戳生成的ckpt_dir和stats_path替换为指定时间戳的路径(save_dir不需要替换)
        if args.get("time_stamp", None):
            time_stamp = args["time_stamp"]
            all_config["ckpt_dir"] = replace_timestamp(
                all_config["ckpt_dir"], time_stamp
            )
            all_config["stats_path"] = replace_timestamp(
                all_config["stats_path"], time_stamp
            )
        # 检查路径（支持task_name/ts/task_name/ts两级嵌套和仅task_name/ts一级两种目录结构）
        stats_dir, stats_path = get_stats_path(
            all_config["stats_path"], all_config["task_name"]
        )
        all_config["stats_path"] = stats_path
        # 评估时如果start_joint为AUTO，则从统计数据中读取初始动作
        if all_config["start_joint"] == "AUTO":
            assert use_stats, "start_joint=AUTO requires stats_path"
            # 读取统计数据
            init_states = get_init_states(stats_dir)
            # 设置start_joint为初始action
            all_config["start_joint"] = init_states[1]
        all_config["learning_rate"] = (
            -1
        )  # TODO：there should not be learning_rate in policy_config
    else:
        raise ValueError(f"stage {stage} not supported, must be 'train' or 'eval'")
    # set policy class and config
    all_config["policy_class"] = all_config["policy_config"]["policy_class"]
    policy_config = config_policy(all_config)
    all_config["policy_config"] = policy_config
    # all_config["policy_maker"] = task_funcs["policy_maker"]
    all_config["state_dim"] = policy_config["state_dim"]
    all_config["action_dim"] = policy_config["action_dim"]
    all_config["stage"] = stage
    return all_config
