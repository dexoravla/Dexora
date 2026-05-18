import argparse
import os
from configurations.task_configs.config_tools.basic_configer import (
    get_all_config,
    replace_timestamp,
    get_stats_path,
)
from utils.utils import get_pkl_info
from pprint import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tn", "--task_name", action="store", type=str, help="task_name", required=True
    )
    parser.add_argument(
        "-ts",
        "--time_stamp",
        action="store",
        type=str,
        help="time_stamp",
        required=True,
    )
    parser.add_argument(
        "-in",
        "--info_name",
        action="store",
        type=str,
        help="name for showing the info",
        required=True,
    )
    args = vars(parser.parse_args())
    name = args["info_name"]
    time_stamp = args["time_stamp"]
    # get stats path
    all_config = get_all_config(args, "eval")
    stats_path = all_config["stats_path"]
    # show info
    if name in ["key_info", "key_info.pkl","all"]:
        key_info = get_pkl_info(os.path.dirname(stats_path) + "/key_info.pkl")
        pprint(key_info)
    elif name in ["dataset_stats", "dataset_stats.pkl", "stats", "all"]:
        stats = get_pkl_info(stats_path)
        pprint(stats)