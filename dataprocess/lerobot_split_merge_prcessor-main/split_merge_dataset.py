#!/usr/bin/env python3
"""
数据集拆分与合并工具 - 修复版本
支持按帧数或episode数量拆分单个数据集，以及合并多个数据集
修复了task_index在episodes_stats.jsonl和tasks.jsonl中的一致性问题

使用示例:
# 拆分模式 - 按episode数量
python split_merge_dataset.py split \
   --input /mnt/nas/synnas/docker2/robocoin-datasets/ruantong_a2d_box_storage_e \
   --output /mnt/nas/synnas/docker2/robocoin-datasets/ruantong_a2d_box_storage_e_fix \
   --max_episodes 58 

# 拆分模式 - 按帧数
python split_merge_dataset.py split \
   --input /home/kemove/robotics-data-processor/lerobot/test \
   --output /home/kemove/robotics-data-processor/lerobot/test \
   --start_episodes 2 \
   --max_entries  \
   --fps 20

# 合并模式
python split_merge_dataset.py merge \
   --sources /home/kemove/robotics-data-processor/lerobot/test1 \
             /home/kemove/robotics-data-processor/lerobot/test2 \
   --output /home/kemove/robotics-data-processor/lerobot/test3 \
   --max_episodes 550 \
   --fps 20 \
   --max_dim 32

# 拆分：从第 100 个 episode 开始取 50 个
python split_merge_dataset.py split \
    --input /mnt/nas/synnas/docker2/robocoin-datasets/realman_rmc_aidal_box_up_down \
    --output /home/kemove/robotics-data-processor/lerobot/box_up_down \
    --start_episodes 2 \
    --max_episodes 300

# 拆分：从第 20,000 帧开始取 10,000 帧（按整 episode 对齐）
python split_merge_dataset.py split \
    --input /path/to/ds \
    --output /path/to/out \
    --start_frames 20000 \
    --max_frames 10000

# 合并：跨多源数据集，从整体第 5,000 帧后开始合并 500 个 episode
python split_merge_dataset.py merge \
    --sources /path/to/ds1 /path/to/ds2 /path/to/ds3 \
    --output /path/to/merged \
    --start_frames 5000 \
    --max_episodes 500

# 合并：扫描一、二级子目录并合并
python split_merge_dataset.py merge \
    --sources_dir /home/kemove/mcap_to_lerobot/Industry_Move_industrial_parts_to_different_plastic_boxes \
    --output /home/kemove/mcap_to_lerobot/Industry_Move_industrial_parts_to_different_plastic_boxes_merged 

# 同时指定部分路径并自动发现其余 
python split_merge_dataset.py merge \
    --sources /data/ds_a /data/ds_b \
    --sources_dir /data/more_datasets \
    --output /data/merge
"""

import argparse
import json
import os
import shutil
import traceback
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# 导入合并相关函数
from lerobot_dataset_lib import (
    load_jsonl,
    save_jsonl,
    copy_videos,
    copy_data_files,
    merge_stats,
    get_info,
    select_episodes,
    write_meta_and_copy,
)

def mode_merge(args: argparse.Namespace):
    """合并模式：将多个数据集合并为一个（CLI 调用库函数）。"""
    source_folders = list(args.sources) if getattr(args, "sources", None) else []
    src_dir = getattr(args, "sources_dir", None)
    if src_dir:
        for name in os.listdir(src_dir):
            p = os.path.join(src_dir, name)
            if not os.path.isdir(p):
                continue
            if os.path.exists(os.path.join(p, "meta", "info.json")):
                source_folders.append(p)
                continue
            for name2 in os.listdir(p):
                p2 = os.path.join(p, name2)
                if os.path.isdir(p2) and os.path.exists(os.path.join(p2, "meta", "info.json")):
                    source_folders.append(p2)
    source_folders = sorted(set(source_folders))
    if not source_folders:
        raise RuntimeError("No valid sources found")
    fps = args.fps if args.fps is not None else get_info(source_folders[0]).get("fps", 20)
    max_episodes = args.max_episodes
    max_dim_cli = args.max_dim
    start_entries = getattr(args, "start_entries", None)
    start_episodes = getattr(args, "start_episodes", None)

    (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        all_tasks,
        all_stats_data,
        total_frames,
    ) = select_episodes(
        source_folders,
        max_entries=None,
        max_episodes=max_episodes,
        start_entries=getattr(args, "start_entries", None),         # 修复：传入起始帧偏移
        start_episodes=getattr(args, "start_episodes", None),      # 修复：传入起始 episode 偏移
    )

    write_meta_and_copy(
        source_folders=source_folders,
        output_folder=args.output,
        episode_mapping=episode_mapping,
        all_episodes=all_episodes,
        all_episodes_stats=all_episodes_stats,
        folder_dimensions=folder_dimensions,
        folder_task_mapping=folder_task_mapping,
        episode_to_frame_index=episode_to_frame_index,
        all_stats_data=all_stats_data,
        all_tasks=all_tasks,
        total_frames=total_frames,
        max_dim_cli=max_dim_cli,
        fps=fps,
    )


def mode_split(args: argparse.Namespace):
    """拆分模式：对单个数据集选择前 N 帧或前 N 个 episode 输出（CLI 调用库函数）。"""
    input_folder = args.input
    fps = args.fps if args.fps is not None else get_info(input_folder).get("fps", 20)
    max_entries = args.max_entries
    max_episodes = args.max_episodes
    max_dim_cli = args.max_dim
    start_entries = getattr(args, "start_entries", None)
    start_episodes = getattr(args, "start_episodes", None)

    source_folders = [input_folder]

    (
        episode_mapping,
        all_episodes,
        all_episodes_stats,
        episode_to_frame_index,
        folder_dimensions,
        folder_task_mapping,
        all_tasks,
        all_stats_data,
        total_frames,
    ) = select_episodes(
        source_folders,
        max_entries=max_entries,
        max_episodes=max_episodes,
        start_entries=start_entries,
        start_episodes=start_episodes,
    )

    write_meta_and_copy(
        source_folders=source_folders,
        output_folder=args.output,
        episode_mapping=episode_mapping,
        all_episodes=all_episodes,
        all_episodes_stats=all_episodes_stats,
        folder_dimensions=folder_dimensions,
        folder_task_mapping=folder_task_mapping,
        episode_to_frame_index=episode_to_frame_index,
        all_stats_data=all_stats_data,
        all_tasks=all_tasks,
        total_frames=total_frames,
        max_dim_cli=max_dim_cli,
        fps=fps,
    )


def main():
    parser = argparse.ArgumentParser(description="数据集拆分与合并 CLI")
    subparsers = parser.add_subparsers(dest="mode", help="操作模式")

    # 拆分模式
    split_parser = subparsers.add_parser("split", help="拆分数据集")
    split_parser.add_argument("--input", required=True, help="输入数据集路径")
    split_parser.add_argument("--output", required=True, help="输出数据集路径")
    split_parser.add_argument("--max_entries", type=int, help="最大帧数限制")
    split_parser.add_argument("--max_episodes", type=int, help="最大episode数量限制")
    split_parser.add_argument("--fps", type=int, help="帧率设置")
    split_parser.add_argument("--max_dim", type=int, help="最大维度设置")
    split_parser.add_argument("--start_entries", type=int, help="起始帧偏移（跳过前N帧）")
    split_parser.add_argument("--start_episodes", type=int, help="起始episode偏移（跳过前N个episode）")

    # 合并模式
    merge_parser = subparsers.add_parser("merge", help="合并数据集")
    merge_parser.add_argument("--sources", nargs="+", required=False, help="源数据集路径列表")
    merge_parser.add_argument("--sources_dir", help="源数据集父目录（扫描一二级子目录）")
    merge_parser.add_argument("--output", required=True, help="输出数据集路径")
    merge_parser.add_argument("--max_episodes", type=int, help="最大episode数量限制")
    merge_parser.add_argument("--fps", type=int, help="帧率设置")
    merge_parser.add_argument("--max_dim", type=int, help="最大维度设置")
    merge_parser.add_argument("--start_entries", type=int, help="起始帧偏移（跳过前N帧）")
    merge_parser.add_argument("--start_episodes", type=int, help="起始episode偏移（跳过前N个episode）")

    args = parser.parse_args()
    if args.mode == "split":
        mode_split(args)
    elif args.mode == "merge":
        mode_merge(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()