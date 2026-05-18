#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性验证脚本
验证每个action文件夹下4个相机文件夹的图片数量和bson文件的数据帧数是否一致
记录每个action的每条数据里面有多少帧的数据
"""

import os
import bson
import glob
from pathlib import Path
from collections import defaultdict
import argparse
import json
from datetime import datetime


def count_images_in_camera_folder(camera_folder_path):
    if not os.path.exists(camera_folder_path):
        return 0
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_count = 0
    for ext in image_extensions:
        image_count += len(glob.glob(os.path.join(camera_folder_path, ext)))
    return image_count


def count_bson_data_frames(bson_file_path):
    if not os.path.exists(bson_file_path):
        return {}
    try:
        with open(bson_file_path, 'rb') as f:
            data = list(bson.decode_file_iter(f))
        if not data:
            return {}
        record = data[0]
        result = {}
        if 'frames' in record and isinstance(record['frames'], list):
            result['frames_count'] = len(record['frames'])
        if 'data' in record and isinstance(record['data'], dict):
            result['data_fields'] = {}
            for key, value in record['data'].items():
                if isinstance(value, list):
                    result['data_fields'][key] = len(value)
        return result
    except Exception as e:
        print(f"警告: 无法读取bson文件 {bson_file_path}: {e}")
        return {}


def get_camera_folders(episode_path):
    camera_folders = []
    expected_cameras = ['camera0_head', 'camera1_left_wrist', 'camera2_right_wrist', 'camera3_third_view']
    for camera in expected_cameras:
        camera_path = os.path.join(episode_path, camera)
        if os.path.exists(camera_path):
            camera_folders.append(camera_path)
        else:
            print(f"警告: 相机文件夹不存在: {camera_path}")
    return camera_folders


def validate_episode(episode_path):
    result = {
        'episode_path': episode_path,
        'camera_image_counts': {},
        'bson_data_frames': {},
        'camera_consistent': True,
        'bson_consistent': True,
        'errors': []
    }
    camera_folders = get_camera_folders(episode_path)
    if len(camera_folders) != 4:
        result['errors'].append(f"相机文件夹数量不正确: 期望4个，实际{len(camera_folders)}个")
        result['camera_consistent'] = False
        return result
    image_counts = []
    for camera_folder in camera_folders:
        camera_name = os.path.basename(camera_folder)
        image_count = count_images_in_camera_folder(camera_folder)
        result['camera_image_counts'][camera_name] = image_count
        image_counts.append(image_count)
    if len(set(image_counts)) > 1:
        result['camera_consistent'] = False
        result['errors'].append(f"相机图片数量不一致: {dict(zip([os.path.basename(f) for f in camera_folders], image_counts))}")
    bson_files = ['episode_0.bson', 'xhand_control_data.bson']
    bson_frame_counts = []
    for bson_file in bson_files:
        bson_path = os.path.join(episode_path, bson_file)
        if os.path.exists(bson_path):
            frame_info = count_bson_data_frames(bson_path)
            result['bson_data_frames'][bson_file] = frame_info
            if 'frames_count' in frame_info:
                bson_frame_counts.append(frame_info['frames_count'])
            elif 'data_fields' in frame_info and frame_info['data_fields']:
                first_field_count = list(frame_info['data_fields'].values())[0]
                bson_frame_counts.append(first_field_count)
        else:
            result['errors'].append(f"bson文件不存在: {bson_path}")
            result['bson_consistent'] = False
    if image_counts and bson_frame_counts:
        expected_count = image_counts[0]
        for i, bson_count in enumerate(bson_frame_counts):
            if bson_count != expected_count:
                result['bson_consistent'] = False
                result['errors'].append(f"{bson_files[i]}中的数据帧数({bson_count})与图片数量({expected_count})不一致")
    return result


def validate_action_folder(action_path):
    result = {
        'action_path': action_path,
        'episodes': [],
        'total_episodes': 0,
        'consistent_episodes': 0,
        'inconsistent_episodes': 0,
        'summary': {}
    }
    episode_folders = []
    for item in os.listdir(action_path):
        item_path = os.path.join(action_path, item)
        if os.path.isdir(item_path) and item.startswith('episode_'):
            episode_folders.append(item_path)
    episode_folders.sort()
    result['total_episodes'] = len(episode_folders)
    if result['total_episodes'] == 0:
        result['summary']['error'] = "未找到episode文件夹"
        return result
    for episode_path in episode_folders:
        episode_result = validate_episode(episode_path)
        result['episodes'].append(episode_result)
        if episode_result['camera_consistent'] and episode_result['bson_consistent']:
            result['consistent_episodes'] += 1
        else:
            result['inconsistent_episodes'] += 1
    if result['consistent_episodes'] == result['total_episodes']:
        result['summary']['status'] = "完全一致"
    elif result['consistent_episodes'] > 0:
        result['summary']['status'] = "部分一致"
    else:
        result['summary']['status'] = "完全不一致"
    return result


def save_log_to_file(results, log_file_path):
    """
    只将有问题的action和episode输出到日志文件
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"数据一致性验证日志\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(f"总体统计:\n")
        f.write(f"  总action文件夹数: {results['total_actions']}\n")
        f.write(f"  完全一致的action: {results['consistent_actions']}\n")
        f.write(f"  存在不一致的action: {results['inconsistent_actions']}\n\n")
        for action in results['actions']:
            # 只输出有不一致episode的action
            inconsistent_episodes = [ep for ep in action['episodes'] if not (ep['camera_consistent'] and ep['bson_consistent'])]
            if not inconsistent_episodes:
                continue
            action_name = os.path.basename(action['action_path'])
            f.write(f"Action: {action_name}\n")
            f.write(f"  总episode数: {action['total_episodes']}\n")
            f.write(f"  一致的episode: {action['consistent_episodes']}\n")
            f.write(f"  不一致的episode: {action['inconsistent_episodes']}\n")
            for episode in inconsistent_episodes:
                episode_name = os.path.basename(episode['episode_path'])
                f.write(f"    Episode: {episode_name}\n")
                f.write(f"      相机图片数量:\n")
                for camera, count in episode['camera_image_counts'].items():
                    f.write(f"        {camera}: {count} 张\n")
                f.write(f"      BSON数据帧数:\n")
                for bson_file, frame_info in episode['bson_data_frames'].items():
                    f.write(f"        {bson_file}:\n")
                    if 'frames_count' in frame_info:
                        f.write(f"          frames: {frame_info['frames_count']} 条\n")
                    if 'data_fields' in frame_info:
                        for field, count in frame_info['data_fields'].items():
                            f.write(f"          {field}: {count} 条\n")
                if episode['errors']:
                    f.write(f"      错误信息:\n")
                    for error in episode['errors']:
                        f.write(f"        - {error}\n")
                f.write("\n")
            f.write("-" * 60 + "\n\n")


def validate_all_actions(base_path="."):
    results = {
        'total_actions': 0,
        'consistent_actions': 0,
        'inconsistent_actions': 0,
        'actions': []
    }
    action_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.startswith('action'):
            action_folders.append(item_path)
    action_folders.sort()
    results['total_actions'] = len(action_folders)
    print(f"找到 {results['total_actions']} 个action文件夹")
    for action_path in action_folders:
        action_name = os.path.basename(action_path)
        print(f"正在验证 {action_name}...")
        action_result = validate_action_folder(action_path)
        results['actions'].append(action_result)
        if action_result['inconsistent_episodes'] == 0:
            results['consistent_actions'] += 1
        else:
            results['inconsistent_actions'] += 1
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(base_path, 'logs', f'data_validation_{timestamp}.log')
    save_log_to_file(results, log_file_path)
    print(f"\n验证日志已保存到: {log_file_path}")
    return results


def print_detailed_report(results):
    print("\n" + "="*80)
    print("数据一致性验证报告")
    print("="*80)
    print(f"\n总体统计:")
    print(f"  总action文件夹数: {results['total_actions']}")
    print(f"  完全一致的action: {results['consistent_actions']}")
    print(f"  存在不一致的action: {results['inconsistent_actions']}")
    inconsistent_actions = [a for a in results['actions'] if a['inconsistent_episodes'] > 0]
    if inconsistent_actions:
        print(f"\n不一致的action文件夹详情:")
        print("-" * 60)
        for action in inconsistent_actions:
            action_name = os.path.basename(action['action_path'])
            print(f"\n{action_name}:")
            print(f"  总episode数: {action['total_episodes']}")
            print(f"  一致的episode: {action['consistent_episodes']}")
            print(f"  不一致的episode: {action['inconsistent_episodes']}")
            for episode in action['episodes']:
                if not (episode['camera_consistent'] and episode['bson_consistent']):
                    episode_name = os.path.basename(episode['episode_path'])
                    print(f"    {episode_name}:")
                    if not episode['camera_consistent']:
                        print(f"      相机图片数量: {episode['camera_image_counts']}")
                    if not episode['bson_consistent']:
                        print(f"      bson数据帧数: {episode['bson_data_frames']}")
                    for error in episode['errors']:
                        print(f"      错误: {error}")
    else:
        print(f"\n✅ 所有action文件夹的数据都是一致的！")


def main():
    parser = argparse.ArgumentParser(description='验证action文件夹数据一致性')
    parser.add_argument('--path', '-p', default='/media/air/data', 
                       help='要验证的基础路径 (默认: 当前目录)')
    parser.add_argument('--action', '-a', 
                       help='只验证指定的action文件夹 (例如: action8)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='显示详细输出')
    args = parser.parse_args()
    if args.action:
        action_path = os.path.join(args.path, args.action)
        if not os.path.exists(action_path):
            print(f"错误: action文件夹不存在: {action_path}")
            return
        print(f"验证单个action文件夹: {args.action}")
        result = validate_action_folder(action_path)
        print(f"\n{args.action} 验证结果:")
        print(f"  总episode数: {result['total_episodes']}")
        print(f"  一致的episode: {result['consistent_episodes']}")
        print(f"  不一致的episode: {result['inconsistent_episodes']}")
        if args.verbose:
            for episode in result['episodes']:
                if not (episode['camera_consistent'] and episode['bson_consistent']):
                    episode_name = os.path.basename(episode['episode_path'])
                    print(f"\n  {episode_name} 不一致:")
                    for error in episode['errors']:
                        print(f"    {error}")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = os.path.join(args.path, 'logs', f'action_{args.action}_validation_{timestamp}.log')
        # 只保留有问题的episode
        filtered_episodes = [ep for ep in result['episodes'] if not (ep['camera_consistent'] and ep['bson_consistent'])]
        single_action_results = {
            'total_actions': 1,
            'consistent_actions': 1 if result['inconsistent_episodes'] == 0 else 0,
            'inconsistent_actions': 1 if result['inconsistent_episodes'] > 0 else 0,
            'actions': [{
                'action_path': result['action_path'],
                'episodes': filtered_episodes,
                'total_episodes': result['total_episodes'],
                'consistent_episodes': result['consistent_episodes'],
                'inconsistent_episodes': result['inconsistent_episodes'],
                'summary': result['summary']
            }]
        }
        save_log_to_file(single_action_results, log_file_path)
        print(f"\n验证日志已保存到: {log_file_path}")
    else:
        print("开始验证所有action文件夹...")
        results = validate_all_actions(args.path)
        print_detailed_report(results)


if __name__ == "__main__":
    main()