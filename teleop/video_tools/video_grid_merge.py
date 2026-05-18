#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频网格合并工具
将四个视频文件拼接成2×2布局的大视频
"""

import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

def get_video_info(video_path):
    """获取视频信息"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            info = json.loads(result.stdout)
            # 获取视频流信息
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'video':
                    return {
                        'width': int(stream.get('width', 0)),
                        'height': int(stream.get('height', 0)),
                        'duration': float(info.get('format', {}).get('duration', 0))
                    }
    except Exception as e:
        print(f"获取视频信息失败: {e}")
    return None

def create_2x2_grid_video(video_files, output_path, target_width=1920, target_height=1080):
    """
    创建2×2网格视频
    
    Args:
        video_files: 包含4个视频文件路径的列表
        output_path: 输出文件路径
        target_width: 目标宽度
        target_height: 目标高度
    """
    
    if len(video_files) != 4:
        raise ValueError("需要恰好4个视频文件")
    
    # 计算每个视频的尺寸
    cell_width = target_width // 2
    cell_height = target_height // 2
    
    # 构建ffmpeg命令
    # 使用hstack和vstack来创建2×2网格
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-i', video_files[0],  # 左上
        '-i', video_files[1],  # 右上
        '-i', video_files[2],  # 左下
        '-i', video_files[3],  # 右下
        '-filter_complex', f'''
        [0:v]scale={cell_width}:{cell_height}[v0];
        [1:v]scale={cell_width}:{cell_height}[v1];
        [2:v]scale={cell_width}:{cell_height}[v2];
        [3:v]scale={cell_width}:{cell_height}[v3];
        [v0][v1]hstack[top];
        [v2][v3]hstack[bottom];
        [top][bottom]vstack[out]
        ''',
        '-map', '[out]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    # 执行ffmpeg命令
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 创建进度条
        with tqdm(total=100, desc="合并视频", unit="%") as pbar:
            last_progress = 0
            while True:
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    # 尝试从ffmpeg输出中提取进度信息
                    if 'time=' in output:
                        try:
                            # 解析时间信息来估算进度
                            time_str = output.split('time=')[1].split()[0]
                            # 这里可以进一步解析时间来计算进度
                            # 简化处理，每10%更新一次
                            current_progress = min(95, last_progress + 5)
                            if current_progress > last_progress:
                                pbar.update(current_progress - last_progress)
                                last_progress = current_progress
                        except:
                            pass
        
        # 完成
        pbar.update(100 - last_progress)
        
        if process.returncode == 0:
            print(f"\n✅ 视频合并完成: {output_path}")
            return True
        else:
            print(f"\n❌ 视频合并失败")
            return False
            
    except Exception as e:
        print(f"\n❌ 执行ffmpeg命令时出错: {e}")
        return False

def process_action_folder(action_path, output_dir):
    """处理单个action文件夹"""
    action_name = os.path.basename(action_path)
    print(f"\n📁 处理action文件夹: {action_name}")
    
    # 查找所有episode文件夹
    episode_dirs = []
    for item in os.listdir(action_path):
        item_path = os.path.join(action_path, item)
        if os.path.isdir(item_path) and item.startswith('episode_'):
            episode_dirs.append(item_path)
    
    episode_dirs.sort()  # 按名称排序
    
    if not episode_dirs:
        print(f"⚠️  在 {action_name} 中没有找到episode文件夹")
        return
    
    print(f"找到 {len(episode_dirs)} 个episode文件夹")
    
    # 处理每个episode
    for episode_dir in tqdm(episode_dirs, desc=f"处理 {action_name} 的episodes"):
        episode_name = os.path.basename(episode_dir)
        
        # 查找四个视频文件
        video_files = []
        expected_files = ['camera_0.mp4', 'camera_4.mp4', 'camera_6.mp4', 'camera_11.mp4']
        
        for filename in expected_files:
            file_path = os.path.join(episode_dir, filename)
            if os.path.exists(file_path):
                video_files.append(file_path)
            else:
                print(f"⚠️  缺少文件: {file_path}")
        
        if len(video_files) != 4:
            print(f"⚠️  {episode_name} 中视频文件数量不正确，跳过")
            continue
        
        # 创建输出文件名
        output_filename = f"审查_{action_name}_{episode_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 检查是否已存在
        if os.path.exists(output_path):
            print(f"⏭️  {output_filename} 已存在，跳过")
            continue
        
        print(f"\n🎬 处理: {episode_name}")
        print(f"   输入文件: {[os.path.basename(f) for f in video_files]}")
        print(f"   输出文件: {output_filename}")
        
        # 合并视频
        success = create_2x2_grid_video(video_files, output_path)
        
        if success:
            print(f"✅ {episode_name} 处理完成")
        else:
            print(f"❌ {episode_name} 处理失败")

def main():
    parser = argparse.ArgumentParser(description='将视频文件合并成2×2网格布局')
    parser.add_argument('--input', '-i', default='MP4', help='输入目录路径 (默认: MP4)')
    parser.add_argument('--output', '-o', default='output', help='输出目录路径 (默认: output)')
    parser.add_argument('--action', '-a', help='指定处理单个action文件夹')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input):
        print(f"❌ 输入目录不存在: {args.input}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查ffmpeg是否可用
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 未找到ffmpeg，请确保已安装ffmpeg")
        sys.exit(1)
    
    print("🎬 视频网格合并工具")
    print(f"📁 输入目录: {args.input}")
    print(f"📁 输出目录: {args.output}")
    
    if args.action:
        # 处理单个action文件夹
        action_path = os.path.join(args.input, args.action)
        if not os.path.exists(action_path):
            print(f"❌ action文件夹不存在: {action_path}")
            sys.exit(1)
        process_action_folder(action_path, args.output)
    else:
        # 处理所有action文件夹
        action_dirs = []
        for item in os.listdir(args.input):
            item_path = os.path.join(args.input, item)
            if os.path.isdir(item_path) and item.startswith('action'):
                action_dirs.append(item_path)
        
        action_dirs.sort()
        
        if not action_dirs:
            print(f"⚠️  在 {args.input} 中没有找到action文件夹")
            sys.exit(1)
        
        print(f"找到 {len(action_dirs)} 个action文件夹")
        
        for action_dir in action_dirs:
            process_action_folder(action_dir, args.output)
    
    print("\n🎉 所有处理完成！")

if __name__ == "__main__":
    main() 