#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频审查生成器
直接从图片生成2×2布局的审查视频，跳过中间的四视频步骤
"""

import os
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import traceback

def create_video_from_images(image_folder, output_video_path, fps=20, should_rotate=False):
    """
    从图片文件夹创建视频
    
    Args:
        image_folder: 图片文件夹路径
        output_video_path: 输出视频路径
        fps: 帧率
        should_rotate: 是否需要旋转
    """
    try:
        # 检查图片文件夹是否存在
        if not os.path.exists(image_folder):
            print(f"❌ 图片文件夹不存在: {image_folder}")
            return False
        
        # 获取所有图片文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_folder).glob(f'*{ext}'))
            image_files.extend(Path(image_folder).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"❌ 在 {image_folder} 中没有找到图片文件")
            return False
        
        # 按文件名排序
        image_files.sort()
        
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', os.path.join(image_folder, 'frame_*.jpg'),  # 适配frame_XXXXXX.jpg格式
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p'
        ]
        
        # 如果需要旋转，添加旋转滤镜
        if should_rotate:
            cmd.extend(['-vf', 'transpose=1'])
        
        cmd.append(output_video_path)
        
        # 执行ffmpeg命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"❌ 创建视频失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 创建视频时出错: {e}")
        return False

def create_grid_video(video_files, output_path, target_width=1920, target_height=1080):
    """
    创建网格视频，支持3-4个视频文件
    
    Args:
        video_files: 包含3-4个视频文件路径的列表
        output_path: 输出文件路径
        target_width: 目标宽度
        target_height: 目标高度
    """
    
    if len(video_files) < 3 or len(video_files) > 4:
        raise ValueError("需要3-4个视频文件")
    
    # 根据视频数量决定布局
    if len(video_files) == 4:
        # 2×2布局
        cell_width = target_width // 2
        cell_height = target_height // 2
        
        filter_complex = f'''
        [0:v]scale={cell_width}:{cell_height}[v0];
        [1:v]scale={cell_width}:{cell_height}[v1];
        [2:v]scale={cell_width}:{cell_height}[v2];
        [3:v]scale={cell_width}:{cell_height}[v3];
        [v0][v1]hstack[top];
        [v2][v3]hstack[bottom];
        [top][bottom]vstack[out]
        '''
    else:
        # 3个视频：2×1.5布局（上面2个，下面1个居中）
        cell_width = target_width // 2
        cell_height = target_height // 2
        
        filter_complex = f'''
        [0:v]scale={cell_width}:{cell_height}[v0];
        [1:v]scale={cell_width}:{cell_height}[v1];
        [2:v]scale={target_width}:{cell_height}[v2];
        [v0][v1]hstack[top];
        [top][v2]vstack[out]
        '''
    
    # 构建ffmpeg命令
    cmd = ['ffmpeg', '-y']  # 覆盖输出文件
    
    # 添加输入文件
    for video_file in video_files:
        cmd.extend(['-i', video_file])
    
    # 添加滤镜和输出参数
    cmd.extend([
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ])
    
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

def process_episode_from_images(episode_path, output_dir, fps=20):
    """
    直接从图片处理单个episode，生成2×2审查视频
    
    Args:
        episode_path: episode图片文件夹路径
        output_dir: 输出目录
        fps: 帧率
    """
    episode_name = os.path.basename(episode_path)
    
    # 动态查找camera文件夹
    camera_folders = []
    for item in os.listdir(episode_path):
        item_path = os.path.join(episode_path, item)
        if os.path.isdir(item_path) and item.startswith('camera_'):
            camera_folders.append((item, item_path))
    
    # 按camera编号排序
    camera_folders.sort(key=lambda x: int(x[0].split('_')[1]))
    
    if len(camera_folders) < 3:
        print(f"⚠️  {episode_name} 中camera文件夹数量不足，需要至少3个，找到{len(camera_folders)}个")
        return False
    elif len(camera_folders) > 4:
        print(f"⚠️  {episode_name} 中camera文件夹数量过多，只使用前4个")
        camera_folders = camera_folders[:4]
    
    print(f"   使用camera: {[name for name, _ in camera_folders]}")
    
    # 创建临时视频文件路径
    temp_videos = []
    for camera_name, camera_path in camera_folders:
        temp_video = os.path.join(output_dir, f"temp_{camera_name}.mp4")
        temp_videos.append(temp_video)
    
    try:
        # 为每个camera创建临时视频
        print(f"\n🎬 处理: {episode_name}")
        print(f"   输入文件夹: {[name for name, _ in camera_folders]}")
        
        # 创建临时视频
        for i, (camera_name, camera_path) in enumerate(camera_folders):
            temp_video = temp_videos[i]
            # 根据camera编号判断是否需要旋转（通常是较大的编号需要旋转）
            camera_num = int(camera_name.split('_')[1])
            should_rotate = camera_num >= 6  # camera_6及以上需要旋转
            
            print(f"   生成临时视频: {camera_name}")
            success = create_video_from_images(
                image_folder=camera_path,
                output_video_path=temp_video,
                fps=fps,
                should_rotate=should_rotate
            )
            
            if not success:
                print(f"❌ 创建临时视频失败: {camera_name}")
                return False
        
        # 创建输出文件名
        output_filename = f"审查_{episode_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        # 检查是否已存在
        if os.path.exists(output_path):
            print(f"⏭️  {output_filename} 已存在，跳过")
            return True
        
        # 合并成网格视频
        layout_text = "2×2网格" if len(temp_videos) == 4 else "2×1.5网格"
        print(f"   合并为{layout_text}视频: {output_filename}")
        success = create_grid_video(temp_videos, output_path)
        
        if success:
            print(f"✅ {episode_name} 处理完成")
            return True
        else:
            print(f"❌ {episode_name} 处理失败")
            return False
            
    finally:
        # 清理临时文件
        for temp_video in temp_videos:
            if os.path.exists(temp_video):
                os.remove(temp_video)

def process_action_folder(action_path, output_base_dir, fps=20):
    """处理单个action文件夹"""
    action_name = os.path.basename(action_path)
    print(f"\n📁 处理action文件夹: {action_name}")
    
    # 创建对应的输出action目录
    output_action_dir = os.path.join(output_base_dir, action_name)
    os.makedirs(output_action_dir, exist_ok=True)
    
    # 查找所有episode文件夹
    episode_dirs = []
    for item in os.listdir(action_path):
        item_path = os.path.join(action_path, item)
        if os.path.isdir(item_path) and (item.startswith('episode_') or item.startswith('episode')):
            episode_dirs.append(item_path)
    
    episode_dirs.sort()  # 按名称排序
    
    if not episode_dirs:
        print(f"⚠️  在 {action_name} 中没有找到episode文件夹")
        return
    
    print(f"找到 {len(episode_dirs)} 个episode文件夹")
    print(f"输出目录: {output_action_dir}")
    
    # 处理每个episode
    success_count = 0
    for episode_dir in tqdm(episode_dirs, desc=f"处理 {action_name} 的episodes"):
        success = process_episode_from_images(episode_dir, output_action_dir, fps)
        if success:
            success_count += 1
    
    print(f"✅ {action_name} 处理完成: {success_count}/{len(episode_dirs)} 个episode成功")

def main():
    parser = argparse.ArgumentParser(description='直接从图片生成2×2布局的审查视频')
    parser.add_argument('--input', '-i', default='/media/air/7a64a8f1-f098-4ddb-8ead-e2fc98703fef/data', help='输入目录路径 (默认: /media/air/7a64a8f1-f098-4ddb-8ead-e2fc98703fef/data)')
    parser.add_argument('--output', '-o', default='review_videos', help='输出目录路径 (默认: review_videos)')
    parser.add_argument('--action', '-a', help='指定处理单个action文件夹')
    parser.add_argument('--fps', '-f', type=int, default=20, help='视频帧率 (默认: 20)')
    
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
    
    print("🎬 视频审查生成器")
    print(f"📁 输入目录: {args.input}")
    print(f"📁 输出目录: {args.output}")
    print(f"🎞️  帧率: {args.fps} FPS")
    
    try:
        if args.action:
            # 处理单个action文件夹
            action_path = os.path.join(args.input, args.action)
            if not os.path.exists(action_path):
                print(f"❌ action文件夹不存在: {action_path}")
                sys.exit(1)
            process_action_folder(action_path, args.output, args.fps)
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
                process_action_folder(action_dir, args.output, args.fps)
        
        print("\n🎉 所有处理完成！")
        
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 