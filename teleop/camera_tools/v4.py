import cv2
import os
import argparse
import imageio
from natsort import natsorted

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert camera images to GIFs.')
    parser.add_argument('action_num', type=str, help='Action number (e.g., 0 for action0)')
    return parser.parse_args()

def create_gif(image_folder, output_gif_path, original_fps, should_rotate=False):
    """
    将图片转换为GIF动画，可选是否旋转180°
    :param image_folder: 图片所在文件夹
    :param output_gif_path: 输出GIF路径
    :param original_fps: 原始帧率(用于计算采样间隔)
    :param should_rotate: 是否旋转180°（True/False）
    """
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    images = natsorted(images)
    
    if not images:
        print(f"Warning: No image files found in {image_folder}!")
        return
    
    # 准备GIF帧列表
    gif_frames = []
    
    # 计算采样间隔 (20fps → 4fps = 每5帧取1帧)
    sample_interval = 5
    target_fps = max(1, original_fps // sample_interval)  # 确保至少1fps
    
    # 逐帧处理图片
    for i, image_name in enumerate(images):
        if i % sample_interval != 0:  # 跳过非采样帧
            continue
            
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is not None:
            if should_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)  # 旋转180°
            
            # 转换颜色空间（BGR→RGB）并添加到GIF帧列表
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(frame_rgb)
        else:
            print(f"Warning: Could not read image {image_path}, skipping frame")
    
    # 保存GIF（如果至少有一帧）
    if gif_frames:
        imageio.mimsave(output_gif_path, gif_frames, fps=target_fps, loop=0)  # loop=0表示无限循环
        print(f"Compressed GIF ({target_fps}fps) saved to: {output_gif_path}")
    else:
        print(f"Error: No valid frames to create GIF")

def main():
    args = parse_arguments()
    action_num = args.action_num
    
    input_base_path = f"/home/air/Desktop/wzr/data_collection/action8/episode_{action_num}"
    output_base_path = "/home/air/action"
    original_fps = 20  # 原始帧率

    # 相机文件夹与输出文件的对应关系
    camera_mapping = {
        "camera_0": f"gif_action{action_num}_left_hand",  # 需要旋转
        "camera_2": f"gif_action{action_num}_opposite",   # 不需要旋转
        "camera_6": f"gif_action{action_num}_right_hand" # 需要旋转
    }

    # 处理每个相机文件夹
    for camera_folder, output_prefix in camera_mapping.items():
        input_folder = os.path.join(input_base_path, camera_folder)
        output_gif_path = os.path.join(output_base_path, f"{output_prefix}.gif")
        
        if os.path.exists(input_folder):
            print(f"\nProcessing: {camera_folder}...")
            should_rotate = camera_folder in ["camera_0", "camera_6"]  # 仅旋转camera_0和camera_6
            create_gif(input_folder, output_gif_path, original_fps, should_rotate)
        else:
            print(f"Warning: Folder {input_folder} does not exist, skipping")

if __name__ == "__main__":
    main()