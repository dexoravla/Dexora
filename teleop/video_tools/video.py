import cv2
import os
import argparse
import imageio  # 用于生成GIF
from natsort import natsorted

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert camera images to MP4 videos and GIFs.')
    parser.add_argument('action_num', type=str, help='Action number (e.g., 0 for action0)')
    return parser.parse_args()





def create_video_and_gif(image_folder, output_video_path, output_gif_path, fps, should_rotate=False):
    """
    将图片转换为MP4视频和GIF动画，可选是否旋转180°
    :param image_folder: 图片所在文件夹
    :param output_video_path: 输出MP4路径
    :param output_gif_path: 输出GIF路径（为None时不生成GIF）
    :param fps: 帧率
    :param should_rotate: 是否旋转180°（True/False）
    """
    images = [img for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))]
    images = natsorted(images)
    
    if not images:
        print(f"Warning: No image files found in {image_folder}!")
        return
    
    # 读取第一张图片获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read image {first_image_path}!")
        return
    
    height, width, _ = frame.shape
    size = (width, height)
    
    # 创建视频写入器（MP4）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # 准备GIF帧列表
    gif_frames = []
    
    # 逐帧处理图片
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is not None:
            if should_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)  # 旋转180°
            
            # 写入MP4
            video_writer.write(frame)
            
            # 转换颜色空间（BGR→RGB）并添加到GIF帧列表
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(frame_rgb)
        else:
            print(f"Warning: Could not read image {image_path}, skipping frame")
    
    # 释放MP4写入器
    video_writer.release()
    # print(f"MP4 video saved to: {output_video_path}")
    
    # 保存GIF（如果至少有一帧且output_gif_path不为None）
    if output_gif_path is not None and gif_frames:
        imageio.mimsave(output_gif_path, gif_frames, fps=fps, loop=0)  # loop=0表示无限循环
        print(f"GIF animation saved to: {output_gif_path}")

def main():
    args = parse_arguments()
    action_num = args.action_num
    
    input_base_path = f"/home/air/Desktop/wzr/data_collection/action15/episode_{action_num}"
    output_base_path = input_base_path
    fps = 20

    # 相机文件夹与输出文件的对应关系
    camera_mapping = {
        "camera_2": f"left_hand_action{action_num}",  # 需要旋转
        "camera_4": f"opposite_action{action_num}",    # 不需要旋转
        "camera_6": f"right_hand_action{action_num}"  # 需要旋转
    }

    # 处理每个相机文件夹
    for camera_folder, output_prefix in camera_mapping.items():
        input_folder = os.path.join(input_base_path, camera_folder)
        output_mp4_path = os.path.join(output_base_path, f"{output_prefix}.mp4")
        output_gif_path = os.path.join(output_base_path, f"{output_prefix}.gif")
        
        if os.path.exists(input_folder):
            print(f"\nProcessing: {camera_folder}...")
            should_rotate = camera_folder in ["camera_0", "camera_6"]  # 仅旋转camera_0和camera_6
            create_video_and_gif(input_folder, output_mp4_path, output_gif_path, fps, should_rotate)
        else:
            print(f"Warning: Folder {input_folder} does not exist, skipping")

if __name__ == "__main__":
    main()