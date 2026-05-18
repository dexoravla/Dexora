import cv2
import os
import argparse
from natsort import natsorted

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert camera images to MP4 videos.')
    parser.add_argument('action_num', type=str, help='Action number (e.g., 0 for action0)')
    return parser.parse_args()

def create_video_from_images(image_folder, output_video_path, fps, should_rotate=False):
    """
    将图片转换为视频，可选是否旋转180°
    :param image_folder: 图片所在文件夹
    :param output_video_path: 输出视频路径
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
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    
    # 逐帧处理图片
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is not None:
            if should_rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_180)  # 旋转180°
            out.write(frame)
        else:
            print(f"Warning: Could not read image {image_path}, skipping frame")
    
    out.release()
    print(f"Video saved to: {output_video_path}")

def main():
    args = parse_arguments()
    action_num = args.action_num
    
    input_base_path = f"/home/air/Desktop/wzr/data_collection/action27/episode_{action_num}"
    output_base_path = input_base_path
    fps = 20

    # 相机文件夹与输出视频的对应关系
    camera_mapping = {
        "camera_0": f"head_camera_action{action_num}.mp4",   # 不需要旋转
        "camera_6": f"left_hand_action{action_num}.mp4",  # 需要旋转
        "camera_4": f"opposite_action{action_num}.mp4",   # 不需要旋转
        "camera_11": f"right_hand_action{action_num}.mp4"  # 需要旋转
    }

    # 处理每个相机文件夹
    for camera_folder, video_name in camera_mapping.items():
        input_folder = os.path.join(input_base_path, camera_folder)
        output_path = os.path.join(output_base_path, video_name)
        
        if os.path.exists(input_folder):
            print(f"Processing: {camera_folder}...")
            should_rotate = camera_folder in ["camera_6", "camera_11"]  # 仅旋转camera_0和camera_6
            create_video_from_images(input_folder, output_path, fps, should_rotate)
        else:
            print(f"Warning: Folder {input_folder} does not exist, skipping")

if __name__ == "__main__":
    main()