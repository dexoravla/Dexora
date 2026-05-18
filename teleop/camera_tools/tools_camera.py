import contextlib
import os
import numpy as np
from pyapriltags import Detector

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
import cv2


def open_camera(cam_list):
    if cam_list == []:
        current = 0
        while current < 100:
            cap = cv2.VideoCapture(current)
            if cap.isOpened():
                cam_list.append(current)
            if cap is not None:
                cap.release()
            current += 1

        if cam_list == []:
            print("No available camera found")
            exit()

        print(f"Cameras found: {cam_list}")

    cap = cv2.VideoCapture(cam_list[0])
    if cap.isOpened():
        print(f"Open camera {cam_list[0]}")
    else:
        print(f"Open camera {cam_list[0]} failed")
        exit()
    cam_list.append(cam_list.pop(0))
    return cap


@contextlib.contextmanager
def redirect_stderr():
    original_stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)


def detect_apriltag_pose(detector, image, camera_params, tag_size, tag_id=None):
    """
    使用 pyapriltags 检测 apriltag 并返回姿态矩阵。
    camera_params: [fx, fy, cx, cy] 相机内参
    tag_size: AprilTag 的物理尺寸（米）
    tag_id: 可选的特定tag ID，如果为None则返回第一个检测到的tag
    """
    # apriltag 的输入必须是灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # pyapriltags的detect方法可以直接估计pose，需要传入camera_params和tag_size
    with redirect_stderr():
        try:
            # 将camera_params转换为numpy数组格式: [fx, fy, cx, cy]
            camera_params_array = np.array(camera_params, dtype=np.float64)
            results = detector.detect(gray, estimate_tag_pose=True, 
                                     camera_params=camera_params_array, 
                                     tag_size=tag_size)
        except Exception as e:
            print(f"apriltag.detector.detect 调用异常: {e}")
            return None

    # 检查每个检测结果
    for result in results:
        # 调试输出
        # print(result)
        if tag_id is not None and getattr(result, 'tag_id', None) != tag_id:
            continue
        
        # 检查pose是否成功估计
        if not hasattr(result, 'pose_R') or not hasattr(result, 'pose_t'):
            print("警告: Detection对象缺少pose信息")
            continue
            
        pose_R = result.pose_R  # 3x3旋转矩阵
        pose_t = result.pose_t   # 3x1平移向量
        
        # 构建4x4变换矩阵
        pose = np.eye(4)
        pose[:3, :3] = pose_R
        pose[:3, 3] = pose_t.flatten()
        
        # 检查pose矩阵
        if pose.shape != (4, 4):
            print(f"警告: pose矩阵尺寸异常: {pose.shape}")
            continue
            
        return pose
    return None


def load_reference_image(id):
    reference_image = reference_pose = None
    reference_image = cv2.imread(f"ref{id}.jpg")
    if reference_image is None:
        print(f"No reference image found for camera {id}")
    else:
        # 全局变量 detector, camera_params, tag_size 必须在调用处定义
        reference_pose = detect_apriltag_pose(detector, reference_image, camera_params, tag_size)
        if reference_pose is None:
            print(f"Failed to detect AprilTag in reference image for camera {id}")
    return reference_image, reference_pose


def ncc(imageA, imageB):
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"
    
    meanA = np.mean(imageA)
    meanB = np.mean(imageB)
    
    ncc_value = np.sum((imageA - meanA) * (imageB - meanB))
    ncc_value /= np.sqrt(np.sum((imageA - meanA) ** 2) * np.sum((imageB - meanB) ** 2))
    
    return ncc_value

def calculate_pose_difference(current_pose, reference_pose):
    translation_diff = current_pose[:3, 3] - reference_pose[:3, 3]
    rotation_diff = current_pose[:3, :3] @ np.linalg.inv(reference_pose[:3, :3])

    # cv2.Rodrigues返回旋转向量（非欧拉角）
    rotation_diff_euler = cv2.Rodrigues(rotation_diff)[0].flatten()

    return translation_diff, rotation_diff_euler


def normalize_array(arr, val):
    for i in range(len(arr)):
        arr[i] = np.clip(arr[i] / val, -1, 1)


def draw_adjustment_guidance(frame, translation_diff, rotation_diff_euler,metric=None):
    h, w, _ = frame.shape
    center = (w // 2, h // 2)
    scale = int(h / 3)
    edge_dist = 50
    marker_radius = 5
    line_thickness = 2
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)

    # Translation adjustment indicator (green arrow)
    arrow_length = int(np.linalg.norm(translation_diff[:2]) * scale)
    angle = np.arctan2(translation_diff[1], translation_diff[0])
    end_point = (
        int(center[0] + arrow_length * np.cos(angle)),
        int(center[1] + arrow_length * np.sin(angle)),
    )
    cv2.arrowedLine(frame, center, end_point, GREEN, line_thickness)

    # Z-axis movement indicator
    z_arrow_length = int(abs(translation_diff[2]) * scale)
    z_direction = -1 if translation_diff[2] > 0 else 1
    z_start_point = (edge_dist, center[1])
    z_end_point = (edge_dist, center[1] + z_direction * z_arrow_length)
    cv2.arrowedLine(frame, z_start_point, z_end_point, GREEN, line_thickness)
    cv2.putText(frame, "Z", (10, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)

    # Rotation indicator lines (pitch, yaw, roll) in red
    yaw_line_start = (center[0] - scale, center[1])
    yaw_line_end = (center[0] + scale, center[1])
    cv2.line(frame, yaw_line_start, yaw_line_end, RED, line_thickness, cv2.LINE_AA)
    text_pos = (center[0] + scale + 20, center[1])
    cv2.putText(frame, "Yaw", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
    yaw_marker = np.array([int(center[0] + rotation_diff_euler[1] * scale), center[1]])
    marker_pos = (yaw_marker - [0, marker_radius], yaw_marker + [0, marker_radius])
    cv2.line(frame, *marker_pos, RED, line_thickness, cv2.LINE_AA)

    pitch_line_start = (center[0], center[1] - scale)
    pitch_line_end = (center[0], center[1] + scale)
    cv2.line(frame, pitch_line_start, pitch_line_end, RED, line_thickness)
    text_pos = (center[0] + 20, center[1] - scale - 10)
    cv2.putText(frame, "Pitch", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
    pitch_marker = np.array([center[0], int(center[1] - rotation_diff_euler[0] * scale)])
    marker_pos = (pitch_marker - [marker_radius, 0], pitch_marker + [marker_radius, 0])
    cv2.line(frame, *marker_pos, RED, line_thickness, cv2.LINE_AA)

    tilt_scale = int(scale / np.sqrt(2))
    roll_line_start = (center[0] - tilt_scale, center[1] - tilt_scale)
    roll_line_end = (center[0] + tilt_scale, center[1] + tilt_scale)
    cv2.line(frame, roll_line_start, roll_line_end, RED, line_thickness)
    text_pos = (center[0] + tilt_scale + 20, center[1] + tilt_scale + 20)
    cv2.putText(frame, "Roll", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2, cv2.LINE_AA)
    roll_marker = np.array(
        [
            int(center[0] + rotation_diff_euler[2] * tilt_scale),
            int(center[1] + rotation_diff_euler[2] * tilt_scale),
        ]
    )
    tilt_marker_radius = (np.array([marker_radius, -marker_radius]) / np.sqrt(2)).astype(int)
    marker_pos = (roll_marker - tilt_marker_radius, roll_marker + tilt_marker_radius)
    cv2.line(frame, *marker_pos, RED, line_thickness, cv2.LINE_AA)

    if metric is not None:
        cv2.putText(frame, f"NCC: {metric:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2, cv2.LINE_AA)


if __name__ == "__main__":
    # pyapriltags的Detector直接在初始化时传入参数，不需要DetectorOptions
    detector = Detector(families="tag36h11")
    tag_size = 0.1  # AprilTag大小(单位:米)，注意：精度要和现场一致

    # D455 RGB
    # camera_params = [384.904, 384.387, 321.351, 243.619]  # [fx, fy, cx, cy]
    # camera_width = 1280
    # camera_height = 800

    # Default parameters
    camera_params = [1000, 1000, 500, 500]
    camera_width = 10000
    camera_height = 10000

    cam_list = [9]
    cap = open_camera(cam_list)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    reference_image, reference_pose = load_reference_image(cam_list[-1])
    overlay = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 检查 apriltag 检测过程的返回值和运行时输出
        current_pose = detect_apriltag_pose(detector, frame, camera_params, tag_size)
        # 会把检测细节和异常打印出来

        ncc_value = None
        if overlay and reference_image is not None:
            frame = np.mean([reference_image, frame], axis=0).astype(np.uint8)
            ncc_value = ncc(reference_image, frame)

        if current_pose is not None and reference_pose is not None:
            translation_diff, rotation_diff_euler = calculate_pose_difference(
                current_pose, reference_pose
            )

            normalize_array(translation_diff, 0.2)
            normalize_array(rotation_diff_euler, 0.2)
            draw_adjustment_guidance(frame, translation_diff, rotation_diff_euler, ncc_value)

        cv2.imshow("Current Camera View", frame)

        # q 退出，o 切换叠加，s 保存新参考，n 换相机
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("o"):
            overlay = not overlay
        elif key == ord("s"):
            ret, reference_image = cap.read()
            cv2.imwrite(f"ref{cam_list[-1]}.jpg", reference_image)
            reference_pose = detect_apriltag_pose(
                detector, reference_image, camera_params, tag_size
            )
            print(
                f"Reference image updated for camera {cam_list[-1]}{'' if reference_pose is not None else ' (but failed to detect AprilTag)'}"
            )
        elif key == ord("n"):
            cap.release()
            cap = open_camera(cam_list)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            reference_image, reference_pose = load_reference_image(cam_list[-1])

    cap.release()
    cv2.destroyAllWindows()
