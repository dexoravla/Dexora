import time
import numpy as np
import os
from pathlib import Path
import argparse
from typing import Optional, Union, Any
from image_processor import save_images_concurrently, encode_video_frames


iMPORT_ERRORS = []
try:
    import pyrealsense2 as rs
except ImportError:
    iMPORT_ERRORS.append("pyrealsense2")
    print("pyrealsense2 not found. You can install it with `pip install pyrealsense2`")

try:
    import cv2
except ImportError:
    iMPORT_ERRORS.append("cv2")
    print("cv2 not found. You can install it with `pip install opencv-python`")

try:
    from pyorbbecsdk import (
        Config,
        VideoStreamProfile,
        OBError,
        OBSensorType,
        FormatConvertFilter,
        VideoFrame,
        OBFormat,
        OBConvertFormat,
        Pipeline,
        FrameSet,
    )
except Exception as e:
    iMPORT_ERRORS.append("pyorbbecsdk")
    print(f"Error with pyorbbecsdk package: {e}")

class RealsenseCamera(object):
    """
    realsense 相机处理类
    """

    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, fps
        )
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, fps
        )
        # self.align = rs.align(rs.stream.color) # depth2rgb
        self.pipeline.start(self.config)  # 开始连接相机

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()  # 获得frame (包括彩色，深度图)
        # 创建对齐对象
        align_to = rs.stream.color  # rs.align允许我们执行深度帧与其他帧的对齐
        align = rs.align(align_to)  # “align_to”是我们计划对齐深度帧的流类型。
        aligned_frames = align.process(frames)
        # 获取对齐的帧
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame是对齐的深度图
        color_frame = aligned_frames.get_color_frame()
        colorizer = rs.colorizer()
        depthx_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorizer_depth = np.asanyarray(
            colorizer.colorize(aligned_depth_frame).get_data()
        )
        return color_image, depthx_image, colorizer_depth

    def release(self):
        self.pipeline.stop()


class OrbbecCamera(object):
    """
    orbbec 相机处理类
    """

    def __init__(self, width, height="auto", fps="max") -> None:
        self.config = Config()
        self.pipeline = Pipeline()
        profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        assert width in [640, 1280, 1920], "Width must be 640, 1280, or 1920"
        fps = 30 if fps == "max" else fps
        try:
            color_profile: VideoStreamProfile = profile_list.get_video_stream_profile(width, 0, OBFormat.RGB, fps)
        except OBError as e:
            print(e)
            color_profile = profile_list.get_default_video_stream_profile()
            # print("color profile: ", color_profile)
        self.config.enable_stream(color_profile)
        self.pipeline.start(self.config)

    def get_color_image(self):
        frames: FrameSet = self.pipeline.wait_for_frames(100)
        if frames is None:
            return None
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
        # covert to RGB format
        color_image = self.frame_to_bgr_image(color_frame)
        return color_image

    @staticmethod
    def determine_convert_format(frame: VideoFrame):
        if frame.get_format() == OBFormat.I420:
            return OBConvertFormat.I420_TO_RGB888
        elif frame.get_format() == OBFormat.MJPG:
            return OBConvertFormat.MJPG_TO_RGB888
        elif frame.get_format() == OBFormat.YUYV:
            return OBConvertFormat.YUYV_TO_RGB888
        elif frame.get_format() == OBFormat.NV21:
            return OBConvertFormat.NV21_TO_RGB888
        elif frame.get_format() == OBFormat.NV12:
            return OBConvertFormat.NV12_TO_RGB888
        elif frame.get_format() == OBFormat.UYVY:
            return OBConvertFormat.UYVY_TO_RGB888
        else:
            return None

    @classmethod
    def frame_to_rgb_frame(cls, frame: VideoFrame) -> Union[Optional[VideoFrame], Any]:
        if frame.get_format() == OBFormat.RGB:
            return frame
        convert_format = cls.determine_convert_format(frame)
        if convert_format is None:
            print("Unsupported format")
            return None
        print("covert format: {}".format(convert_format))
        convert_filter = FormatConvertFilter()
        convert_filter.set_format_convert_format(convert_format)
        rgb_frame = convert_filter.process(frame)
        if rgb_frame is None:
            print("Convert {} to RGB failed".format(frame.get_format()))
        return rgb_frame


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--imgs_dir", type=str, default="data/Realsense/images")
    parser.add_argument(
        "-out", "--video_dir", type=str, default="data/Realsense/rgb_data"
    )
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--vcodec", type=str, default="libx264")
    parser.add_argument("--pix_fmt", type=str, default="yuv420p")
    parser.add_argument("--g", type=int, default=2)
    parser.add_argument("--crf", type=int, default=30)
    parser.add_argument("--fast_decode", type=int, default=0)
    parser.add_argument("--log_level", type=str, default="error")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-ct", "--camera_type", type=str, default="realsense")
    args = parser.parse_args()
    # 视频保存路径
    imgs_dir = args.imgs_dir
    video_dir = args.video_dir
    fps = args.fps
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs("data/Realsense/depth_data", exist_ok=True)
    os.makedirs("data/Realsense/depthcolor_data", exist_ok=True)
    os.makedirs("data/Realsense/camera_colordepth", exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    video_path = f"{video_dir}/{int(time.time())}.mp4"
    video_depth_path = f"data/Realsense/depth_data/{int(time.time())}_depth.mp4"
    video_depthcolor_path = (
        f"data/Realsense/depthcolor_data/{int(time.time())}_depthcolor.mp4"
    )
    video_depthcolor_camera_path = (
        f"data/Realsense/camera_colordepth/{int(time.time())}_depthcolor.mp4"
    )
    # 初始化参数
    w, h = 640, 480
    mp4 = cv2.VideoWriter_fourcc(*"mp4v")  # 视频格式
    # 视频保存而建立对象
    color_images = []
    # wr_color = cv2.VideoWriter(video_path, mp4, fps, (w, h), isColor=True)
    wr_depth = cv2.VideoWriter(video_depth_path, mp4, fps, (w, h), isColor=False)
    wr_depthcolor = cv2.VideoWriter(
        video_depthcolor_path, mp4, fps, (w, h), isColor=True
    )
    wr_camera_colordepth = cv2.VideoWriter(
        video_depthcolor_camera_path, mp4, fps, (w, h), isColor=True
    )
    # 初始化相机
    if args.camera_type == "orbbec":
        cam = OrbbecCamera(w, h, fps)
    elif args.camera_type == "realsense":
        cam = RealsenseCamera(w, h, fps)
    print("录制视频请按: s, 保存视频或退出请按：q")
    flag_V = 0
    cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
    while True:
        # 读取图像帧，包括RGB图和深度图
        if args.camera_type == "orbbec":
            color_image = cam.get_color_image()
        elif args.camera_type == "realsense":
            color_image, depth_image, colorizer_depth = cam.get_frame()
            # 深度图数据格式转换，uint16→uint8
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
        # images = np.hstack((color_image, depth_colormap))
        cv2.imshow("RealSense", color_image)

        key = cv2.waitKey(1)
        if key == ord("s"):
            flag_V = 1
            print("...录制视频中...")
        elif key == ord("q"):
            print(
                f"Saving images concurrently to {imgs_dir}, frames:", len(color_images)
            )
            images_array = np.array(color_images)
            save_images_concurrently(
                images_array, out_dir=Path(imgs_dir), max_workers=4
            )
            # Encode a video from a directory of images
            encode_video_frames(
                Path(imgs_dir),
                Path(video_path),
                fps,
                args.vcodec,
                args.pix_fmt,
                args.g,
                args.crf,
                args.fast_decode,
                args.log_level,
                args.overwrite,
            )
            print("...录制结束...")
            break
        elif key == 27:
            print("不保存退出")
            break
        elif key != -1:
            print(f"按键{key}无功能")
        if flag_V == 1:
            color_images.append(color_image[:, :, ::-1])
            # wr_depth.write(depth_image)  # 保存基于灰度深度图
            # wr_depthcolor.write(depth_colormap)  # 保存计算所得着色深度图
            # wr_camera_colordepth.write(colorizer_depth)  # 保存相机自行计算的着色深度图

    # 释放资源
    cv2.destroyAllWindows()
    wr_depthcolor.release()
    wr_depth.release()
    wr_camera_colordepth.release()
    cam.release()
