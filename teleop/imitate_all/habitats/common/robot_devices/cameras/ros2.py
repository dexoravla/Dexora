import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from threading import Thread
import time


class ROS2Camera(Node):
    def __init__(self, topic_name):
        super().__init__("camera_node")
        self.subscription = self.create_subscription(
            Image, topic_name, self.image_callback, 10
        )
        self.bridge = CvBridge()
        self.latest_frame = None  # 保存最新图像帧
        self.get_logger().info(f"Subscribed to topic: {topic_name}")
        Thread(target=rclpy.spin, args=(self,), daemon=True).start()
        while self.latest_frame is None:
            self.get_logger().info(f"Waiting for topic: {topic_name}")
            time.sleep(0.5)

    def image_callback(self, msg):
        """ROS 回调函数，将消息转换为 OpenCV 格式的图像并保存。"""
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def isOpened(self):
        """检查是否有新图像。"""
        return self.latest_frame is not None

    def read(self):
        """模仿 OpenCV 的 `cap.read()` 方法。

        Returns:
            ret (bool): 图像是否成功读取。
            frame (numpy.ndarray): 图像数据，如果读取失败则为 None。
        """
        if self.latest_frame is not None:
            return True, self.latest_frame
        else:
            return False, None

    def release(self):
        """释放资源。"""
        self.destroy_node()
