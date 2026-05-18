#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ROS1Camera(object):
    def __init__(self, topic_name):
        """ROS 1 相机节点初始化"""
        if rospy.get_name() == "/unnamed":
            rospy.init_node("camera_node", anonymous=True)
        self.bridge = CvBridge()
        self.latest_frame = None  # 存储最新图像帧
        self.subscriber = rospy.Subscriber(
            topic_name, Image, self.image_callback, queue_size=10
        )
        rospy.loginfo(f"Waiting for topic: {topic_name}")
        rospy.wait_for_message(topic_name, Image)
        rospy.loginfo(f"Subscribed to topic: {topic_name}")

    def image_callback(self, msg):
        """订阅回调，将ROS图像消息转换为OpenCV格式的图像。"""
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def isOpened(self):
        """检查是否已经接收到图像。"""
        return self.latest_frame is not None

    def read(self):
        """模仿OpenCV的`cap.read()`接口。

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
        self.subscriber.unregister()


def main():
    rospy.init_node("camera_node", anonymous=True)
    topic_name = rospy.get_param("~topic_name", "/camera/image_raw")
    camera_node = ROS1Camera(topic_name)

    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        ret, frame = camera_node.read()
        if ret:
            cv2.imshow("Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            rospy.loginfo("Waiting for frames...")
        rate.sleep()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
