import rospy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from typing import Dict
import numpy as np


class LoggerROS1(object):
    def __init__(self, name):
        self.name = name
        if rospy.get_name() == "/unnamed":
            rospy.init_node(name, anonymous=True)
        self._pubers: Dict[str, rospy.Publisher] = {}

    def log_1D(self, topic, data_1d: list):
        if topic not in self._pubers:
            self._pubers[topic] = rospy.Publisher(
                topic, Float64MultiArray, queue_size=10
            )
        self._pubers[topic].publish(Float64MultiArray(data=data_1d))

    def log_2D(self, topic, data_2d: np.ndarray):
        if topic not in self._pubers:
            self._pubers[topic] = rospy.Publisher(topic, Image, queue_size=10)
        self._pubers[topic].publish(CvBridge().cv2_to_imgmsg(data_2d))


if __name__ == "__main__":

    logger = LoggerROS1("test_logger")
    while not rospy.is_shutdown():
        print("logging...")
        logger.log_1D("test_array", [1, 2, 3])
        logger.log_2D("test_image", np.zeros((100, 100, 3), dtype=np.uint8))
        rospy.sleep(1)
