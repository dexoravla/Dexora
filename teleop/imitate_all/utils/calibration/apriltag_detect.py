import cv2
import apriltag
import os
import contextlib


class ApriTagDetector(object):
    def __init__(self, tag_families, camera_matrix, dist_coeffs) -> None:
        options = apriltag.DetectorOptions(families=tag_families)
        self.detector = apriltag.Detector(options)
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    @contextlib.contextmanager
    @staticmethod
    def __redirect_stderr():
        original_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(original_stderr_fd, 2)
            os.close(original_stderr_fd)

    def detect(self, image, camera_params, tag_size, tag_id=None):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with self.__redirect_stderr():
            results = self.detector.detect(gray)
        for result in results:
            result: apriltag.Detection
            if tag_id is not None and result.tag_id != tag_id:
                continue
            pose, *_ = self.detector.detection_pose(result, camera_params, tag_size)
            return pose
        return None


if __name__ == "__main__":
    import numpy as np

    camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dist_coeffs = np.array([0, 0, 0, 0])
    detector = ApriTagDetector(camera_matrix, dist_coeffs)
    image = cv2.imread("image.jpg")
    pose = detector.detect(image, camera_matrix, 0.1)
    print(pose)
