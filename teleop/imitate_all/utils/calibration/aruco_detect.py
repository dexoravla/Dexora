import cv2
import numpy as np
from typing import Union
from cv2.aruco import Dictionary


class ArucoDetector(object):
    def __init__(self, aruco_dict: Union[int, Dictionary]):
        if isinstance(aruco_dict, int):
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.cam_params = {}

    def set_camera_params(self, camera, camera_matrix, dist_coeffs):
        if camera not in self.cam_params.keys():
            self.cam_params[camera] = {}
        self.cam_params[camera]["camera_matrix"] = np.array(camera_matrix)
        self.cam_params[camera]["dist_coeffs"] = np.array(dist_coeffs)

    @staticmethod
    def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
        """
        This will estimate the rvec and tvec for each of the marker corners detected by:
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        corners - is an array of detected corners for each detected marker in the image
        marker_size - is the size of the detected markers
        mtx - is the camera matrix
        distortion - is the camera distortion matrix
        RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
        """
        marker_points = np.array(
            [
                [-marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, marker_size / 2, 0],
                [marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0],
            ],
            dtype=np.float32,
        )
        rvecs = []
        Rmats = []
        tvecs = []
        for c in corners:
            _, r, t = cv2.solvePnP(
                marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
            )
            R, _ = cv2.Rodrigues(r)
            rvecs.append(r)
            Rmats.append(R)
            tvecs.append(t)
        return rvecs, Rmats, tvecs

    def detect(self, image, camera, marker_size, putAxis=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if len(corners) > 0:
            # Estimate the pose of each marker
            rvecs, Rmats, tvecs = self.estimatePoseSingleMarkers(
                corners,
                marker_size,
                self.cam_params[camera]["camera_matrix"],
                self.cam_params[camera]["dist_coeffs"],
            )

            if putAxis:
                corners_int = np.array(corners).astype(np.int32)
                image = cv2.drawContours(image, corners_int, -1, (0, 255, 0), 3)
                # Draw the marker axes on the image
                for i in range(len(rvecs)):
                    cv2.drawFrameAxes(
                        image,
                        #self.camera_matrix,
                        #self.dist_coeffs,
                        self.cam_params[camera]["camera_matrix"],
                        self.cam_params[camera]["dist_coeffs"],
                        rvecs[i],
                        tvecs[i],
                        0.02,
                    )
                    cv2.putText(
                        image,
                        str(ids[i][0]),
                        tuple(corners_int[i][0][0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
        else:
            Rmats = []
            tvecs = []
        # 对一副图片中的所有二维码检测的结果
        results = {
            "image": image,  # （带有标记的）图像
            "tvecs": tvecs,  # 二维码中心的位置
            "Rmats": Rmats,  # 二维码的旋转矩阵
            "ids": ids,  # 二维码的id
            "corners": corners,  # 二维码的四个角点
        }
        return results


if __name__ == "__main__":

    np.set_printoptions(precision=5, suppress=True, linewidth=500)

    cameras_intri_dict = {
        "x1": {
            "id": 1,
            "camera_matrix": [
                [1651.95478, 0.0, 946.51219],
                [0.0, 1652.26405, 542.23279],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [0.17503, -0.3395, -0.00023, -0.00228, -0.51048],
        },
        "x2": {
            "id": 2,
            "camera_matrix": [
                [3303.62945, 0.0, 978.12415],
                [0.0, 3312.68043, 537.40249],
                [0.0, 0.0, 1.0],
            ],
            "dist_coeffs": [0.16144, -1.18852, 0.0003, 0.00305, 17.45546],
        },
        "kinect": {
            "id": 3,
            "camera_matrix": [  # 手标的
                [947.44872, 0.0, 975.80396],
                [0.0, 930.4648, 518.00402],
                [0.0, 0.0, 1.0],
            ],
            # "camera_matrix" : [ # 从设备读出来的
            #     [919.2274169921875,   0.    ,  962.2535400390625],
            #     [  0.     , 919.3020629882812,  555.55419921875],
            #     [  0.     ,   0.    ,    1.     ],
            # ],
            "dist_coeffs": [0.08584, 0.04115, -0.00389, 0.0067, -0.20302],
        },
    }

    camera = "x1"
    camera_matrix = np.array(cameras_intri_dict[camera]["camera_matrix"])
    dist_coeffs = np.array(cameras_intri_dict[camera]["dist_coeffs"])
    cam_id = cameras_intri_dict[camera]["id"]
    marker_size = 0.053
    print(cv2.aruco.DICT_4X4_50)
    det = ArucoDetector(cv2.aruco.DICT_4X4_50, camera_matrix, dist_coeffs, marker_size)
    image = None
    img, Rmats, tvecs, ids, _ = det.detect(image)
