import os
import cv2
import numpy as np


class CheckBoard(object):
    def __init__(self, points_per_col, points_per_row, square_size) -> None:
        self.points_per_col = points_per_col
        self.points_per_row = points_per_row
        self.square_size = square_size

        self.patten_size = (self.points_per_row, self.points_per_col)

        self.corners_world = []
        for row in range(self.points_per_col):
            for col in range(self.points_per_row):
                self.corners_world.append(
                    [float(col * self.square_size), float(row * self.square_size), 0.0]
                )
        self.corners_pixel_all = []

    def add_corners_pixel(self, img):
        self.img_size = (img.shape[1], img.shape[0])

        corners_pixel = []
        # 进行角点粗检测
        found, corners_pixel = cv2.findChessboardCorners(img, self.patten_size, None)

        if found:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 角点亚像素优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            cv2.cornerSubPix(img_gray, corners_pixel, (5, 5), (-1, -1), criteria)
            self.corners_pixel_all.append(corners_pixel)
            return True
        else:
            return False

    def caliCam(self):
        if len(self.corners_pixel_all) < 3:
            print("Not enough images")
            return None

        obj_points = []
        img_points = []
        for i in range(len(self.corners_pixel_all)):
            obj_points.append(self.corners_world)
            img_points.append(self.corners_pixel_all[i])

        # print("obj_points: ", obj_points)
        # print("img_points: ", img_points)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            np.array(obj_points, dtype=np.float32),
            img_points,
            self.img_size,
            None,
            None,
        )
        return ret, mtx, dist, rvecs, tvecs


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5, linewidth=500)
    show = True

    img_forder = "/home/leexuanyi/study/Imitate-All/calibpicture/PCcam"
    img_files = os.listdir(img_forder)
    imgs = []
    for img_file in img_files:
        img = cv2.imread(os.path.join(img_forder, img_file))
        imgs.append(img)

    if show:
        cv2.namedWindow("img")

    cb = CheckBoard(8, 11, 20)
    for i, img in enumerate(imgs):
        res = cb.add_corners_pixel(img)
        if res:
            if show:
                cv2.drawChessboardCorners(
                    img, cb.patten_size, cb.corners_pixel_all[-1], True
                )
                cv2.imshow("img", img)
                key = cv2.waitKey(0)
                if key == ord("q"):
                    break
        else:
            print("{} : failed".format(img_files[i]))

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cb.caliCam()
    print("ret:\n", ret)
    print("mtx:\n", mtx)
    print("dist:\n", np.array2string(dist[0], separator=", "))
    # print("rvecs: ", rvecs)
    # print("tvecs: ", tvecs)
