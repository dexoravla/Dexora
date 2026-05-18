import cv2
import time
import threading
from queue import Queue
from pathlib import Path

class ImageSaver(threading.Thread):
    def __init__(self, name="cam_saver", quality=95):
        super().__init__()
        self.queue = Queue()
        self.quality = quality
        self.running = True
        self.name = name
        self.start()

    def save(self, path, image):
        self.queue.put((path, image))

    def run(self):
        while self.running or not self.queue.empty():
            try:
                path, img = self.queue.get(timeout=0.1)
                cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            except:
                continue

    def stop(self):
        self.running = False
        self.join()

def main():
    # === 参数设置 ===
    fps = 20
    duration_s = 2
    save_root = Path("camera_test_output")
    cam_ids = [0,2,6]  # 你可以改为你实际的摄像头编号

    num_frames = int(fps * duration_s)
    frame_interval = 1.0 / fps

    # === 初始化摄像头 ===
    caps = {}
    savers = {}
    for cam_id in cam_ids:
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[video{cam_id}] ❌ 无法打开")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps[cam_id] = cap
        savers[cam_id] = ImageSaver(name=f"saver_cam{cam_id}")

    # === 创建保存目录 ===
    for cam_id in cam_ids:
        (save_root / f"camera_{cam_id}").mkdir(parents=True, exist_ok=True)

    # === 开始采图 ===
    for frame_idx in range(num_frames):
        start_time = time.perf_counter()

        for cam_id, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                print(f"[video{cam_id}] ⚠️ 采图失败")
                continue

            save_path = save_root / f"camera_{cam_id}" / f"frame_{frame_idx:06d}.jpg"
            savers[cam_id].save(save_path, frame)

        elapsed = time.perf_counter() - start_time
        sleep_time = max(0, frame_interval - elapsed)
        print(f"Frame {frame_idx} done in {elapsed*1000:.2f} ms")
        time.sleep(sleep_time)

    # === 清理资源 ===
    for cap in caps.values():
        cap.release()
    for saver in savers.values():
        saver.stop()
    print("✅ 完成图像保存测试")

if __name__ == "__main__":
    main()
