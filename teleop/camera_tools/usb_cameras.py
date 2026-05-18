import cv2
import argparse
import time

def open_camera(device, width, height, fps, format):
    cap = cv2.VideoCapture(device)
    
    if format == "MJPEG":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif format == "YUYV":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    if not cap.isOpened():
        print(f"Error: Cannot open camera {device}")
        return None

    return cap


def main():
    parser = argparse.ArgumentParser(description="Multiple USB camera video capture")
    parser.add_argument("--devices", nargs='+', type=str, default=("0",), help="Device numbers of the cameras")
    parser.add_argument("--width", type=int, default=640, help="Width of the video frames")
    parser.add_argument("--height", type=int, default=480, help="Height of the video frames")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate of the video")
    parser.add_argument("--format", choices=["MJPEG", "YUYV"], default="MJPEG", help="Video format")
    parser.add_argument("-sf", "--show_fps", action="store_true", help="Show FPS")
    parser.add_argument("-s", "--show", action="store_true", help="Show video windows")

    args = parser.parse_args()

    cameras = []
    for device in args.devices:
        cam = open_camera(device, args.width, args.height, args.fps, args.format)
        if cam:
            cameras.append((device, cam))

    if not cameras:
        print("Error: No cameras were opened successfully")
        return
    print(f"Opened {len(cameras)} cameras")
    print("Press q to exit")
    while True:
        start_time = time.time()
        for device, cam in cameras:
            ret, frame = cam.read()
            if ret:
                if args.show:
                    cv2.imshow(f"Camera {device}", frame)
                else:
                    print(f"Image shape from camera {device}: {frame.shape}")
            else:
                print(f"Error: Cannot read frame from camera {device}")
        if args.show_fps:
            fps = 1.0 / (time.time() - start_time)
            print(f"FPS: {fps:.2f}")
        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for _, cam in cameras:
        cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()