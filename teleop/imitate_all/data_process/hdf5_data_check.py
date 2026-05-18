import argparse
import convert_all as crd
import cv2
import os, sys


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task_name", type=str, required=True)
parser.add_argument(
    "-vn", "--video_names", type=str, nargs="+", default=["/observations/images/0"]
)
parser.add_argument("-ep", "--episode", type=int, default=0)
args = parser.parse_args()

task_name = args.task_name
episode = args.episode
video_names = args.video_names


target_root_dir = "data/hdf5"
target_dir = f"{target_root_dir}/{task_name}"
data = crd.hdf5_to_dict(f"{target_dir}/episode_{episode}.hdf5")
print(data.keys())
data_flat = crd.flatten_dict(data, prefix="/")
print(data_flat.keys())

# show actions
start = 0
end = start + 2

action = data_flat["/action"][start:end]
obs = data_flat["/observations/qpos"][start:end]
print("action")
print(action)
print("obs")
print(obs)
print("diff")
print((obs - action))

print("action 1")
print(action[1])
print("obs 1")
print(obs[1])
print("diff 1 in degree")
print((obs[1] - action[1]) * 180 / 3.1415926)


for video_name in video_names:
    print(video_name)
    image = data_flat[video_name][0]
    print(image.shape)
    compresser = crd.Compresser("jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 50], True)
    image_dc = compresser.decompress(image, "jpg")
    print(image_dc.shape)

    cv2.imshow("image", image_dc)
    cv2.waitKey(0)
