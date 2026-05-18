import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import data_process.convert_all as crd
import argparse

parser = argparse.ArgumentParser(description="Convert mmk2 data to hdf5")

parser.add_argument("--raw_dir", type=str, default="data/raw", help="input directory")
parser.add_argument("--output", type=str, default="data/hdf5", help="output directory")
parser.add_argument(
    "-tn", "--task_name", type=str, default="example_task", help="task name"
)
parser.add_argument(
    "--cameras", type=str, nargs="+", default=["0"], help="camera names"
)
parser.add_argument("-pad", "--padding", action="store_true", help="pad the hdf5 data")

args = parser.parse_args()

# get all low_dim data (head&spine velocity control)
task_name = args.task_name
raw_root_dir = args.raw_dir
raw_dir = f"{raw_root_dir}/{task_name}"
data = crd.raw_to_dict(
    raw_dir,
    ["low_dim.json"],
    video_file_names=None,
    flatten_mode="hdf5",
    concatenater={
        "/observations/qpos": (
            "/observation/arm/left/joint_position",
            "/observation/eef/left/joint_position",
            "/observation/arm/right/joint_position",
            "/observation/eef/right/joint_position",
        ),
        "/action": (
            "/action/arm/left/joint_position",
            "/action/eef/left/joint_position",
            "/action/arm/right/joint_position",
            "/action/eef/right/joint_position",
        ),
    },
    key_filter=[
        "/observation/ts_diff_with_head_color_img",
        "/observation/arm/left/joint_velocity",
        "/observation/arm/right/joint_velocity",
        "/observation/arm/left/joint_effort",
        "/observation/arm/right/joint_effort",
        "/observation/eef/left/joint_velocity",
        "/observation/eef/right/joint_velocity",
        "/observation/eef/left/joint_effort",
        "/observation/eef/right/joint_effort",
        "/observation/head/joint_position",
        "/observation/head/joint_velocity",
        "/observation/head/joint_effort",
        "/observation/spine/joint_position",
        "/observation/spine/joint_velocity",
        "/observation/joint_states/time",
        "/observation/time",
        "/action/time",
        "/action/arm/left/time",
        "/action/arm/right/time",
        "/action/head/color/time",
        "/action/head/joint_position",
        "/action/spine/joint_position",
        "/action/base/velocity",
        # "/action/head/joint_velocity",
        # "/action/spine/joint_velocity"
    ],
)

import os
import cv2

# merge high_dim data and save
raw_dir
names = args.cameras
video_names = [f"{name}.mp4" for name in names]
target_root_dir = args.output
target_dir = f"{target_root_dir}/{task_name}"
low_dim_data = data
name_converter = {names[i]: f"/observations/images/{i}" for i in range(len(names))}
target_namer = lambda i: f"episode_{i}.hdf5"

compresser = crd.Compresser("jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 50], True)

os.makedirs(target_dir, exist_ok=True)

# get max episode length
episode_lens = []
for key, low_d in low_dim_data.items():
    length = len(list(low_d.values())[0])
    episode_lens.append(length)
    # if length < 200:
    #     print(f"{key} has length {length}")

max_pad_length = max(episode_lens) if args.padding else None

# save all data
episode_names = list(low_dim_data.keys())
print(f"Episode lengths: {episode_lens}")
print(f"Max episode length: {max_pad_length}")
print(f"All episodes: {episode_names}")
print(f"episode number: {len(episode_names)}")
downsampling = 0


def save_one(index, ep_name):
    crd.merge_video_and_save(
        low_dim_data[ep_name],
        f"{raw_dir}/{ep_name}",
        video_names,
        crd.save_dict_to_hdf5,
        name_converter,
        compresser,
        f"{target_dir}/" + target_namer(index),
        max_pad_length,
        downsampling,
    )
    data.pop(ep_name)


# save all
from concurrent.futures import ThreadPoolExecutor

futures = []
with ThreadPoolExecutor(max_workers=25) as executor:
    for index, ep_name in enumerate(episode_names):
        # silent execution, no print
        futures.append(executor.submit(save_one, index, ep_name))

print(f"All data saved to {target_dir}")
