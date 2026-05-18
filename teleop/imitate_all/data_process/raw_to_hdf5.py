import convert_all as crd
import os
import cv2
import argparse
from concurrent.futures import ThreadPoolExecutor
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("-tn", "--task_name", type=str)
parser.add_argument("-vn", "--video_names", type=str, nargs="+")
parser.add_argument("-ds", "--downsampling", type=int, default=0)
parser.add_argument("-md", "--mode", type=str, default="real3")
parser.add_argument("-pad", "--padding", action="store_true")
parser.add_argument("-dir", "--raw_dir", type=str, default="data/raw")
# parser.add_argument("-bson", "--use_bson_style", action="store_true")
args = parser.parse_args()

task_name = args.task_name
downsampling = args.downsampling
mode = args.mode
padding = args.padding
raw_dir = args.raw_dir
# use_bson_style = args.use_bson_style

task_dir = os.path.abspath(f"{raw_dir}/{task_name}")
assert os.path.exists(task_dir), f"task_dir {task_dir} not exists"
# raw_dir = os.path.abspath(raw_dir)
# assert os.path.exists(raw_dir)


def load_raw_real_data(raw_dir, downsampling=0):
    data = crd.raw_to_dict(
        raw_dir,
        ["low_dim.json"],
        video_file_names=None,
        flatten_mode=None,
        concatenater={
            "/observations/qpos": (
                "observation/arm/joint_position",
                "observation/eef/joint_position",
            ),
            "/action": (
                "action/arm/joint_position",
                "action/eef/joint_position",
                # "observation/base/velocity",
            ),
        },
        key_filter=[
            "observation/eef/pose",
            "action/eef/pose",
            # "/time",
        ],
    )
    return crd.downsample(data, downsampling)


def load_raw_mujoco_data(raw_dir, downsampling=0):
    data = crd.raw_to_dict(
        raw_dir,
        ["obs_action.json"],
        video_file_names=None,
        flatten_mode="hdf5",
        name_converter={
            "/obs/jq": "/observations/qpos",
            "/act": "/action",
        },
        pre_process=None,
        concatenater=None,
        key_filter=["/time"],
    )
    return crd.downsample(data, downsampling)


# load raw low dim data
if mode == "real3":
    low_dim_data = load_raw_real_data(task_dir, downsampling)
elif mode == "mujoco":
    low_dim_data = load_raw_mujoco_data(task_dir, downsampling)
else:
    raise ValueError(f"mode {mode} is not supported")

# merge high_dim data and save
raw_names = args.video_names
video_names = [name + ".mp4" for name in raw_names]
target_dir = f"data/hdf5/{task_name}/"
name_converter = {
    raw_names[i]: f"/observations/images/{i}" for i in range(len(raw_names))
}
print(f"name_converter: {name_converter}")
target_namer = lambda i: f"episode_{i}.hdf5"
compresser = crd.Compresser("jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 50], True)

# get max episode length
episode_lens = []
for low_d in low_dim_data.values():
    episode_lens.append(len(list(low_d.values())[0]))
max_pad_length = max(episode_lens) if padding else None

print(f"Episode flatten keys: {list(low_dim_data.values())[0].keys()}")
episode_names = list(low_dim_data.keys())
print(f"Episode number: {len(episode_names)}")
print(f"Max episode length: {max_pad_length}")
print(f"All episodes: {episode_names}")

# create target dir
os.makedirs(target_dir, exist_ok=True)


def save_one(index, ep_name):
    crd.merge_video_and_save(
        low_dim_data[ep_name],
        f"{task_dir}/{ep_name}",
        video_names,
        crd.save_dict_to_hdf5,
        name_converter,
        compresser,
        f"{target_dir}/" + target_namer(index),
        max_pad_length,
        downsampling,
    )
    low_dim_data.pop(ep_name)


# save all data
print(f"Start saving all data to {target_dir}...")
futures = []
with ThreadPoolExecutor(max_workers=25) as executor:
    for index, ep_name in enumerate(episode_names):
        futures.append(executor.submit(save_one, index, ep_name))
print(f"All data saved to {target_dir}")

# # save one data
# index = 0
# ep_name = episode_names[index]
# save_one(index, ep_name)