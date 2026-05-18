# pad hdf5 data
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import convert_all as crd
import argparse
import random


parser = argparse.ArgumentParser(description="Pad hdf5 data")
parser.add_argument(
    "-ins", "--input_directories", type=str, nargs="+", help="directories to pad"
)
parser.add_argument(
    "-out",
    "--output_directory",
    type=str,
    help="output directory to save the padded data",
)
parser.add_argument(
    "-rt", "--ratio", type=int, nargs="+", help="ratio for each directory"
)

args = parser.parse_args()

in_dirs = args.input_directories
out_dir = args.output_directory
ratio = args.ratio
if ratio is not None:
    assert len(in_dirs) == len(
        ratio
    ), "The number of input directories and ratios should be the same"
    # rt_sum = sum(ratio)
    # if rt_sum != 10:
    #     print("The sum of the ratios is not 10, a mistake?")
    # ratio = np.array(ratio) / rt_sum

os.makedirs(out_dir, exist_ok=True)

# get the max length of the data
total_episodes = 0
max_length = 0
for d in in_dirs:
    max_length = max(max_length, len(crd.hdf5_to_dict(d + "/episode_0.hdf5")["action"]))
    total_episodes += len(crd.get_files_name_by_suffix(d, ".hdf5"))
print("total episodes: ", total_episodes)
print("max episode length: ", max_length)

# pad the data
print(f"Padding data with ratio: {ratio}...")
random.seed(0)
cnt = 0
for i, d in enumerate(in_dirs):
    hdf5_files = crd.get_files_name_by_suffix(d, ".hdf5")
    file_num = len(hdf5_files)
    print(f"Found {file_num} hdf5 files in {d}")
    if ratio is not None:
        rt = ratio[i]
        assert 1 <= rt <= 10, "The ratio should be integer between 1 and 10"
        used_file_num = int(rt / 10 * file_num)
        print(f"Using {used_file_num} hdf5 files in {d}")
        hdf5_files = random.sample(hdf5_files, used_file_num)
        print(f"Files used: {hdf5_files}")

    for file in hdf5_files:
        data = crd.flatten_dict(crd.hdf5_to_dict(f"{d}/{file}"), prefix="/")
        # print(data.keys())
        crd.save_dict_to_hdf5(
            data,
            f"{out_dir}/episode_{cnt}.hdf5",
            max_length,
        )
        cnt += 1
print(f"Saved {cnt} hdf5 files to {out_dir}")
