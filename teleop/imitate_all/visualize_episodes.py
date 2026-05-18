import os
import numpy as np
import cv2
import h5py
import argparse
import matplotlib.pyplot as plt
from data_process.convert_all import Compresser
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset does not exist at \n{dataset_path}\n")

    with h5py.File(dataset_path, "r") as root:
        qpos = root["/observations/qpos"][()]
        # qvel = root["/observations/qvel"][()]
        qvel = None
        action = root["/action"][()]
        image_dict = dict()
        for cam_name in root[f"/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

    return qpos, qvel, action, image_dict


def save_videos(video, dt, video_path=None, swap_channel=False, decompress=False):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                if swap_channel:
                    image = image[:, :, [2, 1, 0]]  # swap B and R channel
                if decompress:
                    image = Compresser.decompress(image, "jpg")
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f"Saved video to: {video_path}")
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f"Saved video to: {video_path}")


def visualize_joints(
    state_list,
    action_list,
    plot_path=None,
    ylim=None,
    label_overwrite=None,
    joint_names=None,
):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = "State", "Action"

    qpos = np.array(state_list)  # ts, dim
    action = np.array(action_list)
    if qpos.shape != action.shape:
        logger.warning(f"qpos and action have different shapes: {qpos.shape} vs {action.shape}")
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))
    assert num_dim == len(
        joint_names
    ), "joint_names must match the number of dimensions"
    # plot joint state
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f"{joint_names[dim_idx]}")
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(action[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved qpos plot to: {plot_path}")
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace(".pkl", "_timestamp.png")
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h * 2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10e-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f"Camera frame timestamps")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    ax = axs[1]
    ax.plot(np.arange(len(t_float) - 1), t_float[:-1] - t_float[1:])
    ax.set_title(f"dt")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved timestamp plot to: {plot_path}")
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tn",
        "--task_name",
        action="store",
        type=str,
        help="task_name",
        required=False,
    )
    parser.add_argument(
        "-dt",
        "--sample_freq",
        action="store",
        type=float,
        help="Time step.",
        required=True,
    )
    parser.add_argument(
        "-dir",
        "--dataset_dir",
        action="store",
        type=str,
        help="Dataset dir.",
        required=False,
        default="data",
    )
    parser.add_argument(
        "-dn",
        "--dataset_name",
        action="store",
        type=str,
        help="Dataset name.",
        required=True,
    )
    parser.add_argument(
        "-jn",
        "--joint_names",
        action="store",
        nargs="+",
        type=str,
        help="State names.",
        required=False,
        default=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
    )
    parser.add_argument(
        "-sv",
        "--save_video",
        action="store_true",
        help="Save video.",
    )
    parser.add_argument(
        "-sj",
        "--save_joints",
        action="store_true",
        help="Save joint states and actions.",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        action="store",
        type=str,
        help="Output dir.",
        required=False,
        default="visualizations",
    )
    parser.add_argument(
        "-dc",
        "--decompress",
        action="store_true",
        help="Decompress image.",
    )

    args = vars(parser.parse_args())

    dt = args["sample_freq"]
    task_name = args["task_name"]
    dataset_dir = args["dataset_dir"] + "/" + task_name
    dataset_name = args["dataset_name"]
    joint_names = args["joint_names"]
    output_dir = args["output_dir"]
    decompress = args["decompress"]

    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args["save_video"]:
        save_videos(
            image_dict,
            dt,
            video_path=os.path.join(output_dir, dataset_name + "_video.mp4"),
            decompress=decompress,
        )
    if args["save_joints"]:
        visualize_joints(
            qpos,
            action,
            plot_path=os.path.join(output_dir, dataset_name + "_joint.png"),
            joint_names=joint_names,
        )
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back
