"""The pipline for augmenting images in hdf5 files using pytorch and saving to new hdf5 files"""

# 1. read data from hdf5 files
# 2. decompress image data
# 3. change image data to torch tensor
# 4. augment images
# 5. change torch tensor to image data
# 6. compress image data
# 7. pad image data so that all images have the same length
# 8. save data to hdf5 files

import h5py
import torch
import numpy as np
from configurations.task_configs.config_augmentation.image.basic import color_transforms_1
from typing import List, Dict, Tuple
import cv2
import time
import shutil
import os


def create_test_hdf5(file_path):
    """
    创建一个测试用的 HDF5 文件，并添加一些组和数据集
    :param file_path: 要创建的 HDF5 文件路径
    """
    with h5py.File(file_path, 'w') as f:  # 'w' 模式用于创建一个新文件或覆盖现有文件
        # 创建组
        group1 = f.create_group('group1')
        group2 = group1.create_group('group2')

        # 创建数据集
        data1 = np.random.rand(10, 10)
        data2 = np.random.rand(20, 20)
        data3 = np.random.rand(30, 30)
        data4 = np.random.rand(5, 5)

        group1.create_dataset('dataset1', data=data1)
        group2.create_dataset('dataset2', data=data2)
        group2.create_dataset('dataset3', data=data3)
        f.create_dataset('dataset4', data=data4)

        print(f"测试 HDF5 文件 {file_path} 已成功创建。")

def get_hdf5_structure_dict(hf5_file:h5py.File):
    """
    获取HDF5文件的层级结构并保存为字典
    :param file_path: HDF5文件路径
    :return: 层级结构的字典
    """
    def add_to_dict(h5obj, path, result_dict):
        """
        将HDF5文件的层级结构添加到字典中
        :param h5obj: 当前对象 (组或数据集)
        :param path: 当前对象的路径
        :param result_dict: 存储结构的字典
        """
        parts = path.split('/')
        sub_dict = result_dict
        for part in parts:
            if part not in sub_dict:
                sub_dict[part] = {}
            sub_dict = sub_dict[part]

    structure_dict = {}
    hf5_file.visititems(lambda name, obj: add_to_dict(obj, name, structure_dict))
    return structure_dict

def get_hdf5_dataset_keys(hf5_file:h5py.File):
    """
    获取HDF5文件中所有数据集的路径
    :param file_path: HDF5文件路径
    :return: 数据集路径的列表
    """
    def collect_datasets(name, obj, datasets_list:list):
        """
        收集HDF5文件中的所有数据集路径
        :param name: 当前对象的路径
        :param obj: 当前对象 (组或数据集)
        :param datasets_list: 存储数据集路径的列表
        """
        if isinstance(obj, h5py.Dataset):
            datasets_list.append(name)

    datasets_keys = []
    hf5_file.visititems(lambda name, obj: collect_datasets(name, obj, datasets_keys))
    return datasets_keys

def read_sub_data_from_hdf5(hf5_file:h5py.File, data_structure:dict) -> dict:
    """Read data from hdf5 file, only support 2 level of nested dictionary"""
    data = {}
    for key, value in data_structure.items():
        if isinstance(value, dict):
            data[key] = {}
            for key2, value2 in value.items():
                data[key][key2] = hf5_file[key][key2][()]
        else:
            data[key] = hf5_file[key][()]
    return data

def decompress_image_data(data:np.ndarray) -> np.ndarray:
    data_decompressed = []
    for img in data:
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        data_decompressed.append(img)
    data_decompressed = np.array(data_decompressed)
    return data_decompressed

def change_image_data_to_torch_tensor(data) -> torch.Tensor:
    data = torch.from_numpy(data)
    if len(data.shape) == 4:
        data = torch.einsum('k h w c -> k c h w', data)
    elif len(data.shape) == 5:
        data = torch.einsum('n k h w c -> n k c h w', data)
    else:
        raise ValueError(f"Data shape {data.shape} not supported")
    return data

def augment_images(image:torch.Tensor) -> torch.Tensor:
    return color_transforms_1(image)

def change_torch_tensor_to_image_data(data:torch.Tensor) -> np.ndarray:
    if len(data.shape) == 4:
        data = torch.einsum('k c h w -> k h w c', data)
    elif len(data.shape) == 5:
        data = torch.einsum('n k c h w -> n k h w c', data)
    else:
        raise ValueError(f"Data shape {data.shape} not supported")
    data = data.numpy()
    return data

def compress_image_data(data:np.ndarray) -> Tuple[list, np.ndarray]:
    """Compress image data to jpg format, return compressed data and compressed length
    Since the compressed data may have different length, we need to record the length of each compressed image,
    and return a list instead of a np array
    """
    t0 = time.time()
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        50,
    ]  # TODO: tried as low as 20, seems fine
    compressed_len = []
    all_compressed = []
    print(data.shape)
    for cam_imgs in data:
        print(cam_imgs.shape)
        compressed_list = []
        compressed_len.append([])
        for image in cam_imgs:
            result, encoded_image = cv2.imencode(
                ".jpg", image, encode_param
            )
            compressed_list.append(encoded_image)
            compressed_len[-1].append(len(encoded_image))
        all_compressed.append(compressed_list)
        print(image.shape)
        print(encoded_image.shape)
    print(f"compression: {time.time() - t0:.2f}s")
    return all_compressed, np.array(compressed_len)

def pad_image_data(data:list, max_len:int) -> np.ndarray:
    """Pad image data so that all compressed images have the same length"""
    t0 = time.time()
    padded_size = max_len
    all_padded_data = []
    for cam_images in data:
        compressed_image_list = cam_images
        padded_compressed_image_list = []
        for compressed_image in compressed_image_list:
            padded_compressed_image = np.zeros(padded_size, dtype="uint8")
            image_len = len(compressed_image)
            padded_compressed_image[:image_len] = compressed_image
            padded_compressed_image_list.append(padded_compressed_image)
        all_padded_data.append(padded_compressed_image_list)
    print(f"padding: {time.time() - t0:.2f}s")
    return np.array(all_padded_data)

def copy_and_rename_file(src_file_path, dst_file_path):
    """
    复制一个文件并重命名
    :param src_file_path: 原始文件路径
    :param dst_file_path: 目标文件路径
    """
    # 使用 shutil.copy2 来复制文件并保留元数据
    shutil.copy2(src_file_path, dst_file_path)


if __name__ == "__main__":
    # TODO: check if compressed
    import argparse

    # specify data structure
    data_structure = {
        "/observations/images/0":[],
        # "/observations/images/1":[],
        # "/observations/qpos":[],
        # "/observations/qvel":[],
    }
    img_data_keys = list(data_structure.keys())
    # get file handle
    raw_file_path = "/home/ghz/Work/ALOHA/act/data/hdf5/stack_cups/episode_0.hdf5"
    all_data = {}
    root_raw = h5py.File(raw_file_path, "r")
    # read sub data from handle using data_structure
    data_keys = get_hdf5_dataset_keys(root_raw)
    print(data_keys)  # ['/observations/images/0', '/observations/qpos']
    image_data:Dict[str, np.ndarray] = read_sub_data_from_hdf5(root_raw, data_structure)
    # decompress image data
    img_raw_decompressed:List[np.ndarray] = []
    for key in img_data_keys:
        # get image data from data dict
        img_raw = image_data[key]
        print(img_raw.shape)  # (200, 15125)
        # decompress image data
        img_raw_decompressed.append(decompress_image_data(img_raw))
        print(img_raw_decompressed[-1].shape)  # (200, 480, 640, 3)
    # merge all image data to one np array
    img_all_np = np.array(img_raw_decompressed)
    print(img_all_np.shape)  # (1, 200, 480, 640, 3)
    # change image data to usually used torch tensor
    img_torch = change_image_data_to_torch_tensor(img_all_np)
    print(img_torch.shape)  # torch.Size([1, 200, 3, 480, 640])
    # augment images
    img_torch_augmented = augment_images(img_torch)
    print(img_torch_augmented.shape)  # torch.Size([1, 200, 3, 480, 640])
    # change torch tensor to image data
    img_np_augmented = change_torch_tensor_to_image_data(img_torch_augmented)
    print(img_np_augmented.shape)  # (1, 200, 480, 640, 3)
    # compress image data
    img_compressed, img_compressed_len = compress_image_data(img_np_augmented)
    print((len(img_compressed), len(img_compressed[0])))  # (1, 200)
    print(img_compressed_len.shape)  # (1, 200)
    # pad image data so that all images have the same length
    padded_size = img_compressed_len.max()
    img_padded = pad_image_data(img_compressed, padded_size)
    print(img_padded.shape)  # (1, 200, 1xxxx), 1xxxx is the max length of compressed image
    root_raw.close()
    # save data to target hdf5 files
    max_timesteps = img_padded.shape[1]
    print("Saving augmented data to hdf5 file...")
    target_file_path = "/home/ghz/Work/ALOHA/act/data/hdf5/stack_cups/episode_0_augmented.hdf5"
    if target_file_path != raw_file_path:
        copy_and_rename_file(raw_file_path, target_file_path)
    else:
        target_file_path = raw_file_path
        print("Warning: target file path is the same as raw file path, data will be overwritten.")
    try:
        with h5py.File(target_file_path, "r+", rdcc_nbytes=1024**2 * 2) as root_target:
            print(get_hdf5_dataset_keys(root_target))
            for i, key in enumerate(img_data_keys):
                del root_target[key]
                root_target.create_dataset(key, (max_timesteps, padded_size), dtype="uint8", chunks=(1, padded_size))
                root_target[key][...] = img_padded[i]
            assert get_hdf5_dataset_keys(root_target) == data_keys
        assert i == img_padded.shape[0] - 1
    except Exception as e:
        print(e)
        # remove target file if saving failed
        if target_file_path != raw_file_path:
            os.remove(target_file_path)
    else:
        print(f"Successfully saved augmented data to {target_file_path}")