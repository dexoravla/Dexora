import json
from pathlib import Path
from typing import List, Union
import cv2
import logging


"""
- data/raw (root)
    - stack_blocks (name)
        - 0 (episode)
            - meta.json
            - low_dim.json
            - 0.mp4/avi
            - 1.mp4/avi
            - ...
        - 1
            - meta.json
            - low_dim.json
            - 0.mp4/avi
            - 1.mp4/avi
            - ...
        - ...
    - ...
"""


class RawDataset(object):
    def __init__(self, name, root=None, episodes=None) -> None:
        self.root = root
        self.name = name
        if root is not None:
            self.data_path = Path(root) / name
        else:
            self.data_path = Path(name)
        self.raw_data = {}
        self.episode_data_index = {"from": {}, "to": {}}
        self.__last_index = -1
        if episodes is not None:
            self.warm_up_episodes(episodes)

    def warm_up_episodes(self, episodes: List[int], low_dim_only=False):
        if isinstance(episodes, int):
            episodes = [episodes]
        for episode in episodes:
            self.load_episode(episode, low_dim_only)

    def load_episode(self, episode:int, low_dim_only=False):
        if episode not in self.raw_data:
            self.raw_data[episode] = {}
        else:
            print(f"Episode {episode} already loaded.")
        # load the meta.json
        with open(self.data_path / f"{episode}/meta.json", "r") as f:
            meta = json.load(f)
            self.raw_data[episode]["meta"] = meta
        # load the low_dim.json
        with open(self.data_path / f"{episode}/low_dim.json", "r") as f:
            low_dim = json.load(f)
            self.raw_data[episode]["low_dim"] = low_dim
        # check the video files
        if not low_dim_only:
            video_files = list((self.data_path / f"episode").glob("*.mp4")) + list(
                (self.data_path / episode).glob("*.avi")
            )
            video_files.sort()
            print("sorted video files", video_files)
            # load the video files as cv2.VideoCapture
            for i, video_file in enumerate(video_files):
                cap = cv2.VideoCapture(str(video_file))
                self.raw_data[episode][f"images/{i}"] = cap
        self.episode_data_index["from"][episode] = self.__last_index + 1
        self.episode_data_index["to"][episode] = self.__last_index + meta["length"] - 1
        self.__last_index = self.episode_data_index["to"][episode]

    def select_columns(self, keys: Union[str, List[str]]) -> List[dict]:
        if isinstance(keys, str):
            keys = [keys]
        items: List[dict] = []
        for episode, value in self.raw_data["low_dim"].items():
            from_index = self.episode_data_index[episode]["from"]
            to_index = self.episode_data_index[episode]["to"]
            for _ in range(from_index, to_index + 1):
                item = {}
                for key in keys:
                    item[key] = value[key]
                items.append(item)
        return items


if __name__ == "__main__":
    # TODO: test the RawDataset class
    dataset = RawDataset("stack_blocks", "data/raw")
    dataset.warm_up_episodes([0, 1])
    print(dataset.raw_data)
    print(dataset.select_columns(["meta", "low_dim"]))
