import json
import numpy as np
import torch

from data.lerobot_vla_dataset import LeRobotVLADataset


class LeRobotVLADatasetWithLogpi(LeRobotVLADataset):
    """
    Extended LeRobotVLADataset that attaches precomputed logpi values.

    Supports two logpi JSON formats:
    - Path-keyed: {"/path/to/episode": {"frame": value, ...}, ...}
    - Index-keyed: {"0": {"frame": value, ...}, ...}
    For LeRobot, we generally use index-keyed outputs (ep_idx as string),
    but both are supported. If path keys are provided but cannot be resolved,
    we fall back to index keys using ep_idx from the sample meta.
    """

    def __init__(self, logpi_file: str = "new_lerobot_logpi_values.json", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load precomputed logpi values
        self.logpi_dict = {}
        self._use_path_keys = True
        with open(logpi_file, 'r') as f:
            raw_logpi_dict = json.load(f)
            first_key = next(iter(raw_logpi_dict.keys())) if len(raw_logpi_dict) > 0 else None
            if first_key is not None:
                # If key looks like an integer index, treat as old/index format
                try:
                    int(first_key)
                    self._use_path_keys = False
                except (TypeError, ValueError):
                    self._use_path_keys = True
            # Normalize into internal dict with consistent int frame keys
            if self._use_path_keys:
                for ep_path, frames in raw_logpi_dict.items():
                    ep_key = str(ep_path)
                    self.logpi_dict[ep_key] = {}
                    if isinstance(frames, dict):
                        for frame_idx_str, logpi_val in frames.items():
                            try:
                                frame_idx = int(frame_idx_str)
                                self.logpi_dict[ep_key][frame_idx] = float(logpi_val)
                            except (TypeError, ValueError):
                                continue
            else:
                for ep_idx_str, frames in raw_logpi_dict.items():
                    try:
                        ep_idx = int(ep_idx_str)
                    except (TypeError, ValueError):
                        continue
                    self.logpi_dict[ep_idx] = {}
                    if isinstance(frames, dict):
                        for frame_idx_str, logpi_val in frames.items():
                            try:
                                frame_idx = int(frame_idx_str)
                                self.logpi_dict[ep_idx][frame_idx] = float(logpi_val)
                            except (TypeError, ValueError):
                                continue

        # Precompute global mean as a fallback
        all_values = []
        for ep_frames in self.logpi_dict.values():
            if isinstance(ep_frames, dict):
                all_values.extend(ep_frames.values())
            else:
                all_values.append(ep_frames)
        self.global_mean = float(np.mean(all_values)) if all_values else 0.0
        key_type = "path" if self._use_path_keys else "index"
        print(f"Loaded logpi values for {len(self.logpi_dict)} episodes from {logpi_file} (keyed by {key_type})")

    def __getitem__(self, idx):
        # Delegate to get_item treating idx as global frame index
        data = self.get_item(index=idx)

        # Resolve episode and frame
        ep_idx = None
        frame_idx = None
        if isinstance(data, dict) and 'meta' in data:
            ep_idx = data['meta'].get('episode_idx', None)
            frame_idx = data['meta'].get('step_id', None)

        # Determine episode key
        if self._use_path_keys:
            # We don't have episode file paths readily for LeRobot; try a synthetic path then fallback to ep_idx
            episode_key = f"episode_{int(ep_idx) if ep_idx is not None else -1:06d}"
            if episode_key not in self.logpi_dict and ep_idx in self.logpi_dict:
                # Some logs might actually be index keyed although we detected path; fallback
                episode_key = ep_idx
        else:
            episode_key = ep_idx

        # Compute logpi value with fallback/interpolation
        logpi_value = self.global_mean
        if frame_idx is not None and episode_key in self.logpi_dict:
            ep_frames = self.logpi_dict[episode_key]
            if isinstance(ep_frames, dict) and frame_idx in ep_frames:
                logpi_value = ep_frames[frame_idx]
            else:
                if isinstance(ep_frames, dict) and len(ep_frames) > 0:
                    keys = sorted(ep_frames.keys())
                    if frame_idx <= keys[0]:
                        logpi_value = ep_frames[keys[0]]
                    elif frame_idx >= keys[-1]:
                        logpi_value = ep_frames[keys[-1]]
                    else:
                        import bisect
                        r = bisect.bisect_right(keys, frame_idx)
                        l = r - 1
                        f0, f1 = keys[l], keys[r]
                        v0, v1 = ep_frames[f0], ep_frames[f1]
                        t = (frame_idx - f0) / max(1, (f1 - f0))
                        logpi_value = float(v0 + (v1 - v0) * t)

        data['logpi'] = np.array([logpi_value], dtype=np.float32)
        return data

    def get_logpi_statistics(self):
        all_logpi = []
        for ep_frames in self.logpi_dict.values():
            if isinstance(ep_frames, dict):
                all_logpi.extend(ep_frames.values())
            else:
                all_logpi.append(ep_frames)
        if all_logpi:
            return {
                'count': len(all_logpi),
                'mean': float(np.mean(all_logpi)),
                'std': float(np.std(all_logpi)),
                'min': float(np.min(all_logpi)),
                'max': float(np.max(all_logpi)),
            }
        else:
            return {'count': 0, 'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
