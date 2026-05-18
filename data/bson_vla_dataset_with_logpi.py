import json
import os
import numpy as np
import torch
from data.bson_vla_dataset import BsonVLADataset


class BsonVLADatasetWithLogpi(BsonVLADataset):
    """
    Extended BsonVLADataset that includes precomputed logpi values.
    """
    
    def __init__(self, logpi_file="logpi_values.json", *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load precomputed logpi values
        # Support both formats:
        # - NEW: {"/path/to/episode": {"frame": value, ...}, ...}
        # - OLD: {"0": {"frame": value, ...}, ...}
        self.logpi_dict = {}
        self._use_path_keys = True  # assume new format by default
        with open(logpi_file, 'r') as f:
            raw_logpi_dict = json.load(f)
            # Decide keying scheme by inspecting the first key
            first_key = next(iter(raw_logpi_dict.keys())) if len(raw_logpi_dict) > 0 else None
            if first_key is not None:
                # If key looks like an integer index, treat as old format
                try:
                    int(first_key)
                    self._use_path_keys = False
                except (TypeError, ValueError):
                    self._use_path_keys = True
            # Normalize into internal dict with consistent inner int frame keys
            if self._use_path_keys:
                for ep_path, frames in raw_logpi_dict.items():
                    self.logpi_dict[str(ep_path)] = {}
                    if isinstance(frames, dict):
                        for frame_idx_str, logpi_val in frames.items():
                            try:
                                frame_idx = int(frame_idx_str)
                                self.logpi_dict[str(ep_path)][frame_idx] = float(logpi_val)
                            except (TypeError, ValueError):
                                continue
            else:
                for ep_idx_str, frames in raw_logpi_dict.items():
                    try:
                        ep_idx = int(ep_idx_str)
                    except (TypeError, ValueError):
                        # Skip invalid keys
                        continue
                    self.logpi_dict[ep_idx] = {}
                    if isinstance(frames, dict):
                        for frame_idx_str, logpi_val in frames.items():
                            try:
                                frame_idx = int(frame_idx_str)
                                self.logpi_dict[ep_idx][frame_idx] = float(logpi_val)
                            except (TypeError, ValueError):
                                continue
        
        # Precompute global mean as a very last-resort fallback
        self._all_values = []
        for ep_frames in self.logpi_dict.values():
            if isinstance(ep_frames, dict):
                self._all_values.extend(ep_frames.values())
            else:
                # single value per episode case
                self._all_values.append(ep_frames)
        self.global_mean = float(np.mean(self._all_values)) if self._all_values else 0.0
        key_type = "path" if self._use_path_keys else "index"
        print(f"Loaded logpi values for {len(self.logpi_dict)} episodes from {logpi_file} (keyed by {key_type})")
    
    def __getitem__(self, idx):
        # Get original data using get_item method
        data = self.get_item(idx)
        
        # Extract episode and frame information
        # Resolve the episode key based on the detected format
        if self._use_path_keys:
            episode_key = str(self.episode_infos[idx].path)
        else:
            episode_key = idx  # use numeric index
        frame_idx = None
        if 'meta' in data:
            frame_idx = data['meta'].get('step_id', None)
        
        # Get logpi value
        logpi_value = self.global_mean  # Default to global mean if nothing else available
        if episode_key is not None and frame_idx is not None and episode_key in self.logpi_dict:
            ep_frames = self.logpi_dict[episode_key]
            if isinstance(ep_frames, dict) and frame_idx in ep_frames:
                logpi_value = ep_frames[frame_idx]
            else:
                # Interpolate within the episode: linear between nearest frames; clamp at boundaries
                if isinstance(ep_frames, dict) and len(ep_frames) > 0:
                    keys = sorted(ep_frames.keys())
                    # Find neighbors
                    # If below min, clamp
                    if frame_idx <= keys[0]:
                        logpi_value = ep_frames[keys[0]]
                    # If above max, clamp
                    elif frame_idx >= keys[-1]:
                        logpi_value = ep_frames[keys[-1]]
                    else:
                        # Find right neighbor index
                        import bisect
                        r = bisect.bisect_right(keys, frame_idx)
                        l = r - 1
                        f0, f1 = keys[l], keys[r]
                        v0, v1 = ep_frames[f0], ep_frames[f1]
                        t = (frame_idx - f0) / max(1, (f1 - f0))
                        logpi_value = float(v0 + (v1 - v0) * t)
        
        # Add logpi to data
        data['logpi'] = np.array([logpi_value], dtype=np.float32)
        
        return data
    
    def get_logpi_statistics(self):
        """Get statistics about loaded logpi values"""
        all_logpi = []
        for ep_frames in self.logpi_dict.values():
            if isinstance(ep_frames, dict):
                all_logpi.extend(ep_frames.values())
            else:
                all_logpi.append(ep_frames)
        
        if all_logpi:
            return {
                'count': len(all_logpi),
                'mean': np.mean(all_logpi),
                'std': np.std(all_logpi),
                'min': np.min(all_logpi),
                'max': np.max(all_logpi)
            }
        else:
            return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}
