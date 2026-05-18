#!/usr/bin/env python3
"""
Stage-2a: episode-level pre-screening (Dexora §III-C, Eq.(1)-(3)).

For every episode tau we min-max normalize the proprioceptive state per
dimension, then use **centered finite differences** to compute velocity /
acceleration / jerk and collapse them to scalar episode-level RMS values:

    Aep(tau) = sqrt( 1/((T-6) D) * sum_{t,k} a_{t,k}^2 )       Eq.(2)
    Jep(tau) = sqrt( 1/((T-6) D) * sum_{t,k} j_{t,k}^2 )       Eq.(3)

The pre-screening set is then

    Spre = Low-r(Aep)  ∩  Low-r(Jep)             (r = target_ratio, paper: 0.2)

which the paper reports retains ~18% of episodes. ``Spre`` is the input to
``replay_validate.py`` (Stage-2b), whose output is the discriminator's
positive set ``Shigh``.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm

# Add lerobot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lerobot', 'src'))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from data.bson_vla_dataset import BsonVLADataset


class DataNormalizer:
    """
    Data normalizer following the approach from LeRobot VLA dataset.
    """
    
    def __init__(self, stats_file: Optional[str] = None, normalize_mode: str = "min_max"):
        self.normalize_mode = normalize_mode
        self.stats = None
        if stats_file:
            self._load_statistics(stats_file)
    
    def _load_statistics(self, stats_file: str):
        """Load statistics from JSON file for normalization."""
        try:
            with open(stats_file, 'r') as f:
                self.stats = json.load(f)
            print(f"Loaded statistics from {stats_file}")
            print(f"State dim: {len(self.stats['state']['mean'])}, Action dim: {len(self.stats['action']['mean'])}")
        except Exception as e:
            print(f"Warning: Failed to load statistics from {stats_file}: {e}")
            self.stats = None
    
    def normalize_data(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """Normalize data using loaded statistics."""
        if self.stats is None or self.normalize_mode is None:
            return data
            
        if data_type not in self.stats:
            print(f"Warning: No statistics found for {data_type}")
            return data
            
        stats_data = self.stats[data_type]
        
        if self.normalize_mode == 'mean_std':
            # Normalize using mean and standard deviation: (x - mean) / std
            mean = np.array(stats_data['mean'])
            std = np.array(stats_data['std'])
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            normalized = (data - mean) / std
            
        elif self.normalize_mode == 'min_max':
            # Normalize using percentiles as min/max: (x - min) / (max - min)
            min_val = np.array(stats_data['percentile_1'])
            max_val = np.array(stats_data['percentile_99'])
            # Avoid division by zero
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val
            
        else:
            print(f"Warning: Unknown normalization mode {self.normalize_mode}")
            return data
            
        return normalized


def disable_video_loading(dataset):
    """Disable video loading to speed up data access."""
    object.__setattr__(dataset, 'video_keys', [])
    object.__setattr__(dataset, 'image_transforms', None)
    dataset.meta.info["features"]={}
    object.__setattr__(dataset, 'image_transforms', None)


def calculate_episode_metrics(states: np.ndarray, dt: float) -> Dict[str, float]:
    """
    Per-episode kinematic smoothness metrics following Dexora Eq.(1)-(3).

    Given a (T, D) trajectory of per-dimension min-max normalized states and a
    sampling interval ``dt``, we use **centered finite differences** to obtain
    velocity / acceleration / jerk:

        v_t = (s_{t+1} - s_{t-1}) / (2 dt)
        a_t = (v_{t+1} - v_{t-1}) / (2 dt)
        j_t = (a_{t+1} - a_{t-1}) / (2 dt)

    Episode-level Aep / Jep are RMS values across **both time and dimension**
    (the inner window t = 4..T-3, i.e. length T-6, exactly as in the paper):

        Aep = sqrt( 1/((T-6)D) * sum_{t,k} a_{t,k}^2 )
        Jep = sqrt( 1/((T-6)D) * sum_{t,k} j_{t,k}^2 )

    Returns a dict with scalar Aep / Jep (or +inf for too-short episodes).
    """
    T = len(states)
    if T < 7:
        return {
            'Aep': float('inf'),
            'Jep': float('inf'),
            'is_buggy': False,
            'T': int(T),
        }

    v = (states[2:] - states[:-2]) / (2.0 * dt)              # length T-2
    a = (v[2:] - v[:-2]) / (2.0 * dt)                         # length T-4
    j = (a[2:] - a[:-2]) / (2.0 * dt)                         # length T-6

    # Align Acc to the same inner (T-6) window as Jerk (paper: t=4..T-3).
    a_inner = a[1:-1]

    Aep = float(np.sqrt(np.mean(a_inner ** 2)))               # RMS over time x dim
    Jep = float(np.sqrt(np.mean(j ** 2)))

    is_buggy = bool(np.allclose(Aep, 0.0) and np.allclose(Jep, 0.0))

    return {
        'Aep': Aep,
        'Jep': Jep,
        'is_buggy': is_buggy,
        'T': int(T),
    }


def sample_episodes(dataset: LeRobotDataset, num_episodes: int = 100, normalizer: Optional[DataNormalizer] = None, state_dim_keep: Optional[int] = 36) -> List[Tuple[int, np.ndarray]]:
    """
    Sample random episodes from the dataset with optional normalization.
    
    Args:
        dataset: LeRobotDataset instance
        num_episodes: Number of episodes to sample
        normalizer: Optional data normalizer
        
    Returns:
        List of (episode_idx, states) tuples
    """
    if hasattr(dataset, "episode_data_index"):
        total_episodes = len(dataset.episode_data_index["from"])
    else:
        total_episodes = len(dataset)
    episode_indices = random.sample(range(total_episodes), min(num_episodes, total_episodes))
    
    sampled_episodes = []
    
    for ep_idx in tqdm(episode_indices, desc="Sampling episodes"):
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        
        # Extract states for this episode. LeRobot v2.1 column is
        # ``observation.state``; older custom converters used plural ``states``.
        # Probe once per episode and stick with whichever works.
        episode_states = []
        state_key = None
        for i in tqdm(range(ep_start, ep_end), desc=f"Loading episode {ep_idx}", leave=False):
            try:
                sample = dataset[i]
                if state_key is None:
                    state_key = (
                        'observation.state' if 'observation.state' in sample
                        else 'states'
                    )
                state = sample[state_key]
                state = state.numpy() if hasattr(state, 'numpy') else np.asarray(state)
                episode_states.append(state)
            except Exception as e:
                print(f"Error loading sample {i}: {e}")
                continue
        
        if len(episode_states) > 2:  # Need at least 3 timesteps
            states_array = np.array(episode_states)

            # Slice to the paper's 36-D layout if requested. The Dexora HF
            # release is 39-D (adds head_joint_1, head_joint_2, spine_joint);
            # the paper policy and the released stats files are both 36-D.
            if state_dim_keep is not None and states_array.shape[-1] > int(state_dim_keep):
                states_array = states_array[..., :int(state_dim_keep)]

            # Apply normalization if provided
            if normalizer is not None:
                states_array = normalizer.normalize_data(states_array, 'state')

            sampled_episodes.append((ep_idx, states_array))
    
    return sampled_episodes


def analyze_episodes(
    episodes: List[Tuple[int, np.ndarray]],
    dt: float,
) -> Dict[str, np.ndarray]:
    """
    Compute Aep / Jep for every episode (Dexora Eq.(2)-(3)).

    ``dt = 1/fps`` is fixed across the dataset (the paper logs at 20 Hz).
    Output arrays are indexed by ``episode_indices`` and contain **scalars**
    per episode (not per-dim arrays), so downstream ranking matches Eq.(2)(3).
    """
    Aep_list: List[float] = []
    Jep_list: List[float] = []
    episode_lengths: List[int] = []
    episode_indices: List[int] = []
    buggy_episodes: List[int] = []
    valid_episodes: List[int] = []

    for ep_idx, states in tqdm(episodes, desc="Analyzing episodes"):
        metrics = calculate_episode_metrics(states, dt=dt)

        if metrics['is_buggy']:
            buggy_episodes.append(ep_idx)
            print(f"Episode {ep_idx}: BUGGY (all zeros) - Length={metrics['T']}")
            continue

        Aep_list.append(metrics['Aep'])
        Jep_list.append(metrics['Jep'])
        episode_lengths.append(metrics['T'])
        episode_indices.append(ep_idx)
        valid_episodes.append(ep_idx)

        print(
            f"Episode {ep_idx}: T={metrics['T']}, "
            f"Aep={metrics['Aep']:.4f}, Jep={metrics['Jep']:.4f}"
        )

    print(f"\nFound {len(buggy_episodes)} buggy episodes: {buggy_episodes}")
    print(f"Valid episodes for analysis: {len(valid_episodes)}")

    return {
        'Aep': np.asarray(Aep_list, dtype=np.float64),
        'Jep': np.asarray(Jep_list, dtype=np.float64),
        'episode_lengths': np.asarray(episode_lengths, dtype=np.int64),
        'episode_indices': np.asarray(episode_indices, dtype=np.int64),
        'buggy_episodes': buggy_episodes,
        'valid_episodes': valid_episodes,
        'dt': float(dt),
    }


def calculate_filtering_thresholds(
    metrics: Dict[str, np.ndarray],
    target_ratio: float = 0.2,
) -> Dict[str, object]:
    """
    Pre-screening set ``Spre`` per Dexora §III-C:

        Spre = { tau : tau in Low-r(Aep)  AND  tau in Low-r(Jep) }

    where ``r = target_ratio`` (paper: 0.2). We rank episodes by Aep and by
    Jep **separately**, keep the lowest-``r`` fraction of each list, then take
    the intersection. The paper reports that this retains ~18% of episodes
    when r=0.2; we print the same statistic so users can sanity-check.

    For backward compatibility the returned dict still exposes the
    per-quantity thresholds ``jerk_threshold`` / ``acceleration_threshold``
    (i.e. the values at the r-th percentile), but they are reported, not
    used as a filter (the rank-based intersection is the filter).
    """
    Aep = metrics['Aep']
    Jep = metrics['Jep']
    ep_indices = metrics['episode_indices']
    n = len(ep_indices)
    assert Aep.shape == Jep.shape == (n,), (
        f"Aep/Jep/index shape mismatch: {Aep.shape}/{Jep.shape}/{ep_indices.shape}"
    )

    # Number of episodes to keep in each Low-r(.) list (paper: floor(r * N)).
    keep_n = max(1, int(np.floor(target_ratio * n)))

    # Lowest-keep_n indices for each criterion.
    low_a = set(np.argsort(Aep, kind='stable')[:keep_n].tolist())
    low_j = set(np.argsort(Jep, kind='stable')[:keep_n].tolist())

    spre_local = sorted(low_a & low_j)  # positions into ep_indices
    spre_episodes = ep_indices[np.array(spre_local, dtype=np.int64)] if spre_local else np.array([], dtype=np.int64)

    # Report-only thresholds: the Aep / Jep values at rank ``keep_n - 1``.
    a_thr = float(np.sort(Aep)[keep_n - 1])
    j_thr = float(np.sort(Jep)[keep_n - 1])

    retained_ratio = len(spre_episodes) / max(1, n)
    print(
        f"\n=== Pre-screening (Spre) results (target r={target_ratio*100:.1f}%) ===\n"
        f"  N (valid episodes)        : {n}\n"
        f"  |Low-r(Aep)|              : {keep_n}\n"
        f"  |Low-r(Jep)|              : {keep_n}\n"
        f"  |Spre| = intersection     : {len(spre_episodes)}  ({retained_ratio*100:.2f}% of N)\n"
        f"  Aep r-percentile (report) : {a_thr:.6f}\n"
        f"  Jep r-percentile (report) : {j_thr:.6f}"
    )

    return {
        'jerk_threshold': j_thr,
        'acceleration_threshold': a_thr,
        'valid_episodes': spre_episodes.tolist(),
        'num_valid': int(len(spre_episodes)),
        'total_analyzed': int(n),
        'actual_ratio': float(retained_ratio),
        'target_ratio': float(target_ratio),
        'low_keep_per_axis': int(keep_n),
    }


def create_filtering_plots(metrics: Dict[str, np.ndarray], output_dir: str):
    """
    Plot CDFs of episode-level Aep / Jep (Eq.(2)-(3)) and a small JSON sidecar
    of the underlying values so the user can re-derive any threshold downstream.
    """
    Aep = np.asarray(metrics['Aep'], dtype=np.float64)
    Jep = np.asarray(metrics['Jep'], dtype=np.float64)
    n = Aep.size

    if n == 0:
        print("[create_filtering_plots] No episodes to plot; skipping.")
        return

    j_thresholds = np.linspace(Jep.min(), Jep.max(), 50)
    a_thresholds = np.linspace(Aep.min(), Aep.max(), 50)
    j_ratios = [float(np.sum(Jep <= t) / n) for t in j_thresholds]
    a_ratios = [float(np.sum(Aep <= t) / n) for t in a_thresholds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.fill_between(j_thresholds, 0, j_ratios, alpha=0.6, color='cyan')
    ax1.plot(j_thresholds, j_ratios, color='darkcyan', linewidth=2)
    ax1.set_xlabel('Episode Jerk RMS Jep (normalized state)')
    ax1.set_ylabel('Remaining Episode Ratio')
    ax1.set_title('Pre-screening CDF: Jerk (Eq.(3))')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(a_thresholds, 0, a_ratios, alpha=0.6, color='orange')
    ax2.plot(a_thresholds, a_ratios, color='darkorange', linewidth=2)
    ax2.set_xlabel('Episode Acceleration RMS Aep (normalized state)')
    ax2.set_ylabel('Remaining Episode Ratio')
    ax2.set_title('Pre-screening CDF: Acceleration (Eq.(2))')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'episode_quality_filtering_summary.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    results = {
        'jerk_thresholds': j_thresholds.tolist(),
        'acceleration_thresholds': a_thresholds.tolist(),
        'jerk_cdf': j_ratios,
        'acceleration_cdf': a_ratios,
        'Aep_values': Aep.tolist(),
        'Jep_values': Jep.tolist(),
        'episode_indices': np.asarray(metrics['episode_indices'], dtype=np.int64).tolist(),
        'total_episodes_analyzed': int(n),
        'dt': float(metrics.get('dt', float('nan'))),
    }
    with open(os.path.join(output_dir, 'filtering_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Analyze episode quality metrics with normalization')
    parser.add_argument('num_episodes', type=int,
                        help='Number of episodes to sample')
    parser.add_argument('--output_dir', type=str, default='episode_quality_analysis',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for episode sampling')
    parser.add_argument('--stats_file', type=str, default='v4_lerobot_stats/dataset_statistics.json',
                        help='Path to dataset statistics file for normalization')
    parser.add_argument('--normalize_mode', type=str, choices=['mean_std', 'min_max'], default='min_max',
                        help='Normalization mode')
    parser.add_argument('--target_ratio', type=float, default=0.2,
                        help='Target ratio of episodes to keep (default: 0.2 for 20%%)')
    parser.add_argument('--lerobot_root', type=str,
                        default='dataprocess/output/airbot_dexterous_bimanual_dexterous_manipulation',
                        help='Path to the LeRobot dataset root.')
    parser.add_argument('--fps', type=float, default=None,
                        help=("Override sampling frequency. If unset, we read it from the "
                              "LeRobot dataset metadata; if that is also missing, default to "
                              "20 Hz (Dexora paper §III-A)."))
    parser.add_argument('--state_dim_keep', type=int, default=36,
                        help=("Slice each frame's state vector to the first N dims before "
                              "computing Aep / Jep. Default 36 matches the paper's flat "
                              "[left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)] "
                              "layout. Pass 0 to keep the full 39-D vector (HF release adds "
                              "head_joint_1, head_joint_2, spine_joint)."))

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    normalizer = DataNormalizer(stats_file=args.stats_file, normalize_mode=args.normalize_mode)

    # LeRobot v2.1 columns are singular (``observation.state``, ``action``).
    delta_timestamps = {
        'observation.state': [0],
    }
    dataset = LeRobotDataset("", args.lerobot_root, delta_timestamps=delta_timestamps)
    disable_video_loading(dataset)
    print(f"Dataset loaded successfully. Total samples: {len(dataset)}")

    # Resolve dt = 1 / fps. Prefer CLI override, then dataset metadata, then 20 Hz.
    fps = args.fps
    if fps is None:
        ds_fps = getattr(getattr(dataset, "meta", None), "fps", None)
        fps = float(ds_fps) if ds_fps else 20.0
    dt = 1.0 / float(fps)
    print(f"Using fps={fps} Hz (dt={dt:.6f}s) for centered-difference derivatives.")

    state_dim_keep = int(args.state_dim_keep) if args.state_dim_keep and args.state_dim_keep > 0 else None
    print(f"Sampling {args.num_episodes} episodes with normalization (state_dim_keep={state_dim_keep})...")
    episodes = sample_episodes(dataset, args.num_episodes, normalizer, state_dim_keep=state_dim_keep)
    print(f"Successfully sampled {len(episodes)} episodes")

    if len(episodes) == 0:
        print("No valid episodes found!")
        return

    print("Analyzing episode metrics...")
    metrics = analyze_episodes(episodes, dt=dt)

    if len(metrics['valid_episodes']) == 0:
        print("No valid episodes found for analysis!")
        return

    print("Calculating Spre via Low-r(Aep) ∩ Low-r(Jep) ...")
    filtering_results = calculate_filtering_thresholds(metrics, args.target_ratio)

    Aep = metrics['Aep']
    Jep = metrics['Jep']
    print("\n=== Summary Statistics ===")
    print(f"Total episodes sampled: {len(episodes)}")
    print(f"Valid episodes analyzed: {len(metrics['valid_episodes'])}")
    print(f"Buggy episodes found: {len(metrics['buggy_episodes'])}")
    print(f"Average episode length: {np.mean(metrics['episode_lengths']):.1f}")
    print(f"Aep (Eq.(2)) mean/std/min/max: "
          f"{Aep.mean():.6f} / {Aep.std():.6f} / {Aep.min():.6f} / {Aep.max():.6f}")
    print(f"Jep (Eq.(3)) mean/std/min/max: "
          f"{Jep.mean():.6f} / {Jep.std():.6f} / {Jep.min():.6f} / {Jep.max():.6f}")

    print("Creating filtering threshold plots...")
    create_filtering_plots(metrics, args.output_dir)

    complete_results = {
        'filtering_thresholds': filtering_results,
        'buggy_episodes': metrics['buggy_episodes'],
        'all_valid_episodes': metrics['valid_episodes'],
        'summary_stats': {
            'total_sampled': len(episodes),
            'valid_analyzed': len(metrics['valid_episodes']),
            'buggy_found': len(metrics['buggy_episodes']),
            'avg_episode_length': float(np.mean(metrics['episode_lengths'])),
            'Aep_mean': float(Aep.mean()),
            'Aep_std': float(Aep.std()),
            'Jep_mean': float(Jep.mean()),
            'Jep_std': float(Jep.std()),
            'fps': float(fps),
            'dt': float(dt),
        },
        'normalization_settings': {
            'stats_file': args.stats_file,
            'normalize_mode': args.normalize_mode,
            'target_ratio': args.target_ratio,
        },
        # Per-episode Aep / Jep for downstream debugging / re-thresholding.
        'episode_metrics': [
            {
                'episode_index': int(ep),
                'Aep': float(a),
                'Jep': float(j),
                'T': int(t),
            }
            for ep, a, j, t in zip(
                metrics['episode_indices'],
                Aep,
                Jep,
                metrics['episode_lengths'],
            )
        ],
    }

    with open(os.path.join(args.output_dir, 'complete_analysis_results.json'), 'w') as f:
        json.dump(complete_results, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print(f"Complete results saved to: {args.output_dir}/complete_analysis_results.json")


if __name__ == "__main__":
    main()
