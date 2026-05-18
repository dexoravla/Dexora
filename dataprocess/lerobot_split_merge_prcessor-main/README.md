# LeRobot Split & Merge Processor

A command-line toolkit for splitting and merging robotics datasets:
- Split: select the first N frames or the first N episodes from a single dataset and emit a subset
- Merge: select episodes from multiple source datasets and merge them into a single unified dataset

Project layout:
- `lerobot_dataset_lib.py`: function library that provides the low-level capabilities (selection, statistics merging, video and data copying, dimensionality and index alignment)
- `split_merge_dataset.py`: command-line entry point that parses arguments and invokes the helpers in `lerobot_dataset_lib.py`

## Installation and environment

The following environment and packages are required at runtime:
- Python 3.8+ (3.10 recommended)
- pandas (for reading/writing Parquet)
- At least one Parquet backend: pyarrow or fastparquet (one or both)
- numpy

Install via pip:
```bash
pip install -U pandas numpy pyarrow fastparquet
```

Alternative: create an isolated environment with conda:
```bash
conda create -n lerobot-dp python=3.10
conda activate lerobot-dp
conda install -c conda-forge pandas numpy pyarrow fastparquet
```
Note: without any Parquet engine, pandas's `read_parquet`/`to_parquet` will fail.
## Usage

Command-line entry point:
```bash
python split_merge_dataset.py --help
```

### Subcommands

- `split`: split a single dataset into a subset
- `merge`: combine multiple datasets into a single dataset

## Common conventions
- The dataset directory must contain `meta/info.json`, `meta/episodes.jsonl`, `meta/episodes_stats.jsonl`, `meta/tasks.jsonl`, etc. Videos and data frames are organized according to the structure defined in `info.json`.
- Starting-offset priority: `start_entries` (frame offset) takes precedence over `start_episodes` (episode offset); when a frame offset is set, it is aligned to whole episodes.
- Truncation priority: `max_entries` (frame count) takes precedence over `max_episodes`.
- Dimensionality handling: zero-pad `observation.state` and `action` to `max_dim` and update `features.shape` in `info.json`; when unset, the maximum dimensionality across all sources is used.
- Frame-rate handling: when `fps` is not provided, it is read from the source dataset's `meta/info.json` and written into the output dataset.

## Parameter reference

### split parameters

- `--input`: input dataset path (required)
- `--output`: output dataset path (required)
- `--max_entries`: maximum number of frames (truncate by frame count)
- `--max_episodes`: maximum number of episodes (truncate by episode count)
- `--fps`: frame rate of the output dataset; if not provided, read from `input/meta/info.json`
- `--max_dim`: target dimensionality; if not provided, use the maximum dimensionality across the sources
- `--start_entries`: starting frame offset (skip the first N frames; takes priority over the starting episode offset)
- `--start_episodes`: starting episode offset (skip the first N episodes)

Example (split by episode count):
```bash
python split_merge_dataset.py split \
  --input /path/to/source_dataset \
  --output /path/to/output_subset \
  --start_episodes 2 \
  --max_episodes 300 \
  --fps 20 \
  --max_dim 32
```

Example (take 10,000 frames starting from frame 20,000, aligned to whole episodes):
```bash
python split_merge_dataset.py split \
  --input /path/to/source_dataset \
  --output /path/to/output_subset \
  --start_entries 20000 \
  --max_entries 10000 \
  --fps 20 \
  --max_dim 32
```

### merge parameters

- `--sources`: list of source dataset paths (optional, supports multiple paths)
- `--sources_dir`: parent directory of source datasets (scans first- and second-level subdirectories, auto-discovering LeRobot datasets containing `meta/info.json`)
- `--output`: output dataset path (required)
- `--max_episodes`: maximum number of episodes (truncate by count after the global merge)
- `--fps`: frame rate of the output dataset; if not provided, read from the first valid source's `meta/info.json`, falling back to 20 if missing
- `--max_dim`: target dimensionality; if not provided, unify and zero-pad to the maximum dimensionality across all source datasets
- `--start_entries`: starting frame offset (skip the first N frames across all sources, aligned to whole episodes)
- `--start_episodes`: starting episode offset (skip the first N episodes across all sources)

Example (multiple sources listed explicitly):
```bash
python split_merge_dataset.py merge \
  --sources /path/to/source_a \
            /path/to/source_b \
            /path/to/source_c \
  --output /path/to/merged_dataset \
  --max_episodes 550 \
  --fps 20 \
  --max_dim 32
```

Example (auto-discover first- and second-level subdirectories):
```bash
python split_merge_dataset.py merge \
  --sources_dir /path/to/datasets_root \
  --output /path/to/merged_dataset \
  --max_episodes 300 \
  --fps 20 \
  --max_dim 32
```

Example (starting offset + merge count limit):
```bash
python split_merge_dataset.py merge \
  --sources_dir /path/to/datasets_root \
  --output /path/to/merged \
  --start_entries 5000 \
  --max_episodes 500 \
  --fps 20 \
  --max_dim 32
```

### Auto-discovery rules
- Level 1: check whether direct subdirectories of `sources_dir` contain `meta/info.json`; if so, treat them as datasets
- Level 2: if a level-1 subdirectory is not a dataset, check whether its subdirectories contain `meta/info.json`
- The resulting set is deduplicated and sorted; if no valid dataset is found, an error is raised

## Output contents

The generated dataset directory contains:
- `meta/episodes.jsonl`: list of selected episodes, renumbered
- `meta/episodes_stats.jsonl`: per-episode statistics (with dimensionality aligned)
- `meta/tasks.jsonl`: only the tasks that are actually used are kept (filtered by task_index)
- `meta/stats.json`: global statistics, merged and recomputed from per-episode stats (means/counts)
- `meta/info.json`: summary info (total frame count, total episode count, splits, features.shape, fps, chunks_size, total_videos, etc.)
- `videos/`: videos copied according to `info.json`'s `video_path` template
- `data/chunk-XXX/episode_YYYYYY.parquet`: per-chunk frame data, with dimensions and indices aligned

## Directory structure outline

```plaintext
/path/to/output_dataset
.
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── episodes_stats.jsonl
│   ├── tasks.jsonl
│   └── stats.json
├── videos/
│   └── chunk-000/
│       └── {video_key}/episode_{episode_index:06d}.mp4
└── data/
    └── chunk-000/
        └── episode_{episode_index:06d}.parquet
```

- meta/info.json
  - Key fields: `total_episodes`, `total_frames`, `total_tasks`, `total_chunks`, `fps`, `features`, `chunks_size`, `total_videos`
  - `splits` is automatically set to `{"train": "0:total_episodes"}` (covers all episodes)
  - Example `video_path`: `videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4`
  - Inside `features`, the `shape` of `observation.state` and `action` are aligned to `[max_dim]`

- meta/episodes.jsonl
  - One episode object per line; `episode_index` is renumbered to be 0-based and contiguous; keeps `length`, `task_index`, etc.

- meta/episodes_stats.jsonl
  - Per-episode statistics; contains `min/max/mean/std/count`, with `observation.state` and `action` padded to `max_dim`.

- meta/tasks.jsonl
  - Task list (only the actually used `task_index` values are kept); duplicates across sources are deduplicated by description and renumbered.

- meta/stats.json
  - Global statistics: merged from each source's stats and recomputed from per-episode statistics (means/counts), with dimensionality unified to `max_dim`.

- videos/
  - Directory layout matches `info["video_path"]`; organized by `chunk` and `video_key`; filenames contain the merged `episode_index`.

- data/
  - `data/chunk-XXX/episode_YYYYYY.parquet`: sharded by `chunks_size`; `observation.state` and `action` are zero-padded to `max_dim`; `index` is the global frame index, `episode_index` is the merged numbering.

### Naming and numbering conventions
- `episode_index`: 0-based, filenames are zero-padded to 6 digits (e.g. `episode_000123`)
- `chunk`: computed as `episode_index // chunks_size` (e.g. `chunk-000`)
- `splits`: defaults to `train: 0:total_episodes`; modify `info.json` directly if `val`/`test` splits are needed

## Notes

- Running the copy and statistics stages requires `pandas` and a Parquet backend (`pyarrow` or `fastparquet`).
- If a source dataset's `meta/info.json` lacks `video_path`, video copying will be skipped; please ensure the key exists or add a safeguard in the library.
- Make sure that the `features` definitions across source datasets are consistent, especially for video keys (`dtype == "video"`) and `chunks_size`, in order to avoid copy-path issues.
- Other automatic behavior: `tasks.jsonl` is filtered to actually used tasks; `splits` is automatically set to `train: 0:total_episodes`; `chunks_size` is inherited from the source `info.json`.
