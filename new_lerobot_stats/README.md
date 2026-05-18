# Dataset statistics for per-dim min-max normalization

`dataset_statistics.json` here is consumed by:

* `scripts/analyze_episode_quality.py` (Stage 2a, Eq.(1)-(3) pre-screening)
* `scripts/eval_smoothness.py` (Tab. III Acc / Jerk metrics)
* `data/lerobot_vla_dataset.py` (training-time normalization)

The released `Dexora_Real-World_Dataset` does **not** ship this file
because it depends on which task subset you train on. Generate it once
from your subset (≈ 2 min) with:

```bash
python -m data.lerobot_vla_dataset --stat \
    --num_samples 5000 \
    --repo_dir   /path/to/Dexora_Real-World_Dataset/<subset> \
    --output_dir new_lerobot_stats
```

This writes `new_lerobot_stats/dataset_statistics.json` (36-D, paper layout)
plus `state_distributions.png` / `action_distributions.png` for a quick
sanity check.

## Schema

```json
{
  "state":  { "mean": [..36..], "std": [..36..], "percentile_1": [..36..], "percentile_99": [..36..] },
  "action": { "mean": [..36..], "std": [..36..], "percentile_1": [..36..], "percentile_99": [..36..] },
  "metadata": { "num_samples": 5000, "state_dim": 36, "action_dim": 36, "timestamp": "..." }
}
```

The HF release stores **39-D** state/action (the last 3 dims are
`head_joint_1, head_joint_2, spine_joint`, fixed by the AIRBOT platform
SDK but not modelled by the Dexora policy). `data/lerobot_vla_dataset.py`
slices to the first 36 dims by default, so the stats file is 36-D.
