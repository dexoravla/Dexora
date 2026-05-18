#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from data.bson_vla_dataset import BsonVLADataset
from data.lerobot_vla_dataset import LeRobotVLADataset
from train.dataset import DataCollatorForVLAConsumerDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Compute logpi values for dataset frames.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/base.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/dexrdt-400m-v5",
        help="Path to the pretrained RDT model checkpoint.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Path to the dataset directory. For ``--load_from=lerobot`` this "
            "is the LeRobot v2.1 dataset root (e.g. "
            "``data/Dexora_Real-World_Dataset/airbot_pick_and_place``); for "
            "``--load_from=bson`` this is the BSON dataset root."
        ),
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default="lerobot",
        choices=["lerobot", "bson"],
        help="Dataset backend (LeRobot v2.1 release or legacy BSON).",
    )
    parser.add_argument(
        "--stats_file",
        type=str,
        default=None,
        help="Optional dataset_statistics.json for normalization.",
    )
    parser.add_argument(
        "--state_dim_keep",
        type=int,
        default=36,
        help="Slice state/action to first N dims (36 = paper layout).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="logpi_values.json",
        help="Output file to save logpi dictionary.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--num_noise_steps",
        type=int,
        default=4,
        help="Number of noise steps S to sample for logpi calculation.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="Maximum number of episodes to process (-1 for all).",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default="google/t5-v1_1-xxl",
        help="Pretrained text encoder name or path.",
    )
    parser.add_argument(
        "--pretrained_vision_encoder_name_or_path",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Pretrained vision encoder name or path.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="zscore",
        choices=["zscore", "minmax", "none"],
        help=(
            "How to normalize the raw energies into the log-pi proxy used by the discriminator. "
            "'zscore' implements Eq.(5) of the paper: \\hat{log pi} = -zscore(E). "
            "'minmax' rescales to [0,1] (legacy behaviour). 'none' keeps the raw -E values."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Optional temperature applied after normalization (1.0 = no-op).",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Subsample factor n: compute logpi every n frames within each episode (use step_id %% n == 0).",
    )
    parser.add_argument(
        "--verbose_timing",
        action="store_true",
        help="Print detailed timing for each major step.",
    )
    
    return parser.parse_args()


def compute_denoising_residual_energy(model, batch_data, num_noise_steps, vision_encoder, verbose=False):
    """
    Compute the denoising residual energy (logpi) for a batch of data.
    Uses RDTRunner's existing compute_loss method for consistency.
    
    Args:
        model: RDTRunner model
        batch_data: Batch of data samples
        num_noise_steps: Number of noise steps S to sample
        vision_encoder: Vision encoder for processing images
        
    Returns:
        logpi values for each sample in batch
    """
    device = next(model.parameters()).device
    
    # Extract data from batch (use correct keys from collated batch)
    lang_tokens = batch_data.get('lang_embeds')  # [B, lang_len, lang_dim]
    lang_attn_mask = batch_data.get('lang_attn_mask')  # [B, lang_len]
    img_tokens = batch_data.get('images')  # Use 'images' key from collated batch
    state_tokens = batch_data.get('states')  # [B, 1, state_dim]
    action_gt = batch_data.get('actions')  # [B, horizon, action_dim]
    action_mask = batch_data.get('state_elem_mask')  # [B, action_dim] - use state_elem_mask
    ctrl_freqs = batch_data.get('ctrl_freqs')  # [B]
    
    # Assert all required data is present
    assert state_tokens is not None, "state_tokens is required but got None"
    assert action_gt is not None, "action_gt is required but got None"
    assert action_mask is not None, "action_mask is required but got None"
    assert ctrl_freqs is not None, "ctrl_freqs is required but got None"
    assert lang_tokens is not None, "lang_tokens is required but got None"
    assert img_tokens is not None, "img_tokens is required but got None"
    
    batch_size = action_gt.shape[0]
    
    # Process images through vision encoder to get img_tokens BEFORE dtype conversion
    # In our collator, images are [B, N, C, H, W] where N = img_history_size * num_cameras
    B, N, C, H, W = img_tokens.shape

    t0 = time.perf_counter()

    # Preprocess images using vision encoder's image processor (similar to VLAConsumerDataset)
    processed_images = []
    for b in range(B):
        for n in range(N):
            img_tensor = img_tokens[b, n]  # [C, H, W]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            from PIL import Image
            img_pil = Image.fromarray(img_np, mode='RGB')
            processed_img = vision_encoder.image_processor.preprocess(
                img_pil, return_tensors='pt'
            )["pixel_values"][0]
            processed_images.append(processed_img)
    t1 = time.perf_counter()

    # Stack processed images and encode to tokens
    processed_img_tensor = torch.stack(processed_images, dim=0).to(device)

    # Encode images to tokens
    with torch.no_grad():
        image_embeds = vision_encoder(processed_img_tensor).detach()  # [B*N, L, hidden]
        img_tokens = image_embeds.reshape((B, -1, vision_encoder.hidden_size))
    t2 = time.perf_counter()
    
    # Align inputs to the corresponding module dtypes (do not change model weights)
    try:
        core_dtype = next(model.model.parameters()).dtype
    except StopIteration:
        core_dtype = next(model.parameters()).dtype
    lang_ad_dtype = next(model.lang_adaptor.parameters()).dtype
    img_ad_dtype = next(model.img_adaptor.parameters()).dtype
    state_ad_dtype = next(model.state_adaptor.parameters()).dtype

    # Cast per consumer
    lang_tokens = lang_tokens.to(device, dtype=lang_ad_dtype)
    img_tokens = img_tokens.to(device, dtype=img_ad_dtype)
    state_tokens = state_tokens.to(device, dtype=state_ad_dtype)
    action_gt = action_gt.to(device, dtype=state_ad_dtype)
    action_mask = action_mask.to(device, dtype=state_ad_dtype)
    ctrl_freqs = ctrl_freqs.to(device, dtype=core_dtype)
    # lang_attn_mask must be boolean and on the same device for fused attention
    if lang_attn_mask is not None:
        if lang_attn_mask.dtype != torch.bool:
            lang_attn_mask = (lang_attn_mask != 0)
        lang_attn_mask = lang_attn_mask.to(device)
    t3 = time.perf_counter()

    # Expand action_mask to correct dimensions
    if action_mask.dim() == 2:
        action_mask = action_mask.unsqueeze(1)  # [B, action_dim] -> [B, 1, action_dim]
    
    # Compute logpi using denoising score matching theory
    # For diffusion models, logpi(a|s) ≈ -∫ ||ε_θ(a_t, t) - ε||² dt
    # We approximate this integral by sampling multiple timesteps
    
    total_energy = torch.zeros(batch_size, device=device)
    
    with torch.no_grad():
        if verbose and torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()
        t_fwd0 = time.perf_counter()
        for _ in range(num_noise_steps):
            # Sample random timestep for each sample in batch
            timesteps = torch.randint(0, model.num_train_timesteps, (batch_size,), device=device).long()
            
            # Sample noise
            noise = torch.randn_like(action_gt)
            
            # Add noise to actions
            noisy_actions = model.noise_scheduler.add_noise(action_gt, noise, timesteps)
            
            # Prepare input sequence (following RDTRunner.compute_loss logic)
            state_action_traj = torch.cat([state_tokens, noisy_actions], dim=1)
            action_mask_expanded = action_mask.expand(-1, state_action_traj.shape[1], -1)
            state_action_traj = torch.cat([state_action_traj, action_mask_expanded], dim=2)
            
            # Adapt conditions
            lang_cond, img_cond, state_action_traj = model.adapt_conditions(
                lang_tokens, img_tokens, state_action_traj)
            # Safety: ensure adapted conditions are on correct device/dtypes
            if lang_cond is not None:
                lang_cond = lang_cond.to(device, dtype=lang_ad_dtype)
            if img_cond is not None:
                img_cond = img_cond.to(device, dtype=img_ad_dtype)
            state_action_traj = state_action_traj.to(device, dtype=core_dtype)
            
            # Predict noise
            pred_noise = model.model(state_action_traj, ctrl_freqs, timesteps, 
                                   lang_cond, img_cond, lang_mask=lang_attn_mask)
            
            # Compute denoising score matching loss per sample
            # This approximates the negative log probability density
            if model.prediction_type == 'epsilon':
                target = noise
            elif model.prediction_type == 'sample':
                # Convert sample prediction back to noise prediction
                alpha_prod_t = model.noise_scheduler.alphas_cumprod[timesteps]
                beta_prod_t = 1 - alpha_prod_t
                target = (noisy_actions - alpha_prod_t.sqrt().view(-1, 1, 1) * action_gt) / beta_prod_t.sqrt().view(-1, 1, 1)
            else:
                raise ValueError(f"Unknown prediction type: {model.prediction_type}")
            
            # Compute MSE per sample (not averaged across batch)
            mse_per_sample = torch.mean((pred_noise - target) ** 2, dim=(1, 2))  # [B]
            
            # Accumulate energy (negative log probability)
            total_energy += mse_per_sample
        if verbose and torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize()
        t_fwd1 = time.perf_counter()
    
    # Return the **raw** denoising-residual energy E_t = (1/|S|L) * sum_s sum_tau
    # ||eps_theta - eps||^2 (Eq.(4)). The sign flip / z-score that produces the
    # log-pi proxy \hat{log pi}_t = -zscore(E_t) (Eq.(5)) is applied **once** at
    # the end of main(), so this function returns positive energies and is easy
    # to reason about in isolation.
    logpi_values = total_energy / num_noise_steps
    
    if verbose:
        print((
            f"[TIMING] preprocess={(t1 - t0)*1000:.1f} ms, "
            f"vision_encode={(t2 - t1)*1000:.1f} ms, casts={(t3 - t2)*1000:.1f} ms, "
            f"diffusion_loop={(t_fwd1 - t_fwd0)*1000:.1f} ms, B={B}, N={N}, C={C}, H={H}, W={W}"
        ))
        if torch.cuda.is_available():
            try:
                mem = torch.cuda.memory_allocated() / (1024**2)
                print(f"[CUDA] allocated={mem:.1f} MB on {torch.cuda.get_device_name(torch.cuda.current_device())}")
            except Exception:
                pass
    return logpi_values


def main():
    args = parse_args()
    
    # Load config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    # Setup device
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Initialize encoders
    print("Loading encoders...")
    text_encoder = T5Embedder(
        from_pretrained=args.pretrained_text_encoder_name_or_path,
        model_max_length=config['dataset']['tokenizer_max_length'],
        local_files_only=False,
        device=device
    )
    
    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path,
        args=None,
        delay_load=False
    )
    # Move vision encoder to device
    try:
        vision_encoder.vision_tower.to(device)
        vision_encoder.vision_tower.eval()
    except Exception as e:
        print(f"[WARN] Failed to move vision encoder to {device}: {e}")
    
    # Load RDT model
    print(f"Loading RDT model from {args.model_path}...")
    rdt_runner = RDTRunner(
        action_dim=config['model']['state_token_dim'],
        pred_horizon=config['common']['action_chunk_size'],
        config=config['model'],
        lang_token_dim=config['model']['lang_token_dim'],
        img_token_dim=config['model']['img_token_dim'],
        state_token_dim=config['model']['state_token_dim'],
        max_lang_cond_len=config['dataset']['tokenizer_max_length'],
        img_cond_len=27*27*config['common']['num_cameras'],  # 27x27 patches per camera, see train.py L156
        # Use float32 to match most checkpoints unless mixed precision is explicitly needed
        dtype=torch.float32
    )
    
    # Load pretrained weights
    if os.path.isdir(args.model_path):
        checkpoint_path = os.path.join(args.model_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            # Try different checkpoint formats
            checkpoint_files = [f for f in os.listdir(args.model_path) if f.endswith('.pt') or f.endswith('.bin')]
            if checkpoint_files:
                checkpoint_path = os.path.join(args.model_path, checkpoint_files[0])
            else:
                raise FileNotFoundError(f"No checkpoint found in {args.model_path}")
    else:
        checkpoint_path = args.model_path
    
    print(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        model_state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        model_state_dict = state_dict['state_dict']
    else:
        model_state_dict = state_dict
    
    rdt_runner.load_state_dict(model_state_dict, strict=False)
    rdt_runner.to(device)
    rdt_runner.eval()
    
    # Debug: print module dtypes
    try:
        core_dtype = next(rdt_runner.model.parameters()).dtype
        lang_ad_dtype = next(rdt_runner.lang_adaptor.parameters()).dtype
        img_ad_dtype = next(rdt_runner.img_adaptor.parameters()).dtype
        state_ad_dtype = next(rdt_runner.state_adaptor.parameters()).dtype
        print(f"[DEBUG] DTypes -> core(model): {core_dtype}, lang_adaptor: {lang_ad_dtype}, img_adaptor: {img_ad_dtype}, state_adaptor: {state_ad_dtype}")
    except Exception as e:
        print(f"[DEBUG] Failed to read module dtypes: {e}")
    
    print(f"Loading dataset from {args.dataset_path} (backend={args.load_from})...")
    if args.load_from == "lerobot":
        ds_kwargs = {}
        if args.dataset_path is not None:
            ds_kwargs["repo_dir"] = args.dataset_path
        if args.stats_file is not None:
            ds_kwargs["stats_file"] = args.stats_file
        if args.state_dim_keep is not None:
            ds_kwargs["state_dim_keep"] = int(args.state_dim_keep) if args.state_dim_keep > 0 else None
        dataset = LeRobotVLADataset(**ds_kwargs)
    elif args.load_from == "bson":
        ds_kwargs = dict(sub_sample=1.0, normalize_mode="min_max")
        if args.dataset_path is not None:
            ds_kwargs["bson_dir"] = args.dataset_path
        if args.stats_file is not None:
            ds_kwargs["stats_file"] = args.stats_file
        dataset = BsonVLADataset(**ds_kwargs)
    else:
        raise ValueError(f"Unknown --load_from: {args.load_from!r}")
    
    # Data collator
    data_collator = DataCollatorForVLAConsumerDataset(
        tokenizer=text_encoder.tokenizer
    )
    
    print(f"Enumerating episodes and frames (stride={max(1, args.frame_stride)})...")

    # Precompute (episode_idx, frame_idx) pairs with stride, supporting both dataset types
    is_bson = hasattr(dataset, "episode_infos")
    pairs = []
    if is_bson:
        ep_indices = list(range(len(dataset.episode_infos)))
        if args.max_episodes > 0:
            ep_indices = ep_indices[:args.max_episodes]
        for ep_idx in ep_indices:
            ep_info = dataset.episode_infos[ep_idx]
            # Extract episode data once to compute valid frame range
            episode_data = dataset._extract_data_from_episode(ep_info)
            if not episode_data:
                continue
            qpos = episode_data["state"]
            num_steps = episode_data["episode_len"]
            if num_steps < dataset.CHUNK_SIZE:
                continue
            EPS = 1e-2
            qpos_delta = np.abs(qpos - qpos[0:1])
            indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
            first_idx = indices[0] if len(indices) > 0 else 1
            if first_idx >= num_steps:
                continue
            start = max(first_idx - 1, 0)
            for t in range(start, num_steps, max(1, args.frame_stride)):
                pairs.append((ep_idx, t))
    else:
        total_episodes = dataset.dataset.meta.total_episodes
        ep_from_arr = dataset.dataset.episode_data_index["from"]
        ep_to_arr = dataset.dataset.episode_data_index["to"]

        ep_indices = list(range(total_episodes))
        if args.max_episodes > 0:
            ep_indices = ep_indices[:args.max_episodes]

        EPS = 1e-2
        stride = max(1, args.frame_stride)
        for ep_idx in ep_indices:
            ep_from = ep_from_arr[ep_idx].item()
            ep_to = ep_to_arr[ep_idx].item()
            num_steps = ep_to - ep_from
            if num_steps < dataset.CHUNK_SIZE:
                continue
            # Probe states to detect the first frame where movement occurs
            try:
                s0 = dataset.get_item(index=ep_idx, frame_index=0)["state"][0]
            except Exception:
                # Skip episode if first frame cannot be loaded
                continue
            first_idx = 1
            for t in range(1, num_steps):
                try:
                    st = dataset.get_item(index=ep_idx, frame_index=t)["state"][0]
                except Exception:
                    continue
                if np.any(np.abs(st - s0) > EPS):
                    first_idx = t
                    break
            if first_idx >= num_steps:
                continue
            start = max(first_idx - 1, 0)
            for t in range(start, num_steps, stride):
                pairs.append((ep_idx, t))

    print(f"Total frames selected: {len(pairs)}")

    # Process in batches; store as ep_idx -> {frame_idx: value}
    logpi_dict = {}

    for i in tqdm(range(0, len(pairs), args.batch_size), desc="Computing logpi"):
        cur_pairs = pairs[i: i + args.batch_size]
        batch_data = []
        t_batch0 = time.perf_counter()
        for (ep_idx, frame_idx) in cur_pairs:
            # Deterministically fetch the specific frame sample
            result = dataset.get_item(index=ep_idx, frame_index=frame_idx)
            if isinstance(result, tuple) and len(result) == 2:
                success, data = result
                if not success or data is None:
                    continue
            else:
                data = result
                if data is None:
                    continue

            # Convert keys to match DataCollator expectations
            if 'state' in data:
                data['states'] = data.pop('state')
            if 'state_indicator' in data:
                data['state_elem_mask'] = data.pop('state_indicator')
            if 'lang_embed' not in data and 'input_ids' not in data:
                data['lang_embed'] = torch.zeros(1, 4096)

            # Add images if missing, using mean-color background (match train/dataset.py)
            if 'images' not in data:
                cam_keys = [
                    ('cam_high', 'cam_high_mask'),
                    ('cam_right_wrist', 'cam_right_wrist_mask'),
                    ('cam_left_wrist', 'cam_left_wrist_mask'),
                    ('cam_third_view', 'cam_third_view_mask'),
                ]
                cams, masks = [], []
                for img_key, mask_key in cam_keys:
                    if img_key in data and mask_key in data:
                        cams.append(data[img_key])
                        masks.append(data[mask_key])
                if len(cams) == 0:
                    H, W = 480, 640
                    data['images'] = [torch.zeros(3, H, W, dtype=torch.uint8)]
                else:
                    H, W = cams[0].shape[1], cams[0].shape[2]
                    mean_color = np.array([
                        int(x*255) for x in vision_encoder.image_processor.image_mean
                    ], dtype=np.uint8).reshape(1, 1, 3)
                    background = np.ones((H, W, 3), dtype=np.uint8) * mean_color
                    img_history_size = cams[0].shape[0]
                    preprocessed_images = []
                    for i_hist in range(img_history_size):
                        for cam_idx in range(len(cams)):
                            valid = bool(masks[cam_idx][i_hist]) if i_hist < masks[cam_idx].shape[0] else False
                            img = cams[cam_idx][i_hist] if valid and i_hist < cams[cam_idx].shape[0] else background
                            if img.ndim == 2:
                                img = np.stack([img]*3, axis=-1)
                            tensor_img = torch.from_numpy(img).permute(2, 0, 1).to(torch.uint8)
                            preprocessed_images.append(tensor_img)
                    data['images'] = preprocessed_images

            data['data_idx'] = ep_idx
            if 'ctrl_freq' not in data:
                # Dexora paper §III-A: "all sensing streams ... are logged at 20 Hz."
                # Keep the fallback consistent with the canonical Dexora corpus and
                # with scripts/eval_smoothness.py (which also defaults to 20).
                data['ctrl_freq'] = 20.0
            batch_data.append(data)

        t_collate0 = time.perf_counter()
        collated_batch = data_collator(batch_data)
        t_collate1 = time.perf_counter()
        logpi_values = compute_denoising_residual_energy(
            rdt_runner, collated_batch, args.num_noise_steps, vision_encoder, verbose=args.verbose_timing
        )
        t_batch1 = time.perf_counter()
        if args.verbose_timing:
            print(f"[TIMING] collate={(t_collate1 - t_collate0)*1000:.1f} ms, batch_total={(t_batch1 - t_batch0):.3f} s")

        for j in range(len(logpi_values)):
            ep_idx = batch_data[j].get('data_idx')
            frame_idx = batch_data[j]['meta'].get('step_id', 0)
            # Use full episode path as the first-level key
            if hasattr(dataset, 'episode_infos'):
                ep_path = dataset.episode_infos[ep_idx].path if (ep_idx is not None and 0 <= ep_idx < len(dataset.episode_infos)) else str(ep_idx)
                ep_key = str(ep_path)
            else:
                ep_key = str(ep_idx)
            frame_key = str(frame_idx)
            if ep_key not in logpi_dict:
                logpi_dict[ep_key] = {}
            logpi_dict[ep_key][frame_key] = float(logpi_values[j])
    
    # =====================================================================
    # Convert raw energies into the log-pi proxy used by the discriminator.
    #
    # Paper Eq.(5):
    #     \hat{log pi}_t = -zscore(E_t)
    #                    = -(E_t - Mean(E)) / sqrt(Var(E) + eps)
    #
    # ``logpi_dict`` now stores the raw energies E_t (positive, see upstream),
    # so this block implements Eq.(5) literally: the sign flip is applied
    # exactly once, here. Larger \hat{log pi}_t thus means "the diffusion
    # policy explains the chunk better", matching the paper's wording.
    # =====================================================================
    raw_values = []
    key_index = []  # list of (ep_key, frame_key)
    for ep_key, frames in logpi_dict.items():
        if isinstance(frames, dict):
            for frame_key, val in frames.items():
                raw_values.append(float(val))
                key_index.append((ep_key, frame_key))
        else:
            raw_values.append(float(frames))
            key_index.append((str(ep_key), None))
    raw_values = np.array(raw_values, dtype=np.float64)  # E_t per chunk

    eps = 1e-8
    if raw_values.size == 0:
        norm_values = np.array([], dtype=np.float64)
    elif args.normalize_mode == "zscore":
        mu = float(raw_values.mean())
        sigma = float(raw_values.std())
        denom = max(sigma, eps)
        # Eq.(5): \hat{log pi}_t = -zscore(E_t).
        norm_values = -(raw_values - mu) / denom
    elif args.normalize_mode == "minmax":
        # Legacy mode: rescale energies to [0, 1] and then flip so larger
        # values correspond to higher log-pi, matching Eq.(5)'s direction.
        vmin = float(raw_values.min())
        vmax = float(raw_values.max())
        rng = vmax - vmin
        if rng < 1e-12:
            norm_values = np.zeros_like(raw_values, dtype=np.float64)
        else:
            norm_values = 1.0 - (raw_values - vmin) / rng
    elif args.normalize_mode == "none":
        # Keep -E as the proxy so the direction still matches Eq.(5).
        norm_values = -raw_values
    else:
        raise ValueError(f"Unknown normalize_mode: {args.normalize_mode}")

    if args.temperature != 1.0:
        norm_values = norm_values / float(args.temperature)

    # Reconstruct nested dict
    norm_dict = {}
    for (ep_key, frame_key), p in zip(key_index, norm_values.tolist()):
        if ep_key not in norm_dict:
            norm_dict[ep_key] = {}
        if frame_key is None:
            norm_dict[ep_key] = p
        else:
            norm_dict[ep_key][frame_key] = p

    # Save raw and normalized outputs side by side.
    output_path = Path(args.output_file)
    raw_sidecar = output_path.with_name(output_path.stem + "_raw_E" + output_path.suffix)
    print(f"Saving raw energies E_t (Eq.(4)) to {raw_sidecar} ...")
    with open(raw_sidecar, 'w') as f:
        json.dump(logpi_dict, f, indent=2)

    print(
        f"Saving \\hat{{log pi}}_t = -zscore(E_t) (Eq.(5)) "
        f"(normalize_mode={args.normalize_mode}, T={args.temperature}) "
        f"to {args.output_file} ..."
    )
    with open(args.output_file, 'w') as f:
        json.dump(norm_dict, f, indent=2)

    norm_flat = []
    for v in norm_dict.values():
        if isinstance(v, dict):
            norm_flat.extend(list(v.values()))
        else:
            norm_flat.append(v)
    norm_flat = np.array(norm_flat, dtype=np.float64)

    if norm_flat.size > 0:
        print("\nNormalized log-pi statistics:")
        print(f"  Total entries: {norm_flat.size}")
        print(f"  Mean: {norm_flat.mean():.6f}")
        print(f"  Std:  {norm_flat.std():.6f}")
        print(f"  Min:  {norm_flat.min():.6f}")
        print(f"  Max:  {norm_flat.max():.6f}")

    print("Logpi computation completed!")


if __name__ == "__main__":
    main()
