# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import gc
import math
import os
import sys
from ast import literal_eval
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from configs.experiment import load_experiment_config
from dataset import MultiLatentLeRobotDataset
from dataset.lerobot_latent_dataset import va_train_collate_fn
from dataset.clip_latent_cache import (
    clip_latent_paths_exist,
    clip_latent_paths_for_sample,
    load_robotwin_tshape_clip_latent,
    save_clip_latent,
)
from distributed.fsdp import apply_ac
from modules.utils import load_transformer, load_vae
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    get_state_history_grid_id,
    init_logger,
    logger,
    sample_timestep_id,
    warmup_constant_lambda,
)
from utils.debug_profile import DebugTrainProfiler


class StepTracker:
    def __init__(self, step=0):
        self.step = int(step)

    def state_dict(self):
        return {"step": self.step}

    def load_state_dict(self, state_dict):
        self.step = int(state_dict.get("step", 0))


class Trainer:
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.step_tracker = StepTracker()
        self.accelerator.register_for_checkpointing(self.step_tracker)

        self.step = 0
        self.device = accelerator.device
        self.patch_size = config.patch_size
        self.train_loader_iter = None
        self.wandb = None
        self.last_reported_action_loss = None
        self.vae_dtype = (
            torch.float16
            if self.accelerator.mixed_precision == "fp16"
            else torch.bfloat16
        )

        if config.enable_wandb and self.accelerator.is_main_process:
            wandb.login(host=os.environ["WANDB_BASE_URL"], key=os.environ["WANDB_API_KEY"])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                config=config,
                mode=os.getenv("WANDB_MODE", "online"),
                name=os.getenv("WANDB_NAME", "test_lln"),
            )
            logger.info("WandB logging enabled")

        if self.accelerator.is_main_process:
            logger.info("Loading models...")
            logger.info("Loading VAE...")
        vae_path = os.path.join(config.wan22_pretrained_model_name_or_path, "vae")
        self.vae = load_vae(
            vae_path,
            torch_dtype=self.vae_dtype,
            torch_device=self.device,
        )
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_half = None
        if config.env_type == "robotwin_tshape":
            self.vae_half = load_vae(
                vae_path,
                torch_dtype=self.vae_dtype,
                torch_device=self.device,
            )
            self.vae_half.requires_grad_(False)
            self.vae_half.eval()

        if self.accelerator.is_main_process:
            logger.info("Loading transformer...")

        resume_from = getattr(config, "resume_from", None)
        transformer_path = getattr(config, "transformer_source_path", None)
        if not transformer_path:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, "transformer")

        transformer_overrides = {
            "action_head_type": getattr(config, "action_head_type", "shared"),
            "action_head_hidden_size": int(
                getattr(config, "action_head_hidden_size", 768)
            ),
            "action_head_num_attention_heads": int(
                getattr(config, "action_head_num_attention_heads", 12)
            ),
            "action_head_ffn_dim": int(
                getattr(config, "action_head_ffn_dim", 3072)
            ),
            "action_head_num_layers": getattr(config, "action_head_num_layers", None),
            "action_head_dropout": float(getattr(config, "action_head_dropout", 0.0)),
        }
        if transformer_overrides["action_head_type"] == "separate":
            # The pretrained video checkpoint has no action-head tensors.  Disable
            # meta-device loading so newly introduced modules are materialized and
            # initialized normally instead of remaining unusable meta tensors.
            transformer_overrides["materialize_missing_modules"] = True
        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=self.vae_dtype,
            torch_device="cpu",
            attn_mode="flex",
            **transformer_overrides,
        )

        if self.accelerator.is_main_process:
            logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        self.transformer.train()
        self.transformer.requires_grad_(True)
        if getattr(config, "action_head_type", "shared") == "separate":
            # The legacy shared action input/output modules remain present only
            # for checkpoint compatibility. They do not contribute to the
            # separate-head output and should not consume optimizer state.
            for module_name in (
                "action_embedder",
                "condition_embedder_action",
                "action_proj_out",
            ):
                getattr(self.transformer, module_name).requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps),
        )

        if self.accelerator.is_main_process:
            logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        if self.accelerator.is_main_process:
            logger.info(f"Train dataset size: {len(train_dataset)} samples")
            if getattr(config, "enable_clip_latent_cache", False):
                logger.info(
                    "Clip latent cache enabled: "
                    f"read={getattr(config, 'clip_latent_cache_read', True)}, "
                    f"write={getattr(config, 'clip_latent_cache_write', True)}"
                )
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(42)
        train_loader_kwargs = dict(
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.load_worker,
            generator=dataloader_generator,
            pin_memory=True,
            collate_fn=va_train_collate_fn,
        )
        if config.load_worker > 0:
            train_loader_kwargs["prefetch_factor"] = 2
            train_loader_kwargs["persistent_workers"] = True
        self.train_loader = DataLoader(
            train_dataset,
            **train_loader_kwargs,
        )

        (
            self.transformer,
            self.optimizer,
            self.train_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.transformer,
            self.optimizer,
            self.train_loader,
            self.lr_scheduler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=self.config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=self.config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.attn_mask_saved = False

        debug_profile = getattr(config, "debug_profile", False)
        debug_total_steps = getattr(config, "debug_total_steps", 20)
        debug_profile_steps = getattr(config, "debug_profile_steps", 10)
        profile_output_dir = Path(config.save_root) / "debug_profile"
        self.debug_profiler = DebugTrainProfiler(
            enabled=debug_profile,
            total_steps=debug_total_steps,
            profile_steps=debug_profile_steps,
            output_dir=profile_output_dir,
            is_main_process=self.accelerator.is_main_process,
        )

        if resume_from:
            self._load_training_state(resume_from)

    def _get_next_batch(self):
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)

        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        return batch

    @torch.no_grad()
    def _add_noise(
        self,
        latent,
        train_scheduler,
        action_mask=None,
        action_mode=False,
        clean_prefix_frames=0,
    ):
        bsz, _, frames, _, _ = latent.shape

        timestep_ids = sample_timestep_id(batch_size=frames, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)
        loss_mask = torch.ones((bsz, frames), device=self.device, dtype=latent.dtype)

        if clean_prefix_frames > 0:
            clean_prefix_frames = min(int(clean_prefix_frames), frames)
            timesteps = timesteps.clone()
            timesteps[:clean_prefix_frames] = 0
            noisy_latents = noisy_latents.clone()
            noisy_latents[:, :, :clean_prefix_frames] = latent[:, :, :clean_prefix_frames]
            targets = targets.clone()
            targets[:, :, :clean_prefix_frames] = 0
            loss_mask[:, :clean_prefix_frames] = 0

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1

        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,
            latent.shape[-2] // patch_h,
            latent.shape[-1] // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(self.device)
        latent_grid_id = latent_grid_id[None].repeat(bsz, 1, 1)

        if action_mask is not None:
            action_mask = action_mask.to(noisy_latents.dtype)
            noisy_latents *= action_mask
            targets *= action_mask

        return {
            "timesteps": timesteps[None].repeat(bsz, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "grid_id": latent_grid_id,
            "loss_mask": loss_mask,
        }

    @torch.no_grad()
    def _normalize_latents(self, latents):
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        return (latents.float() - latents_mean) * (1.0 / latents_std)

    @torch.no_grad()
    def _encode_video_frames(self, video_frames, return_per_camera=False):
        per_camera = None
        if self.config.env_type == "robotwin_tshape":
            if isinstance(video_frames, dict):
                high_frames = video_frames[self.config.obs_cam_keys[0]].float()
                wrist_frames = torch.stack(
                    [video_frames[key].float() for key in self.config.obs_cam_keys[1:]],
                    dim=1,
                )
            else:
                video_frames = video_frames.float()
                high_frames = video_frames[:, 0]
                wrist_frames = video_frames[:, 1:]

            batch_size, num_frames, _, _, _ = high_frames.shape
            wrist_cameras = wrist_frames.shape[1]
            high_frames = rearrange(high_frames, "b f h w c -> (b f) c h w")
            wrist_frames = rearrange(wrist_frames, "b k f h w c -> (b k f) c h w")

            high_frames = F.interpolate(
                high_frames,
                size=(self.config.height, self.config.width),
                mode="bilinear",
                align_corners=False,
            )
            wrist_frames = F.interpolate(
                wrist_frames,
                size=(self.config.height // 2, self.config.width // 2),
                mode="bilinear",
                align_corners=False,
            )
            high_frames = rearrange(
                high_frames,
                "(b f) c h w -> b c f h w",
                b=batch_size,
                f=num_frames,
            )
            wrist_frames = rearrange(
                wrist_frames,
                "(b k f) c h w -> (b k) c f h w",
                b=batch_size,
                k=wrist_cameras,
                f=num_frames,
            )

            high_frames = (high_frames / 255.0) * 2.0 - 1.0
            wrist_frames = (wrist_frames / 255.0) * 2.0 - 1.0

            vae_device = next(self.vae.parameters()).device
            high_posterior = self.vae.encode(
                high_frames.to(device=vae_device, dtype=self.vae_dtype)
            ).latent_dist
            wrist_posterior = self.vae_half.encode(
                wrist_frames.to(device=vae_device, dtype=self.vae_dtype)
            ).latent_dist

            wrist_mu = wrist_posterior.mean
            wrist_mu = self._normalize_latents(wrist_mu)
            wrist_mu = rearrange(
                wrist_mu,
                "(b k) c f h w -> b k c f h w",
                b=batch_size,
                k=wrist_cameras,
            )
            high_mu = self._normalize_latents(high_posterior.mean)
            if return_per_camera:
                per_camera = {
                    self.config.obs_cam_keys[0]: high_mu.detach(),
                }
                for cam_id, camera_key in enumerate(self.config.obs_cam_keys[1:], start=0):
                    per_camera[camera_key] = wrist_mu[:, cam_id].detach()
            tail_camera_keys = list(self.config.obs_cam_keys[1:])
            crop_view_keys = set(getattr(self.config, "crop_view_keys", []))
            crop_indices = [
                cam_id for cam_id, camera_key in enumerate(tail_camera_keys)
                if camera_key in crop_view_keys
            ]
            arm_indices = [
                cam_id for cam_id, camera_key in enumerate(tail_camera_keys)
                if camera_key not in crop_view_keys
            ]
            latent_rows = []
            if crop_indices:
                latent_rows.append(
                    torch.cat([wrist_mu[:, cam_id] for cam_id in crop_indices], dim=-1)
                )
            if arm_indices:
                latent_rows.append(
                    torch.cat([wrist_mu[:, cam_id] for cam_id in arm_indices], dim=-1)
                )

            video_latent = torch.cat([*latent_rows, high_mu], dim=-2)
        else:
            video_frames = video_frames.float()
            batch_size, num_cameras, num_frames, _, _, _ = video_frames.shape
            frames = rearrange(video_frames, "b k f h w c -> (b k f) c h w")
            frames = F.interpolate(
                frames,
                size=(self.config.height, self.config.width),
                mode="bilinear",
                align_corners=False,
            )
            frames = rearrange(
                frames,
                "(b k f) c h w -> (b k) c f h w",
                b=batch_size,
                k=num_cameras,
                f=num_frames,
            )
            frames = (frames / 255.0) * 2.0 - 1.0

            vae_device = next(self.vae.parameters()).device
            posterior = self.vae.encode(
                frames.to(device=vae_device, dtype=self.vae_dtype)
            ).latent_dist
            mu = self._normalize_latents(posterior.mean)
            mu = rearrange(mu, "(b k) c f h w -> b k c f h w", b=batch_size, k=num_cameras)
            if return_per_camera:
                per_camera = {
                    self.config.obs_cam_keys[cam_id]: mu[:, cam_id].detach()
                    for cam_id in range(num_cameras)
                }
            video_latent = torch.cat([mu[:, cam_id] for cam_id in range(num_cameras)], dim=-1)

        result = video_latent.float().to(self.device)
        if return_per_camera:
            return result, per_camera
        return result

    def _clip_sampling_params(self):
        freq_ratio = int(getattr(self.config, "freq_ratio", 2))
        if freq_ratio <= 0:
            raise ValueError(f"freq_ratio must be positive, got {freq_ratio}")
        video_frame_stride = 4 * freq_ratio
        action_frame_stride = freq_ratio
        raw_window_frames = max(
            (9 - 1) * video_frame_stride + 1,
            (32 - 1) * action_frame_stride + 1,
        )
        return freq_ratio, video_frame_stride, raw_window_frames

    def _image_frame_ids(self, clip_start: int) -> list[int]:
        _, video_frame_stride, raw_window_frames = self._clip_sampling_params()
        return list(
            range(
                int(clip_start),
                int(clip_start) + raw_window_frames,
                video_frame_stride,
            )
        )[:9]

    def _stack_batch_video_frames(self, video_frames_list):
        if isinstance(video_frames_list[0], dict):
            return {
                key: torch.stack([sample[key] for sample in video_frames_list], dim=0)
                for key in video_frames_list[0].keys()
            }
        return torch.stack(video_frames_list, dim=0)

    @torch.no_grad()
    def _save_clip_latent_cache_for_sample(
        self,
        *,
        repo_root: Path,
        episode_index: int,
        clip_start: int,
        per_camera: dict[str, torch.Tensor],
        freq_ratio: int,
        frame_ids: list[int],
        raw_window_frames: int,
        cache_paths: dict[str, Path] | None = None,
    ) -> None:
        if clip_latent_paths_exist(
            repo_root,
            episode_index=episode_index,
            clip_start=clip_start,
            camera_keys=self.config.obs_cam_keys,
            freq_ratio=freq_ratio,
        ):
            return

        video_frame_stride = frame_ids[1] - frame_ids[0] if len(frame_ids) > 1 else 0
        if cache_paths is None:
            cache_paths = clip_latent_paths_for_sample(
                repo_root,
                episode_index=episode_index,
                clip_start=clip_start,
                camera_keys=self.config.obs_cam_keys,
                freq_ratio=freq_ratio,
            )
        for camera_key, latent in per_camera.items():
            path = cache_paths[camera_key]
            if path.is_file():
                continue
            latent_to_save = latent[0] if latent.dim() == 5 else latent
            save_clip_latent(
                path,
                latent=latent_to_save,
                clip_start=clip_start,
                frame_ids=frame_ids,
                episode_index=episode_index,
                camera_key=camera_key,
                video_frame_stride=video_frame_stride,
                raw_window_frames=raw_window_frames,
                freq_ratio=freq_ratio,
            )

    @torch.no_grad()
    def _get_batch_video_latent(self, batch_dict):
        episode_indices = batch_dict["episode_index"]
        clip_starts = batch_dict["clip_start"]
        repo_roots = batch_dict["dataset_repo_root"]
        video_frames_list = batch_dict["video_frames"]
        cache_paths_list = batch_dict.get("clip_latent_cache_paths")
        cache_freq_ratios = batch_dict.get("clip_latent_cache_freq_ratio")
        image_frame_ids_list = batch_dict.get("image_frame_ids")
        raw_window_frames_list = batch_dict.get("clip_raw_window_frames")
        batch_size = len(video_frames_list)

        cache_read = bool(getattr(self.config, "clip_latent_cache_read", True))
        cache_write = bool(getattr(self.config, "clip_latent_cache_write", True))
        batch_latents: list[torch.Tensor | None] = [None] * batch_size
        miss_indices: list[int] = []

        def _item_int(values, idx: int, default: int) -> int:
            if values is None:
                return default
            value = values[idx]
            if torch.is_tensor(value):
                return int(value.item())
            return int(value)

        for batch_idx in range(batch_size):
            episode_index = int(episode_indices[batch_idx].item())
            clip_start = int(clip_starts[batch_idx].item())
            repo_root = Path(repo_roots[batch_idx])
            cache_freq_ratio = _item_int(
                cache_freq_ratios,
                batch_idx,
                int(getattr(self.config, "freq_ratio", 2)),
            )

            latent = None
            if cache_read and cache_freq_ratio > 0:
                latent = load_robotwin_tshape_clip_latent(
                    repo_root,
                    episode_index=episode_index,
                    clip_start=clip_start,
                    camera_keys=self.config.obs_cam_keys,
                    freq_ratio=cache_freq_ratio,
                )
                if latent is None and video_frames_list[batch_idx] is None:
                    latent = load_robotwin_tshape_clip_latent(
                        repo_root,
                        episode_index=episode_index,
                        clip_start=clip_start,
                        camera_keys=self.config.obs_cam_keys,
                        freq_ratio=cache_freq_ratio,
                    )

            if latent is not None:
                batch_latents[batch_idx] = latent
            else:
                miss_indices.append(batch_idx)

        if miss_indices:
            miss_frames = [video_frames_list[batch_idx] for batch_idx in miss_indices]
            if any(video_frames is None for video_frames in miss_frames):
                bad_idx = next(
                    batch_idx
                    for batch_idx in miss_indices
                    if video_frames_list[batch_idx] is None
                )
                raise RuntimeError(
                    "Clip latent cache miss but video_frames is None. "
                    f"repo={repo_roots[bad_idx]}, episode={episode_indices[bad_idx].item()}, "
                    f"clip_start={clip_starts[bad_idx].item()}"
                )

            encoded, per_camera = self._encode_video_frames(
                self._stack_batch_video_frames(miss_frames),
                return_per_camera=True,
            )
            for miss_offset, batch_idx in enumerate(miss_indices):
                batch_latents[batch_idx] = encoded[miss_offset]
                cache_freq_ratio = _item_int(
                    cache_freq_ratios,
                    batch_idx,
                    int(getattr(self.config, "freq_ratio", 2)),
                )
                if cache_write and cache_freq_ratio > 0 and per_camera is not None:
                    sample_per_camera = {
                        camera_key: camera_latent[miss_offset].detach()
                        for camera_key, camera_latent in per_camera.items()
                    }
                    episode_index = int(episode_indices[batch_idx].item())
                    clip_start = int(clip_starts[batch_idx].item())
                    repo_root = Path(repo_roots[batch_idx])
                    sample_cache_paths = None
                    if (
                        cache_paths_list is not None
                        and cache_paths_list[batch_idx] is not None
                    ):
                        sample_cache_paths = {
                            camera_key: Path(path_str)
                            for camera_key, path_str in cache_paths_list[batch_idx].items()
                        }
                    frame_ids = (
                        [int(value) for value in image_frame_ids_list[batch_idx]]
                        if image_frame_ids_list is not None
                        else self._image_frame_ids(clip_start)
                    )
                    raw_window_frames = _item_int(
                        raw_window_frames_list,
                        batch_idx,
                        frame_ids[-1] - frame_ids[0] + 1 if frame_ids else 0,
                    )
                    self._save_clip_latent_cache_for_sample(
                        repo_root=repo_root,
                        episode_index=episode_index,
                        clip_start=clip_start,
                        per_camera=sample_per_camera,
                        freq_ratio=cache_freq_ratio,
                        frame_ids=frame_ids,
                        raw_window_frames=raw_window_frames,
                        cache_paths=sample_cache_paths,
                    )

        if any(latent is None for latent in batch_latents):
            raise RuntimeError("Failed to resolve video latents for every batch item.")

        return torch.stack(
            [latent.float().to(device=self.device) for latent in batch_latents],
            dim=0,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        if (
            getattr(self.config, "enable_clip_latent_cache", False)
            and self.config.env_type == "robotwin_tshape"
        ):
            latents = self._get_batch_video_latent(batch_dict)
        else:
            batched_frames = self._stack_batch_video_frames(batch_dict["video_frames"])
            latents = self._encode_video_frames(batched_frames)
        latent_dict = self._add_noise(
            latent=latents,
            train_scheduler=self.train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            clean_prefix_frames=1,
        )

        action_dict = self._add_noise(
            latent=batch_dict["actions"],
            train_scheduler=self.train_scheduler_action,
            action_mask=batch_dict["actions_mask"],
            action_mode=True,
        )

        latent_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["text_emb"] = batch_dict["text_emb"]
        action_dict["actions_mask"] = batch_dict["actions_mask"]
        state = batch_dict["state"]
        state_mask = batch_dict["state_mask"]
        if torch.rand(1).item() < 0.5:
            state_cond_timestep_ids = sample_timestep_id(
                batch_size=state.shape[-3],
                min_timestep_bd=0.5,
                max_timestep_bd=1.0,
                num_train_timesteps=self.train_scheduler_action.num_train_timesteps,
            )
            state_noise = torch.zeros_like(state).normal_()
            state_timesteps = self.train_scheduler_action.timesteps[
                state_cond_timestep_ids
            ].to(device=self.device)
            state = self.train_scheduler_action.add_noise(
                state,
                state_noise,
                state_timesteps,
                t_dim=2,
            )
        else:
            state_timesteps = torch.zeros(
                (state.shape[-3],),
                device=self.device,
                dtype=action_dict["timesteps"].dtype,
            )
        state = state * state_mask[:, None, :, None, None].to(
            device=state.device,
            dtype=state.dtype,
        )
        state_grid_id = get_state_history_grid_id(
            history_len=state.shape[-3],
            action_per_frame=self.config.action_per_frame,
            t=1,
            device=self.device,
        ).to(self.device)
        state_grid_id = state_grid_id[None].repeat(state.shape[0], 1, 1)
        state_timesteps = state_timesteps[None].repeat(state.shape[0], 1)
        action_dict["state"] = {
            "value": state,
            "grid_id": state_grid_id,
            "timesteps": state_timesteps,
            "mask": state_mask,
        }
        if "example_action_loss_mask" in batch_dict:
            action_dict["example_action_loss_mask"] = batch_dict["example_action_loss_mask"]

        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
        }

    def convert_input_format(self, input_dict):
        target_dtype = None
        if self.accelerator.mixed_precision == "bf16":
            target_dtype = torch.bfloat16
        elif self.accelerator.mixed_precision == "fp16":
            target_dtype = torch.float16

        def _convert_value(value):
            if torch.is_tensor(value):
                if target_dtype is not None and torch.is_floating_point(value):
                    return value.to(device=self.device, dtype=target_dtype, non_blocking=True)
                return value.to(device=self.device, non_blocking=True)
            if isinstance(value, dict):
                return {key: _convert_value(inner_value) for key, inner_value in value.items()}
            if isinstance(value, list):
                return [_convert_value(inner_value) for inner_value in value]
            if isinstance(value, tuple):
                return tuple(_convert_value(inner_value) for inner_value in value)
            return value

        return {key: _convert_value(value) for key, value in input_dict.items()}

    def compute_loss(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred,
            "b (f n) c -> b c f n 1",
            f=input_dict["action_dict"]["targets"].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size,
            latent_pred,
            input_dict["latent_dict"]["targets"].shape[-3],
            input_dict["latent_dict"]["targets"].shape[-2],
            input_dict["latent_dict"]["targets"].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        batch_size, latent_frames = input_dict["latent_dict"]["timesteps"].shape
        _, action_frames = input_dict["action_dict"]["timesteps"].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(batch_size, latent_frames)
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(batch_size, action_frames)

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict["latent_dict"]["targets"].float().detach(),
            reduction="none",
        )
        latent_loss_mask = input_dict["latent_dict"]["loss_mask"][:, None, :, None, None].to(
            latent_loss.dtype
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss * latent_loss_mask
        latent_loss = latent_loss.sum() / (latent_loss_mask.expand_as(latent_loss).sum() + 1e-6)

        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict["action_dict"]["targets"].float().detach(),
            reduction="none",
        )
        final_action_mask = input_dict["action_dict"]["actions_mask"].float()
        example_action_loss_mask = input_dict["action_dict"].get("example_action_loss_mask")
        if example_action_loss_mask is not None:
            final_action_mask = (
                final_action_mask
                * example_action_loss_mask.float()[:, None, None, None, None]
            )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * final_action_mask
        if final_action_mask.any():
            action_loss = action_loss.sum() / (final_action_mask.sum() + 1e-6)
            has_valid_action_loss = True
        else:
            action_loss = action_loss.sum() * 0.0
            has_valid_action_loss = False

        latent_loss = latent_loss * float(getattr(self.config, "video_loss_weight", 1.0))
        action_loss = action_loss * float(getattr(self.config, "action_loss_weight", 1.0))
        return latent_loss, action_loss, has_valid_action_loss

    def _reduce_scalar(self, tensor, reduction):
        tensor = tensor.detach().float().reshape(1)
        if self.accelerator.num_processes == 1:
            return tensor[0]
        if reduction == "mean" and hasattr(self.accelerator, "reduce"):
            return self.accelerator.reduce(tensor, reduction=reduction)[0]
        gathered = self.accelerator.gather(tensor)
        if reduction == "mean":
            return gathered.mean()
        if reduction == "max":
            return gathered.max()
        raise ValueError(f"Unsupported reduction: {reduction}")

    def _save_transformer(self, checkpoint_dir):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving transformer to {transformer_dir}")
        self.accelerator.unwrap_model(self.transformer).save_pretrained(
            transformer_dir,
            is_main_process=True,
            safe_serialization=True,
        )

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            checkpoints = sorted(
                self.save_dir.glob("checkpoint_step_*"),
                key=lambda path: path.stat().st_mtime,
            )
            if len(checkpoints) >= 3:
                for checkpoint in checkpoints[: len(checkpoints) - 2]:
                    if checkpoint.is_dir():
                        for child in checkpoint.rglob("*"):
                            if child.is_file() or child.is_symlink():
                                child.unlink()
                        for child in sorted(checkpoint.rglob("*"), reverse=True):
                            if child.is_dir():
                                child.rmdir()
                        checkpoint.rmdir()

        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.accelerator.is_main_process:
            logger.info(f"Saving training state to {checkpoint_dir}")

        self.step_tracker.step = self.step
        self.accelerator.save_state(output_dir=str(checkpoint_dir))
        self._save_transformer(checkpoint_dir)

        if self.accelerator.is_main_process:
            last_link = self.save_dir / "last"
            if last_link.exists() or last_link.is_symlink():
                last_link.unlink()
            os.symlink(checkpoint_dir.resolve(), last_link)
            logger.info(f"Checkpoint saved successfully at step {self.step}")

    def _load_training_state(self, checkpoint_path):
        checkpoint_dir = Path(checkpoint_path)
        if self.accelerator.is_main_process:
            logger.info(f"Loading training state from {checkpoint_dir}")

        self.accelerator.load_state(str(checkpoint_dir))
        self.accelerator.wait_for_everyone()
        self.step = self.step_tracker.step

        if self.accelerator.is_main_process:
            logger.info(f"Training state loaded, resuming from step {self.step}")

    def train(self):
        if self.accelerator.is_main_process:
            logger.info(f"Starting training for {self.config.num_steps} steps...")
            if getattr(self.config, "debug_profile", False):
                logger.info(
                    "Debug timing enabled: "
                    f"{getattr(self.config, 'debug_total_steps', 20)} steps total, "
                    f"measuring last {getattr(self.config, 'debug_profile_steps', 10)} steps"
                )

        self.transformer.train()
        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process,
            leave=True,
            dynamic_ncols=True,
            initial=self.step,
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []

        while self.step < self.config.num_steps:
            train_step = self.step
            self.debug_profiler.on_step_start(train_step)

            with self.debug_profiler.record("train/data_next_batch"):
                batch = self._get_next_batch()
            with self.debug_profiler.record("train/convert_input_format"):
                batch = self.convert_input_format(batch)
            with self.debug_profiler.record("train/prepare_input_dict"):
                input_dict = self._prepare_input_dict(batch)

            with self.accelerator.accumulate(self.transformer):
                export_mask = (not self.attn_mask_saved) and self.config.rank == 0
                if export_mask:
                    input_dict["export_mask"] = True
                    input_dict["mask_output_dir"] = self.save_dir

                with self.debug_profiler.record("train/forward"):
                    output = self.transformer(input_dict, train_mode=True)
                if export_mask:
                    self.attn_mask_saved = True
                with self.debug_profiler.record("train/compute_loss"):
                    latent_loss, action_loss, has_valid_action_loss = self.compute_loss(
                        input_dict, output
                    )
                loss = latent_loss + action_loss
                with self.debug_profiler.record("train/backward"):
                    self.accelerator.backward(loss)

                total_norm = None
                if self.accelerator.sync_gradients:
                    with self.debug_profiler.record("train/clip_grad_norm"):
                        total_norm = self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(), 2.0
                        )

                with self.debug_profiler.record("train/optimizer_step"):
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            accumulated_latent_losses.append(latent_loss.detach())
            accumulated_action_losses.append(action_loss.detach())

            if not self.accelerator.sync_gradients:
                continue

            lr = self.lr_scheduler.get_last_lr()[0]

            latent_loss_sum = torch.stack(accumulated_latent_losses).sum()
            action_loss_sum = torch.stack(accumulated_action_losses).sum()
            latent_loss_show = self._reduce_scalar(latent_loss_sum, "mean").cpu().item()
            action_loss_show = self._reduce_scalar(action_loss_sum, "mean").cpu().item()
            max_latent_loss_show = self._reduce_scalar(latent_loss_sum, "max").cpu().item()
            max_action_loss_show = self._reduce_scalar(action_loss_sum, "max").cpu().item()

            if has_valid_action_loss:
                self.last_reported_action_loss = {
                    "avg": action_loss_show,
                    "max": max_action_loss_show,
                }
            elif self.last_reported_action_loss is not None:
                action_loss_show = self.last_reported_action_loss["avg"]
                max_action_loss_show = self.last_reported_action_loss["max"]

            accumulated_latent_losses = []
            accumulated_action_losses = []

            self.step += 1
            self.step_tracker.step = self.step

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            if self.accelerator.is_main_process:
                grad_norm = float(total_norm) if total_norm is not None else float("nan")
                progress_bar.update(1)
                progress_bar.set_postfix(
                    {
                        "latent_loss": f"{latent_loss_show:.4f}",
                        "action_loss": f"{action_loss_show:.4f}",
                        "step": self.step,
                        "grad_norm": f"{grad_norm:.2f}",
                        "lr": f"{lr:.2e}",
                    }
                )
                if self.config.enable_wandb and self.wandb is not None:
                    self.wandb.log(
                        {
                            "loss_metrics/global_avg_video_loss": latent_loss_show,
                            "loss_metrics/global_avg_action_loss": action_loss_show,
                            "loss_metrics/global_max_video_loss": max_latent_loss_show,
                            "loss_metrics/global_max_action_loss": max_action_loss_show,
                            "grad_norm": grad_norm,
                            "lr": lr,
                        },
                        step=self.step,
                    )

            if self.step % self.config.save_interval == 0:
                if not getattr(self.config, "debug_profile", False):
                    self.save_checkpoint()

            self.debug_profiler.on_step_end(train_step)

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        self.debug_profiler.finish()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if getattr(self.config, "debug_profile", False):
                logger.info("Debug timing run completed!")
            else:
                logger.info("Training completed!")


def run(args):
    config = (
        load_experiment_config(args.experiment_config, VA_CONFIGS)
        if args.experiment_config is not None
        else VA_CONFIGS[args.config_name]
    )
    if not hasattr(config, "norm_stat"):
        raise ValueError(
            f"Config '{args.config_name}' does not include task norm_stat. "
            "Use a task-specific Astribot config such as "
            "'astribot_pick_white_plate', 'astribot_sort_bottles', "
            "or 'astribot_lixinji'."
        )

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.dataset_paths is not None:
        config.dataset_paths = args.dataset_paths
        if len(config.dataset_paths) > 0:
            config.empty_emb_path = str(Path(config.dataset_paths[0]) / "empty_emb.pt")
    if args.pretrained_model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.pretrained_model_path
    if args.enable_wandb is not None:
        config.enable_wandb = args.enable_wandb
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.episode_skip_first is not None:
        config.episode_skip_first = args.episode_skip_first
    if args.freq_ratio is not None:
        config.freq_ratio = args.freq_ratio
    if args.variant is not None:
        config.variant = args.variant
        if args.variant == "add_crop_views":
            config.obs_cam_keys = [
                "observation.images.cam_main",
                "observation.images.cam_left_wrist",
                "observation.images.cam_right_wrist",
                "observation.images.right_in_left",
                "observation.images.right_in_main",
            ]
            config.crop_view_keys = [
                "observation.images.right_in_left",
                "observation.images.right_in_main",
            ]
    if args.enable_clip_latent_cache is not None:
        config.enable_clip_latent_cache = args.enable_clip_latent_cache
    if args.clip_latent_cache_read is not None:
        config.clip_latent_cache_read = args.clip_latent_cache_read
    if args.clip_latent_cache_write is not None:
        config.clip_latent_cache_write = args.clip_latent_cache_write
    if args.debug:
        config.debug_profile = True
        config.debug_total_steps = 20
        config.debug_profile_steps = 10
        config.num_steps = config.debug_total_steps
        config.enable_wandb = False
        config.save_interval = max(config.num_steps + 1, 10**9)
        config.load_worker = 0

    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            getattr(config, "gradient_accumulation_steps", 1)
        )
    )

    config.rank = accelerator.process_index
    config.local_rank = accelerator.local_process_index
    config.world_size = accelerator.num_processes
    config.learning_rate = config.learning_rate * math.sqrt(max(config.world_size / 8, 1.0))

    if accelerator.is_main_process:
        logger.info(f"Using config: {args.config_name}")
        logger.info(
            f"World size: {config.world_size}, "
            f"Local rank: {config.local_rank}, Rank: {config.rank}"
        )
        logger.info(f"Scaled learning rate by WORLD_SIZE: {config.learning_rate}")

    trainer = Trainer(config, accelerator)
    trainer.train()


def main():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        lowered = value.lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

    def parse_dataset_paths(value):
        if value is None:
            return None
        try:
            parsed = literal_eval(value)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed]

        raise argparse.ArgumentTypeError(
            "--dataset-paths must be a Python/JSON-style list, for example "
            "\"['/path/repo_a', '/path/parent_dir']\""
        )

    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default="robotwin_train",
        help="Config name",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="JSON experiment config shared by training and inference.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--dataset-paths",
        type=parse_dataset_paths,
        default=None,
        help="Dataset root/repo list. Must be a Python/JSON-style list string.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Path to pretrained LingBot-VA checkpoint root",
    )
    parser.add_argument(
        "--enable-wandb",
        type=str2bool,
        default=None,
        help="Enable or disable Weights & Biases logging",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Accelerate checkpoint directory to resume from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config batch size",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override config training steps",
    )
    parser.add_argument(
        "--episode-skip-first",
        type=int,
        default=None,
        help="Skip the first N episode/action_config entries after sorting by episode_index.",
    )
    parser.add_argument(
        "--freq-ratio",
        type=int,
        default=None,
        help=(
            "Data sampling frequency ratio. The base video stride is 4; "
            "video stride becomes 4 * freq_ratio and action stride becomes freq_ratio."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=("default", "add_crop_views", "keyframe_sample"),
        help="Dataset/training variant to use.",
    )
    parser.add_argument(
        "--enable-clip-latent-cache",
        type=str2bool,
        default=None,
        help=(
            "Enable on-disk clip VAE latent cache during training. "
            "Reads cache when present and writes cache on miss."
        ),
    )
    parser.add_argument(
        "--clip-latent-cache-read",
        type=str2bool,
        default=None,
        help="Read clip VAE latents from disk when cache files exist.",
    )
    parser.add_argument(
        "--clip-latent-cache-write",
        type=str2bool,
        default=None,
        help="Write clip VAE latents to disk when cache files are missing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Run 20 steps and print wall-clock timing for the last 10 steps "
            "to save_root/debug_profile/profile_report.txt."
        ),
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    # import debugpy; debugpy.listen(5678); logger.info("Waiting for debugger to attach..."); debugpy.wait_for_client()
    main()
