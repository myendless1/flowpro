# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from configs.experiment import load_experiment_config
from action_representation import (
    EXECUTION_CHANNEL_IDS,
    encode_absolute_history,
    validate_action_representation,
)
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    get_state_history_grid_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


CAMERA_KEY_ALIASES = {
    "observation.images.cam_high": ("observation.images.cam_high", "observation.images.cam_main"),
    "observation.images.cam_main": ("observation.images.cam_main", "observation.images.cam_high"),
}


def _sync_cuda_for_timing(device):
    if isinstance(device, torch.device) and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@contextmanager
def log_timing(name, device=None):
    _sync_cuda_for_timing(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_cuda_for_timing(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"[timing] {name}: {elapsed_ms:.1f} ms")


def _module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return "no-params"


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram
        self.use_cfg = self._guidance_requires_cfg()
        self._runtime_initialized = False
        self.current_prompt_key = None
        logger.info(
            f"[device] local_rank={job_config.local_rank}, device={self.device}, "
            f"dtype={self.dtype}, enable_offload={self.enable_offload}"
        )

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        with log_timing("load primary vae", self.device):
            self.vae = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
        logger.info(f"[device] primary vae device={_module_device(self.vae)}")

        with log_timing("load tokenizer"):
            self.tokenizer = load_tokenizer(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'tokenizer'), )

        with log_timing("load text encoder", self.device):
            self.text_encoder = load_text_encoder(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'text_encoder'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
        logger.info(f"[device] text encoder device={_module_device(self.text_encoder)}")

        with log_timing("load transformer", self.device):
            transformer_path = getattr(job_config, "transformer_source_path", None)
            if not transformer_path:
                transformer_path = os.path.join(
                    job_config.wan22_pretrained_model_name_or_path, "transformer"
                )
            self.transformer = load_transformer(
                transformer_path,
                torch_dtype=self.dtype,
                torch_device=self.device,
                attn_mode="torch"
            )
        logger.info(f"[device] transformer device={_module_device(self.transformer)}")
        shard_fn = shard_model
        with log_timing("configure/shard transformer", self.device):
            self.transformer = _configure_model(model=self.transformer,
                                                shard_fn=shard_fn,
                                                param_dtype=self.dtype,
                                                device=self.device,
                                                eval_mode=True,
                                                )
        logger.info(f"[device] transformer after configure device={_module_device(self.transformer)}")

        self.env_type = job_config.env_type
        self.vae_half = None
        if self.env_type == 'robotwin_tshape':
            with log_timing("load wrist vae", self.device):
                self.vae_half = load_vae(
                    os.path.join(job_config.wan22_pretrained_model_name_or_path,
                                 'vae'),
                    torch_dtype=self.dtype,
                    torch_device='cpu' if self.enable_offload else self.device,
                )
            logger.info(f"[device] wrist vae device={_module_device(self.vae_half)}")

    def _guidance_requires_cfg(self):
        return (self.job_config.guidance_scale > 1) or (
            self.job_config.action_guidance_scale > 1
        )

    def _update_guidance_scales(self, obs, allow_cfg_mode_change=False):
        if "video_guidance_scale" in obs:
            self.job_config.guidance_scale = float(obs["video_guidance_scale"])
        if "action_guidance_scale" in obs:
            self.job_config.action_guidance_scale = float(obs["action_guidance_scale"])

        if (
            not allow_cfg_mode_change
            and self._runtime_initialized
            and self._guidance_requires_cfg()
            and not self.use_cfg
        ):
            raise ValueError(
                "Guidance scale > 1 requires CFG cache/text embeddings. "
                "Send the scale values together with reset before compute_kv_cache/infer_action."
            )

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]

    def preprocess_state(self, state):
        if torch.is_tensor(state):
            state_model_input = state.detach().cpu().to(torch.float32)
        else:
            state_model_input = torch.from_numpy(np.asarray(state, dtype=np.float32))
        if state_model_input.ndim != 2:
            raise ValueError(
                f"Expected state to have shape [T,D], but got {tuple(state_model_input.shape)}"
            )
        state_model_input = torch.from_numpy(
            encode_absolute_history(state_model_input.numpy())
        )

        if self.action_norm_method == 'quantiles':
            state_model_input = (state_model_input - self.state_q01[:, 0, 0][None]) / (
                self.state_q99[:, 0, 0][None] - self.state_q01[:, 0, 0][None] + 1e-6) * 2. - 1.
            state_model_input = torch.clamp(state_model_input, -1.5, 1.5)
        else:
            raise NotImplementedError
        state_channel_mask = torch.zeros_like(self.action_mask)
        state_channel_mask[list(EXECUTION_CHANNEL_IDS)] = True
        state_model_input *= state_channel_mask.to(dtype=state_model_input.dtype)[None]
        return state_model_input.T.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # B, C, F, H, W
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _camera_key_candidates(self, key):
        return CAMERA_KEY_ALIASES.get(key, (key,))

    def _get_obs_image(self, image_dict, key):
        for candidate in self._camera_key_candidates(key):
            if candidate in image_dict:
                return image_dict[candidate]
        available_keys = sorted(image_dict.keys())
        raise KeyError(
            f"Missing camera image for config key '{key}'. "
            f"Tried {list(self._camera_key_candidates(key))}; "
            f"available keys: {available_keys}"
        )

    def _load_image_file_for_key(self, key):
        for candidate in self._camera_key_candidates(key):
            image_path = os.path.join(self.job_config.input_img_path, f"{candidate}.png")
            if os.path.exists(image_path):
                return np.array(Image.open(image_path).convert("RGB"))
        raise FileNotFoundError(
            f"Missing init image for config key '{key}' under "
            f"{self.job_config.input_img_path}. Tried "
            f"{[f'{candidate}.png' for candidate in self._camera_key_candidates(key)]}"
        )

    def _encode_obs(self, obs):
        with log_timing("encode_obs total", self.device):
            with log_timing("encode_obs select/preprocess images"):
                images = self._select_history_obs(obs)
                if not isinstance(images, list):
                    images = [images]
                if len(images) < 1:
                    return None
                videos = []
                for k_i, k in enumerate(self.job_config.obs_cam_keys):
                    if self.env_type == 'robotwin_tshape':
                        if k_i == 0:  # camera high
                            height_i, width_i = self.height, self.width
                        else:
                            height_i, width_i = self.height // 2, self.width // 2
                    else:
                        height_i, width_i = self.height, self.width

                    history_video_k = torch.from_numpy(
                        np.stack([self._get_obs_image(each, k)
                                  for each in images])).float().permute(3, 0, 1, 2)
                    history_video_k = F.interpolate(history_video_k,
                                                    size=(height_i, width_i),
                                                    mode='bilinear',
                                                    align_corners=False).unsqueeze(0)
                    videos.append(history_video_k)

            if self.env_type == 'robotwin_tshape':
                videos_high = videos[0] / 255.0 * 2.0 - 1.0
                videos_left_and_right = torch.cat(videos[1:],
                                                  dim=0) / 255.0 * 2.0 - 1.0
                vae_device = next(self.vae.parameters()).device
                with log_timing("encode_obs primary vae encode", vae_device):
                    high_posterior = self.vae.encode(
                        videos_high.to(vae_device).to(self.dtype)
                    ).latent_dist
                    enc_out_high = self.normalize_latents(
                        high_posterior.mean,
                        torch.tensor(self.vae.config.latents_mean).to(high_posterior.mean.device),
                        1.0 / torch.tensor(self.vae.config.latents_std).to(high_posterior.mean.device),
                    )

                wrist_vae = self.vae_half
                wrist_vae_device = next(wrist_vae.parameters()).device
                with log_timing("encode_obs wrist vae encode", wrist_vae_device):
                    wrist_posterior = wrist_vae.encode(
                        videos_left_and_right.to(wrist_vae_device).to(self.dtype)
                    ).latent_dist
                    enc_out_left_and_right = self.normalize_latents(
                        wrist_posterior.mean,
                        torch.tensor(wrist_vae.config.latents_mean).to(wrist_posterior.mean.device),
                        1.0 / torch.tensor(wrist_vae.config.latents_std).to(wrist_posterior.mean.device),
                    )
                with log_timing("encode_obs concat latents", self.device):
                    wrist_latents = list(enc_out_left_and_right.split(1, dim=0))
                    tail_keys = list(self.job_config.obs_cam_keys[1:])
                    crop_keys = set(getattr(self.job_config, "crop_view_keys", []))
                    crop_indices = [
                        index for index, key in enumerate(tail_keys) if key in crop_keys
                    ]
                    arm_indices = [
                        index for index, key in enumerate(tail_keys) if key not in crop_keys
                    ]
                    latent_rows = []
                    if crop_indices:
                        latent_rows.append(torch.cat(
                            [wrist_latents[index] for index in crop_indices], dim=-1
                        ))
                    if arm_indices:
                        latent_rows.append(torch.cat(
                            [wrist_latents[index] for index in arm_indices], dim=-1
                        ))
                    video_latent = torch.cat([*latent_rows, enc_out_high], dim=-2)
            else:
                videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
                vae_device = next(self.vae.parameters()).device
                with log_timing("encode_obs vae encode", vae_device):
                    videos_chunk = videos.to(vae_device).to(self.dtype)
                    posterior = self.vae.encode(videos_chunk).latent_dist
                    latents_mean = torch.tensor(self.vae.config.latents_mean).to(posterior.mean.device)
                    latents_std = torch.tensor(self.vae.config.latents_std).to(posterior.mean.device)
                    mu_norm = self.normalize_latents(posterior.mean, latents_mean, 1.0 / latents_std)
                    video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
            with log_timing("encode_obs move latent to transformer device", self.device):
                video_latent = video_latent.to(self.device)
            logger.info(
                f"[timing] encode_obs result shape={tuple(video_latent.shape)} "
                f"device={video_latent.device}"
            )
            return video_latent

    def _select_history_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return images
        available_len = min(len(images), 9)
        history_len = available_len - (available_len - 1) % 4
        return images[-history_len:]

    def _extract_prompt_from_obs(self, obs):
        for key in ("prompt", "instruction", "task"):
            if key in obs and obs[key] is not None:
                return obs[key]

        payload = obs.get('obs')
        if isinstance(payload, list):
            payloads = reversed(payload)
        else:
            payloads = (payload,)
        for item in payloads:
            if isinstance(item, dict):
                for key in ("task", "instruction", "prompt"):
                    if key in item and item[key] is not None:
                        return item[key]
        return None

    def _extract_state_from_obs(self, obs):
        payload = obs.get('obs')
        if isinstance(payload, list) and len(payload) > 0:
            payload = payload[-1]
        if isinstance(payload, dict):
            executed_history = payload.get('observation.executed_action_history')
            if executed_history is not None:
                logger.info("Using observation.executed_action_history for state/action history.")
                return executed_history
            if payload.get('observation.state') is not None:
                logger.info("Using legacy observation.state for state/action history.")
            return payload.get('observation.state')

        return None

    def _clear_runtime_caches(self):
        pass

    def _create_runtime_cache(self):
        patch_size = self.job_config.patch_size
        max_history_latents = 3
        latent_token_per_chunk = (max_history_latents *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        state_token_per_chunk = self.job_config.state_history_len
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk + state_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size=2 if self.use_cfg else 1)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        with log_timing("reset total", self.device):
            self.use_cfg = self._guidance_requires_cfg()
            #### Reset all parameters
            self.frame_st_id = 0
            self.init_latent = None
            self._runtime_initialized = False
            #### clean vae and transformer cache
            with log_timing("reset clear runtime caches", self.device):
                self._clear_runtime_caches()

            self.action_per_frame = self.job_config.action_per_frame
            self.height, self.width = self.job_config.height, self.job_config.width

            if self.env_type == 'robotwin_tshape':
                tail_keys = list(self.job_config.obs_cam_keys[1:])
                crop_keys = set(getattr(self.job_config, "crop_view_keys", []))
                has_crop_row = any(key in crop_keys for key in tail_keys)
                has_arm_row = any(key not in crop_keys for key in tail_keys)
                tail_rows = int(has_crop_row) + int(has_arm_row)
                self.latent_height = (
                    self.height // 16 + tail_rows * (self.height // 32)
                )
                self.latent_width = self.width // 16
            else:
                self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                    self.job_config.obs_cam_keys)

            with log_timing("reset create transformer cache", self.device):
                self._create_runtime_cache()

            with log_timing("reset action norm tensors"):
                self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
                training_ids = getattr(
                    self.job_config,
                    "training_action_channel_ids",
                    self.job_config.used_action_channel_ids,
                )
                self.action_mask[list(training_ids)] = True

                self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                                dtype=torch.float32).reshape(-1, 1, 1)
                self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                                dtype=torch.float32).reshape(-1, 1, 1)
                state_norm_stat = getattr(
                    self.job_config, 'state_norm_stat', self.job_config.norm_stat
                )
                self.state_q01 = torch.tensor(state_norm_stat['q01'],
                                              dtype=torch.float32).reshape(-1, 1, 1)
                self.state_q99 = torch.tensor(state_norm_stat['q99'],
                                              dtype=torch.float32).reshape(-1, 1, 1)
                self.action_norm_method = self.job_config.action_norm_method

            ##### get prompt
            with log_timing("reset encode prompt", self.device):
                if prompt is None:
                    self.prompt_embeds = self.negative_prompt_embeds = None
                else:
                    self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                        prompt=prompt,
                        negative_prompt=None,
                        do_classifier_free_guidance=self.use_cfg,
                        num_videos_per_prompt=1,
                        prompt_embeds=None,
                        negative_prompt_embeds=None,
                        max_sequence_length=512,
                        device=self.device,
                        dtype=self.dtype,
                    )

            self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
            self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
            # os.makedirs(self.exp_save_root, exist_ok=True)
            self._runtime_initialized = True
            self.current_prompt_key = self._prompt_key(prompt)
            with log_timing("reset torch cuda empty_cache", self.device):
                torch.cuda.empty_cache()

    def _prompt_key(self, prompt):
        if prompt is None:
            return None
        if isinstance(prompt, (list, tuple)):
            return tuple(str(item) for item in prompt)
        return str(prompt)

    def _ensure_prompt(self, prompt):
        prompt_key = self._prompt_key(prompt)
        if prompt_key is None or prompt_key == self.current_prompt_key:
            return
        logger.info("Prompt changed; resetting server prompt/cache before inference.")
        self._reset(prompt=prompt)

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        # save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        # save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions, latents

    def _infer_action_chunk(self):
        with log_timing("infer_action_chunk total", self.device):
            frame_chunk_size = self.job_config.frame_chunk_size

            with log_timing("infer_action_chunk init noise/scheduler", self.device):
                actions = torch.randn(1,
                                      self.job_config.action_dim,
                                      frame_chunk_size,
                                      self.action_per_frame,
                                      1,
                                      device=self.device,
                                      dtype=self.dtype)

                action_inference_step = self.job_config.action_num_inference_steps
                self.action_scheduler.set_timesteps(action_inference_step)
                action_timesteps = self.action_scheduler.timesteps
                action_timesteps = F.pad(action_timesteps,
                                         (0, 1),
                                         mode='constant',
                                         value=0)

            with log_timing(
                f"infer_action_chunk denoise loop steps={len(action_timesteps)}",
                self.device,
            ):
                with torch.no_grad():
                    for i, t in enumerate(tqdm(action_timesteps)):
                        last_step = i == len(action_timesteps) - 1
                        action_cond = None

                        input_dict = self._prepare_latent_input(
                            None,
                            actions,
                            action_t=t,
                            action_cond=action_cond,
                            frame_st_id=0)
                        action_noise_pred = self.transformer(
                            self._repeat_input_for_cfg(input_dict['action_res_lst']),
                            update_cache=0, #1 if last_step else 0,
                            cache_name=self.cache_name,
                            action_mode=True)

                        if not last_step:
                            action_noise_pred = rearrange(action_noise_pred,
                                                          'b (f n) c -> b c f n 1',
                                                          f=frame_chunk_size)
                            if self.job_config.action_guidance_scale > 1:
                                action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                            else:
                                action_noise_pred = action_noise_pred[:1]
                            actions = self.action_scheduler.step(action_noise_pred,
                                                                 t,
                                                                 actions,
                                                                 return_dict=False)

            with log_timing("infer_action_chunk postprocess", self.device):
                actions[:, ~self.action_mask] *= 0
                # save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))
                actions = self.postprocess_action(actions)
            with log_timing("infer_action_chunk torch cuda empty_cache", self.device):
                torch.cuda.empty_cache()
            return actions

    def _compute_kv_cache(self, obs):
        with log_timing("compute_kv_cache total", self.device):
            self.frame_st_id = 0
            self.init_latent = None
            with log_timing("compute_kv_cache clear caches", self.device):
                self._clear_runtime_caches()
                self.transformer.clear_pred_cache(self.cache_name)

            latent_model_input = self._encode_obs(obs)
            if latent_model_input is None:
                raise ValueError("Observation is required to compute KV cache.")
            self.init_latent = latent_model_input
            with log_timing("compute_kv_cache preprocess state", self.device):
                state = self._extract_state_from_obs(obs)
                if state is None or state.shape[0] == 0:
                    valid_state_len = 0
                    state_model_input = None
                else:
                    state_model_input = self.preprocess_state(state)
                    state_model_input = state_model_input.to(device=self.device, dtype=self.dtype)
                    valid_state_len = state_model_input.shape[2]
            logger.info(
                f"get KV cache obs: {latent_model_input.shape}, state: {valid_state_len}"
            )
            history_frame_st_id = 1 - latent_model_input.shape[2]
            with log_timing("compute_kv_cache prepare latent input", self.device):
                input_dict = self._prepare_latent_input(latent_model_input,
                                                        None,
                                                        frame_st_id=history_frame_st_id)
                state_input_dict = None
                if valid_state_len > 0:
                    state_input_dict = {
                        'noisy_latents':
                        state_model_input,
                        'timesteps':
                        torch.zeros([state_model_input.shape[2]],
                                    dtype=torch.float32,
                                    device=self.device),
                        'grid_id':
                        get_state_history_grid_id(
                            history_len=state_model_input.shape[-3],
                            action_per_frame=self.job_config.action_per_frame,
                            t=1,
                            device=self.device,
                        ).to(self.device),
                        'text_emb':
                        self.prompt_embeds.to(self.dtype).clone(),
                    }

            with (
                    torch.no_grad(),
            ):
                with log_timing("compute_kv_cache transformer visual history", self.device):
                    self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                                     update_cache=1,
                                     cache_name=self.cache_name,
                                     action_mode=False,
                                     history=True)
                if state_input_dict is not None:
                    with log_timing("compute_kv_cache transformer state history", self.device):
                        self.transformer(self._repeat_input_for_cfg(state_input_dict),
                                         update_cache=1,
                                         cache_name=self.cache_name,
                                         action_mode=True,
                                         history=True)
            with log_timing("compute_kv_cache torch cuda empty_cache", self.device):
                torch.cuda.empty_cache()

    @torch.no_grad()
    def infer(self, obs):
        with log_timing("infer request total", self.device):
            reset = obs.get('reset', False)
            prompt = self._extract_prompt_from_obs(obs)
            compute_kv_cache = obs.get('compute_kv_cache', False)
            infer_action = obs.get('infer_action', False)
            prompt_key = self._prompt_key(prompt)
            prompt_changed = prompt_key is not None and prompt_key != self.current_prompt_key
            self._update_guidance_scales(obs, allow_cfg_mode_change=reset or prompt_changed)


            if reset:
                logger.info(f"******************* Reset server ******************")
                with log_timing("infer branch reset", self.device):
                    self._reset(prompt=prompt)
                return dict()
            with log_timing("infer ensure prompt", self.device):
                self._ensure_prompt(prompt)
            if infer_action and obs.get('obs') is not None:
                logger.info(
                    f"################# Compute KV Cache + Infer Action #################")
                with log_timing("infer branch compute_kv_cache + infer_action", self.device):
                    self._compute_kv_cache(obs)
                    action = self._infer_action_chunk()
                return dict(action=action)
            elif compute_kv_cache:
                logger.info(
                    f"################# Compute KV Cache #################")
                with log_timing("infer branch compute_kv_cache", self.device):
                    self._compute_kv_cache(obs)
                return dict()
            else:
                logger.info(
                    f"################# Infer Action #################")

                with log_timing("infer branch infer_action", self.device):
                    action = self._infer_action_chunk()
                return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {
            v: self._load_image_file_for_key(v)
            for v in self.job_config.obs_cam_keys
        }
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        del self.transformer
        if self.vae_half is not None:
            del self.vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

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
    config.action_representation = validate_action_representation(
        getattr(config, "action_representation", "absolute")
    )
    if getattr(config, "state_action_representation", "absolute") != "absolute":
        raise ValueError("state/action history must use absolute cmd actions")
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.pretrained_model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.pretrained_model_path
    if args.transformer_source is not None:
        source = Path(args.transformer_source).expanduser().resolve()
        candidates = (source / "checkpoints" / "last" / "transformer", source / "transformer")
        resolved = next((path for path in candidates if path.is_dir()), None)
        if resolved is None and source.is_dir() and (source / "config.json").is_file():
            resolved = source
        if resolved is None:
            raise FileNotFoundError(f"Cannot resolve transformer checkpoint from {source}")
        config.transformer_source_path = str(resolved)
    if args.state_history_len is not None:
        config.state_history_len = int(args.state_history_len)
    if args.num_inference_steps is not None:
        config.num_inference_steps = int(args.num_inference_steps)
    if args.action_num_inference_steps is not None:
        config.action_num_inference_steps = int(args.action_num_inference_steps)
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(
            model,
            local_rank,
            config.host,
            port,
            metadata={
                "experiment_name": str(getattr(config, "experiment_name", args.config_name)),
                "action_representation": config.action_representation,
                "state_action_representation": "absolute",
                "action_head_type": str(getattr(config, "action_head_type", "shared")),
                "obs_cam_keys": list(config.obs_cam_keys),
            },
        )
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--experiment-config",
        type=str,
        default=None,
        help="JSON experiment config shared by training and inference.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="Base Wan2.2 directory containing VAE, tokenizer and text encoder.",
    )
    parser.add_argument(
        "--transformer-source",
        type=str,
        default=None,
        help="Fine-tuned checkpoint root or transformer directory.",
    )
    parser.add_argument(
        "--state-history-len",
        type=int,
        default=None,
        help='override state history length',
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help='override video denoising steps',
    )
    parser.add_argument(
        "--action-num-inference-steps",
        type=int,
        default=None,
        help='override action denoising steps',
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    # import debugpy; debugpy.listen(5678); logger.info("Waiting for debugger attach..."); debugpy.wait_for_client()
    init_logger()
    main()
