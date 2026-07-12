# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from pathlib import Path
from typing import Callable, ClassVar

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention

__all__ = ["FlexAttnMaskManager"]


class FlexAttnMaskManager:
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(
        create_block_mask
    )
    compiled_flex_attention: ClassVar[Callable] = torch.compile(
        flex_attention,
        dynamic=True,
    )
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    @classmethod
    def run_attention(
        cls,
        query,
        key,
        value,
        is_cross=False,
        dtype=torch.bfloat16,
    ):
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = (
            cls.get_cross_attention_mask()
            if is_cross
            else cls.get_attention_mask()
        )
        x_out = cls.compiled_flex_attention(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )
        return rearrange(x_out, "b n s d -> b s n d")

    @staticmethod
    def _build_mask_inputs(
        latent_shape,
        action_shape,
        padded_length,
        patch_size,
        text_seq_len,
        device,
        state_shape,
        state_mask=None,
    ):
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape
        _, _, S_F, S_H, S_W = state_shape
        latent_tokens_h = L_H // patch_size[1]
        latent_tokens_w = L_W // patch_size[2]
        latent_frames = L_F // patch_size[0]

        latent_seq_id = torch.arange(B, device=device)[:, None, None, None].expand(
            -1, latent_frames, latent_tokens_h, latent_tokens_w
        ).flatten()
        action_seq_id = torch.arange(B, device=device)[:, None, None, None].expand(
            -1, A_F, A_H, A_W
        ).flatten()
        state_seq_id = torch.arange(B, device=device)[:, None, None, None].expand(
            -1, S_F, S_H, S_W
        ).flatten()
        if state_mask is not None:
            state_mask = state_mask.to(device=device, dtype=torch.bool)
            expanded_state_mask = state_mask[:, :, None, None].expand(-1, -1, S_H, S_W).flatten()
            state_seq_id = torch.where(
                expanded_state_mask,
                state_seq_id,
                torch.full_like(state_seq_id, -1),
            )
        seq_ids = torch.cat([latent_seq_id, action_seq_id, state_seq_id])

        latent_raw_frame_ids = torch.arange(L_F, device=device)[None, :, None, None].expand(
            B, -1, latent_tokens_h, latent_tokens_w
        )[None].flatten()
        action_raw_frame_ids = torch.arange(A_F, device=device)[None, :, None, None].expand(
            B, -1, A_H, A_W
        )[None].flatten()
        state_base_frame_ids = torch.arange(-S_F, 0, device=device)[None, :, None, None].expand(
            B, -1, S_H, S_W
        )
        state_raw_frame_ids = state_base_frame_ids.flatten()

        raw_frame_parts = [latent_raw_frame_ids, action_raw_frame_ids]
        modality_parts = [
            torch.zeros_like(latent_raw_frame_ids),
            torch.ones_like(action_raw_frame_ids),
        ]
        video_noise_ids = (latent_raw_frame_ids > 0).long()
        noise_parts = [
            video_noise_ids,
            torch.ones_like(action_raw_frame_ids),
        ]
        raw_frame_parts.append(state_raw_frame_ids)
        modality_parts.append(torch.ones_like(state_raw_frame_ids))
        noise_parts.append(torch.zeros_like(state_raw_frame_ids))

        raw_frame_ids = torch.cat(raw_frame_parts)
        modality_ids = torch.cat(modality_parts)
        noise_ids = torch.cat(noise_parts)

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1).long().to(device)
        raw_frame_ids = F.pad(raw_frame_ids, (0, padded_length), value=-1).long().to(
            device
        )
        modality_ids = F.pad(modality_ids, (0, padded_length), value=-1).long().to(
            device
        )
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1).long().to(device)
        text_seq_ids = torch.arange(B)[:, None].expand(-1, text_seq_len).flatten()
        text_seq_ids = text_seq_ids.long().to(device)

        return (
            seq_ids,
            raw_frame_ids,
            modality_ids,
            noise_ids,
            text_seq_ids,
        )

    @staticmethod
    def _self_visible(
        seq_ids,
        raw_frame_ids,
        modality_ids,
        noise_ids,
        q_idx,
        kv_idx,
    ):
        same_seq = (
            (seq_ids[q_idx] == seq_ids[kv_idx])
            & (seq_ids[q_idx] >= 0)
            & (seq_ids[kv_idx] >= 0)
        )

        video_query = modality_ids[q_idx] == 0
        action_query = modality_ids[q_idx] == 1
        video_key = modality_ids[kv_idx] == 0
        action_key = modality_ids[kv_idx] == 1
        noisy_query = noise_ids[q_idx] == 1
        noisy_key = noise_ids[kv_idx] == 1
        clean_query = noise_ids[q_idx] == 0
        clean_key = noise_ids[kv_idx] == 0

        video_pair = video_query & video_key
        video_visible = (
            (video_pair & clean_query & (raw_frame_ids[q_idx] == 0) & (raw_frame_ids[kv_idx] == 0))
            | (video_pair & noisy_query & (raw_frame_ids[q_idx] > 0) & (raw_frame_ids[kv_idx] >= 0))
        )
        action_from_video_visible = (
            action_query
            & noisy_query
            & clean_key
            & video_key
            & (raw_frame_ids[kv_idx] == 0)
        )
        state_visible = (
            action_query
            & clean_query
            & clean_key
            & action_key
            & (raw_frame_ids[q_idx] < 0)
            & (raw_frame_ids[kv_idx] < 0)
        )
        action_from_state_visible = (
            action_query
            & noisy_query
            & clean_key
            & action_key
            & (raw_frame_ids[kv_idx] < 0)
        )
        action_from_action_visible = (
            action_query
            & action_key
            & noisy_query
            & noisy_key
        )
        action_visible = (
            action_from_video_visible
            | state_visible
            | action_from_state_visible
            | action_from_action_visible
        )
        return (video_visible | action_visible) & same_seq

    @staticmethod
    def _cross_visible(seq_ids, text_seq_ids, q_idx, kv_idx):
        return (
            (seq_ids[q_idx] == text_seq_ids[kv_idx])
            & (seq_ids[q_idx] >= 0)
            & (text_seq_ids[kv_idx] >= 0)
        )

    @classmethod
    def _get_mask_mod(
        cls,
        seq_ids,
        raw_frame_ids,
        modality_ids,
        noise_ids,
    ):
        def mask_mod(b, h, q_idx, kv_idx):
            return cls._self_visible(
                seq_ids,
                raw_frame_ids,
                modality_ids,
                noise_ids,
                q_idx,
                kv_idx,
            )

        return mask_mod

    @classmethod
    def _get_cross_mask_mod(cls, seq_ids, text_seq_ids):
        def mask_mod(b, h, q_idx, kv_idx):
            return cls._cross_visible(seq_ids, text_seq_ids, q_idx, kv_idx)

        return mask_mod

    @classmethod
    @torch.no_grad()
    def init_masks(
        cls,
        latent_shape,
        action_shape,
        padded_length,
        patch_size,
        device,
        state_shape,
        state_mask=None,
        text_seq_len=512,
        export_mask=False,
        output_dir=None,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        (
            seq_ids,
            raw_frame_ids,
            modality_ids,
            noise_ids,
            text_seq_ids,
        ) = cls._build_mask_inputs(
            latent_shape,
            action_shape,
            padded_length,
            patch_size,
            text_seq_len,
            device,
            state_shape=state_shape,
            state_mask=state_mask,
        )
        cls.attention_mask = cls.compiled_create_block_mask(
            cls._get_mask_mod(
                seq_ids,
                raw_frame_ids,
                modality_ids,
                noise_ids,
            ),
            1,
            1,
            len(seq_ids),
            len(seq_ids),
            device=device,
        )
        cls.cross_attention_mask = cls.compiled_create_block_mask(
            cls._get_cross_mask_mod(seq_ids, text_seq_ids),
            1,
            1,
            len(seq_ids),
            len(text_seq_ids),
            device=device,
        )
        if export_mask:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            q_idx = torch.arange(len(seq_ids), device=device)[:, None]
            Image.fromarray(
                cls._self_visible(
                    seq_ids,
                    raw_frame_ids,
                    modality_ids,
                    noise_ids,
                    q_idx,
                    torch.arange(len(seq_ids), device=device)[None, :],
                ).to(torch.uint8).mul(255).cpu().numpy(),
                mode="L",
            ).save(output_dir / "attention_mask.png")
            Image.fromarray(
                cls._cross_visible(
                    seq_ids,
                    text_seq_ids,
                    q_idx,
                    torch.arange(len(text_seq_ids), device=device)[None, :],
                ).to(torch.uint8).mul(255).cpu().numpy(),
                mode="L",
            ).save(output_dir / "cross_attention_mask.png")

    @classmethod
    def get_attention_mask(cls):
        return cls.attention_mask

    @classmethod
    def get_cross_attention_mask(cls):
        return cls.cross_attention_mask

    @classmethod
    @torch.no_grad()
    def export_dense_mask(
        cls,
        latent_shape,
        action_shape,
        padded_length,
        patch_size,
        state_shape,
        state_mask=None,
        kind="self",
        text_seq_len=512,
        device="cpu",
    ):
        (
            seq_ids,
            raw_frame_ids,
            modality_ids,
            noise_ids,
            text_seq_ids,
        ) = cls._build_mask_inputs(
            latent_shape,
            action_shape,
            padded_length,
            patch_size,
            text_seq_len,
            device,
            state_shape=state_shape,
            state_mask=state_mask,
        )
        q_idx = torch.arange(len(seq_ids), device=device)[:, None]
        if kind == "self":
            kv_idx = torch.arange(len(seq_ids), device=device)[None, :]
            return cls._self_visible(
                seq_ids,
                raw_frame_ids,
                modality_ids,
                noise_ids,
                q_idx,
                kv_idx,
            ).to(torch.bool)
        if kind == "cross":
            kv_idx = torch.arange(len(text_seq_ids), device=device)[None, :]
            return cls._cross_visible(seq_ids, text_seq_ids, q_idx, kv_idx).to(
                torch.bool
            )
        raise ValueError(f"Unsupported mask kind: {kind}")
