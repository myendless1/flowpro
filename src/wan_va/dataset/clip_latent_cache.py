"""On-disk clip VAE latent cache used by training and precompute."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import torch

CAMERA_KEY_ALIASES = {
    "observation.images.cam_high": (
        "observation.images.cam_high",
        "observation.images.cam_main",
    ),
    "observation.images.cam_main": (
        "observation.images.cam_main",
        "observation.images.cam_high",
    ),
}


def _latent_root(repo_root: Path, freq_ratio: int) -> Path:
    return repo_root / "latent" / f"fr{int(freq_ratio)}"


def resolve_cache_camera_key(repo_root: Path, camera_key: str, freq_ratio: int = 2) -> str:
    """Resolve on-disk camera folder name; fall back to config key before cache exists."""
    candidates = CAMERA_KEY_ALIASES.get(camera_key, (camera_key,))
    latent_root = _latent_root(repo_root, freq_ratio)
    if latent_root.is_dir():
        found = {
            path.name
            for path in latent_root.glob("chunk-*/*")
            if path.is_dir()
        }
        for candidate in candidates:
            if candidate in found:
                return candidate
    return candidates[0]


def clip_latent_path(
    repo_root: Path,
    *,
    episode_index: int,
    clip_start: int,
    camera_key: str,
    freq_ratio: int = 2,
    chunk_size: int = 1000,
) -> Path:
    resolved_key = resolve_cache_camera_key(repo_root, camera_key, freq_ratio=freq_ratio)
    chunk_name = f"chunk-{int(episode_index) // int(chunk_size):03d}"
    return (
        _latent_root(repo_root, freq_ratio)
        / chunk_name
        / resolved_key
        / f"episode_{int(episode_index):06d}_{int(clip_start)}.pth"
    )


def clip_latent_paths_for_sample(
    repo_root: Path,
    *,
    episode_index: int,
    clip_start: int,
    camera_keys: Sequence[str],
    freq_ratio: int = 2,
    chunk_size: int = 1000,
) -> dict[str, Path]:
    return {
        camera_key: clip_latent_path(
            repo_root,
            episode_index=episode_index,
            clip_start=clip_start,
            camera_key=camera_key,
            freq_ratio=freq_ratio,
            chunk_size=chunk_size,
        )
        for camera_key in camera_keys
    }


def clip_latent_paths_exist(
    repo_root: Path,
    *,
    episode_index: int,
    clip_start: int,
    camera_keys: Sequence[str],
    freq_ratio: int = 2,
    chunk_size: int = 1000,
) -> bool:
    repo_root = Path(repo_root)
    for camera_key in camera_keys:
        path = clip_latent_path(
            repo_root,
            episode_index=episode_index,
            clip_start=clip_start,
            camera_key=camera_key,
            freq_ratio=freq_ratio,
            chunk_size=chunk_size,
        )
        if not path.is_file():
            return False
    return True


def load_clip_latent(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    latent = payload["latent"]
    if not torch.is_tensor(latent):
        raise TypeError(f"Invalid latent payload in {path}")
    return latent


def save_clip_latent(
    path: Path,
    *,
    latent: torch.Tensor,
    clip_start: int,
    frame_ids: Sequence[int],
    episode_index: int,
    camera_key: str,
    video_frame_stride: int,
    raw_window_frames: int,
    freq_ratio: int,
    source_video: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    payload = {
        "latent": latent.detach().cpu().to(torch.bfloat16).contiguous(),
        "clip_start": int(clip_start),
        "frame_ids": [int(value) for value in frame_ids],
        "episode_index": int(episode_index),
        "camera_key": camera_key,
        "freq_ratio": int(freq_ratio),
        "video_frame_stride": int(video_frame_stride),
        "raw_window_frames": int(raw_window_frames),
        "source_video": source_video,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _as_clip_latent_cthw(latent: torch.Tensor) -> torch.Tensor:
    """Normalize per-camera latent to [C, T, H, W]."""
    if latent.ndim == 5:
        if latent.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got shape {tuple(latent.shape)}")
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(f"Expected [C, T, H, W], got shape {tuple(latent.shape)}")
    return latent


def merge_robotwin_tshape_latent(
    high_latent: torch.Tensor,
    wrist_latents: Sequence[torch.Tensor],
    wrist_camera_keys: Sequence[str] | None = None,
) -> torch.Tensor:
    """Merge per-camera [C, T, H, W] latents into one tshape tensor [C, T, H, W]."""
    high_mu = _as_clip_latent_cthw(high_latent)
    if not wrist_latents:
        return high_mu

    wrist_stack = torch.stack(
        [_as_clip_latent_cthw(latent) for latent in wrist_latents],
        dim=0,
    )
    wrist_camera_keys = list(wrist_camera_keys or [])
    crop_indices = [
        cam_id
        for cam_id, camera_key in enumerate(wrist_camera_keys)
        if "right_in_" in camera_key
    ]
    arm_indices = [
        cam_id
        for cam_id in range(wrist_stack.shape[0])
        if cam_id not in set(crop_indices)
    ]

    latent_rows = []
    if crop_indices:
        latent_rows.append(torch.cat([wrist_stack[cam_id] for cam_id in crop_indices], dim=-1))
    if arm_indices:
        latent_rows.append(torch.cat([wrist_stack[cam_id] for cam_id in arm_indices], dim=-1))
    return torch.cat([*latent_rows, high_mu], dim=-2)


def load_robotwin_tshape_clip_latent(
    repo_root: Path,
    *,
    episode_index: int,
    clip_start: int,
    camera_keys: Sequence[str],
    freq_ratio: int = 2,
    chunk_size: int = 1000,
) -> torch.Tensor | None:
    repo_root = Path(repo_root)
    high_key = camera_keys[0]
    high_path = clip_latent_path(
        repo_root,
        episode_index=episode_index,
        clip_start=clip_start,
        camera_key=high_key,
        freq_ratio=freq_ratio,
        chunk_size=chunk_size,
    )
    if not high_path.is_file():
        return None

    wrist_paths = []
    for camera_key in camera_keys[1:]:
        path = clip_latent_path(
            repo_root,
            episode_index=episode_index,
            clip_start=clip_start,
            camera_key=camera_key,
            freq_ratio=freq_ratio,
            chunk_size=chunk_size,
        )
        if not path.is_file():
            return None
        wrist_paths.append(path)

    high_latent = load_clip_latent(high_path)
    wrist_latents = [load_clip_latent(path) for path in wrist_paths]
    return merge_robotwin_tshape_latent(
        high_latent,
        wrist_latents,
        wrist_camera_keys=camera_keys[1:],
    )
