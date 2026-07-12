#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.dataset.curation.base import (  # noqa: E402
    resolve_lerobot_videos_root,
)


@dataclass(frozen=True)
class VideoInfo:
    video_path: str
    chunk_name: str
    num_frames: int
    image_height: int
    image_width: int
    video_fps: float


@dataclass(frozen=True)
class LatentTask:
    repo_dir: str
    video_path: str
    save_path: str
    text_emb_path: str
    camera_key: str
    episode_index: int
    start_frame: int
    end_frame: int
    frame_ids: tuple[int, ...]
    image_height: int
    image_width: int
    text: str
    fps: float
    ori_fps: float
    temporal_stride: int
    overwrite: bool


@dataclass(frozen=True)
class BuildPlan:
    repo_dirs: tuple[str, ...]
    tasks: tuple[LatentTask, ...]


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("Missing runtime dependency: torch")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    _require_torch()
    name = str(dtype_name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_text_device(device_arg: str) -> torch.device:
    _require_torch()
    if torch.cuda.is_available() and str(device_arg).startswith("cuda"):
        return torch.device(device_arg)
    return torch.device("cpu")


def _available_gpu_ids(device_ids_arg: str) -> list[int]:
    _require_torch()
    explicit = _parse_csv(device_ids_arg)
    if explicit:
        return [int(value) for value in explicit]
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def _repo_task_slug(repo_dir: Path) -> str:
    name = repo_dir.name
    for marker in ("-demo_", "-piper_", "-kuavo_", "-sim_"):
        marker_idx = name.find(marker)
        if marker_idx > 0:
            return name[:marker_idx]
    return name


def _default_text_for_repo(repo_dir: Path) -> str:
    return _repo_task_slug(repo_dir).replace("_", " ").strip()


def _first_episode_text(repo_dir: Path, episode_row: dict[str, Any]) -> str:
    action_config = episode_row.get("action_config")
    if isinstance(action_config, list):
        for item in action_config:
            if not isinstance(item, dict):
                continue
            text = str(item.get("action_text", "")).strip()
            if text:
                return text

    tasks = episode_row.get("tasks")
    if isinstance(tasks, list):
        for item in tasks:
            text = str(item).strip()
            if text:
                return text

    text = str(episode_row.get("task", "")).strip()
    if text:
        return text
    return _default_text_for_repo(repo_dir)


def _segment_text(repo_dir: Path, episode_row: dict[str, Any], segment_row: dict[str, Any]) -> str:
    text = str(segment_row.get("action_text", "")).strip()
    if text:
        return text
    return _first_episode_text(repo_dir, episode_row)


def _episode_segments(repo_dir: Path, episode_row: dict[str, Any]) -> list[dict[str, Any]]:
    segments = episode_row.get("action_config")
    if isinstance(segments, list) and segments:
        out = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            if "start_frame" not in segment or "end_frame" not in segment:
                continue
            out.append(segment)
        if out:
            return out

    length = int(episode_row.get("length", 0))
    return [
        {
            "start_frame": 0,
            "end_frame": length,
            "action_text": _first_episode_text(repo_dir, episode_row),
        }
    ]


def _discover_repo_roots(dataset_root: Path) -> list[Path]:
    root = dataset_root.resolve()
    try:
        direct_videos_root = resolve_lerobot_videos_root(root, must_exist=True)
    except FileNotFoundError:
        direct_videos_root = None
    if (
        (root / "meta" / "episodes.jsonl").is_file()
        and (root / "meta" / "info.json").is_file()
        and direct_videos_root is not None
    ):
        return [root]

    repo_roots: set[Path] = set()
    for meta_path in root.rglob("meta/episodes.jsonl"):
        repo_dir = meta_path.parent.parent.resolve()
        if not (repo_dir / "meta" / "info.json").is_file():
            continue
        try:
            resolve_lerobot_videos_root(repo_dir, must_exist=True)
        except FileNotFoundError:
            continue
        repo_roots.add(repo_dir)
    return sorted(repo_roots, key=lambda path: str(path))


def _discover_camera_keys(repo_dir: Path, requested_cameras: Sequence[str]) -> list[str]:
    videos_root = resolve_lerobot_videos_root(repo_dir, must_exist=True)
    found = {
        path.name
        for path in videos_root.glob("chunk-*/*")
        if path.is_dir()
    }
    if requested_cameras:
        return [camera_key for camera_key in requested_cameras if camera_key in found]
    return sorted(found)


def _find_episode_video_path(repo_dir: Path, episode_index: int, camera_key: str, chunk_size: int) -> Path | None:
    expected_chunk = f"chunk-{episode_index // max(1, int(chunk_size)):03d}"
    videos_root = resolve_lerobot_videos_root(repo_dir, must_exist=True)
    expected = videos_root / expected_chunk / camera_key / f"episode_{episode_index:06d}.mp4"
    if expected.is_file():
        return expected

    candidates = sorted(
        videos_root.glob(f"chunk-*/{camera_key}/episode_{episode_index:06d}.mp4")
    )
    if not candidates:
        return None
    return candidates[0]


def _probe_video(video_path: Path) -> tuple[int, int, int, float | None]:
    frame_count = width = height = 0
    fps: float | None = None

    try:
        import av

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        if stream.frames is not None:
            frame_count = max(frame_count, int(stream.frames))
        width = max(width, int(stream.codec_context.width or 0), int(getattr(stream, "width", 0) or 0))
        height = max(height, int(stream.codec_context.height or 0), int(getattr(stream, "height", 0) or 0))
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        container.close()
    except Exception:
        pass

    if frame_count <= 0 or width <= 0 or height <= 0 or fps is None or fps <= 0:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            if frame_count <= 0:
                frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            if width <= 0:
                width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            if height <= 0:
                height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if fps is None or fps <= 0:
                fps_value = float(cap.get(cv2.CAP_PROP_FPS))
                if fps_value > 0:
                    fps = fps_value
        cap.release()

    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(
            f"Failed to probe video stats for {video_path}: "
            f"frames={frame_count}, width={width}, height={height}, fps={fps}"
        )
    return frame_count, height, width, fps


def _text_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class _TextEmbedderCache:
    def __init__(self, pretrained_root: Path, device: torch.device, dtype: torch.dtype):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from wan_va.modules.utils import load_text_encoder, load_tokenizer

        self._prompt_clean = prompt_clean
        self.device = device
        self.dtype = dtype
        self.tokenizer = load_tokenizer(str(pretrained_root / "tokenizer"))
        self.text_encoder = load_text_encoder(
            str(pretrained_root / "text_encoder"),
            torch_dtype=dtype,
            torch_device=device,
        )
        self.memory_cache: dict[str, torch.Tensor] = {}

    def encode(self, prompt: str) -> torch.Tensor:
        clean_prompt = self._prompt_clean(prompt or "")
        if clean_prompt in self.memory_cache:
            return self.memory_cache[clean_prompt]

        text_inputs = self.tokenizer(
            [clean_prompt],
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)
        seq_len = int(mask[0].gt(0).sum().item())

        with torch.no_grad():
            prompt_embeds = self.text_encoder(input_ids, mask).last_hidden_state.to(
                dtype=self.dtype,
                device=self.device,
            )

        prompt_embeds = prompt_embeds[:, :seq_len]
        if seq_len < 512:
            pad = torch.zeros(
                (1, 512 - seq_len, prompt_embeds.shape[-1]),
                device=self.device,
                dtype=self.dtype,
            )
            prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)

        out = prompt_embeds[0].detach().cpu().to(torch.bfloat16).contiguous()
        self.memory_cache[clean_prompt] = out
        return out


def _ensure_empty_emb(
    *,
    repo_dir: Path,
    text_mode: str,
    text_embedder: _TextEmbedderCache | None,
    overwrite: bool,
) -> torch.Tensor:
    if text_mode == "encode":
        if text_embedder is None:
            raise RuntimeError("text_embedder is required when text_mode=encode")
        empty_emb = text_embedder.encode("")
    elif text_mode == "empty":
        empty_emb = torch.zeros((512, 4096), dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported text_mode: {text_mode}")

    empty_emb_path = repo_dir / "empty_emb.pt"
    if overwrite or not empty_emb_path.exists():
        torch.save(empty_emb.to(torch.bfloat16).cpu().contiguous(), empty_emb_path)
    return empty_emb


def _ensure_text_emb_file(
    *,
    repo_dir: Path,
    text: str,
    text_mode: str,
    text_embedder: _TextEmbedderCache | None,
    empty_emb: torch.Tensor,
    overwrite: bool,
) -> Path:
    cache_dir = repo_dir / "latents" / "_text_emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if text_mode == "empty":
        cache_path = cache_dir / "empty.pt"
        if overwrite or not cache_path.exists():
            torch.save(empty_emb.to(torch.bfloat16).cpu().contiguous(), cache_path)
        return cache_path

    if text_mode != "encode":
        raise ValueError(f"Unsupported text_mode: {text_mode}")
    if text_embedder is None:
        raise RuntimeError("text_embedder is required when text_mode=encode")

    cache_path = cache_dir / f"{_text_cache_key(text)}.pt"
    if overwrite or not cache_path.exists():
        torch.save(text_embedder.encode(text), cache_path)
    return cache_path


def _build_video_infos(
    repo_dir: Path,
    episode_index: int,
    camera_keys: Sequence[str],
    chunk_size: int,
) -> dict[str, VideoInfo] | None:
    out: dict[str, VideoInfo] = {}
    chunk_names: set[str] = set()
    for camera_key in camera_keys:
        video_path = _find_episode_video_path(repo_dir, episode_index, camera_key, chunk_size)
        if video_path is None:
            return None
        num_frames, image_height, image_width, video_fps = _probe_video(video_path)
        chunk_name = video_path.parent.parent.name
        chunk_names.add(chunk_name)
        out[camera_key] = VideoInfo(
            video_path=str(video_path),
            chunk_name=chunk_name,
            num_frames=int(num_frames),
            image_height=int(image_height),
            image_width=int(image_width),
            video_fps=float(video_fps) if video_fps is not None and video_fps > 0 else 0.0,
        )

    if len(chunk_names) > 1:
        raise RuntimeError(
            f"Episode {episode_index} in {repo_dir} resolved to multiple chunk names: {sorted(chunk_names)}"
        )
    return out


def _effective_clip_end(
    *,
    requested_end: int,
    episode_length: int,
    video_infos: dict[str, VideoInfo],
) -> int:
    end_frame = int(requested_end)
    if episode_length > 0:
        end_frame = min(end_frame, int(episode_length))
    if video_infos:
        end_frame = min(end_frame, min(info.num_frames for info in video_infos.values()))
    return int(end_frame)


def _build_tasks(args: argparse.Namespace, text_embedder: _TextEmbedderCache | None) -> BuildPlan:
    requested_cameras = _parse_csv(args.camera_keys)
    repo_dirs = _discover_repo_roots(args.dataset_root)
    if not repo_dirs:
        raise RuntimeError(f"No LeRobot repos found under {args.dataset_root}")

    if int(args.max_repos) > 0:
        repo_dirs = repo_dirs[: int(args.max_repos)]

    overwrite = bool(args.overwrite)
    tasks: list[LatentTask] = []

    for repo_dir in repo_dirs:
        info_path = repo_dir / "meta" / "info.json"
        episodes_path = repo_dir / "meta" / "episodes.jsonl"
        if not info_path.is_file() or not episodes_path.is_file():
            print(f"[WARN] skip {repo_dir}: missing meta/info.json or meta/episodes.jsonl")
            continue

        info = _load_json(info_path)
        chunk_size = int(info.get("chunks_size", 1000) or 1000)
        camera_keys = _discover_camera_keys(repo_dir, requested_cameras)
        if not camera_keys:
            print(f"[WARN] skip {repo_dir}: no matching camera dirs under videos/")
            continue

        empty_emb = _ensure_empty_emb(
            repo_dir=repo_dir,
            text_mode=args.text_mode,
            text_embedder=text_embedder,
            overwrite=overwrite,
        )

        episode_rows = sorted(
            _load_jsonl_rows(episodes_path),
            key=lambda row: int(row["episode_index"]),
        )
        produced_episodes = 0
        for episode_row in episode_rows:
            episode_index = int(episode_row["episode_index"])
            video_infos = _build_video_infos(
                repo_dir=repo_dir,
                episode_index=episode_index,
                camera_keys=camera_keys,
                chunk_size=chunk_size,
            )
            if video_infos is None:
                print(f"[WARN] skip {repo_dir} episode={episode_index}: missing one or more requested videos")
                continue

            produced_episodes += 1
            if int(args.max_episodes) > 0 and produced_episodes > int(args.max_episodes):
                break

            episode_length = int(episode_row.get("length", 0) or 0)
            segments = _episode_segments(repo_dir, episode_row)
            for segment in segments:
                start_frame = int(segment["start_frame"])
                requested_end = int(segment["end_frame"])
                end_frame = _effective_clip_end(
                    requested_end=requested_end,
                    episode_length=episode_length,
                    video_infos=video_infos,
                )
                if end_frame <= start_frame:
                    continue

                frame_ids = tuple(range(start_frame, end_frame, int(args.frame_stride)))
                if not frame_ids:
                    continue

                text = _segment_text(repo_dir, episode_row, segment)
                text_emb_path = _ensure_text_emb_file(
                    repo_dir=repo_dir,
                    text=text,
                    text_mode=args.text_mode,
                    text_embedder=text_embedder,
                    empty_emb=empty_emb,
                    overwrite=overwrite,
                )

                for camera_key, video_info in video_infos.items():
                    ori_fps = float(video_info.video_fps) if video_info.video_fps > 0 else float(args.ori_fps)
                    if float(args.fps) > 0:
                        fps = float(args.fps)
                    else:
                        fps = float(ori_fps) / max(1, int(args.frame_stride))

                    save_path = (
                        repo_dir
                        / "latents"
                        / video_info.chunk_name
                        / camera_key
                        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
                    )
                    if save_path.exists() and not overwrite:
                        continue

                    tasks.append(
                        LatentTask(
                            repo_dir=str(repo_dir),
                            video_path=video_info.video_path,
                            save_path=str(save_path),
                            text_emb_path=str(text_emb_path),
                            camera_key=camera_key,
                            episode_index=episode_index,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            frame_ids=frame_ids,
                            image_height=video_info.image_height,
                            image_width=video_info.image_width,
                            text=text,
                            fps=float(fps),
                            ori_fps=float(ori_fps),
                            temporal_stride=int(args.temporal_stride),
                            overwrite=overwrite,
                        )
                    )

    return BuildPlan(
        repo_dirs=tuple(str(path) for path in repo_dirs),
        tasks=tuple(tasks),
    )


def _cleanup_text_cache_dirs(repo_dirs: Sequence[str | Path]) -> int:
    removed = 0
    for repo_dir in repo_dirs:
        cache_dir = Path(repo_dir) / "latents" / "_text_emb_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed += 1
    return removed


def _resize_frames(frames: torch.Tensor, image_height: int, image_width: int) -> torch.Tensor:
    if frames.shape[-2] == image_height and frames.shape[-1] == image_width:
        return frames.contiguous()
    return F.interpolate(
        frames.float(),
        size=(image_height, image_width),
        mode="bilinear",
        align_corners=False,
    ).to(frames.dtype).contiguous()


def _decode_selected_frames_hdf5(
    hdf5_path: Path,
    frame_ids: Sequence[int],
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    decoded: list[torch.Tensor] = []
    used_ids: list[int] = []
    with h5py.File(hdf5_path, "r") as h5_file:
        if "frames" not in h5_file:
            raise RuntimeError(f"Missing 'frames' dataset in {hdf5_path}")
        frames_dataset = h5_file["frames"]
        num_frames = len(frames_dataset)
        for frame_id in frame_ids:
            if frame_id < 0 or frame_id >= num_frames:
                continue
            encoded = np.asarray(frames_dataset[frame_id], dtype=np.uint8)
            frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"cv2.imdecode failed for {hdf5_path} at frame_id={frame_id}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            decoded.append(torch.from_numpy(np.ascontiguousarray(frame_rgb)).permute(2, 0, 1))
            used_ids.append(int(frame_id))

    if not decoded:
        raise RuntimeError(f"Failed to decode any requested frames from {hdf5_path}")
    return _resize_frames(torch.stack(decoded, dim=0), image_height, image_width), tuple(used_ids)


def _decode_selected_frames_mp4(
    video_path: Path,
    frame_ids: Sequence[int],
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    wanted_ids = sorted({int(value) for value in frame_ids if int(value) >= 0})
    if not wanted_ids:
        raise RuntimeError(f"No non-negative frame ids requested for {video_path}")

    try:
        import av

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        decoded: list[torch.Tensor] = []
        used_ids: list[int] = []
        wanted_pos = 0
        for frame_idx, frame in enumerate(container.decode(stream)):
            if wanted_pos >= len(wanted_ids):
                break
            if frame_idx != wanted_ids[wanted_pos]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            decoded.append(torch.from_numpy(rgb).permute(2, 0, 1))
            used_ids.append(frame_idx)
            wanted_pos += 1
        container.close()
        if decoded:
            return _resize_frames(torch.stack(decoded, dim=0), image_height, image_width), tuple(used_ids)
    except Exception:
        pass

    cap = cv2.VideoCapture(str(video_path))
    decoded = []
    used_ids = []
    wanted_pos = 0
    frame_idx = 0
    while wanted_pos < len(wanted_ids):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx != wanted_ids[wanted_pos]:
            frame_idx += 1
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        decoded.append(torch.from_numpy(frame).permute(2, 0, 1))
        used_ids.append(frame_idx)
        wanted_pos += 1
        frame_idx += 1
    cap.release()

    if not decoded:
        raise RuntimeError(f"Failed to decode requested frames from {video_path}")
    return _resize_frames(torch.stack(decoded, dim=0), image_height, image_width), tuple(used_ids)


def _load_video_selected_frames(
    video_path: Path,
    frame_ids: Sequence[int],
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    hdf5_path = video_path.with_suffix(".hdf5")
    if hdf5_path.is_file():
        return _decode_selected_frames_hdf5(
            hdf5_path=hdf5_path,
            frame_ids=frame_ids,
            image_height=image_height,
            image_width=image_width,
        )
    return _decode_selected_frames_mp4(
        video_path=video_path,
        frame_ids=frame_ids,
        image_height=image_height,
        image_width=image_width,
    )


def _normalize_latents(mu: torch.Tensor, vae: Any) -> torch.Tensor:
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=mu.device,
        dtype=mu.dtype,
    ).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std,
        device=mu.device,
        dtype=mu.dtype,
    ).view(1, -1, 1, 1, 1)
    return ((mu.float() - latents_mean) * (1.0 / latents_std)).to(mu.dtype)


def _extract_vae_latent(encoded: Any) -> torch.Tensor:
    if hasattr(encoded, "latent_dist"):
        latent_dist = encoded.latent_dist
        if hasattr(latent_dist, "mode"):
            return latent_dist.mode()
        if hasattr(latent_dist, "mean"):
            return latent_dist.mean
    if isinstance(encoded, (tuple, list)) and encoded:
        return _extract_vae_latent(encoded[0])
    if isinstance(encoded, torch.Tensor):
        return encoded
    raise RuntimeError(f"Cannot extract latent tensor from type={type(encoded).__name__}")


def _encode_video_latent(
    frames: torch.Tensor,
    *,
    vae: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise RuntimeError(f"Expected frames [T,3,H,W], got {tuple(frames.shape)}")

    video = frames.to(device=device, dtype=torch.float32) / 255.0 * 2.0 - 1.0
    video = video.permute(1, 0, 2, 3).unsqueeze(0).contiguous().to(dtype=dtype)
    with torch.no_grad():
        try:
            encoded = vae.encode(video, return_dict=True)
        except TypeError:
            encoded = vae.encode(video)
        latent = _extract_vae_latent(encoded)
        if latent.ndim == 5 and int(latent.shape[1]) == int(vae.config.z_dim) * 2:
            latent, _ = torch.chunk(latent, 2, dim=1)
        latent = _normalize_latents(latent, vae)
    return latent[0].detach().cpu().contiguous()


def _trim_to_valid_vae_frames(
    frames: torch.Tensor,
    frame_ids: Sequence[int],
    temporal_stride: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    count = min(int(frames.shape[0]), len(frame_ids))
    if count <= 0:
        raise RuntimeError("No frames available for VAE encoding")

    stride = max(1, int(temporal_stride))
    valid_count = ((count - 1) // stride) * stride + 1
    valid_count = max(valid_count, 1)
    return frames[:valid_count].contiguous(), tuple(frame_ids[:valid_count])


def _target_latent_frame_count(video_num_frames: int, temporal_stride: int) -> int:
    return max(1, (int(video_num_frames) - 1) // max(1, int(temporal_stride)) + 1)


def _slice_or_pad_latent_time(latent_bcfhw: torch.Tensor, target_frames: int) -> torch.Tensor:
    current_frames = int(latent_bcfhw.shape[1])
    if current_frames == target_frames:
        return latent_bcfhw.contiguous()
    if current_frames > target_frames:
        return latent_bcfhw[:, :target_frames].contiguous()
    pad = latent_bcfhw[:, -1:].expand(-1, target_frames - current_frames, -1, -1)
    return torch.cat([latent_bcfhw, pad], dim=1).contiguous()


def _flatten_latent(latent_bcfhw: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
    latent = latent_bcfhw.permute(1, 2, 3, 0).contiguous()
    latent_num_frames, latent_height, latent_width, channels = latent.shape
    latent_flat = latent.reshape(latent_num_frames * latent_height * latent_width, channels)
    return (
        latent_flat.to(torch.bfloat16).contiguous(),
        int(latent_num_frames),
        int(latent_height),
        int(latent_width),
    )


def _atomic_torch_save(payload: dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = save_path.with_name(f"{save_path.name}.tmp.{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, save_path)


def _process_task(
    task: LatentTask,
    *,
    vae: Any,
    device: torch.device,
    dtype: torch.dtype,
    text_cache: dict[str, torch.Tensor],
) -> str:
    save_path = Path(task.save_path)
    if save_path.exists() and not task.overwrite:
        return "skip"

    frames, used_frame_ids = _load_video_selected_frames(
        video_path=Path(task.video_path),
        frame_ids=task.frame_ids,
        image_height=int(task.image_height),
        image_width=int(task.image_width),
    )
    frames, used_frame_ids = _trim_to_valid_vae_frames(
        frames=frames,
        frame_ids=used_frame_ids,
        temporal_stride=int(task.temporal_stride),
    )
    latent_bcfhw = _encode_video_latent(
        frames=frames,
        vae=vae,
        device=device,
        dtype=dtype,
    )
    target_latent_frames = _target_latent_frame_count(
        video_num_frames=len(used_frame_ids),
        temporal_stride=int(task.temporal_stride),
    )
    latent_bcfhw = _slice_or_pad_latent_time(
        latent_bcfhw=latent_bcfhw,
        target_frames=target_latent_frames,
    )
    latent_flat, latent_num_frames, latent_height, latent_width = _flatten_latent(latent_bcfhw)

    if task.text_emb_path not in text_cache:
        text_cache[task.text_emb_path] = torch.load(
            task.text_emb_path,
            map_location="cpu",
            weights_only=False,
        )

    payload = {
        "latent": latent_flat,
        "latent_num_frames": int(latent_num_frames),
        "latent_height": int(latent_height),
        "latent_width": int(latent_width),
        "video_num_frames": int(len(used_frame_ids)),
        "text_emb": text_cache[task.text_emb_path].to(torch.bfloat16).cpu().contiguous(),
        "text": task.text,
        "frame_ids": list(used_frame_ids),
        "start_frame": int(task.start_frame),
        "end_frame": int(task.end_frame),
        "fps": float(task.fps),
        "ori_fps": float(task.ori_fps),
        "camera_key": task.camera_key,
        "source_video": task.video_path,
    }
    _atomic_torch_save(payload, save_path)
    return "ok"


def _consumer_loop(
    *,
    worker_id: int,
    device_id: int | None,
    task_queue: Any,
    result_queue: Any,
    wan_pretrained_root: str,
    dtype_name: str,
) -> None:
    from wan_va.modules.utils import load_vae

    if device_id is None:
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")

    dtype = _resolve_dtype(dtype_name)
    vae = load_vae(
        str(Path(wan_pretrained_root) / "vae"),
        torch_dtype=dtype,
        torch_device=device,
    )
    vae.eval()
    text_cache: dict[str, torch.Tensor] = {}

    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put({"status": "done", "worker_id": worker_id})
            return
        try:
            status = _process_task(
                task=task,
                vae=vae,
                device=device,
                dtype=dtype,
                text_cache=text_cache,
            )
            result_queue.put(
                {
                    "status": status,
                    "worker_id": worker_id,
                    "save_path": task.save_path,
                }
            )
        except Exception as exc:
            result_queue.put(
                {
                    "status": "error",
                    "worker_id": worker_id,
                    "save_path": task.save_path,
                    "error": repr(exc),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build LeRobot latent files in parallel and only save the minimal "
            "payload documented in wan_va/dataset/curation/data_structure.md."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Single LeRobot repo or a directory that contains multiple repos.",
    )
    parser.add_argument(
        "--wan-pretrained-root",
        type=Path,
        required=True,
        help="Directory that contains vae/, tokenizer/, and text_encoder/.",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        default="",
        help="Comma-separated camera keys. Empty means all discovered cameras.",
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default="",
        help="Comma-separated CUDA ids. Empty means all visible GPUs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override worker count. Default: number of selected GPUs, or 1 on CPU.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--text-mode",
        choices=["encode", "empty"],
        default="encode",
        help="encode: real text embeddings, empty: save unconditional embeddings for all samples.",
    )
    parser.add_argument(
        "--text-device",
        type=str,
        default="cuda:0",
        help="Device used by the producer to precompute cached text embeddings.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Saved fps in payload. Default 0 means auto = ori_fps / frame_stride.",
    )
    parser.add_argument(
        "--ori-fps",
        type=float,
        default=50.0,
        help="Fallback original fps if video probing fails to provide one.",
    )
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--max-repos", type=int, default=0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Per-repo episode cap for debugging.",
    )
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument(
        "--keep-text-cache",
        action="store_true",
        help="Keep latents/_text_emb_cache after the run.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.wan_pretrained_root = args.wan_pretrained_root.resolve()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")
    if not args.wan_pretrained_root.exists():
        raise FileNotFoundError(f"wan pretrained root does not exist: {args.wan_pretrained_root}")
    if int(args.frame_stride) <= 0:
        raise ValueError(f"--frame-stride must be positive, got {args.frame_stride}")
    if int(args.temporal_stride) <= 0:
        raise ValueError(f"--temporal-stride must be positive, got {args.temporal_stride}")

    dtype = _resolve_dtype(args.dtype)

    text_embedder = None
    if args.text_mode == "encode":
        text_device = _resolve_text_device(args.text_device)
        text_embedder = _TextEmbedderCache(
            pretrained_root=args.wan_pretrained_root,
            device=text_device,
            dtype=dtype,
        )

    print("[INFO] scanning dataset and preparing text embedding cache")
    build_plan = _build_tasks(args, text_embedder)
    tasks = list(build_plan.tasks)
    if not tasks:
        if not bool(args.keep_text_cache):
            removed = _cleanup_text_cache_dirs(build_plan.repo_dirs)
            if removed > 0:
                print(f"[INFO] removed temporary text embedding cache dirs: {removed}")
        print("[INFO] no latent tasks to run")
        return
    print(f"[INFO] produced latent tasks: {len(tasks)}")

    del text_embedder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gpu_ids = _available_gpu_ids(args.device_ids)
    if int(args.num_workers) > 0:
        worker_count = int(args.num_workers)
    else:
        worker_count = len(gpu_ids) if gpu_ids else 1

    if gpu_ids:
        worker_device_ids: list[int | None] = [
            gpu_ids[idx % len(gpu_ids)]
            for idx in range(worker_count)
        ]
    else:
        worker_device_ids = [None] * worker_count

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(1, int(args.queue_size)))
    result_queue = ctx.Queue()
    workers = []
    for worker_id, device_id in enumerate(worker_device_ids):
        proc = ctx.Process(
            target=_consumer_loop,
            kwargs={
                "worker_id": worker_id,
                "device_id": device_id,
                "task_queue": task_queue,
                "result_queue": result_queue,
                "wan_pretrained_root": str(args.wan_pretrained_root),
                "dtype_name": str(args.dtype),
            },
        )
        proc.start()
        workers.append(proc)
        label = "cpu" if device_id is None else f"cuda:{device_id}"
        print(f"[INFO] started worker {worker_id} on {label}")

    for task in tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put(None)

    counts = {"ok": 0, "skip": 0, "error": 0}
    finished_tasks = 0
    finished_workers = 0
    with tqdm(total=len(tasks), desc="Encoding latents", dynamic_ncols=True) as pbar:
        pbar.set_postfix(ok=0, skip=0, error=0)
        while finished_tasks < len(tasks) or finished_workers < len(workers):
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                if not any(proc.is_alive() for proc in workers) and finished_workers < len(workers):
                    raise RuntimeError("All workers exited before reporting completion")
                continue

            status = result.get("status")
            if status == "done":
                finished_workers += 1
                continue

            counts[status] = counts.get(status, 0) + 1
            finished_tasks += 1
            pbar.update(1)
            pbar.set_postfix(
                ok=counts.get("ok", 0),
                skip=counts.get("skip", 0),
                error=counts.get("error", 0),
            )

            if status == "error":
                tqdm.write(
                    f"[ERROR] worker={result.get('worker_id')} "
                    f"save={result.get('save_path')} error={result.get('error')}"
                )

    for proc in workers:
        proc.join()

    bad_exits = [
        (idx, proc.exitcode)
        for idx, proc in enumerate(workers)
        if proc.exitcode not in (0, None)
    ]
    if bad_exits:
        raise RuntimeError(f"Worker process failures: {bad_exits}")

    print(
        f"[DONE] total={len(tasks)} ok={counts.get('ok', 0)} "
        f"skip={counts.get('skip', 0)} error={counts.get('error', 0)}"
    )
    if counts.get("error", 0) > 0:
        raise RuntimeError("Some latent tasks failed; see [ERROR] lines above")

    if not bool(args.keep_text_cache):
        removed = _cleanup_text_cache_dirs(build_plan.repo_dirs)
        if removed > 0:
            print(f"[INFO] removed temporary text embedding cache dirs: {removed}")


if __name__ == "__main__":
    main()
