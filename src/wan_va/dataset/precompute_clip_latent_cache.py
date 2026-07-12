"""Precompute on-disk clip VAE latent cache for LeRobot video datasets."""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs import VA_CONFIGS
from dataset.clip_latent_cache import clip_latent_paths_for_sample, save_clip_latent
from dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from modules.utils import load_vae
from utils import init_logger, logger


@dataclass(frozen=True)
class CacheJob:
    dataset_id: int
    meta_index: int
    repo_root: str
    episode_index: int
    clip_start: int
    frame_ids: tuple[int, ...]
    freq_ratio: int
    raw_window_frames: int


def parse_dataset_paths(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise argparse.ArgumentTypeError("--dataset-paths must be a Python-style list")
    return [str(item) for item in parsed]


def parse_freq_ratios(value: str) -> list[int]:
    ratios = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not ratios:
        raise argparse.ArgumentTypeError("--freq-ratios must not be empty")
    if any(ratio <= 0 for ratio in ratios):
        raise argparse.ArgumentTypeError("--freq-ratios must be positive")
    return sorted(set(ratios))


def configure_cache_cameras(config) -> None:
    config.variant = "default"
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


def normalize_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    latents_mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(
        vae.config.latents_std,
        device=latents.device,
        dtype=latents.dtype,
    ).view(1, -1, 1, 1, 1)
    return (latents.float() - latents_mean) * (1.0 / latents_std)


@torch.inference_mode()
def encode_full_resolution(vae, frames: torch.Tensor, height: int, width: int, dtype) -> torch.Tensor:
    batch_size, num_frames = frames.shape[:2]
    frames = rearrange(frames.float(), "b f h w c -> (b f) c h w")
    frames = F.interpolate(
        frames,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    frames = rearrange(frames, "(b f) c h w -> b c f h w", b=batch_size, f=num_frames)
    frames = (frames / 255.0) * 2.0 - 1.0
    vae_device = next(vae.parameters()).device
    posterior = vae.encode(frames.to(device=vae_device, dtype=dtype)).latent_dist
    return normalize_latents(vae, posterior.mean)


@torch.inference_mode()
def encode_half_resolution(
    vae,
    frames: torch.Tensor,
    height: int,
    width: int,
    dtype,
) -> torch.Tensor:
    batch_size, num_cameras, num_frames = frames.shape[:3]
    frames = rearrange(frames.float(), "b k f h w c -> (b k f) c h w")
    frames = F.interpolate(
        frames,
        size=(height // 2, width // 2),
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
    vae_device = next(vae.parameters()).device
    posterior = vae.encode(frames.to(device=vae_device, dtype=dtype)).latent_dist
    mu = normalize_latents(vae, posterior.mean)
    return rearrange(mu, "(b k) c f h w -> b k c f h w", b=batch_size, k=num_cameras)


def get_video_frames_for_keys(dataset, episode_index: int, frame_ids: Sequence[int], camera_keys: Sequence[str]):
    frames = {}
    for camera_key in camera_keys:
        video_file = dataset._get_video_file(episode_index, camera_key)
        hdf5_file = dataset._get_hdf5_file(video_file)
        if hdf5_file.exists():
            decoded = dataset._decode_selected_frames_hdf5(
                hdf5_file,
                frame_ids,
                episode_index=episode_index,
                camera_key=camera_key,
            )
        else:
            decoded = dataset._decode_selected_frames(
                video_file,
                frame_ids,
                episode_index=episode_index,
                camera_key=camera_key,
            )
        frames[camera_key] = torch.from_numpy(np.stack(decoded, axis=0))
    return frames


class CacheJobDataset(Dataset):
    def __init__(self, multi_dataset, jobs: Sequence[CacheJob], skip_existing: bool):
        self.multi_dataset = multi_dataset
        self.jobs = list(jobs)
        self.skip_existing = bool(skip_existing)

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, index: int) -> dict:
        job = self.jobs[int(index)]
        dataset = self.multi_dataset._datasets[job.dataset_id]
        missing_keys = (
            missing_camera_keys(dataset, job)
            if self.skip_existing
            else list(dataset.used_video_keys)
        )
        if self.skip_existing and not missing_keys:
            return {
                "job": job,
                "missing_keys": [],
                "frames": {},
            }

        frames = get_video_frames_for_keys(
            dataset,
            episode_index=job.episode_index,
            frame_ids=job.frame_ids,
            camera_keys=missing_keys,
        )
        return {
            "job": job,
            "missing_keys": missing_keys,
            "frames": frames,
        }


def cache_collate_fn(samples: list[dict]) -> list[dict]:
    return samples


def cache_paths_for_job(dataset, job: CacheJob) -> dict[str, Path]:
    return clip_latent_paths_for_sample(
        Path(job.repo_root),
        episode_index=job.episode_index,
        clip_start=job.clip_start,
        camera_keys=dataset.used_video_keys,
        freq_ratio=job.freq_ratio,
    )


def missing_camera_keys(dataset, job: CacheJob) -> list[str]:
    paths = cache_paths_for_job(dataset, job)
    return [camera_key for camera_key, path in paths.items() if not path.is_file()]


def save_batch_latents(dataset, jobs: Sequence[CacheJob], per_camera: dict[str, torch.Tensor]) -> int:
    saved = 0
    for batch_idx, job in enumerate(jobs):
        paths = cache_paths_for_job(dataset, job)
        video_frame_stride = job.frame_ids[1] - job.frame_ids[0] if len(job.frame_ids) > 1 else 0
        for camera_key, latents in per_camera.items():
            path = paths[camera_key]
            if path.is_file():
                continue
            save_clip_latent(
                path,
                latent=latents[batch_idx],
                clip_start=job.clip_start,
                frame_ids=job.frame_ids,
                episode_index=job.episode_index,
                camera_key=camera_key,
                video_frame_stride=video_frame_stride,
                raw_window_frames=job.raw_window_frames,
                freq_ratio=job.freq_ratio,
                source_video=str(dataset._get_video_file(job.episode_index, camera_key)),
            )
            saved += 1
    return saved


def existing_cache_mask(dataset, job: CacheJob) -> dict[str, bool]:
    paths = cache_paths_for_job(dataset, job)
    return {camera_key: path.is_file() for camera_key, path in paths.items()}


def encode_and_save_batch(vae, vae_half, dataset, jobs: Sequence[CacheJob], config, dtype) -> int:
    if not jobs:
        return 0
    camera_keys = list(dataset.used_video_keys)
    high_key = camera_keys[0]
    half_keys = camera_keys[1:]

    missing_by_job = [set(missing_camera_keys(dataset, job)) for job in jobs]
    needed_keys = sorted(set().union(*missing_by_job))
    if not needed_keys:
        return 0

    frames_by_key = {
        camera_key: []
        for camera_key in needed_keys
    }
    for job, missing in zip(jobs, missing_by_job, strict=True):
        frames = get_video_frames_for_keys(
            dataset,
            episode_index=job.episode_index,
            frame_ids=job.frame_ids,
            camera_keys=[key for key in needed_keys if key in missing],
        )
        for camera_key in needed_keys:
            if camera_key in missing:
                frames_by_key[camera_key].append(frames[camera_key])
            else:
                frames_by_key[camera_key].append(None)

    per_camera: dict[str, torch.Tensor] = {}
    if high_key in needed_keys:
        high_jobs = [idx for idx, frame in enumerate(frames_by_key[high_key]) if frame is not None]
        high_frames = torch.stack([frames_by_key[high_key][idx] for idx in high_jobs], dim=0)
        high_latents = encode_full_resolution(
            vae,
            high_frames,
            height=config.height,
            width=config.width,
            dtype=dtype,
        ).detach().cpu()
        output = torch.empty(
            (len(jobs), *high_latents.shape[1:]),
            dtype=high_latents.dtype,
        )
        for out_idx, latent in zip(high_jobs, high_latents, strict=True):
            output[out_idx] = latent
        per_camera[high_key] = output

    half_needed = [key for key in half_keys if key in needed_keys]
    if half_needed:
        # Cameras with the same target resolution share one VAE forward.
        flat_frames = []
        flat_mapping = []
        for job_idx in range(len(jobs)):
            for camera_key in half_needed:
                frame = frames_by_key[camera_key][job_idx]
                if frame is None:
                    continue
                flat_mapping.append((job_idx, camera_key))
                flat_frames.append(frame)
        half_frames = torch.stack(flat_frames, dim=0)[:, None]
        half_latents = encode_half_resolution(
            vae_half,
            half_frames,
            height=config.height,
            width=config.width,
            dtype=dtype,
        )[:, 0].detach().cpu()
        by_key = {
            camera_key: torch.empty(
                (len(jobs), *half_latents.shape[1:]),
                dtype=half_latents.dtype,
            )
            for camera_key in half_needed
        }
        for (job_idx, camera_key), latent in zip(flat_mapping, half_latents, strict=True):
            by_key[camera_key][job_idx] = latent
        per_camera.update(by_key)

    return save_batch_latents(dataset, jobs, per_camera)


def encode_and_save_samples(vae, vae_half, dataset, samples: Sequence[dict], config, dtype) -> int:
    samples = [sample for sample in samples if sample["missing_keys"]]
    if not samples:
        return 0

    jobs = [sample["job"] for sample in samples]
    camera_keys = list(dataset.used_video_keys)
    high_key = camera_keys[0]
    half_keys = camera_keys[1:]
    needed_keys = sorted(set().union(*(set(sample["missing_keys"]) for sample in samples)))

    per_camera: dict[str, torch.Tensor] = {}
    if high_key in needed_keys:
        high_jobs = [
            idx for idx, sample in enumerate(samples)
            if high_key in sample["frames"]
        ]
        if high_jobs:
            high_frames = torch.stack(
                [samples[idx]["frames"][high_key] for idx in high_jobs],
                dim=0,
            )
            high_latents = encode_full_resolution(
                vae,
                high_frames,
                height=config.height,
                width=config.width,
                dtype=dtype,
            ).detach().cpu()
            output = torch.empty(
                (len(samples), *high_latents.shape[1:]),
                dtype=high_latents.dtype,
            )
            for out_idx, latent in zip(high_jobs, high_latents, strict=True):
                output[out_idx] = latent
            per_camera[high_key] = output

    half_needed = [key for key in half_keys if key in needed_keys]
    if half_needed:
        flat_frames = []
        flat_mapping = []
        for sample_idx, sample in enumerate(samples):
            for camera_key in half_needed:
                frame = sample["frames"].get(camera_key)
                if frame is None:
                    continue
                flat_mapping.append((sample_idx, camera_key))
                flat_frames.append(frame)
        if flat_frames:
            half_frames = torch.stack(flat_frames, dim=0)[:, None]
            half_latents = encode_half_resolution(
                vae_half,
                half_frames,
                height=config.height,
                width=config.width,
                dtype=dtype,
            )[:, 0].detach().cpu()
            by_key = {
                camera_key: torch.empty(
                    (len(samples), *half_latents.shape[1:]),
                    dtype=half_latents.dtype,
                )
                for camera_key in half_needed
            }
            for (sample_idx, camera_key), latent in zip(flat_mapping, half_latents, strict=True):
                by_key[camera_key][sample_idx] = latent
            per_camera.update(by_key)

    return save_batch_latents(dataset, jobs, per_camera)


def regular_jobs_for_meta(dataset, dataset_id: int, meta_index: int, meta: dict, freq_ratios: Sequence[int]) -> Iterable[CacheJob]:
    episode_index = int(meta["episode_index"])
    start_frame = int(meta["start_frame"])
    end_frame = int(meta.get("latent_end_frame", meta["end_frame"]))
    available = dataset._get_available_video_frames(episode_index)
    if available is not None:
        end_frame = min(end_frame, int(available))
    for freq_ratio in freq_ratios:
        video_frame_stride = dataset.BASE_VIDEO_FRAME_STRIDE * int(freq_ratio)
        action_frame_stride = int(freq_ratio)
        raw_window_frames = max(
            (dataset.RETURN_VIDEO_FRAMES - 1) * video_frame_stride + 1,
            (dataset.RETURN_ACTIONS - 1) * action_frame_stride + 1,
        )
        max_clip_start = end_frame - raw_window_frames
        if max_clip_start < start_frame:
            continue
        for clip_start in range(start_frame, max_clip_start + 1):
            frame_ids = tuple(
                range(
                    int(clip_start),
                    int(clip_start) + raw_window_frames,
                    video_frame_stride,
                )
            )[: dataset.RETURN_VIDEO_FRAMES]
            yield CacheJob(
                dataset_id=dataset_id,
                meta_index=meta_index,
                repo_root=str(Path(dataset.repo_id).resolve()),
                episode_index=episode_index,
                clip_start=int(clip_start),
                frame_ids=frame_ids,
                freq_ratio=int(freq_ratio),
                raw_window_frames=int(raw_window_frames),
            )


def iter_jobs(multi_dataset, freq_ratios: Sequence[int]) -> Iterable[CacheJob]:
    for dataset_id, dataset in enumerate(multi_dataset._datasets):
        for meta_index, meta in enumerate(dataset.new_metas):
            yield from regular_jobs_for_meta(
                dataset,
                dataset_id=dataset_id,
                meta_index=meta_index,
                meta=meta,
                freq_ratios=freq_ratios,
            )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", type=str, default="astribot_centrifuge_multidrop")
    parser.add_argument("--dataset-paths", type=parse_dataset_paths, required=True)
    parser.add_argument("--pretrained-model-path", type=str, required=True)
    parser.add_argument("--freq-ratios", type=parse_freq_ratios, default=[1, 2])
    parser.add_argument("--cache-batch-size", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-init-worker", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Shuffle seed for cache jobs. Defaults to current time on rank 0 and is broadcast.",
    )
    return parser


def shared_time_seed(accelerator: Accelerator, override: int | None) -> int:
    seed = int(override) if override is not None else int(time.time())
    if accelerator.num_processes <= 1 or not torch.distributed.is_initialized():
        return seed
    payload = [seed if accelerator.is_main_process else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return int(payload[0])


def main() -> None:
    init_logger()
    args = build_argparser().parse_args()
    if args.cache_batch_size <= 0:
        raise ValueError("--cache-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")

    accelerator = Accelerator()
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16

    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.dataset_paths = args.dataset_paths
    if args.dataset_paths:
        config.empty_emb_path = str(Path(args.dataset_paths[0]) / "empty_emb.pt")
    config.wan22_pretrained_model_name_or_path = args.pretrained_model_path
    config.freq_ratio = min(args.freq_ratios)
    config.cfg_prob = 0.0
    config.require_latents_for_sampling = False
    configure_cache_cameras(config)

    if accelerator.is_main_process:
        logger.info(
            "Precomputing clip latent cache: "
            f"freq_ratios={args.freq_ratios}, "
            f"camera_keys={config.obs_cam_keys}, "
            f"dataset_paths={args.dataset_paths}, world_size={accelerator.num_processes}"
        )

    multi_dataset = MultiLatentLeRobotDataset(
        config=config,
        num_init_worker=args.num_init_worker,
    )

    vae_path = Path(args.pretrained_model_path) / "vae"
    vae = load_vae(str(vae_path), torch_dtype=dtype, torch_device=accelerator.device)
    vae.requires_grad_(False)
    vae.eval()
    vae.zero_grad(set_to_none=True)
    vae_half = load_vae(str(vae_path), torch_dtype=dtype, torch_device=accelerator.device)
    vae_half.requires_grad_(False)
    vae_half.eval()
    vae_half.zero_grad(set_to_none=True)

    shuffle_seed = shared_time_seed(accelerator, args.shuffle_seed)
    jobs = list(iter_jobs(multi_dataset, args.freq_ratios))
    random.Random(shuffle_seed).shuffle(jobs)
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    rank_jobs = [
        job
        for global_job_idx, job in enumerate(jobs)
        if global_job_idx % accelerator.num_processes == accelerator.process_index
    ]
    if accelerator.is_main_process:
        logger.info(f"Prepared {len(jobs)} cache jobs with shuffle_seed={shuffle_seed}")
    logger.info(
        f"Rank {accelerator.process_index} assigned {len(rank_jobs)} cache jobs"
    )

    processed = 0
    saved = 0
    skipped = 0
    errors = 0
    cache_dataset = CacheJobDataset(
        multi_dataset=multi_dataset,
        jobs=rank_jobs,
        skip_existing=args.skip_existing,
    )
    dataloader_kwargs = dict(
        batch_size=args.cache_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cache_collate_fn,
        pin_memory=False,
    )
    if args.num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2
        dataloader_kwargs["persistent_workers"] = True
    cache_loader = DataLoader(cache_dataset, **dataloader_kwargs)
    progress = tqdm(
        disable=not accelerator.is_local_main_process,
        desc=f"rank {accelerator.process_index} cache",
        total=len(cache_dataset),
        unit="job",
    )

    for samples in cache_loader:
        processed += len(samples)
        skipped += sum(1 for sample in samples if not sample["missing_keys"])
        try:
            samples_by_dataset: dict[int, list[dict]] = {}
            for sample in samples:
                samples_by_dataset.setdefault(sample["job"].dataset_id, []).append(sample)
            for dataset_id, dataset_samples in samples_by_dataset.items():
                dataset = multi_dataset._datasets[dataset_id]
                saved += encode_and_save_samples(
                    vae,
                    vae_half,
                    dataset,
                    dataset_samples,
                    config,
                    dtype,
                )
        except Exception as exc:
            errors += len(samples)
            logger.exception(f"Failed cache batch on rank={accelerator.process_index}: {exc}")
        finally:
            progress.update(len(samples))

    progress.close()
    accelerator.wait_for_everyone()
    logger.info(
        "Clip latent cache precompute done: "
        f"rank={accelerator.process_index}, processed={processed}, "
        f"skipped={skipped}, saved_files={saved}, errors={errors}"
    )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
