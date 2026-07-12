#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


DA3_SRC_ROOT = Path("/media/damoxing/fileset/md4d/third_parties/lingbot-va/wan_wa/Depth-Anything-3/src")
if str(DA3_SRC_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(DA3_SRC_ROOT))

from depth_anything_3.api import DepthAnything3
from depth_anything_3.model.utils.transform import extri_intri_to_pose_encoding


DEPTH_MAX_MM = np.iinfo(np.uint16).max
DEPTH_PNG_COMPRESSION = 9


@dataclass(frozen=True)
class RepoItem:
    family: str
    repo_name: str
    repo_dir: Path


@dataclass(frozen=True)
class CameraItem:
    camera_key: str
    camera_dir: Path


@dataclass(frozen=True)
class TaskItem:
    family: str
    repo_name: str
    camera_key: str
    video_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        type=str,
        default="upper_bound_resize",
        choices=("upper_bound_resize", "lower_bound_resize"),
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="middle",
        choices=("first", "middle", "saddle_balanced", "saddle_sim_range"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-repos", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--total-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0)
    return parser.parse_args()


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_device(device_arg: str) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def resolve_num_gpus(requested: int, device_arg: str) -> int:
    if not torch.cuda.is_available() or not device_arg.startswith("cuda"):
        return 0
    count = torch.cuda.device_count()
    if requested > 0:
        return min(int(requested), count)
    return count


def validate_machine_sharding(total_machines: int, machine_rank: int) -> tuple[int, int]:
    total_machines = int(total_machines)
    machine_rank = int(machine_rank)
    if total_machines <= 0:
        raise ValueError(f"total_machines must be positive, got {total_machines}")
    if machine_rank < 0 or machine_rank >= total_machines:
        raise ValueError(
            f"machine_rank must be in [0, {total_machines}), got {machine_rank}"
        )
    return total_machines, machine_rank


def resolve_videos_root(repo_dir: Path) -> Path | None:
    direct = repo_dir / "videos"
    if direct.exists():
        return direct
    candidates = sorted(path for path in repo_dir.iterdir() if path.is_dir() and path.name.startswith("videos"))
    return candidates[0] if candidates else None


def iter_repos(dataset_root: Path, families: set[str]) -> Iterator[RepoItem]:
    family_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    for family_dir in family_dirs:
        family = family_dir.name
        if families and family not in families:
            continue
        repo_dirs = sorted(path for path in family_dir.iterdir() if path.is_dir())
        for repo_dir in repo_dirs:
            yield RepoItem(family=family, repo_name=repo_dir.name, repo_dir=repo_dir)


def iter_cameras(repo_dir: Path) -> list[CameraItem]:
    videos_root = resolve_videos_root(repo_dir)
    if videos_root is None:
        return []
    cameras: dict[str, Path] = {}
    for chunk_dir in sorted(videos_root.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        for camera_dir in sorted(path for path in chunk_dir.iterdir() if path.is_dir()):
            if next(camera_dir.glob("episode_*.mp4"), None) is not None:
                cameras.setdefault(camera_dir.name, camera_dir)
    return [CameraItem(camera_key=key, camera_dir=path) for key, path in sorted(cameras.items())]


def build_tasks(dataset_root: Path, families: set[str], max_repos: int, max_episodes: int) -> list[TaskItem]:
    tasks: list[TaskItem] = []
    repo_count = 0
    for repo in iter_repos(dataset_root, families):
        cameras = iter_cameras(repo.repo_dir)
        if not cameras:
            print(f"[WARN] skip repo without videos: {repo.repo_dir}")
            continue
        repo_count += 1
        print(f"[REPO] family={repo.family} repo={repo.repo_name}")
        for camera in cameras:
            print(f"[CAM] {camera.camera_key}")
            video_paths = sorted(camera.camera_dir.glob("episode_*.mp4"))
            if max_episodes > 0:
                video_paths = video_paths[:max_episodes]
            for video_path in video_paths:
                tasks.append(
                    TaskItem(
                        family=repo.family,
                        repo_name=repo.repo_name,
                        camera_key=camera.camera_key,
                        video_path=str(video_path),
                    )
                )
        if max_repos > 0 and repo_count >= max_repos:
            break
    return tasks


def shard_tasks(tasks: list[TaskItem], total_machines: int, machine_rank: int) -> list[TaskItem]:
    total_machines, machine_rank = validate_machine_sharding(total_machines, machine_rank)
    if total_machines == 1:
        print(f"[SHARD] total_machines=1 machine_rank=0 selected={len(tasks)}/{len(tasks)}")
        return tasks

    tasks_by_family: dict[str, list[TaskItem]] = {}
    for task in tasks:
        tasks_by_family.setdefault(task.family, []).append(task)

    sharded: list[TaskItem] = []
    for family, family_tasks in tasks_by_family.items():
        family_sharded = family_tasks[machine_rank::total_machines]
        print(
            "[SHARD] "
            f"family={family} total_machines={total_machines} machine_rank={machine_rank} "
            f"selected={len(family_sharded)}/{len(family_tasks)}"
        )
        sharded.extend(family_sharded)

    print(
        "[SHARD] "
        f"total_machines={total_machines} machine_rank={machine_rank} "
        f"selected_total={len(sharded)}/{len(tasks)}"
    )
    return sharded


def probe_video_hw(video_path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if height <= 0 or width <= 0:
        raise RuntimeError(f"Invalid video size: {video_path}")
    return height, width


def decode_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {video_path}")
    return frames


def as_3x4_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics[:, :3, :]
    if extrinsics.shape[-2:] == (3, 4):
        return extrinsics
    raise ValueError(f"Unsupported extrinsics shape: {extrinsics.shape}")


def resize_depth_map(depth: np.ndarray, target_hw: tuple[int, int]) -> torch.Tensor:
    depth_t = torch.from_numpy(np.asarray(depth, dtype=np.float32)).unsqueeze(1)
    depth_t = F.interpolate(depth_t, size=target_hw, mode="bilinear", align_corners=False)
    return depth_t[:, 0].contiguous()


def scale_intrinsics_to_target(
    intrinsics: np.ndarray,
    src_hw: tuple[int, int],
    dst_hw: tuple[int, int],
) -> torch.Tensor:
    intr_t = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).clone()
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scale_x = float(dst_w) / float(src_w)
    scale_y = float(dst_h) / float(src_h)
    intr_t[:, 0, 0] *= scale_x
    intr_t[:, 1, 1] *= scale_y
    intr_t[:, 0, 2] *= scale_x
    intr_t[:, 1, 2] *= scale_y
    intr_t[:, 2, 2] = 1.0
    return intr_t.contiguous()


def encode_pose(extrinsics_3x4: torch.Tensor, intrinsics_3x3: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    pose = extri_intri_to_pose_encoding(
        extrinsics_3x4.unsqueeze(0),
        intrinsics_3x3.unsqueeze(0),
        image_size_hw=image_hw,
    )
    return pose[0].contiguous()


def load_model(model_path: str, device: torch.device):
    model = DepthAnything3.from_pretrained(model_path)
    model = model.to(device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "cuda out of memory" in str(exc).lower()


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_da3_batch(
    model,
    frames_rgb: list[np.ndarray],
    target_hw: tuple[int, int],
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
) -> dict:
    prediction = model.inference(
        image=frames_rgb,
        process_res=process_res,
        process_res_method=process_res_method,
        export_dir=None,
        export_format="mini_npz",
        use_ray_pose=False,
        ref_view_strategy=ref_view_strategy,
    )
    if prediction.extrinsics is None or prediction.intrinsics is None:
        raise RuntimeError("DA3 prediction missing camera parameters")

    src_hw = tuple(int(v) for v in prediction.depth.shape[-2:])
    extrinsics = torch.from_numpy(as_3x4_extrinsics(prediction.extrinsics)).to(torch.float32).cpu()
    intrinsics = scale_intrinsics_to_target(prediction.intrinsics, src_hw, target_hw).to(torch.float32).cpu()
    pose_enc = encode_pose(extrinsics, intrinsics, target_hw).to(torch.float16).cpu()
    depth = resize_depth_map(prediction.depth, target_hw).unsqueeze(-1).to(torch.float16).cpu()
    return {
        "camera_extrinsics": extrinsics.contiguous(),
        "camera_intrinsics": intrinsics.contiguous(),
        "pose_enc": pose_enc.contiguous(),
        "rasterized_depth": depth.contiguous(),
    }


def run_da3_resilient(
    model,
    frames_rgb: list[np.ndarray],
    target_hw: tuple[int, int],
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
) -> dict:
    try:
        return run_da3_batch(
            model=model,
            frames_rgb=frames_rgb,
            target_hw=target_hw,
            process_res=process_res,
            process_res_method=process_res_method,
            ref_view_strategy=ref_view_strategy,
        )
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise

    total = len(frames_rgb)
    if total <= 1:
        raise

    empty_cuda_cache()
    chunk_size = max(1, total // 2)
    while chunk_size >= 1:
        parts: list[dict] = []
        try:
            for start in range(0, total, chunk_size):
                end = min(total, start + chunk_size)
                parts.append(
                    run_da3_batch(
                        model=model,
                        frames_rgb=frames_rgb[start:end],
                        target_hw=target_hw,
                        process_res=process_res,
                        process_res_method=process_res_method,
                        ref_view_strategy=ref_view_strategy,
                    )
                )
                empty_cuda_cache()
            return {
                "camera_extrinsics": torch.cat([part["camera_extrinsics"] for part in parts], dim=0).contiguous(),
                "camera_intrinsics": torch.cat([part["camera_intrinsics"] for part in parts], dim=0).contiguous(),
                "pose_enc": torch.cat([part["pose_enc"] for part in parts], dim=0).contiguous(),
                "rasterized_depth": torch.cat([part["rasterized_depth"] for part in parts], dim=0).contiguous(),
            }
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            empty_cuda_cache()
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)

    raise RuntimeError("DA3 inference failed after chunk fallback")


@torch.no_grad()
def estimate_video_depth(
    model,
    video_path: Path,
    target_hw: tuple[int, int],
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
) -> dict:
    frames_rgb = decode_all_frames(video_path)
    dense = run_da3_resilient(
        model=model,
        frames_rgb=frames_rgb,
        target_hw=target_hw,
        process_res=process_res,
        process_res_method=process_res_method,
        ref_view_strategy=ref_view_strategy,
    )

    return {
        "camera_extrinsics": dense["camera_extrinsics"],
        "camera_intrinsics": dense["camera_intrinsics"],
        "pose_enc": dense["pose_enc"],
        "rasterized_depth": dense["rasterized_depth"],
        "image_height": int(target_hw[0]),
        "image_width": int(target_hw[1]),
        "num_frames": int(dense["rasterized_depth"].shape[0]),
        "source_video": str(video_path),
    }


def quantize_depth_to_uint16_mm(depth: torch.Tensor) -> np.ndarray:
    depth_np = depth.detach().cpu().numpy()
    if depth_np.ndim == 4 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    depth_mm = np.rint(np.nan_to_num(depth_np, nan=0.0, posinf=float(DEPTH_MAX_MM), neginf=0.0) * 1000.0)
    return np.clip(depth_mm, 0, DEPTH_MAX_MM).astype(np.uint16, copy=False)


def encode_depth_frame_png(depth_mm_u16: np.ndarray) -> np.ndarray:
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(DEPTH_PNG_COMPRESSION)]
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(depth_mm_u16), encode_params)
    if not ok:
        raise RuntimeError("cv2.imencode('.png') failed for uint16 depth frame")
    return encoded.reshape(-1).astype(np.uint8, copy=False)


def write_hdf5_episode(save_path: Path, payload: dict, camera_key: str) -> None:
    tmp_path = save_path.with_name(f"{save_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    depth_mm = quantize_depth_to_uint16_mm(payload["rasterized_depth"])
    num_frames = int(depth_mm.shape[0])
    encoded_lengths: list[int] = []

    try:
        with h5py.File(tmp_path, "w") as h5_file:
            h5_file.attrs["format"] = "da3_depth_hdf5_v1"
            h5_file.attrs["camera_key"] = camera_key
            h5_file.attrs["source_video"] = str(payload["source_video"])
            h5_file.attrs["image_height"] = int(payload["image_height"])
            h5_file.attrs["image_width"] = int(payload["image_width"])
            h5_file.attrs["num_frames"] = int(payload["num_frames"])
            h5_file.attrs["depth_unit"] = "mm"
            h5_file.attrs["depth_dtype"] = "uint16"
            h5_file.attrs["depth_min_mm"] = 0
            h5_file.attrs["depth_max_mm"] = int(DEPTH_MAX_MM)
            h5_file.attrs["depth_codec"] = "png"
            h5_file.attrs["depth_png_compression"] = int(DEPTH_PNG_COMPRESSION)

            h5_file.create_dataset(
                "camera_extrinsics",
                data=payload["camera_extrinsics"].detach().cpu().numpy(),
                compression="gzip",
                shuffle=True,
            )
            h5_file.create_dataset(
                "camera_intrinsics",
                data=payload["camera_intrinsics"].detach().cpu().numpy(),
                compression="gzip",
                shuffle=True,
            )
            h5_file.create_dataset(
                "pose_enc",
                data=payload["pose_enc"].detach().cpu().numpy(),
                compression="gzip",
                shuffle=True,
            )

            encoded_depth = h5_file.create_dataset(
                "depth_mm_png",
                shape=(num_frames,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
            )
            for frame_idx, frame_depth_mm in enumerate(depth_mm):
                encoded = encode_depth_frame_png(frame_depth_mm)
                encoded_depth[frame_idx] = encoded
                encoded_lengths.append(int(encoded.size))

            if encoded_lengths:
                h5_file.attrs["depth_encoded_bytes_min"] = int(min(encoded_lengths))
                h5_file.attrs["depth_encoded_bytes_max"] = int(max(encoded_lengths))
                h5_file.attrs["depth_encoded_bytes_mean"] = float(sum(encoded_lengths) / len(encoded_lengths))
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    os.replace(tmp_path, save_path)


def save_camera_episode(
    output_root: Path,
    task: TaskItem,
    camera_key: str,
    video_path: Path,
    payload: dict,
    overwrite: bool,
) -> None:
    save_dir = output_root / task.family / task.repo_name / camera_key
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{video_path.stem}.hdf5"
    if save_path.exists() and not overwrite:
        print(f"[SKIP] {save_path}")
        return
    write_hdf5_episode(save_path, payload, camera_key)
    print(f"[OK] {save_path}")


def worker_loop(
    worker_idx: int,
    task_queue: mp.Queue,
    args_dict: dict,
) -> None:
    device = resolve_device(f"cuda:{worker_idx}" if args_dict["num_gpus"] > 0 else args_dict["device"])
    model = load_model(args_dict["model_path"], device)
    output_root = Path(args_dict["output_root"])

    while True:
        task = task_queue.get()
        if task is None:
            break
        task = TaskItem(**task)
        video_path = Path(task.video_path)
        target_hw = probe_video_hw(video_path)
        payload = estimate_video_depth(
            model=model,
            video_path=video_path,
            target_hw=target_hw,
            process_res=args_dict["process_res"],
            process_res_method=args_dict["process_res_method"],
            ref_view_strategy=args_dict["ref_view_strategy"],
        )
        payload["camera_key"] = task.camera_key
        save_camera_episode(
            output_root=output_root,
            task=task,
            camera_key=task.camera_key,
            video_path=video_path,
            payload=payload,
            overwrite=bool(args_dict["overwrite"]),
        )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    total_machines, machine_rank = validate_machine_sharding(args.total_machines, args.machine_rank)

    families = parse_csv(args.families)
    tasks = build_tasks(
        dataset_root=dataset_root,
        families=families,
        max_repos=args.max_repos,
        max_episodes=args.max_episodes,
    )
    tasks = shard_tasks(tasks, total_machines=total_machines, machine_rank=machine_rank)
    if not tasks:
        print("[WARN] no tasks found")
        return

    num_gpus = resolve_num_gpus(args.num_gpus, args.device)
    if num_gpus <= 0:
        num_gpus = 1

    task_queue: mp.Queue = mp.Queue(maxsize=num_gpus * 2)
    args_dict = {
        "device": args.device,
        "model_path": args.model_path,
        "output_root": str(output_root),
        "process_res": args.process_res,
        "process_res_method": args.process_res_method,
        "ref_view_strategy": args.ref_view_strategy,
        "overwrite": bool(args.overwrite),
        "num_gpus": num_gpus if torch.cuda.is_available() and args.device.startswith("cuda") else 0,
    }

    workers = [
        mp.Process(target=worker_loop, args=(worker_idx, task_queue, args_dict), daemon=False)
        for worker_idx in range(num_gpus)
    ]
    for worker in workers:
        worker.start()

    for task in tasks:
        task_queue.put(task.__dict__)
    for _ in workers:
        task_queue.put(None)
    for worker in workers:
        worker.join()
        if worker.exitcode != 0:
            raise RuntimeError(f"worker failed: pid={worker.pid} exitcode={worker.exitcode}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
