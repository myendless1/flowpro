# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import numpy as np
from pathlib import Path
from collections import OrderedDict
from collections.abc import Callable
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import cv2
import h5py
import imageio
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from lerobot.constants import HF_LEROBOT_HOME

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import logger
from dataset.curation.base import resolve_lerobot_videos_root
from dataset.clip_latent_cache import clip_latent_paths_for_sample
from action_representation import (
    EXECUTION_CHANNEL_IDS,
    encode_absolute_history,
    encode_action_targets,
    validate_action_representation,
)

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def collect_lerobot_repo_roots(dataset_paths=None, filename='info.json'):
    if not isinstance(dataset_paths, (list, tuple)) or len(dataset_paths) == 0:
        raise ValueError("dataset_paths must be a non-empty list or tuple of paths")

    repo_roots = []
    seen = set()

    for dataset_root in dataset_paths:
        if not dataset_root:
            raise ValueError("dataset_paths contains an empty path")
        dataset_root = os.path.abspath(os.path.expanduser(os.fspath(dataset_root)))
        root_path = Path(dataset_root)
        if not root_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_root}")

        direct_meta = root_path / 'meta' / filename
        if direct_meta.is_file():
            repo_root = str(root_path)
            if repo_root not in seen:
                seen.add(repo_root)
                repo_roots.append(repo_root)
            continue

        repo_infos = recursive_find_file(dataset_root, filename)
        for info_path in repo_infos:
            repo_root = info_path.split(f'/meta/{filename}')[0]
            if repo_root not in seen:
                seen.add(repo_root)
                repo_roots.append(repo_root)

    if not repo_roots:
        raise FileNotFoundError(
            f"No LeRobot repositories found from dataset roots: {list(dataset_paths)}"
        )

    return repo_roots

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = collect_lerobot_repo_roots(
        dataset_paths=getattr(config, 'dataset_paths', None),
        filename='info.json',
    )
    max_workers = min(num_init_worker, len(repo_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        datasets_out_lst = list(
            tqdm(
                executor.map(construct_func, repo_list),
                total=len(repo_list),
                desc="Initializing datasets",
                unit="repo",
            )
        )

    return datasets_out_lst

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    BASE_VIDEO_FRAME_STRIDE = 4
    RETURN_VIDEO_FRAMES = 9
    RETURN_ACTIONS = 32
    ACTION_GROUPS = 2
    ACTIONS_PER_GROUP = 16 # frame_stride * vae_compression_ratio
    DEFAULT_FREQ_RATIO = 2
    MAX_SAMPLE_RETRIES = 20

    def __init__(
        self,
        repo_id,
        config=None,
    ):
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'imageio_ffmpeg'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.text_emb_path = Path(repo_id) / 'text_emb'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.freq_ratio = int(getattr(config, "freq_ratio", self.DEFAULT_FREQ_RATIO))
        if self.freq_ratio <= 0:
            raise ValueError(f"freq_ratio must be positive, got {self.freq_ratio}")
        self.video_frame_stride = self.BASE_VIDEO_FRAME_STRIDE * self.freq_ratio
        self.action_frame_stride = self.freq_ratio
        self.raw_window_frames = max(
            (self.RETURN_VIDEO_FRAMES - 1) * self.video_frame_stride + 1,
            (self.RETURN_ACTIONS - 1) * self.action_frame_stride + 1,
        )
        self.used_video_keys = config.obs_cam_keys
        self.require_latents_for_sampling = bool(
            getattr(config, "require_latents_for_sampling", True)
        )
        # self.enable_action_loss = 'aug_500' not in Path(os.fspath(repo_id)).parent.name.lower()
        self.enable_action_loss = True
        logger.info(f"Use action loss: {self.enable_action_loss}: Dataset {repo_id}")
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        state_norm_stat = getattr(config, "state_norm_stat", config.norm_stat)
        self.state_q01 = np.array(state_norm_stat['q01'], dtype='float')[None]
        self.state_q99 = np.array(state_norm_stat['q99'], dtype='float')[None]
        self.action_representation = validate_action_representation(
            getattr(config, "action_representation", "absolute")
        )
        self.release_pose_aux = bool(getattr(config, "release_pose_aux", False))
        self.release_open_threshold = float(
            getattr(config, "release_open_threshold", 0.5)
        )
        training_channel_ids = list(
            getattr(config, "training_action_channel_ids", config.used_action_channel_ids)
        )
        self.channel_mask = np.zeros(config.action_dim, dtype=np.float32)
        self.channel_mask[training_channel_ids] = 1.0
        self.state_channel_mask = np.zeros(config.action_dim, dtype=np.float32)
        self.state_channel_mask[list(EXECUTION_CHANNEL_IDS)] = 1.0
        self._episode_action_cache = OrderedDict()
        self.state_history_source = getattr(config, "state_history_source", "action")
        self.state_history_len = int(getattr(config, "state_history_len", 1))
        if self.state_history_len <= 0:
            raise ValueError(
                f"state_history_len must be positive, got {self.state_history_len}"
            )
        self.variant = getattr(config, "variant", "default")
        if self.variant not in {"default", "add_crop_views", "keyframe_sample"}:
            raise ValueError(f"Unsupported dataset variant: {self.variant}")
        self.keyframe_non_key_stride = int(getattr(config, "keyframe_non_key_stride", 2))
        if self.keyframe_non_key_stride <= 0:
            raise ValueError(
                f"keyframe_non_key_stride must be positive, got {self.keyframe_non_key_stride}"
            )
        self.video_path = resolve_lerobot_videos_root(Path(repo_id), must_exist=True)
        self._latent_file_cache = {}
        self._text_emb_file_cache = {}
        self._condition_file_cache = {}
        self._keyframe_manifest_index = None
        self._keyframe_manifest_cache = {}
        self.state_column_name = "observation.state" if "observation.state" in self.hf_dataset.column_names else None
        hf_columns = ['action']
        if self.state_column_name is not None:
            hf_columns.append(self.state_column_name)
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=hf_columns,
                output_all_columns=False
            )
        logger.info(
            "Dataset sampling: "
            f"freq_ratio={self.freq_ratio}, "
            f"variant={self.variant}, "
            f"video_frame_stride={self.video_frame_stride}, "
            f"action_frame_stride={self.action_frame_stride}, "
            f"raw_window_frames={self.raw_window_frames}"
        )
        self.parse_meta()

    def parse_meta(self):
        out = []
        resolved_latent_mismatches = 0
        episode_rows = sorted(
            self.meta.episodes.values(),
            key=lambda value: int(value["episode_index"]),
        )
        episode_skip_first = int(getattr(self.config, "episode_skip_first", 0) or 0)
        if episode_skip_first < 0:
            raise ValueError(f"episode_skip_first must be non-negative, got {episode_skip_first}")
        if episode_skip_first:
            logger.info(
                f"Skipping first {episode_skip_first} episode entries for dataset {self.repo_id}"
            )
            episode_rows = episode_rows[episode_skip_first:]

        for value in episode_rows:
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                resolved_end_frame = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if resolved_end_frame is not None:
                    if resolved_end_frame != cur_meta["end_frame"]:
                        resolved_latent_mismatches += 1
                    cur_meta["latent_end_frame"] = resolved_end_frame
                    out.append(cur_meta)
        self.new_metas = out
        if resolved_latent_mismatches > 0:
            logger.warning(
                "Resolved latent file end_frame mismatches against dataset metadata. "
                f"dataset={self.repo_id}, mismatches={resolved_latent_mismatches}"
            )

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        resolved_end_frame = end_frame
        if self.require_latents_for_sampling:
            text_latent_file, resolved_end_frame = self._resolve_latent_file(
                episode_index=episode_index,
                start_frame=start_frame,
                end_frame=end_frame,
                camera_key=self.used_video_keys[0],
            )
            if text_latent_file is None:
                return None

        min_required_frames = (
            self.RETURN_ACTIONS + 1
            if self.variant == "keyframe_sample"
            else self.raw_window_frames
        )
        if resolved_end_frame - start_frame < min_required_frames:
            return None

        video_chunk_path = self.video_path / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            video_file = video_chunk_path / key / f"episode_{episode_index:06d}.mp4"
            if not video_file.exists():
                return None
        return resolved_end_frame

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _stride_hf_batch(self, batch, stride):
        if stride == 1:
            return batch
        return {key: value[::stride] for key, value in batch.items()}

    def _get_hf_data_at_local_indices(self, episode_index: int, local_indices: list[int]):
        if not local_indices:
            raise ValueError("local_indices must be non-empty")
        local_indices = [int(value) for value in local_indices]
        global_indices = [self._get_global_idx(episode_index, value) for value in local_indices]
        global_start = min(global_indices)
        global_end = max(global_indices) + 1
        batch = self._get_range_hf_data(global_start, global_end)
        offsets = torch.as_tensor(
            [global_index - global_start for global_index in global_indices],
            dtype=torch.long,
        )
        return {key: value[offsets] for key, value in batch.items()}

    _KEYFRAME_MANIFEST_GLOBS = (
        "*_keyframe_sample.json",
        "*_sampled_height_keyposes.json",
    )

    def _build_keyframe_manifest_index(self):
        repo_root = Path(self.repo_id)
        index = {}
        for glob_pattern in self._KEYFRAME_MANIFEST_GLOBS:
            for manifest_path in repo_root.rglob(glob_pattern):
                try:
                    episode_text = manifest_path.name.split("episode_", 1)[1][:6]
                    episode_index = int(episode_text)
                except (IndexError, ValueError):
                    continue
                index.setdefault(episode_index, manifest_path)
        self._keyframe_manifest_index = index
        return index

    def _load_keyframe_manifest(self, episode_index: int) -> dict:
        episode_index = int(episode_index)
        if episode_index in self._keyframe_manifest_cache:
            return self._keyframe_manifest_cache[episode_index]
        index = self._keyframe_manifest_index
        if index is None:
            index = self._build_keyframe_manifest_index()
        manifest_path = index.get(episode_index)
        if manifest_path is None:
            raise FileNotFoundError(
                "Missing keyframe sampling manifest. "
                f"dataset={self.repo_id}, episode_index={episode_index}, "
                "expected a file matching *_keyframe_sample.json "
                "or *_sampled_height_keyposes.json"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._keyframe_manifest_cache[episode_index] = manifest
        return manifest

    @staticmethod
    def _uniform_positive_stride(values: list[int]) -> int | None:
        if len(values) < 2:
            return None
        diffs = np.diff(np.asarray(values, dtype=np.int64))
        if diffs.size == 0 or np.any(diffs <= 0):
            return None
        first = int(diffs[0])
        if np.all(diffs == first):
            return first
        return None

    def _sampled_indices_from_keyframe_manifest(self, manifest: dict) -> list[int]:
        if manifest.get("sampled_indices"):
            return [int(value) for value in manifest["sampled_indices"]]

        n_frames = int(manifest["original_frames"])
        keep = np.zeros(n_frames, dtype=bool)
        key_intervals = []
        for interval in manifest.get("merged_key_intervals", []):
            start = int(interval["start"])
            end = int(interval["end"])
            key_intervals.append((start, end))
            keep[start : end + 1] = True

        cursor = 0
        for start, end in key_intervals + [(n_frames, n_frames - 1)]:
            if cursor < start:
                keep[np.arange(cursor, start, self.keyframe_non_key_stride, dtype=int)] = True
            cursor = max(cursor, end + 1)
        if n_frames > 0:
            keep[0] = True
            keep[-1] = True
        return np.flatnonzero(keep).astype(int).tolist()

    def _sample_keyframe_window(
        self,
        episode_index: int,
        start_frame: int,
        end_frame: int,
    ) -> tuple[int, list[int], list[int], list[int], int | None, int]:
        manifest = self._load_keyframe_manifest(episode_index)
        sampled_indices = [
            int(value)
            for value in self._sampled_indices_from_keyframe_manifest(manifest)
            if int(start_frame) <= int(value) < int(end_frame)
        ]
        required_positions = self.RETURN_ACTIONS + 1
        if len(sampled_indices) < required_positions:
            raise ValueError(
                "Insufficient keyframe-sampled frames. "
                f"episode_index={episode_index}, start_frame={start_frame}, "
                f"end_frame={end_frame}, available={len(sampled_indices)}, "
                f"required={required_positions}"
            )

        max_start_pos = len(sampled_indices) - required_positions
        start_pos = int(np.random.randint(0, max_start_pos + 1))
        sampled_window = sampled_indices[start_pos : start_pos + required_positions]
        action_frame_ids = sampled_window[: self.RETURN_ACTIONS]
        image_frame_ids = [
            sampled_window[video_step * self.BASE_VIDEO_FRAME_STRIDE]
            for video_step in range(self.RETURN_VIDEO_FRAMES)
        ]
        cache_freq_ratio = self._uniform_positive_stride(sampled_window)
        raw_window_frames = int(image_frame_ids[-1] - image_frame_ids[0] + 1)
        return (
            int(action_frame_ids[0]),
            image_frame_ids,
            action_frame_ids,
            sampled_indices[:start_pos],
            cache_freq_ratio,
            raw_window_frames,
        )

    def _get_single_hf_vector(self, global_index, column_name):
        sample = self._hf_torch_view[int(global_index)]
        value = sample[column_name]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def _get_single_hf_action(self, global_index):
        return self._get_single_hf_vector(global_index, "action")

    def _get_single_hf_state(self, global_index):
        if self.state_column_name is None:
            return None
        return self._get_single_hf_vector(global_index, self.state_column_name)

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_latent_dir(self, episode_index, camera_key):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return Path(self.latent_path) / f"chunk-{episode_chunk:03d}" / camera_key

    def _get_text_emb_dir(self, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return Path(self.text_emb_path) / f"chunk-{episode_chunk:03d}"

    def _parse_segment_end_frame(self, file_path, episode_index, start_frame):
        parts = file_path.stem.split("_")
        if len(parts) != 4 or parts[0] != "episode":
            raise ValueError(f"Unexpected segment filename format: {file_path}")
        parsed_episode_index = int(parts[1])
        parsed_start_frame = int(parts[2])
        parsed_end_frame = int(parts[3])
        if parsed_episode_index != episode_index or parsed_start_frame != start_frame:
            raise ValueError(
                "Segment filename does not match requested episode/start frame. "
                f"file_path={file_path}, episode_index={episode_index}, start_frame={start_frame}"
            )
        return parsed_end_frame

    def _parse_latent_end_frame(self, latent_file, episode_index, start_frame):
        return self._parse_segment_end_frame(
            latent_file,
            episode_index,
            start_frame,
        )

    def _resolve_latent_file(self, episode_index, start_frame, end_frame, camera_key):
        cache_key = (camera_key, episode_index, start_frame, end_frame)
        if cache_key in self._latent_file_cache:
            return self._latent_file_cache[cache_key]

        latent_dir = self._get_latent_dir(episode_index, camera_key)
        latent_file = latent_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
        if latent_file.exists():
            result = (latent_file, end_frame)
            self._latent_file_cache[cache_key] = result
            return result

        candidates = sorted(
            latent_dir.glob(f"episode_{episode_index:06d}_{start_frame}_*.pth")
        )
        if not candidates:
            result = (None, None)
            self._latent_file_cache[cache_key] = result
            return result

        candidate_pairs = [
            (self._parse_latent_end_frame(candidate, episode_index, start_frame), candidate)
            for candidate in candidates
        ]
        if len(candidate_pairs) == 1:
            result = (candidate_pairs[0][1], candidate_pairs[0][0])
            self._latent_file_cache[cache_key] = result
            return result

        larger_or_equal = [pair for pair in candidate_pairs if pair[0] >= end_frame]
        if larger_or_equal:
            resolved_end_frame, resolved_file = min(larger_or_equal, key=lambda pair: pair[0])
        else:
            resolved_end_frame, resolved_file = max(candidate_pairs, key=lambda pair: pair[0])
        result = (resolved_file, resolved_end_frame)
        self._latent_file_cache[cache_key] = result
        return result

    def _resolve_text_emb_file(self, episode_index, start_frame, end_frame):
        cache_key = (episode_index, start_frame, end_frame)
        if cache_key in self._text_emb_file_cache:
            return self._text_emb_file_cache[cache_key]

        text_emb_dir = self._get_text_emb_dir(episode_index)
        direct_candidates = [
            text_emb_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pt",
            text_emb_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth",
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                result = (candidate, end_frame)
                self._text_emb_file_cache[cache_key] = result
                return result

        candidates = sorted(text_emb_dir.glob(f"episode_{episode_index:06d}_{start_frame}_*.pt"))
        candidates.extend(
            sorted(text_emb_dir.glob(f"episode_{episode_index:06d}_{start_frame}_*.pth"))
        )
        if not candidates:
            result = (None, None)
            self._text_emb_file_cache[cache_key] = result
            return result

        candidate_pairs = [
            (self._parse_segment_end_frame(candidate, episode_index, start_frame), candidate)
            for candidate in candidates
        ]
        if len(candidate_pairs) == 1:
            result = (candidate_pairs[0][1], candidate_pairs[0][0])
            self._text_emb_file_cache[cache_key] = result
            return result

        larger_or_equal = [pair for pair in candidate_pairs if pair[0] >= end_frame]
        if larger_or_equal:
            resolved_end_frame, resolved_file = min(larger_or_equal, key=lambda pair: pair[0])
        else:
            resolved_end_frame, resolved_file = max(candidate_pairs, key=lambda pair: pair[0])
        result = (resolved_file, resolved_end_frame)
        self._text_emb_file_cache[cache_key] = result
        return result

    def _resolve_condition_file(self, episode_index, start_frame, end_frame, camera_key):
        cache_key = (camera_key, episode_index, start_frame, end_frame)
        if cache_key in self._condition_file_cache:
            return self._condition_file_cache[cache_key]

        text_emb_file, resolved_end_frame = self._resolve_text_emb_file(
            episode_index=episode_index,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        if text_emb_file is not None:
            result = ("text_emb", text_emb_file, resolved_end_frame)
            self._condition_file_cache[cache_key] = result
            return result

        latent_file, resolved_end_frame = self._resolve_latent_file(
            episode_index=episode_index,
            start_frame=start_frame,
            end_frame=end_frame,
            camera_key=camera_key,
        )
        if latent_file is not None:
            result = ("latent", latent_file, resolved_end_frame)
            self._condition_file_cache[cache_key] = result
            return result

        result = (None, None, None)
        self._condition_file_cache[cache_key] = result
        return result

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        out = {}
        for key in self.used_video_keys:
            latent_file, resolved_end_frame = self._resolve_latent_file(
                episode_index=episode_index,
                start_frame=start_frame,
                end_frame=end_frame,
                camera_key=key,
            )
            if latent_file is None:
                raise FileNotFoundError(
                    "Missing latent file for requested sample. "
                    f"episode_index={episode_index}, start_frame={start_frame}, "
                    f"end_frame={end_frame}, camera_key={key}"
                )
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)

    def _get_condition_text_emb(self, start_frame, end_frame, episode_index):
        condition_source, condition_file, resolved_end_frame = self._resolve_condition_file(
            episode_index=episode_index,
            start_frame=start_frame,
            end_frame=end_frame,
            camera_key=self.used_video_keys[0],
        )
        if condition_file is None:
            raise FileNotFoundError(
                "Missing text latent file for requested sample. "
                f"episode_index={episode_index}, start_frame={start_frame}, end_frame={end_frame}"
            )
        payload = torch.load(condition_file, weights_only=False)
        if condition_source == "latent":
            text_emb = payload["text_emb"]
        else:
            if torch.is_tensor(payload):
                text_emb = payload
            elif isinstance(payload, dict) and "text_emb" in payload:
                text_emb = payload["text_emb"]
            else:
                raise ValueError(
                    "Unsupported text embedding payload. "
                    f"file={condition_file}, type={type(payload)}"
                )
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb
        return text_emb

    def _get_video_file(self, episode_index, camera_key):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        return (
            self.video_path
            / f"chunk-{episode_chunk:03d}"
            / camera_key
            / f"episode_{episode_index:06d}.mp4"
        )

    def _get_hdf5_file(self, video_path):
        return video_path.with_suffix(".hdf5")

    def _get_available_video_frames(self, episode_index):
        available_frames = []
        for key in self.used_video_keys:
            video_file = self._get_video_file(episode_index, key)
            hdf5_file = self._get_hdf5_file(video_file)
            if hdf5_file.exists():
                with h5py.File(hdf5_file, "r") as h5_file:
                    if "frames" not in h5_file:
                        raise ValueError(f"Missing 'frames' dataset in {hdf5_file}")
                    available_frames.append(len(h5_file["frames"]))
            else:
                available_frames.append(None)

        known_available = [v for v in available_frames if v is not None]
        if not known_available:
            return None
        return min(known_available)

    def _decode_selected_frames_hdf5(self, hdf5_path, frame_ids, episode_index, camera_key):
        try:
            decoded = []
            with h5py.File(hdf5_path, "r") as h5_file:
                frames_dataset = h5_file["frames"]
                num_frames = len(frames_dataset)
                missing = [frame_id for frame_id in frame_ids if frame_id < 0 or frame_id >= num_frames]
                if missing:
                    raise ValueError(
                        "hdf5 returned incomplete frames. "
                        f"episode_index={episode_index}, camera_key={camera_key}, "
                        f"hdf5_path={hdf5_path}, requested_frame_ids={frame_ids}, missing={missing}, "
                        f"num_frames={num_frames}"
                    )

                for frame_id in frame_ids:
                    encoded = np.asarray(frames_dataset[frame_id], dtype=np.uint8)
                    frame_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        raise RuntimeError(
                            f"cv2.imdecode failed for hdf5_path={hdf5_path} at frame_id={frame_id}"
                        )
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    decoded.append(np.ascontiguousarray(frame_rgb))
            return decoded
        except Exception as exc:
            raise RuntimeError(
                "hdf5 decode failed. "
                f"episode_index={episode_index}, camera_key={camera_key}, "
                f"hdf5_path={hdf5_path}, frame_ids={frame_ids}"
            ) from exc

    def _decode_selected_frames(self, video_path, frame_ids, episode_index, camera_key):
        wanted = set(frame_ids)
        try:
            decoded = {}
            reader = imageio.get_reader(str(video_path), format="ffmpeg")
            try:
                max_frame_id = max(frame_ids)
                for frame_idx, frame in enumerate(reader):
                    if frame_idx in wanted:
                        decoded[frame_idx] = np.ascontiguousarray(frame)
                    if frame_idx >= max_frame_id and len(decoded) == len(wanted):
                        break
            finally:
                reader.close()
        except Exception as exc:
            raise RuntimeError(
                "imageio-ffmpeg decode failed. "
                f"episode_index={episode_index}, camera_key={camera_key}, "
                f"video_path={video_path}, frame_ids={frame_ids}"
            ) from exc

        missing = [frame_id for frame_id in frame_ids if frame_id not in decoded]
        if missing:
            raise ValueError(
                "imageio-ffmpeg returned incomplete frames. "
                f"episode_index={episode_index}, camera_key={camera_key}, "
                f"video_path={video_path}, requested_frame_ids={frame_ids}, missing={missing}"
            )
        return [decoded[frame_id] for frame_id in frame_ids]

    def _sample_clip_start(self, start_frame, end_frame, raw_window_frames):
        max_clip_start = end_frame - raw_window_frames
        return int(np.random.randint(start_frame, max_clip_start + 1))

    def _get_video_frames(self, episode_index, frame_ids):
        per_camera_frames = {}
        for key in self.used_video_keys:
            video_file = self._get_video_file(episode_index, key)
            hdf5_file = self._get_hdf5_file(video_file)
            if hdf5_file.exists():
                decoded_frames = self._decode_selected_frames_hdf5(
                    hdf5_file,
                    frame_ids,
                    episode_index=episode_index,
                    camera_key=key,
                )
            else:
                decoded_frames = self._decode_selected_frames(
                    video_file,
                    frame_ids,
                    episode_index=episode_index,
                    camera_key=key,
                )
            per_camera_frames[key] = torch.from_numpy(np.stack(decoded_frames, axis=0))
        return per_camera_frames

    def _build_sample(self, meta_index):
        cur_meta = self.new_metas[meta_index]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        latent_end_frame = cur_meta.get("latent_end_frame", cur_meta["end_frame"])
        video_end_frame = latent_end_frame
        raw_window_frames = self.raw_window_frames
        num_actions = self.RETURN_ACTIONS
        action_span_frames = (num_actions - 1) * self.action_frame_stride + 1
        available_video_frames = self._get_available_video_frames(episode_index)
        if available_video_frames is not None:
            video_end_frame = min(video_end_frame, available_video_frames)
        min_required_frames = (
            self.RETURN_ACTIONS + 1
            if self.variant == "keyframe_sample"
            else raw_window_frames
        )
        if video_end_frame - start_frame < min_required_frames:
            raise ValueError(
                "Insufficient available video frames after hdf5 truncation. "
                f"episode_index={episode_index}, start_frame={start_frame}, "
                f"latent_end_frame={latent_end_frame}, video_end_frame={video_end_frame}, "
                f"required_frames={min_required_frames}"
            )
        cache_freq_ratio = self.freq_ratio
        if self.variant == "keyframe_sample":
            (
                clip_start,
                image_frame_ids,
                action_frame_ids,
                history_frame_ids,
                cache_freq_ratio,
                raw_window_frames,
            ) = self._sample_keyframe_window(
                episode_index,
                start_frame,
                video_end_frame,
            )
            hf_action_data = self._get_hf_data_at_local_indices(
                episode_index,
                action_frame_ids,
            )
            state_history, state_mask = self._get_state_history_from_indices(
                episode_index,
                history_frame_ids,
                fallback_index=clip_start,
            )
            first_reference_frame_id = (
                int(history_frame_ids[-1]) if history_frame_ids else int(action_frame_ids[0])
            )
        else:
            clip_start = self._sample_clip_start(start_frame, video_end_frame, raw_window_frames)
            action_frame_ids = list(
                range(
                    clip_start,
                    clip_start + action_span_frames,
                    self.action_frame_stride,
                )
            )[: self.RETURN_ACTIONS]
            image_frame_ids = list(
                range(
                    clip_start,
                    clip_start + raw_window_frames,
                    self.video_frame_stride,
                )
            )[: self.RETURN_VIDEO_FRAMES]

            global_action_start = self._get_global_idx(episode_index, clip_start)
            global_action_end = self._get_global_idx(
                episode_index, clip_start + action_span_frames
            )
            hf_action_data = self._get_range_hf_data(global_action_start, global_action_end)
            hf_action_data = self._stride_hf_batch(hf_action_data, self.action_frame_stride)
            state_history, state_mask = self._get_state_history(
                episode_index=episode_index,
                local_action_start=clip_start,
            )
            first_reference_frame_id = max(
                0, int(action_frame_ids[0]) - self.action_frame_stride
            )

        video_frames = self._get_video_frames(episode_index, image_frame_ids)

        enable_cache = bool(getattr(self.config, "enable_clip_latent_cache", False))
        cache_read = bool(getattr(self.config, "clip_latent_cache_read", True))
        clip_latent_cache_paths = None
        if enable_cache and cache_freq_ratio is not None:
            clip_latent_cache_paths = clip_latent_paths_for_sample(
                Path(self.repo_id),
                episode_index=episode_index,
                clip_start=clip_start,
                camera_keys=self.used_video_keys,
                freq_ratio=cache_freq_ratio,
            )
            if cache_read and all(path.is_file() for path in clip_latent_cache_paths.values()):
                video_frames = None

        out_dict = {
            "video_frames": video_frames,
            "episode_index": int(episode_index),
            "clip_start": int(clip_start),
            "dataset_repo_root": str(Path(self.repo_id).resolve()),
            "image_frame_ids": [int(value) for value in image_frame_ids],
            "clip_raw_window_frames": int(raw_window_frames),
            "clip_latent_cache_freq_ratio": (
                int(cache_freq_ratio) if cache_freq_ratio is not None else -1
            ),
            "text_emb": self._get_condition_text_emb(start_frame, latent_end_frame, episode_index),
            "example_action_loss_mask": torch.tensor(self.enable_action_loss, dtype=torch.bool),
            "state": self._state_post_process(state_history, state_mask),
            "state_mask": torch.from_numpy(state_mask).bool(),
            "clip_latent_cache_paths": None,
        }
        if clip_latent_cache_paths is not None:
            out_dict["clip_latent_cache_paths"] = {
                camera_key: str(path)
                for camera_key, path in clip_latent_cache_paths.items()
            }
        if self.state_column_name is not None:
            reference_data = self._get_hf_data_at_local_indices(
                episode_index, action_frame_ids
            )[self.state_column_name]
        else:
            reference_frame_ids = [
                first_reference_frame_id,
                *[int(value) for value in action_frame_ids[:-1]],
            ]
            reference_data = self._get_hf_data_at_local_indices(
                episode_index, reference_frame_ids
            )["action"]
        release_poses = release_valid = None
        if self.release_pose_aux:
            release_poses, release_valid = self._release_targets_for_indices(
                episode_index, action_frame_ids
            )
        out_dict["actions"], out_dict["actions_mask"] = self._action_post_process(
            hf_action_data["action"],
            references=reference_data,
            release_poses=release_poses,
            release_valid=release_valid,
        )
        return out_dict

    def _format_state_history(self, history, valid_count, history_dim):
        padded_history = np.zeros(
            (self.state_history_len, int(history_dim)),
            dtype=np.float32,
        )
        state_mask = np.zeros((self.state_history_len,), dtype=bool)
        if valid_count > 0:
            padded_history[-valid_count:] = history[-valid_count:]
            state_mask[-valid_count:] = True
        return padded_history, state_mask

    def _get_state_history_from_indices(self, episode_index, history_indices, fallback_index):
        history_indices = [int(value) for value in history_indices[-self.state_history_len:]]
        valid_count = len(history_indices)
        if valid_count > 0:
            history_batch = self._get_hf_data_at_local_indices(episode_index, history_indices)
            history = history_batch["action"]
            if torch.is_tensor(history):
                history = history.detach().cpu().numpy()
            history = np.asarray(history, dtype=np.float32)
            history_dim = history.shape[1]
        else:
            first_action = self._get_single_hf_action(
                self._get_global_idx(episode_index, int(fallback_index))
            )
            history_dim = int(np.asarray(first_action, dtype=np.float32).shape[0])
            history = np.empty((0, history_dim), dtype=np.float32)
        return self._format_state_history(history, valid_count, history_dim)

    def _get_state_history(self, episode_index, local_action_start):

        history = []
        history_start = local_action_start - self.action_frame_stride * self.state_history_len
        while history_start < 0:
            history_start += self.action_frame_stride
        history_indices = list(
            range(history_start, local_action_start, self.action_frame_stride)
        )
        valid_count = len(history_indices)
        if valid_count > 0:
            global_history_start = self._get_global_idx(episode_index, history_indices[0])
            global_history_end = self._get_global_idx(episode_index, history_indices[-1] + 1)
            history_batch = self._get_range_hf_data(global_history_start, global_history_end)
            history = history_batch["action"]
            history = history[:: self.action_frame_stride]
            if torch.is_tensor(history):
                history = history.detach().cpu().numpy()
            history = np.asarray(history, dtype=np.float32)
        else:
            history = np.empty((0, self.config.action_dim - 1), dtype=np.float32)

        if history.ndim != 2:
            raise ValueError(
                f"Expected state history to have shape [T,D], but got {history.shape}"
            )

        history_dim = history.shape[1] if history.ndim == 2 and history.shape[0] > 0 else None
        if history_dim is None:
            first_action = self._get_single_hf_action(self._get_global_idx(episode_index, local_action_start))
            history_dim = int(np.asarray(first_action, dtype=np.float32).shape[0])

        return self._format_state_history(history, valid_count, history_dim)

    def _normalize_model_actions(self, values, *, q01=None, q99=None):
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(
                f"Expected values to have shape [T,D], but got {values.shape}"
            )

        if values.shape[1] != self.config.action_dim:
            raise ValueError(
                f"Expected model actions [T,{self.config.action_dim}], got {values.shape}"
            )
        q01 = self.q01 if q01 is None else q01
        q99 = self.q99 if q99 is None else q99
        values = (
            (values - q01) / (q99 - q01 + 1e-6) * 2.0
            - 1.0
        )
        return np.clip(values, -1.5, 1.5)

    def _state_post_process(self, state_action, state_mask):
        state_action = np.asarray(state_action, dtype=np.float32)
        valid_mask = np.asarray(state_mask, dtype=bool)
        represented = np.zeros((state_action.shape[0], self.config.action_dim), dtype=np.float32)
        if valid_mask.any():
            represented[valid_mask] = encode_absolute_history(state_action[valid_mask])
        state_aligned = self._normalize_model_actions(
            represented, q01=self.state_q01, q99=self.state_q99
        )
        state_aligned *= self.state_channel_mask
        return torch.from_numpy(state_aligned.T[:, :, None, None]).float()

    def _action_post_process(
        self,
        action,
        *,
        references,
        release_poses=None,
        release_valid=None,
    ):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        expected_actions = self.RETURN_ACTIONS
        if action.shape[0] != expected_actions:
            raise ValueError(
                f"Expected {expected_actions} actions, but got {action.shape[0]}"
            )
        if torch.is_tensor(references):
            references = references.detach().cpu().numpy()
        model_action, action_mask_aligned = encode_action_targets(
            action,
            representation=self.action_representation,
            references=np.asarray(references, dtype=np.float32),
            release_poses=release_poses,
            release_references=np.asarray(references, dtype=np.float32),
            release_valid=release_valid,
        )
        action_aligned = self._normalize_model_actions(model_action)
        action_mask_aligned &= self.channel_mask[None].astype(bool)
        action_aligned = rearrange(
            action_aligned,
            "(f n) c -> c f n 1",
            f=self.ACTION_GROUPS,
            n=self.ACTIONS_PER_GROUP,
        )
        action_mask_aligned = rearrange(
            action_mask_aligned,
            "(f n) c -> c f n 1",
            f=self.ACTION_GROUPS,
            n=self.ACTIONS_PER_GROUP,
        )
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _episode_actions(self, episode_index: int) -> np.ndarray:
        episode_index = int(episode_index)
        cached = self._episode_action_cache.get(episode_index)
        if cached is not None:
            self._episode_action_cache.move_to_end(episode_index)
            return cached
        length = int(self.meta.episodes[episode_index]["length"])
        actions = self._get_hf_data_at_local_indices(
            episode_index, list(range(length))
        )["action"]
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        self._episode_action_cache[episode_index] = actions
        while len(self._episode_action_cache) > 8:
            self._episode_action_cache.popitem(last=False)
        return actions

    def _release_targets_for_indices(
        self, episode_index: int, local_indices: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        actions = self._episode_actions(episode_index)
        release_poses = np.zeros((len(local_indices), 16), dtype=np.float32)
        release_valid = np.zeros((len(local_indices), 2), dtype=bool)
        for arm_index, gripper_index in enumerate((7, 15)):
            gripper = actions[:, gripper_index]
            previous = np.concatenate([gripper[:1], gripper[:-1]])
            transitions = np.flatnonzero(
                (gripper >= self.release_open_threshold)
                & (previous < self.release_open_threshold)
            )
            for sample_index, local_index in enumerate(local_indices):
                candidate_pos = np.searchsorted(transitions, int(local_index), side="left")
                if candidate_pos >= len(transitions):
                    continue
                release_index = int(transitions[candidate_pos])
                pose_slice = slice(0, 7) if arm_index == 0 else slice(8, 15)
                release_poses[sample_index, pose_slice] = actions[
                    release_index, pose_slice
                ]
                release_valid[sample_index, arm_index] = True
        return release_poses, release_valid

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        current_idx = idx
        last_error = None

        for retry_id in range(self.MAX_SAMPLE_RETRIES):
            try:
                return self._build_sample(current_idx)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed to build training sample, retrying with a new sample. "
                    f"retry={retry_id + 1}/{self.MAX_SAMPLE_RETRIES}, "
                    f"requested_idx={idx}, current_idx={current_idx}, error={exc}"
                )
                current_idx = int(np.random.randint(0, len(self.new_metas)))

        raise RuntimeError(
            "Exceeded max sample retries while building raw-video training sample. "
            f"requested_idx={idx}, last_error={last_error}"
        ) from last_error

    def __len__(self):
        return len(self.new_metas)


def va_train_collate_fn(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("va_train_collate_fn received an empty batch")
    batch: dict = {}
    keys = list(samples[0].keys())
    for sample in samples[1:]:
        keys.extend(key for key in sample.keys() if key not in keys)
    for key in keys:
        values = [sample.get(key) for sample in samples]
        if key in {"video_frames", "dataset_repo_root", "clip_latent_cache_paths"}:
            batch[key] = values
        elif values[0] is None:
            batch[key] = values
        elif isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], bool):
            batch[key] = torch.tensor(values, dtype=torch.bool)
        elif isinstance(values[0], (int, float)):
            batch[key] = torch.tensor(values)
        else:
            batch[key] = values
    return batch

if __name__ == '__main__':
    import copy
    import math

    from configs import VA_CONFIGS

    config_name = 'astribot_train'
    seed = 0
    num_vis_samples = 100
    num_init_worker = 32
    fps = 8
    clean_root = (
        '/media/damoxing/fileset/md4d/third_parties/lingbot-va/data/'
        'robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50'
    )
    aug_root = (
        '/media/damoxing/fileset/md4d/third_parties/lingbot-va/data/'
        'robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_aug_500'
    )
    astribot_root = (
        '/media/damoxing/fileset/md4d/third_parties/lingbot-va/data/'
        'data_4d_wam/astribot-pick_white_plate'
    )
    output_dir = (
        Path(__file__).resolve().parents[2] / 'debug' / 'dataset_action_videos'
    )

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    def _sanitize_name(text):
        return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in text)

    def _resolve_global_index(dataset, global_idx):
        dataset_id = dataset.item_id_to_dataset_id[global_idx]
        local_idx = global_idx - dataset.acc_dset_num[dataset_id]
        leaf_dataset = dataset._datasets[dataset_id]
        return leaf_dataset, dataset_id, local_idx

    def _build_video_strip(video_frames, camera_keys, video_step, target_width):
        tiles = []
        target_tile_h = None
        for cam_idx, camera_key in enumerate(camera_keys):
            if isinstance(video_frames, dict):
                frame_rgb = video_frames[camera_key][video_step]
            else:
                frame_rgb = video_frames[cam_idx, video_step]
            frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
            if target_tile_h is None:
                target_tile_h = frame_bgr.shape[0]
            elif frame_bgr.shape[0] != target_tile_h:
                resized_w = max(1, int(round(frame_bgr.shape[1] * target_tile_h / frame_bgr.shape[0])))
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (resized_w, target_tile_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            cv2.putText(
                frame_bgr,
                camera_key.split('.')[-1],
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            tiles.append(frame_bgr)
        strip = cv2.hconcat(tiles)
        if strip.shape[1] != target_width:
            resized_h = max(1, int(round(strip.shape[0] * target_width / strip.shape[1])))
            strip = cv2.resize(strip, (target_width, resized_h), interpolation=cv2.INTER_LINEAR)
        return strip

    def _draw_actions_panel(actions, action_mask, cursor_step):
        action_dim, num_action_steps = actions.shape
        num_cols = 5
        num_rows = math.ceil(action_dim / num_cols)
        cell_w = 220
        cell_h = 110
        pad = 10
        title_h = 28
        width = num_cols * cell_w + (num_cols + 1) * pad
        height = num_rows * cell_h + (num_rows + 1) * pad + title_h
        canvas = np.full((height, width, 3), 250, dtype=np.uint8)
        cv2.putText(
            canvas,
            'Actions (normalized)',
            (pad, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        colors = [
            (255, 99, 71),
            (60, 179, 113),
            (65, 105, 225),
            (218, 165, 32),
            (186, 85, 211),
            (70, 130, 180),
        ]
        y_min = -1.6
        y_max = 1.6

        def _x_of(step, left, plot_w):
            if num_action_steps <= 1:
                return left
            return left + int(round(step * (plot_w - 1) / (num_action_steps - 1)))

        def _y_of(value, top, plot_h):
            clipped = float(np.clip(value, y_min, y_max))
            ratio = (clipped - y_min) / (y_max - y_min)
            return top + plot_h - 1 - int(round(ratio * (plot_h - 1)))

        for dim_idx in range(action_dim):
            row = dim_idx // num_cols
            col = dim_idx % num_cols
            cell_x = pad + col * (cell_w + pad)
            cell_y = title_h + pad + row * (cell_h + pad)
            plot_x = cell_x + 30
            plot_y = cell_y + 16
            plot_w = cell_w - 40
            plot_h = cell_h - 28
            color = colors[dim_idx % len(colors)]

            cv2.rectangle(canvas, (cell_x, cell_y), (cell_x + cell_w, cell_y + cell_h), (210, 210, 210), 1)
            cv2.putText(
                canvas,
                f'a{dim_idx:02d}',
                (cell_x + 6, cell_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

            zero_y = _y_of(0.0, plot_y, plot_h)
            cv2.line(canvas, (plot_x, zero_y), (plot_x + plot_w, zero_y), (220, 220, 220), 1, cv2.LINE_AA)
            cv2.line(canvas, (plot_x, plot_y), (plot_x, plot_y + plot_h), (220, 220, 220), 1, cv2.LINE_AA)

            cursor_x = _x_of(min(cursor_step, num_action_steps - 1), plot_x, plot_w)
            cv2.line(
                canvas,
                (cursor_x, plot_y),
                (cursor_x, plot_y + plot_h),
                (120, 120, 120),
                1,
                cv2.LINE_AA,
            )

            valid_steps = np.flatnonzero(action_mask[dim_idx])
            for step in range(num_action_steps - 1):
                if not (action_mask[dim_idx, step] and action_mask[dim_idx, step + 1]):
                    continue
                p1 = (
                    _x_of(step, plot_x, plot_w),
                    _y_of(actions[dim_idx, step], plot_y, plot_h),
                )
                p2 = (
                    _x_of(step + 1, plot_x, plot_w),
                    _y_of(actions[dim_idx, step + 1], plot_y, plot_h),
                )
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

            if len(valid_steps) > 0:
                current_step = valid_steps[min(np.searchsorted(valid_steps, cursor_step), len(valid_steps) - 1)]
                current_point = (
                    _x_of(int(current_step), plot_x, plot_w),
                    _y_of(actions[dim_idx, current_step], plot_y, plot_h),
                )
                cv2.circle(canvas, current_point, 3, color, -1, cv2.LINE_AA)

        return canvas

    def _make_visualization_frames(sample, meta, repo_name, global_idx, local_idx, camera_keys):
        raw_video_frames = sample['video_frames']
        if isinstance(raw_video_frames, dict):
            video_frames = {
                key: value.detach().cpu().numpy()
                for key, value in raw_video_frames.items()
            }
            num_video_frames = next(iter(video_frames.values())).shape[0]
        else:
            video_frames = raw_video_frames.detach().cpu().numpy()
            num_video_frames = video_frames.shape[1]
        actions = sample['actions'].detach().cpu().numpy().reshape(sample['actions'].shape[0], -1)
        action_mask = sample['actions_mask'].detach().cpu().numpy().reshape(sample['actions_mask'].shape[0], -1)
        num_action_steps = actions.shape[1]
        action_panel = _draw_actions_panel(actions, action_mask, cursor_step=0)
        frames = []

        task_text = meta.get('tasks', '')
        if isinstance(task_text, (list, tuple)):
            task_text = ', '.join(map(str, task_text[:2]))
        task_text = str(task_text)[:120]

        for action_step in range(num_action_steps):
            video_step = int(round(action_step * max(num_video_frames - 1, 0) / max(num_action_steps - 1, 1)))
            top_strip = _build_video_strip(video_frames, camera_keys, video_step, action_panel.shape[1])
            bottom_panel = _draw_actions_panel(actions, action_mask, cursor_step=action_step)
            header_h = 54
            header = np.full((header_h, action_panel.shape[1], 3), 245, dtype=np.uint8)
            cv2.putText(
                header,
                f'sample={global_idx} local={local_idx} repo={repo_name}',
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                header,
                f'episode={meta["episode_index"]} start={meta["start_frame"]} '
                f'end={meta.get("latent_end_frame", meta["end_frame"])} '
                f'action_step={action_step + 1}/{num_action_steps} task={task_text}',
                (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            frames.append(np.concatenate([header, top_strip, bottom_panel], axis=0))
        return frames

    def _write_video(output_path, frames):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(str(output_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame[:, :, ::-1])

    config = copy.deepcopy(VA_CONFIGS[config_name])
    if not hasattr(config, "dataset_paths") or not hasattr(config, "empty_emb_path"):
        raise ValueError(
            f"Config '{config_name}' must define dataset_paths and empty_emb_path."
        )
    config.dataset_paths = [astribot_root, ]
    config.cfg_prob = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=num_init_worker)
    total_len = len(dataset)
    target_num_samples = min(num_vis_samples, total_len)
    max_attempts = max(target_num_samples * 20, 1000)

    print(f'config={config_name}')
    print(f'seed={seed}')
    print(f'dataset_paths={config.dataset_paths}')
    print(f'num_repo_roots={len(dataset._datasets)}')
    print(f'total_len={total_len}')
    print(f'output_dir={output_dir}')
    print(f'target_num_samples={target_num_samples}')

    used_global_indices = set()
    success_count = 0
    attempt_count = 0
    progress = tqdm(total=target_num_samples, desc='Rendering samples', unit='video')

    try:
        while success_count < target_num_samples and attempt_count < max_attempts:
            global_idx = int(rng.integers(total_len))
            if global_idx in used_global_indices:
                continue
            used_global_indices.add(global_idx)
            attempt_count += 1

            leaf_dataset, dataset_id, local_idx = _resolve_global_index(dataset, global_idx)
            meta = leaf_dataset.new_metas[local_idx]
            repo_name = Path(leaf_dataset.repo_id).name
            try:
                sample = leaf_dataset._build_sample(local_idx)
                frames = _make_visualization_frames(
                    sample=sample,
                    meta=meta,
                    repo_name=repo_name,
                    global_idx=global_idx,
                    local_idx=local_idx,
                    camera_keys=leaf_dataset.used_video_keys,
                )
                output_name = (
                    f'{success_count:03d}_g{global_idx:08d}_d{dataset_id:03d}_l{local_idx:08d}_'
                    f'{_sanitize_name(repo_name)}_ep{meta["episode_index"]:06d}.mp4'
                )
                _write_video(output_dir / output_name, frames)
                success_count += 1
                progress.update(1)
            except Exception as exc:
                logger.warning(
                    'Failed to render visualization sample. '
                    f'global_idx={global_idx}, dataset_id={dataset_id}, local_idx={local_idx}, error={exc}'
                )
    finally:
        progress.close()

    print(f'rendered_videos={success_count}')
    print(f'attempted_unique_indices={len(used_global_indices)}')
    if success_count < target_num_samples:
        print(
            f'warning: requested {target_num_samples} samples but only rendered {success_count}.'
        )
