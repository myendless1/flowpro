#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wan_va.dataset.curation.base import resolve_lerobot_videos_root  # noqa: E402


DEFAULT_WAN_PRETRAINED_ROOT = Path(
    "wam4d-ckpt-1/ckpt_to_infer"
)


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


def _segment_text(
    repo_dir: Path, episode_row: dict[str, Any], segment_row: dict[str, Any]
) -> str:
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


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_text_device(device_arg: str) -> torch.device:
    if torch.cuda.is_available() and str(device_arg).startswith("cuda"):
        return torch.device(device_arg)
    return torch.device("cpu")


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

    def normalize_prompt(self, prompt: str) -> str:
        return self._prompt_clean(prompt or "")

    def encode(self, prompt: str) -> torch.Tensor:
        clean_prompt = self.normalize_prompt(prompt)
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


def _hash_text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _chunk_name(chunk_size: int, episode_index: int) -> str:
    return f"chunk-{episode_index // max(1, int(chunk_size)):03d}"


def _ensure_relative_symlink(link_path: Path, target_path: Path, overwrite: bool) -> bool:
    relative_target = Path(
        os.path.relpath(os.fspath(target_path), os.fspath(link_path.parent))
    )
    if link_path.is_symlink():
        current = Path(os.readlink(link_path))
        if current == relative_target:
            return False
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing symlink without --overwrite: {link_path}"
            )
        link_path.unlink()
    elif link_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to replace existing file without --overwrite: {link_path}"
            )
        if link_path.is_dir():
            raise IsADirectoryError(f"Expected file path but found directory: {link_path}")
        link_path.unlink()

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(relative_target)
    return True


def _compute_repo_text_embs(
    *,
    repo_dir: Path,
    text_mode: str,
    text_embedder: _TextEmbedderCache | None,
    overwrite: bool,
    max_episodes: int,
) -> dict[str, int]:
    info_path = repo_dir / "meta" / "info.json"
    episodes_path = repo_dir / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(
            f"Repo is missing meta/info.json or meta/episodes.jsonl: {repo_dir}"
        )

    info = _load_json(info_path)
    chunk_size = int(info.get("chunks_size", 1000) or 1000)
    episode_rows = sorted(
        _load_jsonl_rows(episodes_path),
        key=lambda row: int(row["episode_index"]),
    )
    if max_episodes > 0:
        episode_rows = episode_rows[:max_episodes]

    global_dir = repo_dir / "global_text_emb"
    text_link_root = repo_dir / "text_emb"
    global_dir.mkdir(parents=True, exist_ok=True)
    text_link_root.mkdir(parents=True, exist_ok=True)

    empty_emb = _ensure_empty_emb(
        repo_dir=repo_dir,
        text_mode=text_mode,
        text_embedder=text_embedder,
        overwrite=overwrite,
    )

    manifest_rows: dict[str, dict[str, Any]] = {}
    embeddings_created = 0
    links_created = 0
    segments_total = 0

    for episode_row in episode_rows:
        episode_index = int(episode_row["episode_index"])
        chunk_dir = text_link_root / _chunk_name(chunk_size, episode_index)
        segments = _episode_segments(repo_dir, episode_row)

        for segment in segments:
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            if end_frame <= start_frame:
                continue

            raw_text = _segment_text(repo_dir, episode_row, segment)
            if text_mode == "empty":
                normalized_text = ""
                key = "empty"
                target_path = global_dir / "empty.pt"
                if overwrite or not target_path.exists():
                    torch.save(empty_emb.to(torch.bfloat16).cpu().contiguous(), target_path)
                    embeddings_created += 1
            else:
                if text_embedder is None:
                    raise RuntimeError("text_embedder is required when text_mode=encode")
                normalized_text = text_embedder.normalize_prompt(raw_text)
                key = _hash_text_key(normalized_text)
                target_path = global_dir / f"{key}.pt"
                if overwrite or not target_path.exists():
                    torch.save(text_embedder.encode(raw_text), target_path)
                    embeddings_created += 1

            manifest = manifest_rows.setdefault(
                key,
                {
                    "key": key,
                    "file": target_path.name,
                    "normalized_text": normalized_text,
                    "raw_texts": [],
                    "num_links": 0,
                },
            )
            if raw_text not in manifest["raw_texts"]:
                manifest["raw_texts"].append(raw_text)
            manifest["num_links"] += 1

            link_path = chunk_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pt"
            if _ensure_relative_symlink(link_path, target_path, overwrite):
                links_created += 1
            segments_total += 1

    manifest_path = global_dir / "manifest.json"
    manifest_data = {
        "repo_dir": os.fspath(repo_dir),
        "text_mode": text_mode,
        "num_embeddings": len(manifest_rows),
        "num_segments": segments_total,
        "items": sorted(manifest_rows.values(), key=lambda item: item["file"]),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2)

    return {
        "episodes": len(episode_rows),
        "segments": segments_total,
        "embeddings": len(manifest_rows),
        "embeddings_created": embeddings_created,
        "links_created": links_created,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute deduplicated text embeddings for LeRobot repos, store them under "
            "`global_text_emb/`, and create per-segment relative symlinks under `text_emb/`."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="A single LeRobot repo root or a directory that contains multiple repo roots.",
    )
    parser.add_argument(
        "--wan-pretrained-root",
        type=Path,
        default=DEFAULT_WAN_PRETRAINED_ROOT,
        help="Directory that contains wan tokenizer/text_encoder checkpoints.",
    )
    parser.add_argument(
        "--text-mode",
        choices=["encode", "empty"],
        default="encode",
        help="`encode` computes embeddings from action text. `empty` writes a shared empty embedding.",
    )
    parser.add_argument(
        "--text-device",
        type=str,
        default="cuda:0",
        help="CUDA device for the text encoder. Falls back to CPU when CUDA is unavailable.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-repos", type=int, default=0)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Per-repo episode cap for debugging.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.wan_pretrained_root = args.wan_pretrained_root.resolve()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")
    if args.text_mode == "encode" and not args.wan_pretrained_root.exists():
        raise FileNotFoundError(
            f"wan pretrained root does not exist: {args.wan_pretrained_root}"
        )

    repo_dirs = _discover_repo_roots(args.dataset_root)
    if not repo_dirs:
        raise RuntimeError(f"No LeRobot repos found under {args.dataset_root}")
    if int(args.max_repos) > 0:
        repo_dirs = repo_dirs[: int(args.max_repos)]

    dtype = _resolve_dtype(args.dtype)
    text_embedder = None
    if args.text_mode == "encode":
        text_embedder = _TextEmbedderCache(
            pretrained_root=args.wan_pretrained_root,
            device=_resolve_text_device(args.text_device),
            dtype=dtype,
        )

    total_segments = 0
    total_embeddings = 0
    total_links_created = 0
    total_embeddings_created = 0

    for repo_dir in tqdm(repo_dirs, desc="Building text embeddings", dynamic_ncols=True):
        stats = _compute_repo_text_embs(
            repo_dir=repo_dir,
            text_mode=args.text_mode,
            text_embedder=text_embedder,
            overwrite=bool(args.overwrite),
            max_episodes=int(args.max_episodes),
        )
        total_segments += int(stats["segments"])
        total_embeddings += int(stats["embeddings"])
        total_links_created += int(stats["links_created"])
        total_embeddings_created += int(stats["embeddings_created"])
        print(
            "[INFO]",
            f"repo={repo_dir}",
            f"episodes={stats['episodes']}",
            f"segments={stats['segments']}",
            f"unique_emb={stats['embeddings']}",
            f"new_emb_files={stats['embeddings_created']}",
            f"new_links={stats['links_created']}",
        )

    print(
        "[INFO] completed",
        f"repos={len(repo_dirs)}",
        f"segments={total_segments}",
        f"unique_emb={total_embeddings}",
        f"new_emb_files={total_embeddings_created}",
        f"new_links={total_links_created}",
    )


if __name__ == "__main__":
    main()
