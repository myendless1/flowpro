#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path("/media/damoxing/datasets/vae4d/lerobot")
DEFAULT_WAN_PRETRAINED_ROOT = Path("wam4d-ckpt-1/ckpt_to_infer")


def _run_command(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(shlex.quote(part) for part in cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def _resolve_repo_dirs(output_root: Path, dataset_names: list[str]) -> list[Path]:
    dataset_to_repo = {
        "astribot": output_root / "astribot-lerobot",
        "bridge": output_root / "bridge-lerobot",
        "droid": output_root / "droid-lerobot",
        "libero": output_root / "libero-lerobot",
        "robocoin": output_root / "robocoin-lerobot",
        "robomind": output_root / "robomind-lerobot",
        "robotwin": output_root / "robotwin-lerobot",
        "rt1": output_root / "rt-1-lerobot",
        "songling": output_root / "songling-lerobot",
    }
    out = []
    for dataset_name in dataset_names:
        if dataset_name not in dataset_to_repo:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")
        out.append(dataset_to_repo[dataset_name])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run raw-dataset curation in two steps: "
            "1) convert to LeRobot repos, 2) build latents."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="robocoin,robomind",
        help=(
            "Comma-separated datasets to process. Default keeps only robocoin,robomind enabled. "
            "Compatible with astribot,bridge,droid,libero,robocoin,robomind,robotwin,rt1,songling."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for converted LeRobot repos.",
    )
    parser.add_argument(
        "--input-roots",
        type=str,
        default="",
        help=(
            "Optional dataset input root overrides as comma-separated "
            "`dataset=/abs/path` pairs. Forwarded to convert_raw_to_lerobot.py."
        ),
    )
    parser.add_argument(
        "--wan-pretrained-root",
        type=Path,
        default=DEFAULT_WAN_PRETRAINED_ROOT,
        help="Directory that contains wan VAE/tokenizer/text_encoder checkpoints.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Per-dataset episode cap for debugging. 0 means all.",
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="opencv",
        help="Compatibility arg forwarded to raw->LeRobot conversion.",
    )
    parser.add_argument("--video-codec", type=str, default="libx264")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", type=str, default="medium")
    parser.add_argument(
        "--skip-image-stats",
        action="store_true",
        help="Skip output-image stats during meta rebuild in conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted repos and latent files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop conversion immediately if any dataset fails.",
    )
    parser.add_argument(
        "--device-ids",
        type=str,
        default="",
        help="Comma-separated CUDA ids forwarded to latent builder.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Override latent-builder worker count.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--text-mode",
        choices=["encode", "empty"],
        default="encode",
        help="Forwarded to latent builder.",
    )
    parser.add_argument(
        "--text-device",
        type=str,
        default="cuda:0",
        help="Forwarded to latent builder.",
    )
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--ori-fps", type=float, default=50.0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument(
        "--keep-text-cache",
        action="store_true",
        help="Keep latent text cache directories after the run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_names = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    if not dataset_names:
        raise ValueError("No datasets selected")

    convert_script = THIS_DIR / "convert_raw_to_lerobot.py"
    latent_script = THIS_DIR / "build_lerobot_latents_parallel.py"
    python_bin = sys.executable

    print("[INFO] selected datasets:", ",".join(dataset_names))
    print("[INFO] output root:", args.output_root)
    print("[INFO] wan pretrained root:", args.wan_pretrained_root)
    print("[INFO] supported but disabled by default:")
    print("# astribot, bridge, droid, libero, robotwin, rt1, songling")

    convert_cmd = [
        python_bin,
        str(convert_script),
        "--datasets",
        ",".join(dataset_names),
        "--output-root",
        str(args.output_root),
        "--max-episodes",
        str(int(args.max_episodes)),
        "--video-backend",
        str(args.video_backend),
        "--video-codec",
        str(args.video_codec),
        "--video-crf",
        str(int(args.video_crf)),
        "--video-preset",
        str(args.video_preset),
        "--num-workers",
        str(int(args.num_workers)),
    ]
    if args.input_roots:
        convert_cmd.extend(["--input-roots", str(args.input_roots)])
    if args.skip_image_stats:
        convert_cmd.append("--skip-image-stats")
    if args.overwrite:
        convert_cmd.append("--overwrite")
    if args.strict:
        convert_cmd.append("--strict")

    _run_command(convert_cmd)

    repo_dirs = _resolve_repo_dirs(Path(args.output_root), dataset_names)
    for repo_dir in repo_dirs:
        latent_cmd = [
            python_bin,
            str(latent_script),
            "--dataset-root",
            str(repo_dir),
            "--wan-pretrained-root",
            str(args.wan_pretrained_root),
            "--device-ids",
            str(args.device_ids),
            "--dtype",
            str(args.dtype),
            "--text-mode",
            str(args.text_mode),
            "--text-device",
            str(args.text_device),
            "--fps",
            str(float(args.fps)),
            "--ori-fps",
            str(float(args.ori_fps)),
            "--frame-stride",
            str(int(args.frame_stride)),
            "--temporal-stride",
            str(int(args.temporal_stride)),
            "--max-repos",
            "1",
            "--max-episodes",
            str(int(args.max_episodes)),
            "--queue-size",
            str(int(args.queue_size)),
        ]
        if int(args.num_workers) > 0:
            latent_cmd.extend(["--num-workers", str(int(args.num_workers))])
        if args.keep_text_cache:
            latent_cmd.append("--keep-text-cache")
        if args.overwrite:
            latent_cmd.append("--overwrite")

        _run_command(latent_cmd)

    print("[DONE] conversion and latent building finished")


if __name__ == "__main__":
    main()
