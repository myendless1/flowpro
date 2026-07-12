#!/usr/bin/env python3

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert all mp4 files under a dataset root into same-name HDF5 files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Root directory to recursively scan for mp4 files.",
    )
    parser.add_argument(
        "--input-list",
        type=Path,
        default=None,
        help=(
            "Optional txt file with one source file per line. Entries may be absolute paths, "
            "paths relative to --input-root, or bare file names. .hdf5 entries are mapped to "
            "same-name .mp4 sources before conversion."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root. If omitted, each .hdf5 file is written next to its source .mp4. "
            "Relative paths from input-root are preserved."
        ),
    )
    parser.add_argument(
        "--codec-ext",
        type=str,
        default=".hdf5",
        help="Output extension for converted files. Default: .hdf5",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality passed to cv2.imencode. Default: 95",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing hdf5 outputs.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first conversion failure.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes used for conversion. Default: 64",
    )
    return parser.parse_args()


def list_mp4_files(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("*.mp4") if p.is_file())


def read_path_list(path: Path) -> list[str]:
    entries = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    if not entries:
        raise ValueError(f"No paths found in input list: {path}")
    return entries


def mp4_name_for_entry(entry: str) -> str:
    path = Path(entry)
    if path.suffix.lower() in {".hdf5", ".h5"}:
        return f"{path.stem}.mp4"
    if path.suffix.lower() == ".mp4":
        return path.name
    return f"{path.name}.mp4"


def build_mp4_name_index(input_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in list_mp4_files(input_root):
        index.setdefault(path.name, []).append(path)
    return index


def resolve_listed_mp4_files(input_list: Path, input_root: Path | None) -> list[Path]:
    entries = read_path_list(input_list)
    name_index: dict[str, list[Path]] | None = None
    resolved: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(candidate: Path) -> bool:
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            return False
        if candidate.suffix.lower() != ".mp4":
            return False
        if candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
        return True

    for entry in entries:
        raw_path = Path(entry)
        candidates: list[Path] = []
        if raw_path.suffix.lower() in {".hdf5", ".h5"}:
            raw_path = raw_path.with_suffix(".mp4")
        elif raw_path.suffix.lower() != ".mp4":
            raw_path = raw_path.with_suffix(".mp4")

        if raw_path.is_absolute():
            candidates.append(raw_path)
        elif input_root is not None:
            candidates.append(input_root / raw_path)

        if any(add_candidate(candidate) for candidate in candidates):
            continue

        if input_root is None:
            raise FileNotFoundError(
                f"Could not resolve listed mp4 entry without --input-root: {entry}"
            )

        if name_index is None:
            name_index = build_mp4_name_index(input_root)
        matches = name_index.get(mp4_name_for_entry(entry), [])
        if len(matches) == 1:
            add_candidate(matches[0])
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous mp4 entry `{entry}` matched multiple files under {input_root}: "
                f"{[str(path) for path in matches[:10]]}"
            )
        else:
            raise FileNotFoundError(f"Could not resolve listed mp4 entry `{entry}`")

    return resolved


def probe_video(video_path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,pix_fmt,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    nb_frames = stream.get("nb_frames")
    nb_frames = int(nb_frames) if nb_frames not in (None, "N/A") else None
    pix_fmt = stream.get("pix_fmt", "unknown")
    frame_rate = stream.get("r_frame_rate", "unknown")
    return {
        "width": width,
        "height": height,
        "nb_frames": nb_frames,
        "pix_fmt": pix_fmt,
        "frame_rate": frame_rate,
    }


def ffmpeg_read_all_frames(video_path: Path, width: int, height: int):
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vsync",
        "0",
        "-",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8,
    )

    frame_size = width * height * 3
    try:
        while True:
            raw = process.stdout.read(frame_size)
            if not raw:
                break
            if len(raw) != frame_size:
                raise RuntimeError(
                    f"Incomplete raw frame read from ffmpeg for {video_path}: "
                    f"expected {frame_size} bytes, got {len(raw)}"
                )
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            yield frame
    finally:
        if process.stdout is not None:
            process.stdout.close()
        stderr = ""
        if process.stderr is not None:
            stderr = process.stderr.read().decode("utf-8", errors="ignore")
            process.stderr.close()
        retcode = process.wait()
        if retcode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path} with code {retcode}: {stderr}")


def build_output_path(video_path: Path, input_root: Path, output_root: Path | None, output_ext: str) -> Path:
    if output_root is None:
        return video_path.with_suffix(output_ext)
    try:
        relative_path = video_path.relative_to(input_root).with_suffix(output_ext)
    except ValueError:
        relative_path = Path(video_path.name).with_suffix(output_ext)
    return output_root / relative_path


def build_temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")


def validate_hdf5_file(hdf5_path: Path, expected_meta: dict) -> tuple[bool, str]:
    try:
        with h5py.File(hdf5_path, "r") as h5_file:
            if "frames" not in h5_file:
                return False, "missing 'frames' dataset"

            frames_dataset = h5_file["frames"]
            num_frames_attr = h5_file.attrs.get("num_frames")
            if num_frames_attr is None:
                return False, "missing 'num_frames' attr"

            num_frames = int(num_frames_attr)
            if len(frames_dataset) != num_frames:
                return False, (
                    f"dataset length mismatch: len(frames)={len(frames_dataset)} "
                    f"!= num_frames={num_frames}"
                )

            expected_nb_frames = expected_meta["nb_frames"]
            if expected_nb_frames is not None and num_frames > expected_nb_frames:
                return False, (
                    f"frame count overflow: num_frames={num_frames} "
                    f"> nb_frames_ffprobe={expected_nb_frames}"
                )

            if int(h5_file.attrs.get("width", -1)) != expected_meta["width"]:
                return False, "width mismatch"
            if int(h5_file.attrs.get("height", -1)) != expected_meta["height"]:
                return False, "height mismatch"
            if num_frames <= 0:
                return False, "empty frames dataset"

        return True, ""
    except Exception as exc:
        return False, repr(exc)


def convert_one_video(
    video_path: Path,
    input_root: Path,
    output_root: Path | None,
    output_ext: str,
    jpeg_quality: int,
    overwrite: bool,
):
    output_path = build_output_path(video_path, input_root, output_root, output_ext)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    meta = probe_video(video_path)
    if output_path.exists() and not overwrite:
        is_valid, reason = validate_hdf5_file(output_path, meta)
        if is_valid:
            return "skipped", output_path

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    lengths = []
    frame_count = 0

    temp_output_path = build_temp_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()

    try:
        with h5py.File(temp_output_path, "w") as h5_file:
            h5_file.attrs["source_mp4"] = str(video_path)
            h5_file.attrs["width"] = meta["width"]
            h5_file.attrs["height"] = meta["height"]
            h5_file.attrs["pix_fmt"] = meta["pix_fmt"]
            h5_file.attrs["frame_rate"] = meta["frame_rate"]
            if meta["nb_frames"] is not None:
                h5_file.attrs["nb_frames_ffprobe"] = meta["nb_frames"]
            h5_file.attrs["jpeg_quality"] = jpeg_quality

            encoded_dataset = h5_file.create_dataset(
                "frames",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
            )

            for frame in ffmpeg_read_all_frames(video_path, meta["width"], meta["height"]):
                ok, encoded = cv2.imencode(".jpg", frame[:, :, ::-1], encode_params)
                if not ok:
                    raise RuntimeError(f"cv2.imencode failed for {video_path} at frame {frame_count}")
                encoded = encoded.reshape(-1).astype(np.uint8)
                encoded_dataset.resize((frame_count + 1,))
                encoded_dataset[frame_count] = encoded
                lengths.append(len(encoded))
                frame_count += 1

            h5_file.attrs["num_frames"] = frame_count
            if meta["nb_frames"] is not None:
                h5_file.attrs["decode_complete"] = int(frame_count == meta["nb_frames"])
            if lengths:
                h5_file.attrs["encoded_bytes_min"] = int(min(lengths))
                h5_file.attrs["encoded_bytes_max"] = int(max(lengths))
                h5_file.attrs["encoded_bytes_mean"] = float(sum(lengths) / len(lengths))

        is_valid, reason = validate_hdf5_file(temp_output_path, meta)
        if not is_valid:
            raise RuntimeError(f"incomplete temporary hdf5: {reason}")

        os.replace(temp_output_path, output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise

    return "converted", output_path


def main():
    args = parse_args()
    input_root = args.input_root.resolve() if args.input_root is not None else None
    output_root = args.output_root.resolve() if args.output_root is not None else None

    if input_root is not None and not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if input_root is None and args.input_list is None:
        raise ValueError("Either --input-root or --input-list must be provided")

    if args.input_list is not None:
        mp4_files = resolve_listed_mp4_files(args.input_list.resolve(), input_root)
        if input_root is None:
            common_parent = Path(os.path.commonpath([str(path.parent) for path in mp4_files]))
            input_root = common_parent.resolve()
    else:
        assert input_root is not None
        mp4_files = list_mp4_files(input_root)
    if not mp4_files:
        print(f"No mp4 files found under {input_root}")
        return

    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")

    converted = 0
    skipped = 0
    failures = []
    submitted = 0
    max_pending = max(args.workers * 2, 1)

    def submit_job(executor, video_path, future_to_video):
        future = executor.submit(
            convert_one_video,
            video_path,
            input_root,
            output_root,
            args.codec_ext,
            args.jpeg_quality,
            args.overwrite,
        )
        future_to_video[future] = video_path

    progress = tqdm(total=len(mp4_files), desc="Converting mp4 -> hdf5", unit="video")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_video = {}
        mp4_iter = iter(mp4_files)

        while len(future_to_video) < max_pending:
            try:
                video_path = next(mp4_iter)
            except StopIteration:
                break
            submit_job(executor, video_path, future_to_video)
            submitted += 1

        progress.set_postfix(
            submitted=submitted,
            converted=converted,
            skipped=skipped,
            failed=len(failures),
        )
        try:
            while future_to_video:
                done, _ = concurrent.futures.wait(
                    future_to_video,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    video_path = future_to_video.pop(future)
                    try:
                        status, _ = future.result()
                        if status == "converted":
                            converted += 1
                        else:
                            skipped += 1
                    except Exception as exc:
                        failures.append((video_path, exc))
                        progress.write(f"[FAIL] {video_path}: {exc}")
                        if args.fail_fast:
                            for pending_future in future_to_video:
                                pending_future.cancel()
                            raise
                    finally:
                        progress.update(1)

                    while len(future_to_video) < max_pending:
                        try:
                            next_video_path = next(mp4_iter)
                        except StopIteration:
                            break
                        submit_job(executor, next_video_path, future_to_video)
                        submitted += 1

                    progress.set_postfix(
                        submitted=submitted,
                        converted=converted,
                        skipped=skipped,
                        failed=len(failures),
                    )
        finally:
            progress.close()

    print(
        f"Done. total={len(mp4_files)} converted={converted} skipped={skipped} failed={len(failures)}"
    )
    if failures:
        print("Failed files:")
        for video_path, exc in failures:
            print(f"  {video_path}: {exc}")


if __name__ == "__main__":
    main()
