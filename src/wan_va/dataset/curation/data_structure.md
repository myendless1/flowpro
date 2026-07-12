# Data Structure Contract

This document defines the minimal dataset contract that is compatible with both:

- the current `wam4d/wan_va` raw-video training path
- the older `lingbot-va/wan_wa` latent-based training path

It intentionally describes the smallest safe common subset.

## Scope

This contract targets a single LeRobot-style repo such as:

```text
/path/to/your_dataset_repo/
```

Example:

```text
/media/damoxing/fileset/md4d/third_parties/lingbot-va/data/data_4d_wam/astribot-pick_white_plate
```

## Required Directory Layout

```text
your_dataset_repo/
├── empty_emb.pt
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       │   ├── episode_000000.mp4
│       │   ├── episode_000000.hdf5          # optional but recommended
│       │   └── ...
│       ├── observation.images.cam_left_wrist/
│       │   ├── episode_000000.mp4
│       │   ├── episode_000000.hdf5          # optional but recommended
│       │   └── ...
│       └── observation.images.cam_right_wrist/
│           ├── episode_000000.mp4
│           ├── episode_000000.hdf5          # optional but recommended
│           └── ...
├── latents/
│   └── chunk-000/
│       ├── observation.images.cam_high/
│       │   ├── episode_000000_0_422.pth
│       │   └── ...
│       ├── observation.images.cam_left_wrist/
│       │   ├── episode_000000_0_422.pth
│       │   └── ...
│       └── observation.images.cam_right_wrist/
│           ├── episode_000000_0_422.pth
│           └── ...
└── meta/
    ├── info.json
    ├── tasks.jsonl
    ├── episodes.jsonl
    └── episodes_stats.jsonl
```

## Hard Requirements

The following files are required for compatibility with both versions:

- `empty_emb.pt`
- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `data/chunk-xxx/episode_xxxxxx.parquet`
- `videos/chunk-xxx/<camera_key>/episode_xxxxxx.mp4`
- `latents/chunk-xxx/<camera_key>/episode_xxxxxx_<start>_<end>.pth`

## Recommended But Not Strictly Required

- `videos/.../*.hdf5`
  - The current raw-video loader prefers these when present for faster random frame access.
- `meta/stats.json`
  - Useful for tooling and compatibility hygiene.
- `meta/cache/valid_samples_*.pt`
  - Cache only, never a source-of-truth file.

## Camera Keys

For the current Astribot/RobotWin setup, the expected camera keys are:

- `observation.images.cam_high`
- `observation.images.cam_left_wrist`
- `observation.images.cam_right_wrist`

The directory names under `videos/` and `latents/` must match these keys exactly.

## `meta/info.json`

`meta/info.json` must at least contain valid values for:

- `codebase_version`
- `fps`
- `chunks_size`
- `data_path`
- `video_path`
- `features`
- `total_episodes`
- `total_frames`
- `total_tasks`
- `total_chunks`

Recommended values for a v2.1 LeRobot repo:

```json
{
  "codebase_version": "v2.1",
  "fps": 50,
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
}
```

## `meta/tasks.jsonl`

Each line maps one natural-language task to a numeric `task_index`.

Minimal example:

```json
{"task_index": 0, "task": "pick white plate"}
```

## `meta/episodes.jsonl`

Each line describes one episode and its trainable segments.

Minimal example:

```json
{
  "episode_index": 0,
  "tasks": ["pick white plate"],
  "length": 422,
  "action_config": [
    {
      "start_frame": 0,
      "end_frame": 422,
      "action_text": "pick white plate",
      "skill": ""
    }
  ]
}
```

Rules:

- `episode_index` must match the parquet/video/latent filenames.
- `length` is the original episode frame count.
- Each `action_config` segment defines one valid training segment.
- The latent filename must align with one `action_config` segment.

## `meta/episodes_stats.jsonl`

This file is required by LeRobot metadata loading for `v2.1` datasets.

Each line must contain:

- `episode_index`
- `stats`

The exact `stats` payload may be produced by your existing converter/writer.

## `data/chunk-xxx/episode_xxxxxx.parquet`

This is the action table for one episode.

The common safe contract is:

- it must contain an `action` column
- it should also contain standard LeRobot bookkeeping columns

Recommended columns:

- `observation.state`
- `action`
- `timestamp`
- `frame_index`
- `episode_index`
- `index`
- `task_index`

For the two training paths discussed here, the only hard requirement is:

- `action`

## `videos/`

Each episode should have one file per camera:

```text
videos/chunk-000/observation.images.cam_high/episode_000000.mp4
videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
```

Rules:

- File naming must be `episode_{episode_index:06d}.mp4`.
- The chunk id must match the repo metadata convention.
- The current raw-video path expects all configured cameras to exist.

Optional sidecar:

```text
videos/chunk-000/<camera_key>/episode_000000.hdf5
```

If present, the current raw-video loader may decode frames from `.hdf5` instead of `.mp4`.

## `latents/`

Each latent file must live at:

```text
latents/chunk-{chunk_id}/{camera_key}/episode_{episode_index:06d}_{start_frame}_{end_frame}.pth
```

Example:

```text
latents/chunk-000/observation.images.cam_high/episode_000000_0_422.pth
```

Rules:

- `camera_key` must match the corresponding `videos/` camera key.
- `episode_index` must match the episode.
- `start_frame` and `end_frame` must match one segment from `meta/episodes.jsonl`.

## Latent `.pth` Payload

To stay compatible with both versions, each latent file should be a `dict` with at least:

- `latent`
- `latent_num_frames`
- `latent_height`
- `latent_width`
- `text_emb`
- `frame_ids`
- `start_frame`
- `end_frame`

Recommended full minimal payload:

```python
{
    "latent": Tensor[N, C],
    "latent_num_frames": int,
    "latent_height": int,
    "latent_width": int,
    "video_num_frames": int,
    "text_emb": Tensor[512, 4096],
    "text": str,
    "frame_ids": list[int],
    "start_frame": int,
    "end_frame": int,
    "fps": float,
    "ori_fps": float,
    "camera_key": str,
    "source_video": str,
}
```

### Meaning of the Key Fields

- `latent`
  - Flattened VAE latent tokens.
  - Required by the old latent-based training path.
- `latent_num_frames`
  - Temporal latent frame count.
- `latent_height`
  - Spatial latent grid height.
- `latent_width`
  - Spatial latent grid width.
- `text_emb`
  - Text condition embedding.
  - Required by both versions.
- `frame_ids`
  - Original video frame indices used to build the latent sequence.
  - Required by the old latent-based path for action alignment.
- `start_frame`
  - Segment start frame.
- `end_frame`
  - Segment end frame.

### Current-Version vs Old-Version Use

Current `wam4d/wan_va` version:

- uses `videos/` as visual input
- uses `text_emb` from latent files
- does not use latent visual tokens for training input

Old `lingbot-va/wan_wa` version:

- uses `latent` as visual input
- uses `text_emb` from latent files
- uses `frame_ids` to align actions with latent time steps

Because of that, the common compatible contract must keep both:

- the video files
- the latent payload fields listed above

## `empty_emb.pt`

This is the unconditional / empty text embedding used for CFG-style dropout.

Rules:

- place it at repo root:
  - `your_dataset_repo/empty_emb.pt`
- keep it as a tensor file compatible with the training code

## Minimal Compatibility Summary

If you want one dataset repo to work for both versions, keep:

- `empty_emb.pt`
- `meta/info.json`
- `meta/tasks.jsonl`
- `meta/episodes.jsonl`
- `meta/episodes_stats.jsonl`
- `data/*.parquet` with at least `action`
- `videos/*.mp4` for all configured cameras
- `latents/*.pth` for all configured cameras with:
  - `latent`
  - `latent_num_frames`
  - `latent_height`
  - `latent_width`
  - `text_emb`
  - `frame_ids`
  - `start_frame`
  - `end_frame`

That is the smallest practical common contract.
