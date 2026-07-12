from __future__ import annotations

"""Wan-VA integration for FlowPRO offline RPRO training."""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from flowpro.training.mixer import batch_counts
from flowpro.training.rpro import RPROConfig, rpro_loss
from flowpro.data.store import _restore_split


class PreferencePool:
    def __init__(self, directories: list[str | Path], seed: int = 0) -> None:
        self.files = [
            path
            for directory in directories
            for path in sorted(Path(directory).glob("*.npz"))
            if path.name != "manifest.npz"
        ]
        self.rng = np.random.default_rng(seed)

    def sample(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not self.files:
            raise ValueError("Preference source contains no augmented .npz samples")
        path = self.files[int(self.rng.integers(0, len(self.files)))]
        with np.load(path) as arrays:
            winner = np.asarray(arrays["winner"], np.float32)
            loser = np.asarray(arrays["loser"], np.float32)
            metadata = json.loads(path.with_suffix(".json").read_text())
            observation = _restore_split(metadata.get("observation", {}), arrays)
        return winner, loser, observation


def _pad_actions(actions: np.ndarray, length: int = 32) -> np.ndarray:
    actions = np.asarray(actions, np.float32)
    if actions.ndim != 2 or actions.shape[1] != 16 or not len(actions):
        raise ValueError(f"Expected non-empty [H,16] actions, got {actions.shape}")
    if len(actions) >= length:
        return actions[:length]
    return np.concatenate([actions, np.repeat(actions[-1:], length - len(actions), axis=0)])


def _observation_image(observation: dict[str, Any], key: str) -> np.ndarray | None:
    candidates = {
        "observation.images.cam_main": ("observation.images.cam_main", "observation.images.cam_high"),
        "observation.images.cam_high": ("observation.images.cam_high", "observation.images.cam_main"),
        "observation.images.cam_left_wrist": ("observation.images.cam_left_wrist",),
        "observation.images.cam_right_wrist": ("observation.images.cam_right_wrist",),
    }.get(key, (key,))
    wam = observation.get("wam4d", observation)
    for candidate in candidates:
        if candidate in wam:
            value = np.asarray(wam[candidate])
            if value.ndim == 3:
                return value.astype(np.uint8)
    return None


def _observation_image_history(observation: dict[str, Any], key: str, length: int) -> np.ndarray | None:
    history = observation.get("wam4d_history", [])
    images = []
    for payload in history:
        image = _observation_image({"wam4d": payload}, key)
        if image is not None:
            images.append(image)
    if not images:
        current = _observation_image(observation, key)
        if current is None:
            return None
        images = [current]
    images = images[-length:]
    if len(images) < length:
        images = [images[0]] * (length - len(images)) + images
    return np.stack(images)


class RPROTrainerMixin:
    """Mixin kept separate so importing FlowPRO does not import CUDA dependencies."""

    def _init_rpro(self, spec: dict[str, Any]) -> None:
        import math
        import torch

        self.rpro_spec = spec
        self.rpro_config = RPROConfig(
            beta=float(spec.get("beta", 1.0)),
            lambda_pro=float(spec.get("lambda_pro", 1.0)),
            lambda_sft=float(spec.get("lambda_sft", 1.0)),
        )
        process_seed = int(spec.get("seed", 0)) + int(self.accelerator.process_index) * 100003
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        self.current_pool = PreferencePool([spec["current_preferences"]], seed=process_seed)
        self.history_pool = PreferencePool(spec.get("historical_preferences", []), seed=process_seed + 1)
        # The frozen model is the exact policy at the start of the round.
        self.reference_transformer = copy.deepcopy(self.accelerator.unwrap_model(self.transformer))
        self.reference_transformer.to(self.device)
        self.reference_transformer.requires_grad_(False)
        self.reference_transformer.eval()
        self._torch = torch
        warmup = int(spec.get("warmup_steps", 100))
        decay = max(1, int(spec.get("cosine_decay_steps", spec["steps"])))
        minimum = float(spec.get("min_learning_rate", 0.0))
        peak = float(spec["learning_rate"])
        floor_ratio = minimum / peak if peak > 0 else 0.0

        def schedule(step: int) -> float:
            if warmup > 0 and step < warmup:
                return max(step, 1) / warmup
            progress = min(max((step - warmup) / decay, 0.0), 1.0)
            return floor_ratio + (1.0 - floor_ratio) * .5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, schedule)

    def _dataset(self):
        loader = self.train_loader
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            raise RuntimeError("Wan-VA training loader does not expose its dataset")
        return dataset

    def _replace_sample(self, batch: dict[str, Any], index: int, sample) -> None:
        import torch

        winner, loser, observation = sample
        winner, loser = _pad_actions(winner), _pad_actions(loser)
        current = observation.get("state_action16")
        if current is None:
            history = observation.get("wam4d", observation).get("observation.state", [])
            current = np.asarray(history[-1], np.float32) if len(history) else loser[0]
        current = np.asarray(current, np.float32).reshape(16)

        dataset = self._dataset()
        references_w = np.concatenate([current[None], winner[:-1]], axis=0)
        references_l = np.concatenate([current[None], loser[:-1]], axis=0)
        winner_model, winner_mask = dataset._action_post_process(winner, references=references_w)
        loser_model, _ = dataset._action_post_process(loser, references=references_l)
        batch["actions"][index] = winner_model
        batch["actions_mask"][index] = winner_mask
        batch["loser_actions"][index] = loser_model

        history = observation.get("wam4d", observation).get("observation.state", [])
        history = np.asarray(history, np.float32)
        if history.ndim != 2 or history.shape[1:] != (16,):
            history = current[None]
        history = history[-dataset.state_history_len:]
        padded = np.zeros((dataset.state_history_len, 16), np.float32)
        state_mask = np.zeros(dataset.state_history_len, bool)
        padded[-len(history):] = history
        state_mask[-len(history):] = True
        batch["state"][index] = dataset._state_post_process(padded, state_mask)
        batch["state_mask"][index] = torch.from_numpy(state_mask)

        frames: dict[str, torch.Tensor] = {}
        fallback = None
        for camera_key in self.config.obs_cam_keys:
            image_history = _observation_image_history(
                observation, camera_key, dataset.RETURN_VIDEO_FRAMES
            )
            if image_history is not None:
                fallback = image_history
            elif fallback is not None:
                image_history = fallback
            if image_history is None:
                # Preserve the SFT frame if a camera was unavailable rather than
                # silently creating a condition with an unrelated shape.
                image_history = np.asarray(batch["video_frames"][index][camera_key])
            frames[camera_key] = torch.from_numpy(np.asarray(image_history).copy())
        batch["video_frames"][index] = frames

    def _get_next_rpro_batch(self):
        batch = super()._get_next_batch()
        import torch

        batch_size = int(batch["actions"].shape[0])
        counts = batch_counts(batch_size, int(self.rpro_spec["round"]))
        batch["loser_actions"] = batch["actions"].clone()
        is_preference = torch.zeros(batch_size, dtype=torch.bool)
        cursor = 0
        for source, count in counts.items():
            if source == "sft":
                cursor += count
                continue
            pool = self.current_pool if source == "current" else self.history_pool
            for index in range(cursor, cursor + count):
                self._replace_sample(batch, index, pool.sample())
                is_preference[index] = True
            cursor += count
        batch["is_preference"] = is_preference
        return batch

    def _prepare_rpro_inputs(self, batch):
        # Winner preparation samples noise and timesteps once.
        winner = super()._prepare_input_dict(batch)
        loser = copy.deepcopy(winner)
        action_w = batch["actions"]
        action_l = batch["loser_actions"]
        # FlowMatchScheduler target is noise - clean action.
        noise = winner["action_dict"]["targets"] + action_w
        timesteps = winner["action_dict"]["timesteps"]
        loser["action_dict"]["value"] = self.train_scheduler_action.add_noise(
            action_l, noise, timesteps, t_dim=2
        )
        loser["action_dict"]["targets"] = self.train_scheduler_action.training_target(
            action_l, noise, timesteps
        )
        return winner, loser

    def _per_example_action_loss(self, inputs, output):
        import torch.nn.functional as F
        from einops import rearrange

        target = inputs["action_dict"]["targets"]
        pred = rearrange(output[1], "b (f n) c -> b c f n 1", f=target.shape[-3])
        batch_size = target.shape[0]
        weight = self.train_scheduler_action.training_weight(
            inputs["action_dict"]["timesteps"].flatten()
        ).reshape(batch_size, 1, -1, 1, 1)
        mask = inputs["action_dict"]["actions_mask"].float()
        loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none") * weight * mask
        return loss.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)

    def _rpro_losses(self, winner, loser, is_preference):
        current_w = self.transformer(winner, train_mode=True)
        current_l = self.transformer(loser, train_mode=True)
        with self._torch.no_grad():
            reference_w = self.reference_transformer(winner, train_mode=True)
            reference_l = self.reference_transformer(loser, train_mode=True)
        cw = self._per_example_action_loss(winner, current_w)
        cl = self._per_example_action_loss(loser, current_l)
        rw = self._per_example_action_loss(winner, reference_w)
        rl = self._per_example_action_loss(loser, reference_l)
        # SFT examples are deliberately represented as identical pairs and go
        # through the same RPRO objective.  Their contrastive gradient cancels,
        # while the proximal and positive regression terms remain (paper Eq. 8).
        return rpro_loss(cw, cl, rw, rl, self.rpro_config)


def run_rpro(spec_path: str | Path, *, config_name: str, experiment_config: str) -> None:
    import math
    from accelerate import Accelerator
    from wan_va.configs import VA_CONFIGS
    from wan_va.configs.experiment import load_experiment_config
    from wan_va.train import Trainer

    spec = json.loads(Path(spec_path).read_text())
    config = load_experiment_config(experiment_config, VA_CONFIGS)
    config.dataset_paths = [spec["sft_dataset"]]
    config.empty_emb_path = str(Path(spec["sft_dataset"]) / "empty_emb.pt")
    reference_root = Path(spec["reference_checkpoint"])
    for candidate in (reference_root / "checkpoints" / "last", reference_root):
        if (candidate / "transformer").exists():
            reference_root = candidate
            break
    if not (reference_root / "transformer").exists():
        raise FileNotFoundError(
            "Reference checkpoint must contain transformer/ or checkpoints/last/transformer: "
            f"{spec['reference_checkpoint']}"
        )
    config.wan22_pretrained_model_name_or_path = spec["base_checkpoint"]
    config.transformer_source_path = str(reference_root / "transformer")
    config.save_root = spec["output"]
    config.num_steps = int(spec["steps"])
    config.batch_size = int(spec["batch_size"])
    config.learning_rate = float(spec["learning_rate"])
    config.enable_wandb = bool(spec.get("enable_wandb", False))
    config.load_worker = int(spec.get("load_worker", 0))
    accelerator = Accelerator(gradient_accumulation_steps=int(config.gradient_accumulation_steps))
    config.rank = accelerator.process_index
    config.local_rank = accelerator.local_process_index
    config.world_size = accelerator.num_processes
    config.learning_rate *= math.sqrt(max(config.world_size / 8, 1.0))

    class WanRPROTrainer(RPROTrainerMixin, Trainer):
        def __init__(self, cfg, acc):
            super().__init__(cfg, acc)
            self._init_rpro(spec)

        def _get_next_batch(self):
            return self._get_next_rpro_batch()

        def train(self):
            from tqdm import tqdm
            self.transformer.train(); self.optimizer.zero_grad()
            bar = tqdm(total=self.config.num_steps, initial=self.step, disable=not self.accelerator.is_main_process)
            while self.step < self.config.num_steps:
                batch = self.convert_input_format(self._get_next_batch())
                winner, loser = self._prepare_rpro_inputs(batch)
                with self.accelerator.accumulate(self.transformer):
                    loss, metrics = self._rpro_losses(winner, loser, batch["is_preference"])
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.transformer.parameters(), 2.0)
                    self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    self.step += 1; bar.update(1)
                    if self.wandb and self.accelerator.is_main_process:
                        self.wandb.log({**metrics, "loss/total": loss.detach()}, step=self.step)
                    if self.step % int(getattr(self.config, "save_interval", 500)) == 0:
                        self.save_checkpoint()
            if self.step % int(getattr(self.config, "save_interval", 500)):
                self.save_checkpoint()
            bar.close(); self.accelerator.wait_for_everyone()

    WanRPROTrainer(config, accelerator).train()
