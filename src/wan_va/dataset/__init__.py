# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

__all__ = ["MultiLatentLeRobotDataset"]


def __getattr__(name):
    if name == "MultiLatentLeRobotDataset":
        from .lerobot_latent_dataset import MultiLatentLeRobotDataset

        return MultiLatentLeRobotDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
