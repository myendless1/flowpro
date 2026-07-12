# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.

__all__ = [
    "logger",
    "init_logger",
    "get_mesh_id",
    "get_state_history_grid_id",
    "save_async",
    "data_seq_to_patch",
    "FlowMatchScheduler",
    "run_async_server_mode",
    "sample_timestep_id",
    "warmup_constant_lambda",
]


def __getattr__(name):
    if name in {"logger", "init_logger"}:
        from . import logging

        return getattr(logging, name)
    if name == "FlowMatchScheduler":
        from .scheduler import FlowMatchScheduler

        return FlowMatchScheduler
    if name == "run_async_server_mode":
        from .sever_utils import run_async_server_mode

        return run_async_server_mode
    if name in {
        "data_seq_to_patch",
        "get_mesh_id",
        "get_state_history_grid_id",
        "save_async",
        "sample_timestep_id",
        "warmup_constant_lambda",
    }:
        from . import utils

        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
