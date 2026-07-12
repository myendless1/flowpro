# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

va_robotwin_cfg.wan22_pretrained_model_name_or_path = "wam4d-ckpt-1/ckpt_to_infer"

va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 2
va_robotwin_cfg.env_type = 'robotwin_tshape'

va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_cfg.guidance_scale = 5
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 50

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim
for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_cfg.action_norm_method = 'quantiles'
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.361486787, -0.313842416, 0.840322477, -1, -1, -1, -1,
        -0.05316277, -0.312802196, 0.818531752, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.052973952, 0.090577993, 1.086075842, 1, 1, 1, 1,
        0.347916308, 0.083497161, 1.112079728, 1, 1, 1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}
