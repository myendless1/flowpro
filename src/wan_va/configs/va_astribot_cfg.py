# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_astribot_cfg = EasyDict(__name__='Config: VA astribot base')
va_astribot_cfg.update(va_shared_cfg)

va_astribot_cfg.wan22_pretrained_model_name_or_path = "wam4d-ckpt-1/ckpt_to_infer"

va_astribot_cfg.attn_window = 72
va_astribot_cfg.frame_chunk_size = 2
va_astribot_cfg.env_type = 'robotwin_tshape'

va_astribot_cfg.height = 256
va_astribot_cfg.width = 320
va_astribot_cfg.action_dim = 30
va_astribot_cfg.action_per_frame = 16
va_astribot_cfg.obs_cam_keys = [
    'observation.images.cam_main', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_astribot_cfg.guidance_scale = 5
va_astribot_cfg.action_guidance_scale = 5

va_astribot_cfg.num_inference_steps = 25
va_astribot_cfg.video_exec_step = -1
va_astribot_cfg.action_num_inference_steps = 5

va_astribot_cfg.snr_shift = 5.0
va_astribot_cfg.action_snr_shift = 1.0

va_astribot_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_astribot_cfg.used_action_channel_ids)
] * va_astribot_cfg.action_dim
for i, j in enumerate(va_astribot_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_astribot_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_astribot_cfg.action_norm_method = 'quantiles'
