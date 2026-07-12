# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_astribot_centrifuge_multidrop_cfg import va_astribot_centrifuge_multidrop_cfg

va_astribot_centrifuge_multidrop_efficient_cfg = EasyDict()
va_astribot_centrifuge_multidrop_efficient_cfg.update(
    va_astribot_centrifuge_multidrop_cfg
)
va_astribot_centrifuge_multidrop_efficient_cfg.__name__ = (
    "Config: VA astribot centrifuge_multidrop efficient"
)

# Compact 12-layer transformer + shared VAE/text components.
va_astribot_centrifuge_multidrop_efficient_cfg.wan22_pretrained_model_name_or_path = (
    "wam4d-ckpt-1/ckpt_efficient_to_infer"
)
va_astribot_centrifuge_multidrop_efficient_cfg.transformer_variant = "efficient"

# On-disk clip VAE latent cache (read on hit, write on miss during training).
va_astribot_centrifuge_multidrop_efficient_cfg.enable_clip_latent_cache = True
va_astribot_centrifuge_multidrop_efficient_cfg.clip_latent_cache_read = True
va_astribot_centrifuge_multidrop_efficient_cfg.clip_latent_cache_write = True

va_astribot_centrifuge_multidrop_efficient_cfg.save_root = (
    "wam4d-ckpt-1/fast-lingbot-astribot-centrifuge-multidrop-efficient"
)
va_astribot_centrifuge_multidrop_efficient_cfg.infer_transformer_source_root = (
    va_astribot_centrifuge_multidrop_efficient_cfg.save_root
)
