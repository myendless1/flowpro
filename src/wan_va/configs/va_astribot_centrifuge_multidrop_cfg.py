# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_astribot_cfg import va_astribot_cfg

va_astribot_centrifuge_multidrop_cfg = EasyDict()
va_astribot_centrifuge_multidrop_cfg.update(va_astribot_cfg)
va_astribot_centrifuge_multidrop_cfg.__name__ = (
    'Config: VA astribot centrifuge_multidrop'
)

va_astribot_centrifuge_multidrop_cfg.dataset_paths = [
    '/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/centrifuge_multidrop'
]
va_astribot_centrifuge_multidrop_cfg.empty_emb_path = (
    '/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/centrifuge_multidrop/empty_emb.pt'
)
va_astribot_centrifuge_multidrop_cfg.require_latents_for_sampling = False
va_astribot_centrifuge_multidrop_cfg.enable_wandb = True
va_astribot_centrifuge_multidrop_cfg.load_worker = 16
va_astribot_centrifuge_multidrop_cfg.save_interval = 500
va_astribot_centrifuge_multidrop_cfg.gc_interval = 50
va_astribot_centrifuge_multidrop_cfg.cfg_prob = 0.1

va_astribot_centrifuge_multidrop_cfg.learning_rate = 1e-5
va_astribot_centrifuge_multidrop_cfg.beta1 = 0.9
va_astribot_centrifuge_multidrop_cfg.beta2 = 0.95
va_astribot_centrifuge_multidrop_cfg.weight_decay = 0.1
va_astribot_centrifuge_multidrop_cfg.warmup_steps = 10
va_astribot_centrifuge_multidrop_cfg.batch_size = 16
va_astribot_centrifuge_multidrop_cfg.gradient_accumulation_steps = 1
va_astribot_centrifuge_multidrop_cfg.num_steps = 30000

va_astribot_centrifuge_multidrop_cfg.norm_stat = {
    "q01": [
        0.1835214936733246, 0.07568305730819702, 0.8383593869209289,
    ] + [-1.] * 4 + [
        0.17685304939746857, -0.4042045760154724, 0.8402429580688476,
    ] + [-1.] * 4 + [0.] * 14 + [-1., -1.],
    "q99": [
        0.35924355864524843, 0.2611021399497986, 0.9642798900604248,
    ] + [1.] * 4 + [
        0.5371230578422547, -0.016670119017362622, 0.9971221375465393,
    ] + [1.] * 4 + [0.] * 14 + [1., 1.],
}

va_astribot_centrifuge_multidrop_cfg.save_root = (
    "wam4d-ckpt-1/fast-lingbot-astribot-centrifuge-multidrop"
)
va_astribot_centrifuge_multidrop_cfg.infer_transformer_source_root = (
    va_astribot_centrifuge_multidrop_cfg.save_root
)
