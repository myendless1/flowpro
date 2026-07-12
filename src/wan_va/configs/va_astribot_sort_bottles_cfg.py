# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_astribot_cfg import va_astribot_cfg

va_astribot_sort_bottles_cfg = EasyDict()
va_astribot_sort_bottles_cfg.update(va_astribot_cfg)
va_astribot_sort_bottles_cfg.__name__ = 'Config: VA astribot sort_bottles'

va_astribot_sort_bottles_cfg.dataset_paths = [
    '/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/sort_bottles'
]
va_astribot_sort_bottles_cfg.empty_emb_path = (
    '/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/sort_bottles/empty_emb.pt'
)
va_astribot_sort_bottles_cfg.require_latents_for_sampling = False
va_astribot_sort_bottles_cfg.enable_wandb = True
va_astribot_sort_bottles_cfg.load_worker = 16
va_astribot_sort_bottles_cfg.save_interval = 500
va_astribot_sort_bottles_cfg.gc_interval = 50
va_astribot_sort_bottles_cfg.cfg_prob = 0.1

va_astribot_sort_bottles_cfg.learning_rate = 1e-5
va_astribot_sort_bottles_cfg.beta1 = 0.9
va_astribot_sort_bottles_cfg.beta2 = 0.95
va_astribot_sort_bottles_cfg.weight_decay = 0.1
va_astribot_sort_bottles_cfg.warmup_steps = 10
va_astribot_sort_bottles_cfg.batch_size = 32
va_astribot_sort_bottles_cfg.gradient_accumulation_steps = 1
va_astribot_sort_bottles_cfg.num_steps = 50000

va_astribot_sort_bottles_cfg.norm_stat = {
    "q01": [
        0.3262523412704468, 0.1473357379436493, 0.8589786291122437,
        0.808113157749176, 0.12879665195941925, 0.02505842223763466,
        0.4405516982078552,
        0.3478730022907257, -0.4162251353263855, 0.8069313764572144,
        0.10854489356279373, -0.05651620775461197, 0.012100216001272202,
        0.45062774419784546,
    ] + [0.] * 14 + [0.0026603825390338898, 0.0],
    "q99": [
        0.37810903787612915, 0.22849710285663605, 0.9233524203300476,
        0.8831123113632202, 0.21067608892917633, 0.11152094602584839,
        0.553494930267334,
        0.5477373600006104, 0.06203742325305939, 0.9769476056098938,
        0.8486496806144714, 0.27957358956336975, 0.5404695272445679,
        0.9424344301223755,
    ] + [0.] * 14 + [0.2953488826751709, 0.9987661242485046],
}

va_astribot_sort_bottles_cfg.save_root = (
    "wam4d-ckpt-1/fast-lingbot-astribot-sort-bottles"
)
va_astribot_sort_bottles_cfg.infer_transformer_source_root = (
    va_astribot_sort_bottles_cfg.save_root
)
