# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_astribot_cfg import va_astribot_cfg

va_astribot_lixinji_cfg = EasyDict()
va_astribot_lixinji_cfg.update(va_astribot_cfg)
va_astribot_lixinji_cfg.__name__ = 'Config: VA astribot lixinji'

va_astribot_lixinji_cfg.dataset_paths = [
    '/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/lixinji'
]
va_astribot_lixinji_cfg.empty_emb_path = (
    '/media/damoxing/datasets/vae4d/lerobot-vae4d-org/astribot/lixinji/empty_emb.pt'
)
va_astribot_lixinji_cfg.require_latents_for_sampling = False
va_astribot_lixinji_cfg.enable_wandb = True
va_astribot_lixinji_cfg.load_worker = 16
va_astribot_lixinji_cfg.save_interval = 500
va_astribot_lixinji_cfg.gc_interval = 50
va_astribot_lixinji_cfg.cfg_prob = 0.1

va_astribot_lixinji_cfg.learning_rate = 1e-5
va_astribot_lixinji_cfg.beta1 = 0.9
va_astribot_lixinji_cfg.beta2 = 0.95
va_astribot_lixinji_cfg.weight_decay = 0.1
va_astribot_lixinji_cfg.warmup_steps = 10
va_astribot_lixinji_cfg.batch_size = 32
va_astribot_lixinji_cfg.gradient_accumulation_steps = 1
va_astribot_lixinji_cfg.num_steps = 50000

va_astribot_lixinji_cfg.norm_stat = {
    "q01": [
        0.3729654848575592, 0.17849580943584442, 0.9257165789604187,
        0.6996694803237915, -0.006048905663192272, -0.04416932910680771,
        0.36922505497932434,
        0.39557161927223206, -0.24359551072120667, 0.7906442284584045,
        0.02622310444712639, -0.03666883334517479, 0.025125138461589813,
        0.7583380341529846,
    ] + [0.] * 14 + [0.0, 0.0],
    "q99": [
        0.3796050548553467, 0.25417426228523254, 0.971005380153656,
        0.9292299151420593, 0.1083352267742157, 0.006976745557039976,
        0.7141262888908386,
        0.47722190618515015, 0.026416461914777756, 0.9428150057792664,
        0.6396191120147705, 0.08129261434078217, 0.4099985659122467,
        0.9874131679534912,
    ] + [0.] * 14 + [0.0, 0.9995884895324707],
}

va_astribot_lixinji_cfg.save_root = (
    "wam4d-ckpt-1/fast-lingbot-astribot-lixinji"
)
va_astribot_lixinji_cfg.infer_transformer_source_root = (
    va_astribot_lixinji_cfg.save_root
)
