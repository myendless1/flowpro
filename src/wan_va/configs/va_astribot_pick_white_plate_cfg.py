# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_astribot_cfg import va_astribot_cfg

va_astribot_pick_white_plate_cfg = EasyDict()
va_astribot_pick_white_plate_cfg.update(va_astribot_cfg)
va_astribot_pick_white_plate_cfg.__name__ = 'Config: VA astribot pick_white_plate'

va_astribot_pick_white_plate_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]

va_astribot_pick_white_plate_cfg.dataset_paths = [
    '/media/damoxing/fileset/md4d/third_parties/lingbot-va/data/data_4d_wam/astribot-pick_white_plate'
]
va_astribot_pick_white_plate_cfg.empty_emb_path = (
    '/media/damoxing/fileset/md4d/third_parties/lingbot-va/data/data_4d_wam/astribot-pick_white_plate/empty_emb.pt'
)
va_astribot_pick_white_plate_cfg.enable_wandb = True
va_astribot_pick_white_plate_cfg.load_worker = 16
va_astribot_pick_white_plate_cfg.save_interval = 500
va_astribot_pick_white_plate_cfg.gc_interval = 50
va_astribot_pick_white_plate_cfg.cfg_prob = 0.1

va_astribot_pick_white_plate_cfg.learning_rate = 1e-5
va_astribot_pick_white_plate_cfg.beta1 = 0.9
va_astribot_pick_white_plate_cfg.beta2 = 0.95
va_astribot_pick_white_plate_cfg.weight_decay = 0.1
va_astribot_pick_white_plate_cfg.warmup_steps = 10
va_astribot_pick_white_plate_cfg.batch_size = 32
va_astribot_pick_white_plate_cfg.gradient_accumulation_steps = 1
va_astribot_pick_white_plate_cfg.num_steps = 50000

va_astribot_pick_white_plate_cfg.norm_stat = {
    "q01": [
        0.35041555762290955, 0.1362064927816391, 0.8490400314331055,
        -1, -1, -1, -1,
        0.287426233291626, -0.4483684301376343, 0.8074554800987244,
        -1, -1, -1, -1
    ] + [0.] * 14 + [0.6443171501159668, 0.0],
    "q99": [
        0.44893038272857666, 0.23793137073516846, 0.9069141149520874,
        1, 1, 1, 1,
        0.5656144022941589, -0.029807401821017265, 0.9818356037139893,
        1, 1, 1, 1
    ] + [0.] * 14 + [0.6443171501159668, 1.0],
}

va_astribot_pick_white_plate_cfg.save_root = (
    "wam4d-ckpt-1/fast-astribot-all-high-freq-abs"
)
va_astribot_pick_white_plate_cfg.infer_transformer_source_root = (
    va_astribot_pick_white_plate_cfg.save_root
)
