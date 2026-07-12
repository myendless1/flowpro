# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_astribot_cfg import va_astribot_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_astribot_train_cfg import va_astribot_train_cfg
from .va_astribot_pick_white_plate_cfg import va_astribot_pick_white_plate_cfg
from .va_astribot_sort_bottles_cfg import va_astribot_sort_bottles_cfg
from .va_astribot_lixinji_cfg import va_astribot_lixinji_cfg
from .va_astribot_sort_bottles_train_cfg import va_astribot_sort_bottles_train_cfg
from .va_astribot_lixinji_train_cfg import va_astribot_lixinji_train_cfg
from .va_astribot_centrifuge_multidrop_cfg import va_astribot_centrifuge_multidrop_cfg
from .va_astribot_centrifuge_multidrop_efficient_cfg import (
    va_astribot_centrifuge_multidrop_efficient_cfg,
)

VA_CONFIGS = {
    'astribot': va_astribot_cfg,
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'astribot_train': va_astribot_train_cfg,
    'astribot_pick_white_plate': va_astribot_pick_white_plate_cfg,
    'astribot_original': va_astribot_pick_white_plate_cfg,
    'astribot_sort_bottles': va_astribot_sort_bottles_cfg,
    'astribot_lixinji': va_astribot_lixinji_cfg,
    'astribot_sort_bottles_train': va_astribot_sort_bottles_train_cfg,
    'astribot_lixinji_train': va_astribot_lixinji_train_cfg,
    'astribot_centrifuge_multidrop': va_astribot_centrifuge_multidrop_cfg,
    'astribot_centrifuge_multidrop_efficient': va_astribot_centrifuge_multidrop_efficient_cfg,
}
