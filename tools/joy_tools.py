#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025, Astribot Co., Ltd.
# License: BSD 3-Clause License
# -----------------------------------------------------------------------------
# Author: Astribot Team
# -----------------------------------------------------------------------------

"""
File: joy_tools.py
Brief: Description of this module
"""

import rospy
import time
import numpy as np
from sensor_msgs.msg import Joy

class XboxController:
    def __init__(self, mode='chassis_control'):
        # 创建订阅者，订阅/joy话题
        if mode == 'chassis_control':
            rospy.Subscriber('/astribot_joy', Joy, self.joy_callback_for_chassis_vel, queue_size=1)
            self.stop_flag = False
            self.last_key = 0
            self.vel = np.zeros(3)
        elif mode == 'traj_replay':
            rospy.Subscriber('/astribot_joy', Joy, self.joy_callback_for_traj_replay, queue_size=1)

        self.timestamp = None
        rospy.Timer(rospy.Duration(0.5), self.check_joy_value)

    def check_joy_value(self, event):
        if not self.timestamp:
            self.vel = np.zeros(3)
            print("Note that no joystick message is received." \
            "Please check whether rostopic '/astribot_joy' exists,"\
            "whether the joystick driver is started, and whether the joystick is powered.")
        elif np.count_nonzero(self.vel) > 0 and time.time() - self.timestamp > 0.5:
            self.vel = np.zeros(3)
            print("Note that no joystick message is received for 0.5 seconds." \
            "Please check whether rostopic '/astribot_joy' exists,"\
            "whether the joystick driver is started, and whether the joystick is powered.")

    def joy_callback_for_chassis_vel(self, joy):
        # 读取axes数组中的值
        x_vel = joy.axes[1] * 0.5
        y_vel = joy.axes[0] * 0.5
        yaw_vel = joy.axes[3] * 0.5
        self.vel[0] = x_vel
        self.vel[1] = y_vel
        self.vel[2] = yaw_vel
        self.timestamp = time.time()

    def get_vel(self):
        return self.vel

    def joy_callback_for_traj_replay(self, joy):
        pass
