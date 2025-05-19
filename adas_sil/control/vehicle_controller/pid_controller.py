import sys
import os
import numpy as np
import cv2
import keyboard
import time
import pygame


opencv_bin = "C:/Users/manue/opencv/build/x64/vc16/bin"
os.environ["PATH"] = opencv_bin + os.pathsep + os.environ["PATH"]

carla_egg = "C:/Users/manue/Documents/SEA_ME/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.10-py3.7-win-amd64.egg"
sys.path.append(carla_egg)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import cpp_postprocessing.build.pid_controller_py as pid_controller_py
    print("Successfully imported controller modules")
except ImportError as e:
    print(f"Failed to import controller modules: {e}")

import carla

class PIDController:
    def __init__(self, kp, ki, kd, throttle, dt):
        self.pid_controller = pid_controller_py.PidController()
        self.pid_controller.init(kp, ki, kd, throttle, dt)
        print("PID controller initialized")

    def update(self, setpoint, measured_value):


        return output